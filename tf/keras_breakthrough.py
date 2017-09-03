# !! ARR Ideas for improvement
#
# - Add reflections to the dataset (being careful about interactions with deduplication)
# - Measure prediction speed (how would it do in a Monte Carlo rollout?)
# - Parallel rollouts

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import little_golem as lg
import nn
import numpy as np
import os
import sys

from logger import log, log_progress
from policy import CNPolicy

PRIMARY_CHECKPOINT = 'model.epoch99.hdf5'

def train():
  log('Creating policy')
  policy = CNPolicy()

  # Load the data  
  all_data = lg.load_data(min_rounds=20)
  samples = len(all_data);
  
  log('  Sorting data')
  states = sorted(all_data.keys())
  nn_states = np.empty((samples, 8, 8, 6), dtype=nn.DATA_TYPE)
  action_probs = np.empty((samples, bt.ACTIONS), dtype=nn.DATA_TYPE)
  for ii, state in enumerate(states):
    policy.convert_state(state, nn_states[ii:ii+1].reshape((8, 8, 6)))
    np.copyto(action_probs[ii:ii+1].reshape(bt.ACTIONS), all_data[state])
    
  # Split into training and validation sets.
  # Use a fixed seed to get reproducibility over different runs.  This is especially important when resuming
  # training.  Otherwise the evaluation data in the 2nd run is data that the network has already seen in the 1st.
  log('  Shuffling data consistently')
  np.random.seed(0)
  pair_shuffle(nn_states, action_probs)

  log('  Splitting data')
  split_point = int(samples * 0.8)
  train_states = nn_states[:split_point]
  train_action_probs = action_probs[:split_point]
  eval_states = nn_states[split_point:]
  eval_action_probs = action_probs[split_point:]
  log('  %d training samples vs %d evaluation samples' % (split_point, samples - split_point))
  
  log('Training')
  policy.train(train_states, train_action_probs, eval_states, eval_action_probs)  
  
def convert_index_to_move(index, player):
  move = bt.convert_index_to_move(index, player)
  return lg.encode_move(move)

def predict():
  # Load the trained policy
  policy = CNPolicy(checkpoint=PRIMARY_CHECKPOINT)
  
  # Advance the game to the desired state
  history = input('Input game history: ')
  state = bt.Breakthrough()
  for part in history.split(' '):
    if len(part) == 5:
      state = bt.Breakthrough(state, lg.decode_move(part))
  print(state)
  
  desired_reward = 1 if state.player == 0 else -1
  
  # Predict the next move
  prediction = policy.get_action_probs(state)
  sorted_indices = np.argsort(prediction)[::-1][0:5]
  for index in sorted_indices:
    trial_state = bt.Breakthrough(state, bt.convert_index_to_move(index, state.player))
    greedy_win = rollout(policy, trial_state, greedy=True) == desired_reward
    win_rate = evaluate_for(trial_state, policy, state.player)
    log("Play %s with probability %f (%s) for win rate %d%%" % 
        (convert_index_to_move(index, state.player), 
         prediction[index], 
         '*' if greedy_win else '!',
         int(win_rate * 100)))
    
  _ = input('Press enter to play on')
  rollout(policy, state, greedy=True, show=True)

def rollout(policy, state, greedy=False, show=False):
  state = bt.Breakthrough(state)
  while not state.terminated:
    # Pick the next action, either greedily or weighted by the policy
    index = -1
    if greedy:
      index = get_best_legal(state, policy)
    else:
      index = policy.get_action_index(state)
    str_move = convert_index_to_move(index, state.player)
    if show: print('Playing %s' % str_move)
    state.apply(lg.decode_move(str_move))
    if show: print(state)
  return state.reward

def get_best_legal(state, policy):
  index = -1
  action_probs = policy.get_action_probs(state)
  legal = False
  while not legal:
    if index != -1:
      action_probs[index] = 0
    index = np.argmax(action_probs)
    legal = state.is_legal(bt.convert_index_to_move(index, state.player))
  return index
  
def evaluate_for(initial_state, policy, player, num_rollouts=100):
  states = [bt.Breakthrough(initial_state) for _ in range(num_rollouts)]
  
  move_made = True
  while move_made:
    move_made = False
    for (state, action) in zip(states, policy.get_action_indicies(states)):
      if not state.terminated:
        state.apply(bt.convert_index_to_move(action, state.player))
        move_made = True
  
  wins = 0
  for state in states:
    if state.is_win_for(player): wins += 1
  return wins / num_rollouts

def compare_policies_in_parallel(our_policy, their_policy, num_matches = 100):
  states = [bt.Breakthrough() for _ in range(num_matches)]

  # We start all the even numbered games, they start all the odd ones.  Advance all the odd numbered games by a turn
  # so that it's our turn in every game.
  for state in states[1::2]:
    index = their_policy.get_action_index(state)
    state.apply(bt.convert_index_to_move(index, state.player))

  # Rollout all the games to completion.
  current_policy = our_policy
  other_policy = their_policy
  
  move_made = True
  while move_made:
    # Compute the next move for each game in parallel
    move_made = False    
    for (state, action) in zip(states, current_policy.get_action_indicies(states)):
      if not state.terminated:
        state.apply(bt.convert_index_to_move(action, state.player))
        move_made = True
      
    # Now the it's the other player's turn, so swap policies.
    current_policy, other_policy = other_policy, current_policy

  wins = 0
  for state in states[0::2]:
    if state.reward == 1: wins += 1
  for state in states[1::2]:
    if state.reward == -1: wins += 1
  
  return wins / num_matches
  
def reinforce_in_parallel(our_policy, their_policy, num_matches = 100):
  match_states = [bt.Breakthrough() for _ in range(num_matches)]

  # We start all the even numbered matches, they start all the odd ones.  Advance all the odd numbered ones by a turn
  # so that it's our turn in every match.
  for match_state in match_states[1::2]:
    index = their_policy.get_action_index(match_state)
    match_state.apply(bt.convert_index_to_move(index, match_state.player))

  # For each match, record the states encountered, actions taken and final outcomes.
  training_states  = [[] for _ in range(num_matches)]
  training_actions = [[] for _ in range(num_matches)]
  training_rewards = [None] * num_matches
  
  # Rollout all the games to completion.
  current_policy = our_policy
  other_policy = their_policy
  
  move_made = True
  while move_made:
    # Compute the next move for each game in parallel
    move_made = False    
    for (index, state, action) in zip(range(num_matches), 
                                      match_states, 
                                      current_policy.get_action_indicies(match_states)):
      if not state.terminated:
        if current_policy is our_policy:
          training_states[index].append(bt.Breakthrough(state))
          training_actions[index].append(action)
        state.apply(bt.convert_index_to_move(action, state.player))
        move_made = True
      
    # Now the it's the other player's turn, so swap policies.
    current_policy, other_policy = other_policy, current_policy

  # Calculate the reward from the point of view of our_policy.
  winning_states = []
  winning_actions = []
  losing_states = []
  losing_actions = []
  wins = 0
  for index, final_state in enumerate(match_states):
    our_player = index % 2
    training_rewards[index] = final_state.reward * (1 if (our_player == 0) else -1)
    if final_state.is_win_for(our_player):
      wins += 1
      winning_states.extend(training_states[index])
      winning_actions.extend(training_actions[index])
    else:
      losing_states.extend(training_states[index])
      losing_actions.extend(training_actions[index])

  # Shuffle states to break similarity between adjacent states in training set.
  # !! ARR Would really like to not put through all the +ve and then all the -ve samples, but don't know how to
  # !! ARR set lr per sample. 
  pair_shuffle(losing_states, losing_actions)
  pair_shuffle(winning_states, winning_actions)

  # Train the policy via reinforcement learning    
  our_policy.reinforce(losing_states, losing_actions, -1.0)
  our_policy.reinforce(winning_states, winning_actions, 1.0)

  return wins / num_matches

''' Shuffle a pair of list keeping matching indicies aligned '''
def pair_shuffle(list1, list2):
  rng_state = np.random.get_state()
  np.random.shuffle(list1)
  np.random.set_state(rng_state)
  np.random.shuffle(list2)
    
def reinforce(num_matches=16, num_eval_matches=1000):
  log('Training policy by (parallel) RL')
  # Load the trained policies
  our_policy = CNPolicy(checkpoint=PRIMARY_CHECKPOINT)
  their_policy = CNPolicy(checkpoint=PRIMARY_CHECKPOINT)

  our_policy.prepare_for_reinforcement()
  for _ in range(1000):    
    for _ in range(100):
      pre_train_win_rate = reinforce_in_parallel(our_policy, their_policy, num_matches)
      log('Our policy won %0.1f%% of the matches' % (pre_train_win_rate * 100))
    in_depth_win_rate = compare_policies_in_parallel(our_policy, their_policy, num_matches=num_eval_matches)
    log('In-depth evaluation yields %0.1f%% win rate' % (in_depth_win_rate * 100))
  
  # !! ARR This will overfit to beating their_policy.  Need to train against self + other epochs. 

def main(argv):
  quit = False
  while not quit:
    log('', end='')
    cmd = input("** Running with Keras **  Train (t), predict (p), reinforce (r) or quit (q)? ").lower()
    if cmd == 'train' or cmd == 't':
      train()
    elif cmd == 'predict' or cmd == 'p':
      predict()
    elif cmd == 'reinforce' or cmd == 'r':
      reinforce()
    elif cmd == 'quit' or cmd == 'q':
      quit= True
  log('Done')
  
if __name__ == "__main__":
  main(sys.argv)
