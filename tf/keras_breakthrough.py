from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
from ggp import run_ggp
import little_golem as lg
import mcts
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
  all_data, all_rewards = lg.load_data()
  samples = len(all_data);
  
  log('  Sorting data')
  states = sorted(all_data.keys())
  nn_states = np.empty((samples, 8, 8, 6), dtype=nn.DATA_TYPE)
  action_probs = np.empty((samples, bt.ACTIONS), dtype=nn.DATA_TYPE)
  rewards = np.empty((samples, 1), dtype=nn.DATA_TYPE)
  for ii, state in enumerate(states):
    policy.convert_state(state, nn_states[ii:ii+1].reshape((8, 8, 6)))
    np.copyto(action_probs[ii:ii+1].reshape(bt.ACTIONS), all_data[state])
    rewards[ii] = all_rewards[state]
    
  # Split into training and validation sets.
  # Use a fixed seed to get reproducibility over different runs.  This is especially important when resuming
  # training.  Otherwise the evaluation data in the 2nd run is data that the network has already seen in the 1st.
  log('  Shuffling data consistently')
  np.random.seed(0)
  shuffle_together(nn_states, action_probs, rewards)

  log('  Splitting data')
  split_point = int(samples * 0.8)
  train_states = nn_states[:split_point]
  train_action_probs = action_probs[:split_point]
  train_rewards = rewards[:split_point]
  eval_states = nn_states[split_point:]
  eval_action_probs = action_probs[split_point:]
  eval_rewards = rewards[split_point:]
  log('  %d training samples vs %d evaluation samples' % (split_point, samples - split_point))
  
  log('Training')
  policy.train(train_states, train_action_probs, train_rewards, eval_states, eval_action_probs, eval_rewards, epochs=100)  
  
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
    state_value = policy.get_state_value(trial_state)
    log("Play %s with probability %f (%s) for win rate %d%% and state-value %d%%" % 
        (convert_index_to_move(index, state.player), 
         prediction[index], 
         '*' if greedy_win else '!',
         int(win_rate * 100),
         int(state_value * 50) + 50)) # Scale from [-1,+1] to [0,100]
    
  log('MCTS evaluation...')
  tree = mcts.MCTSTrainer(policy)
  tree.prepare_for_eval(state)
  tree.iterate(state=state, num_iterations=50000)
  
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
  
def evaluate_for(initial_state, policy, player, num_rollouts=1000):
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
      
    # Now it's the other player's turn, so swap policies.
    current_policy, other_policy = other_policy, current_policy

  wins = 0
  p0_wins = 0
  p1_wins = 0
  for state in states[0::2]:
    if state.is_win_for(0):
      wins += 1
      p0_wins += 1
  for state in states[1::2]:
    if state.is_win_for(1):
      wins += 1
      p1_wins += 1
  
  log('Wins, Wins as p0, Wins as p1 = %d, %d, %d' % (wins, p0_wins, p1_wins))
  return wins / num_matches
  
''' Shuffle a set of lists keeping matching indices aligned '''
def shuffle_together(list1, list2, list3):
  rng_state = np.random.get_state()
  np.random.shuffle(list1)
  np.random.set_state(rng_state)
  np.random.shuffle(list2)
  np.random.set_state(rng_state)
  np.random.shuffle(list3)

def reinforce():
  #policy = CNPolicy(checkpoint=PRIMARY_CHECKPOINT)
  policy = CNPolicy()
  policy.compile(lr=0.01)
  policy.save(filename='pre_reinforcement.hdf5')
  mcts.MCTSTrainer(policy).self_play()
  policy.save(filename='post_reinforcement.hdf5')
  log('Evaluating reinforced policy against original')
  original_policy = CNPolicy(checkpoint='pre_reinforcement.hdf5')
  win_rate = compare_policies_in_parallel(policy, original_policy)
  log('Reinforced policy won %d%% of the matches' % (int(win_rate * 100)))
      
def ggp():
  run_ggp()
  
def main(argv):
  quit = False
  while not quit:
    log('', end='')
    cmd = input("** Running with Keras **  Train (t), reinforce (r), predict (p), GGP (g) or quit (q)? ").lower()
    if cmd == 'train' or cmd == 't':
      train()
    elif cmd == 'reinforce' or cmd == 'r':
      reinforce()
    elif cmd == 'predict' or cmd == 'p':
      predict()
    if cmd == 'ggp' or cmd == 'g':
      ggp()
    elif cmd == 'quit' or cmd == 'q':
      quit= True
  log('Done')
  
if __name__ == "__main__":
  main(sys.argv)
