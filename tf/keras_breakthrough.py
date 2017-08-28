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
import tempfile

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from logger import log, log_progress
from policy import CNPolicy
from tensorflow.python.training.saver import checkpoint_exists

LOG_DIR = os.path.join(tempfile.gettempdir(), 'bt', 'keras')

def train():
  log('Creating model')
  policy = CNPolicy()

  # Load the data  
  all_data = lg.load_data(min_rounds=20)
  samples = len(all_data);
  
  log('  Sorting data')
  states = sorted(all_data.keys())
  nn_states = np.empty((samples, 8, 8, 6), dtype=nn.DATA_TYPE)
  action_probs = np.empty((samples, bt.ACTIONS), dtype=nn.DATA_TYPE)
  ii = 0
  for state in states:
    policy.convert_state(state, nn_states[ii:ii+1].reshape((8, 8, 6)))
    policy.copyto(action_probs[ii:ii+1].reshape(bt.ACTIONS), all_data[state])
    ii += 1
    
  # Split into training and validation sets.
  # Use a fixed seed to get reproducibility over different runs.  This is especially important when resuming
  # training.  Otherwise the evaluation data in the 2nd run is data that the network has already seen in the 1st.
  log('  Shuffling data consistently')
  np.random.seed(0)
  rng_state = np.random.get_state()
  np.random.shuffle(nn_states)
  np.random.set_state(rng_state)
  np.random.shuffle(action_probs)

  log('  Splitting data')
  split_point = int(samples * 0.8)
  train_states = nn_states[:split_point]
  train_action_probs = action_probs[:split_point]
  eval_states = nn_states[split_point:]
  eval_action_probs = action_probs[split_point:]
  log('  %d training samples vs %d evaluation samples' % (split_point, samples - split_point))
  
  log('Training')
  epochs = 40
  # !! ARR "model" doesn't exist any more
  history = model.fit(train_states,
                      train_action_probs,
                      validation_data=(eval_states, eval_action_probs),
                      epochs=epochs,
                      batch_size=1024,
                      callbacks=[TensorBoard(log_dir=LOG_DIR, write_graph=True),
                                 ModelCheckpoint(filepath=os.path.join(LOG_DIR, 'model.epoch{epoch:02d}.hdf5')),
                                 ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=3, verbose=1)])  
  
def convert_index_to_move(index, player):
  move = bt.convert_index_to_move(index, player)
  return lg.encode_move(move)

def predict():
  # Load the trained model  
  checkpoint = os.path.join(LOG_DIR, 'model.epoch99.hdf5') # !! ARR Don't hard-code
  policy = CNPolicy(checkpoint=checkpoint)
  
  # Advance the game to the desired state
  history = input('Input game history: ')
  state = bt.Breakthrough()
  for part in history.split(' '):
    if len(part) == 5:
      state = bt.Breakthrough(state, lg.decode_move(part))
  print(state)
  
  desired_reward = 1 if state.player == 0 else -1
  
  # Predict the next move
  prediction = policy.get_action_probs_for_state(state)
  sorted_indices = np.argsort(prediction)[::-1][0:5]
  for index in sorted_indices:
    trial_state = bt.Breakthrough(state, bt.convert_index_to_move(index, state.player))
    greedy_win = rollout(policy, trial_state, greedy=True) == desired_reward
    avg_score = evaluate(trial_state, policy)
    log("Play %s with probability %f (%s) for avg. score %f" % 
        (convert_index_to_move(index, state.player), 
         prediction[index], 
         '*' if greedy_win else '!',
         avg_score))
    
  _ = input('Press enter to play on')
  rollout(policy, state, greedy=True, show=True)

def rollout(policy, state, greedy=False, show=False):
  while not state.terminated:
    prediction = policy.get_action_probs_for_state(state)
    # Pick the next action, either greedily or weighted by the policy
    index = -1
    if greedy:
      index = np.argmax(prediction)
    else:
      legal = False
      while not legal:
        index = np.random.choice(bt.ACTIONS, p=prediction)
        legal = state.is_legal(bt.convert_index_to_move(index, state.player))
    str_move = convert_index_to_move(index, state.player)
    if show: print('Playing %s' % str_move)
    state = bt.Breakthrough(state, lg.decode_move(str_move))
    if show: print(state)
  return state.reward

def evaluate(state, policy, num_rollouts=10):
  # Run sample games and collect the total reward
  total_reward = 0
  for _ in range(num_rollouts):
    total_reward += rollout(policy, state, greedy=False)
  return total_reward / num_rollouts

''' Rollout a game from the start with 2 different policies (one per player). '''
def rollout2(policies):
  state = bt.Breakthrough()
  while not state.terminated:
    # Pick the next action, according to the weighted by the policy (skipping illegal actions)
    prediction = policies[state.player].get_action_probs_for_state(state)
    index = -1
    legal = False
    while not legal:
      index = np.random.choice(bt.ACTIONS, p=prediction)
      legal = state.is_legal(bt.convert_index_to_move(index, state.player))
    state = bt.Breakthrough(state, bt.convert_index_to_move(index, state.player))
  return state.reward

def compare_models_in_parallel(our_model, their_model, num_matches = 100):
  states = [bt.Breakthrough() for _ in range(num_games)]
  wins = 0

  # We start all the even numbered games, they start all the odd ones.  Advance all the odd numbered games by a turn
  # so that it's our turn in every game.
  odd_states = states[1::2]
  moves = opponent.get_moves(states[1::2])
  for st, mv in zip(odd_states, moves):
      st.do_move(mv)

  current_model = our_model
  other_model = their_model
  idxs_to_unfinished_states = {i: states[i] for i in range(num_games)}
  while len(idxs_to_unfinished_states) > 0:
      # Get next moves by current player for all unfinished states.
      moves = current.get_moves(idxs_to_unfinished_states.values())
      just_finished = []
      # Do each move to each state in order.
      for (idx, state), mv in zip(idxs_to_unfinished_states.iteritems(), moves):
          state.do_move(mv)
          if state.is_end_of_game:
              learner_won[idx] = state.get_winner() == learner_color[idx]
              just_finished.append(idx)

      # Remove games that have finished from dict.
      for idx in just_finished:
          del idxs_to_unfinished_states[idx]

      # Now the it's the other player's turn, so swap models.
      current_model, other_model = other_model, current_model

  # Return the win ratio.
  wins = sum(state.get_winner() == pc for (state, pc) in zip(states, learner_color))
  return float(wins) / num_games
  
def reinforce(num_matches = 100):
  # !! ARR For now, just compare 2 models
  
  # Load the trained models
  our_model = CNPolicy(checkpoint=os.path.join(LOG_DIR, 'model.epoch17.hdf5'))
  their_model = CNPolicy(checkpoint=os.path.join(LOG_DIR, 'model.epoch17.hdf5'))
  
  # Player the models against each other
  log('Comparing models', end='')
  wins = 0
  for match in range(num_matches):
    if match % 2 == 0:
      # We'll play first
      if rollout2([our_model, their_model]) == 1: wins += 1
    else:
      # We'll play second - and want the first player to lose
      if rollout2([their_model, our_model]) == -1: wins += 1
    log_progress()
  print('')
  log('Won %d%% of the matches (%d of %d)' % (int(wins * 100 / num_matches), wins, num_matches))

def main(argv):
  handled = False
  while not handled:
    cmd = input("** Running with Keras **  Train (t), predict (p) or reinforce (r)? ").lower()
    if cmd == 'train' or cmd == 't':
      handled = True
      train()
    elif cmd == 'predict' or cmd == 'p':
      handled = True
      predict()
    elif cmd == 'reinforce' or cmd == 'r':
      handled = True
      reinforce()
  log('Done')
  
if __name__ == "__main__":
  main(sys.argv)
