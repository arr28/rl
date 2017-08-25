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
from keras.models import load_model
from logger import log, log_progress
from tensorflow.python.training.saver import checkpoint_exists

LOG_DIR = os.path.join(tempfile.gettempdir(), 'bt', 'keras')

def train():
  log('Creating model')
  model = nn.create_model()

  # Load the data  
  all_data = lg.load_data(min_rounds=20)
  samples = len(all_data);
  
  log('  Sorting data')
  states = sorted(all_data.keys())
  nn_states = np.empty((samples, 8, 8, 6), dtype=nn.DATA_TYPE)
  action_probs = np.empty((samples, bt.ACTIONS), dtype=nn.DATA_TYPE)
  ii = 0
  for state in states:
    nn.convert_state(state, nn_states[ii:ii+1].reshape((8, 8, 6)))
    np.copyto(action_probs[ii:ii+1].reshape(bt.ACTIONS), all_data[state])
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
  history = model.fit(train_states,
                      train_action_probs,
                      validation_data=(eval_states, eval_action_probs),
                      epochs=epochs,
                      batch_size=1024,
                      callbacks=[TensorBoard(log_dir=LOG_DIR, write_graph=True),
                                 ModelCheckpoint(filepath=os.path.join(LOG_DIR, 'model.epoch{epoch:02d}.hdf5')),
                                 ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=3, verbose=1)])
  
  log('Done')
  
def convert_index_to_move(index, player):
  dir = (index % 3) - 1
  index = int(index / 3)
  src_col_ix = index % 8
  src_row_ix = int(index / 8)
  dst_col_ix = src_col_ix + dir
  dst_row_ix = src_row_ix + 1 if player == 0 else src_row_ix - 1
   
  src_col = chr(src_col_ix + ord('a'))
  src_row = chr(src_row_ix + ord('1'))
  dst_col = chr(dst_col_ix + ord('a'))
  dst_row = chr(dst_row_ix + ord('1'))
  
  # print("%s%s %s" % (src_col, src_row, direction))
  return format("%s%s-%s%s" % (src_col, src_row, dst_col, dst_row))

def greedy_rollout(model, state):
  while not state.terminated:
    predictions = model.predict(nn.convert_state(state).reshape((1, 8, 8, 6)))
    for _, prediction in enumerate(predictions):
      index = np.argmax(prediction) # Always pick the best action
      str_move = convert_index_to_move(index, state.player)
      print(state)
      print("Play %s with probability %f" % (str_move, prediction[index]))
      state = bt.Breakthrough(state, lg.decode_move(str_move))
  print("Game complete.  Final state...\n")
  print(state)
  return state.reward

def predict():
  
  # Load the trained model  
  checkpoint = os.path.join(LOG_DIR, 'model.epoch17.hdf5') # !! ARR Don't hard-code
  model = load_model(checkpoint)
  
  # Advance the game to the desired state
  history = input('Input game history: ')
  state = bt.Breakthrough()
  for part in history.split(' '):
    if len(part) == 5:
      state = bt.Breakthrough(state, lg.decode_move(part))

  # Predict the next move
  predictions = model.predict(nn.convert_state(state).reshape((1, 8, 8, 6)))
  for _, prediction in enumerate(predictions):
    sorted_indices = np.argsort(prediction)[::-1][0:5]
    for index in sorted_indices:
      log("Play %s with probability %f" % (convert_index_to_move(index, state.player), prediction[index]))
      
    _ = input('Press enter to play on')
    greedy_rollout(model, state)
        
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
      evaluate()
  
if __name__ == "__main__":
  main(sys.argv)
