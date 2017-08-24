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

from keras.callbacks import TensorBoard, ModelCheckpoint
from logger import log, log_progress

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
  log_dir = os.path.join(tempfile.gettempdir(), 'bt', 'keras')  
  history = model.fit(train_states,
                      train_action_probs,
                      epochs=30,
                      batch_size=16,
                      callbacks=[TensorBoard(log_dir=log_dir, write_graph=True),
                                 ModelCheckpoint(filepath=os.path.join(log_dir, 'model.epoch{epoch:02d}.hdf5'))],
                      verbose=0)
  
  log('Evaluating')
  (loss, accuracy) = model.evaluate(eval_states, eval_action_probs, verbose=0)
  log('accuracy=%f (loss=%f)' % (accuracy, loss))      
    
  log('Done')
  
def main(argv):
  handled = False;
  train()
  return

  while not handled:
    cmd = input("** Running with Keras **  Train (t), predict (p) or evaluate (e)? ").lower()
    if cmd == 'train' or cmd == 't':
      handled = True
      train()
    elif cmd == 'predict' or cmd == 'p':
      handled = True
      predict()
    elif cmd == 'evaluate' or cmd == 'eval' or cmd == 'e':
      handled = True
      evaluate()
  
if __name__ == "__main__":
  main(sys.argv)
