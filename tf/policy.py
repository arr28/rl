from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import keras.backend as K
import nn
import numpy as np
import os
import tempfile

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam

LOG_DIR = os.path.join(tempfile.gettempdir(), 'bt', 'keras')

class CNPolicy:
  
  def __init__(self, num_conv_layers=3, checkpoint=None):
    if checkpoint:
      self._model = load_model(os.path.join(LOG_DIR, checkpoint))
    else:
      self._model = Sequential()
      self._model.add(Conv2D(input_shape=(8, 8, 6), filters=64, kernel_size=[5,5], padding='same', activation='relu'))
      for _ in range(num_conv_layers - 1):
        self._model.add(Conv2D(filters=128, kernel_size=[3,3], padding='same', activation='relu'))
      self._model.add(Flatten())
      self._model.add(Dropout(0.9))
      self._model.add(Dense(bt.ACTIONS, activation='softmax'))  

  def train(self, train_states, train_action_probs, eval_states, eval_action_probs, epochs=40):
    self._model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    history = self._model.fit(train_states,
                              train_action_probs,
                              validation_data=(eval_states, eval_action_probs),
                              epochs=epochs,
                              batch_size=1024,
                              callbacks=[TensorBoard(log_dir=LOG_DIR, write_graph=True),
                                         ModelCheckpoint(filepath=os.path.join(LOG_DIR, 'model.epoch{epoch:02d}.hdf5')),
                                         ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=3, verbose=1)])
        
  def convert_state(self, state, nn_input=np.empty((8, 8, 6), dtype=nn.DATA_TYPE)):
    # 6-channel 8x8 network
    # 
    # 1. Player's own pieces
    # 2. Opponent's pieces
    # 3. Empty squares
    # 4. Player to play
    # 5. Zeros
    # 6. Ones
  
    if state.player == 0:
      np.copyto(nn_input[:,:,0:1].reshape(8, 8), np.equal(state.grid, np.zeros((8,8))))
      np.copyto(nn_input[:,:,1:2].reshape(8, 8), np.equal(state.grid, np.ones((8,8))))
    else:
      np.copyto(nn_input[:,:,0:1].reshape(8, 8), np.equal(state.grid, np.ones((8,8))))
      np.copyto(nn_input[:,:,1:2].reshape(8, 8), np.equal(state.grid, np.zeros((8,8))))
  
    np.copyto(nn_input[:,:,2:3].reshape(8, 8), np.equal(state.grid, np.full((8,8), 2)))
    np.copyto(nn_input[:,:,3:4].reshape(8, 8), np.full((8,8), state.player))
    np.copyto(nn_input[:,:,4:5].reshape(8, 8), np.zeros((8, 8), dtype=nn.DATA_TYPE))
    np.copyto(nn_input[:,:,5:6].reshape(8, 8), np.ones((8, 8), dtype=nn.DATA_TYPE))
    
    return nn_input

  def get_action_probs(self, state):
    predictions = self._model.predict(self.convert_state(state).reshape((1, 8, 8, 6)))
    for _, prediction in enumerate(predictions):
      return prediction
    
  def _get_weighted_legal(self, state, action_probs):
    index = -1
    legal = False
    while not legal:
      index = np.random.choice(bt.ACTIONS, p=action_probs)
      legal = state.is_legal(bt.convert_index_to_move(index, state.player))
    return index
    
  def get_action_index(self, state):
    action_probs = self.get_action_probs(state)
    return self._get_weighted_legal(state, action_probs)
  
  def get_action_indicies(self, states):
    batch_input = np.empty((len(states), 8, 8, 6), dtype=nn.DATA_TYPE)
    for ii, state in enumerate(states):
      self.convert_state(state, batch_input[ii:ii+1].reshape((8, 8, 6)))
    predictions = self._model.predict(batch_input)
    actions = []
    for state, action_probs in zip(states, predictions):
      if state.terminated:
        actions.append(-1)
      else:      
        actions.append(self._get_weighted_legal(state, action_probs))
    return actions
  
  def reinforce(self, states, actions, reward):
    samples = len(states)    
    nn_states = np.empty((samples, 8, 8, 6), dtype=nn.DATA_TYPE)
    nn_actions = np.zeros((samples, bt.ACTIONS), dtype=nn.DATA_TYPE)
    for ii, state in enumerate(states):
      self.convert_state(state, nn_states[ii:ii+1].reshape((8, 8, 6)))
      nn_actions[ii][actions[ii]] = 1
    
    self._model.compile(loss=reinforcement_loss, optimizer=SGD(lr = 0.01 * reward)) # !! ARR Too hot?
    self._model.train_on_batch(nn_states, nn_actions)
    
''' ========== Static methods ========== '''
def reinforcement_loss(y_true, y_pred):
    '''Loss function for the REINFORCE algorithm.
    
    y_true is a one-hot vector of the action taken.
    y_pred is the network's outputs.
    '''
    # Only adjust the network weights for the played action.  (Multiplying by the one-hot vector achieves this.)
    return -y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()))
