from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import nn
import numpy as np

from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam

class CNPolicy:
  
  def __init__(self, num_conv_layers=3, checkpoint=None):
    if checkpoint:
      self._model = load_model(checkpoint)
    else:
      self._model = Sequential()
      self._model.add(Conv2D(input_shape=(8, 8, 6), filters=64, kernel_size=[5,5], padding='same', activation='relu'))
      for _ in range(num_conv_layers - 1):
        self._model.add(Conv2D(filters=128, kernel_size=[3,3], padding='same', activation='relu'))
      self._model.add(Flatten())
      self._model.add(Dropout(0.9))
      self._model.add(Dense(bt.ACTIONS, activation='softmax'))  
      self._model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

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

  def get_action_probs_for_state(self, state):
    predictions = self._model.predict(self.convert_state(state).reshape((1, 8, 8, 6)))
    for _, prediction in enumerate(predictions):
      return prediction