from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import numpy as np

DATA_TYPE = np.float32

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Reshape
from keras.optimizers import SGD, Adam

def create_model():
  model = Sequential()
  model.add(Conv2D(input_shape=(8, 8, 6), filters=64, kernel_size=[5,5], padding='same', activation='relu'))
  model.add(Conv2D(filters=128, kernel_size=[3,3], padding='same', activation='relu'))
  model.add(Conv2D(filters=128, kernel_size=[3,3], padding='same', activation='relu'))
  model.add(Reshape([8 * 8 * 128]))
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(bt.ACTIONS, activation='softmax'))  
  model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
  return model

def convert_state(state, nn_input=np.empty((8, 8, 6), dtype=DATA_TYPE)):
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
  np.copyto(nn_input[:,:,4:5].reshape(8, 8), np.zeros((8, 8), dtype=DATA_TYPE))
  np.copyto(nn_input[:,:,5:6].reshape(8, 8), np.ones((8, 8), dtype=DATA_TYPE))
  
  return nn_input
