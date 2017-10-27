from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import keras.backend as K
import little_golem as lg
import nn
import numpy as np
import os
import tempfile

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Input, Conv2D, Dense, Dropout, Flatten
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from logger import log, log_progress

LOG_DIR = os.path.join(tempfile.gettempdir(), 'bt', 'keras')
REINFORCEMENT_LEARNING_RATE = 0.0001

class CNPolicy:
  
  def __init__(self, num_conv_layers=3, checkpoint=None):
    if checkpoint:
      self._model = load_model(os.path.join(LOG_DIR, checkpoint))
    else:
      log('Creating model with functional API')

      input = Input(shape=(8, 8, 6), name='board_state')
      model = Conv2D(filters=64, kernel_size=[5,5], padding='same', activation='relu')(input)
      for _ in range(num_conv_layers - 1):
        model = Conv2D(filters=128, kernel_size=[3,3], padding='same', activation='relu')(model)
      model = Flatten()(model)
      model = Dropout(0.9)(model)
      policy = Dense(bt.ACTIONS, activation='softmax', name='policy')(model)
      value = Dense(1, activation='tanh', name='reward')(model)

      self._model = Model(inputs=[input], outputs=[policy, value])

  def train(self, train_states, train_action_probs, train_rewards, eval_states, eval_action_probs, eval_rewards, epochs=40):
    self._model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr=0.001), metrics=['accuracy'])
    history = self._model.fit(train_states,
                              [train_action_probs, train_rewards],
                              validation_data=(eval_states, [eval_action_probs, eval_rewards]),
                              epochs=epochs,
                              batch_size=1024,
                              callbacks=[TensorBoard(log_dir=LOG_DIR, write_graph=True),
                                         ModelCheckpoint(filepath=os.path.join(LOG_DIR, 'model.epoch{epoch:02d}.hdf5')),
                                         ReduceLROnPlateau(monitor='val_policy_acc', factor=0.3, patience=3, verbose=1)])
        
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
    predictions = self._model.predict(self.convert_state(state).reshape((1, 8, 8, 6)))[0]
    for _, prediction in enumerate(predictions):
      return prediction
    
  def get_state_value(self, state):
    predictions = self._model.predict(self.convert_state(state).reshape((1, 8, 8, 6)))[1]
    for _, prediction in enumerate(predictions):
      return prediction
    
  def _get_weighted_legal(self, state, action_probs):
    index = -1
    legal = False
    while not legal:
      if index != -1:
        # Avoid picking this action again and re-normalise the probabilities.
        action_probs[index] = 0
        total_action_probs = action_probs.sum()
        if total_action_probs == 0:
          log('Oh dear - no legal action with any weight in state...')
          print(state)
          log('Action probabilities (after adjustment) are...')
          log(action_probs)
          log('Last cleared action was %s (%d)' % (lg.encode_move(bt.convert_index_to_move(index, state.player)), index))
        action_probs /= total_action_probs
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
    predictions = self._model.predict(batch_input)[0]
    actions = []
    for state, action_probs in zip(states, predictions):
      if state.terminated:
        actions.append(-1)
      else:      
        actions.append(self._get_weighted_legal(state, action_probs))
    return actions
  