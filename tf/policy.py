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
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input
from keras.layers.merge import add
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from logger import log, log_progress
from math import isnan
from keras.layers.core import Activation
from keras.layers.merge import add

LOG_DIR = os.path.join(tempfile.gettempdir(), 'bt', 'keras')
REINFORCEMENT_LEARNING_RATE = 0.0001
L2_FACTOR = 1e-4 # AGZ paper has 1x10^-4 for weight regularization

class CNPolicy:
  
  def __init__(self, checkpoint=None):
    if checkpoint:
      self._model = load_model(os.path.join(LOG_DIR, checkpoint), custom_objects={'top_3_accuracy': top_3_accuracy})
    else:
      log('Creating model with functional API')

      if True:
        num_residual_layers = 4
        num_filters = 32
        value_hidden_size = 32
        dropout_common = 0.05
        dropout_policy = 0.05
        dropout_reward = 0
      else:
        num_residual_layers = 8
        num_filters = 112
        value_hidden_size = 128
        dropout_common = 0.15
        dropout_policy = 0.15
        dropout_reward = 0

      # Start with the initial convolution block.
      input = Input(shape=(8, 8, 6), name='board_state')
      model = Conv2D(filters=num_filters,
                     kernel_size=[3,3],
                     #kernel_regularizer=l2(L2_FACTOR),
                     padding='same')(input)
      model = BatchNormalization()(model)
      model = Activation('relu')(model)
      model = Dropout(dropout_common)(model)
                     
      # Add the residual layers.
      for _ in range(num_residual_layers):
        stage_input = model
        model = Conv2D(filters=num_filters,
                       kernel_size=[3,3],
                       #kernel_regularizer=l2(L2_FACTOR),
                       padding='same')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(dropout_common)(model)
        model = Conv2D(filters=num_filters,
                       kernel_size=[3,3],
                       #kernel_regularizer=l2(L2_FACTOR),
                       padding='same')(model)
        model = BatchNormalization()(model)
        model = add([stage_input, model])
        model = Activation('relu')(model)      
        model = Dropout(dropout_common)(model)
      post_residual = model
      
      # Build the policy head.
      policy = Conv2D(filters=2,
                      kernel_size=[1,1],
                      #kernel_regularizer=l2(L2_FACTOR),                      
                      padding="same")(post_residual)
      policy = BatchNormalization()(policy)
      policy = Activation('relu')(policy)
      policy = Dropout(dropout_policy)(policy)
      policy = Flatten()(policy)
      policy = Dense(bt.ACTIONS,
                     #kernel_regularizer=l2(L2_FACTOR),
                     activation='softmax',
                     name='policy')(policy)
      
      # Build the reward head.
      reward = Conv2D(filters=1,
                      kernel_size=[1,1],
                      #kernel_regularizer=l2(L2_FACTOR),
                      padding="same")(post_residual)
      reward = BatchNormalization()(reward)
      reward = Activation('relu')(reward)
      reward = Dropout(dropout_reward)(reward)
      reward = Flatten()(reward)
      reward = Dense(value_hidden_size,
                     kernel_regularizer=l2(L2_FACTOR),
                     activation='relu')(reward)
      reward = Dropout(dropout_reward)(reward)
      reward = Dense(1,
                     #kernel_regularizer=l2(L2_FACTOR),
                     activation='tanh',
                     name='reward')(reward)

      self._model = Model(inputs=[input], outputs=[policy, reward])

  def train(self, train_states, train_action_probs, train_rewards, eval_states, eval_action_probs, eval_rewards, epochs=40, lr=0.1):
    self.compile(lr=lr)
    if eval_states is not None:
      history = self._model.fit(train_states,
                                [train_action_probs, train_rewards],
                                validation_data=(eval_states, [eval_action_probs, eval_rewards]),
                                epochs=epochs,
                                batch_size=1024,
                                callbacks=[TensorBoard(log_dir=LOG_DIR, write_graph=True),
                                           ModelCheckpoint(filepath=os.path.join(LOG_DIR, 'model.epoch{epoch:02d}.hdf5')),
                                           ReduceLROnPlateau(monitor='val_policy_acc', factor=0.5, patience=5, verbose=1)])
    else:
      history = self._model.fit(train_states,
                                [train_action_probs, train_rewards],
                                validation_split=0.2,
                                epochs=epochs,
                                batch_size=1024,
                                callbacks=[TensorBoard(log_dir=LOG_DIR, write_graph=True)])

  def compile(self, lr, use_sgd=True):
    if use_sgd:
        optimizer = SGD(lr=lr, momentum=0.9)
    else:
        optimizer = Adam(lr=lr)

    self._model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                        loss_weights=[1.0, 0.2],
                        optimizer=optimizer,
                        metrics=['accuracy']) # Adding top_3_accuracy causes lots of CPU use?

  def train_batch(self, train_states, train_action_probs, train_rewards):
    history = self._model.train_on_batch(train_states, [train_action_probs, train_rewards])

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

  def evaluate(self, states):
    batch_input = np.empty((len(states), 8, 8, 6), dtype=nn.DATA_TYPE)
    for ii, state in enumerate(states):
      self.convert_state(state, batch_input[ii:ii+1].reshape((8, 8, 6)))      
    return self._model.predict(batch_input)

  def get_action_probs(self, state):
    predictions = self._model.predict(self.convert_state(state).reshape((1, 8, 8, 6)))[0]
    return predictions[0]
    
  def get_state_value(self, state):
    predictions = self._model.predict(self.convert_state(state).reshape((1, 8, 8, 6)))[1]
    for _, prediction in enumerate(predictions):
      return prediction[0]
    
  def get_weighted_legal(self, state, action_probs):
    failed_attempts = 0
    index = -1
    legal = False
    while not legal:
      if index != -1:
        # Avoid picking this action again and re-normalise the probabilities.
        failed_attempts += 1
        action_probs[index] = 0
        total_action_probs = action_probs.sum()
        if total_action_probs <= 0 or isnan(total_action_probs) or failed_attempts > 192:
          orig_action_probs = self.get_action_probs(state)
          orig_best = np.argmax(orig_action_probs)
          log('Oh dear - problem getting legal action in state...')
          print(state)
          log('Action probabilities (after %d adjustment(s)) are...' % (failed_attempts))
          log(action_probs)
          log('Action probabilities (before adjustment) were...')
          log(orig_action_probs)
          log('Best original action was %s (%d)' % (lg.encode_move(bt.convert_index_to_move(orig_best, state.player)), orig_best))
          log('Last cleared action was %s (%d)' % (lg.encode_move(bt.convert_index_to_move(index, state.player)), index))
          log('Total = %f' % (total_action_probs))
          return self._first_legal_action(state)
        action_probs /= total_action_probs
      index = np.random.choice(bt.ACTIONS, p=action_probs)
      legal = state.is_legal(bt.convert_index_to_move(index, state.player))
    return index
    
  def _first_legal_action(self, state):
    for index in range(bt.ACTIONS):
      if state.is_legal(bt.convert_index_to_move(index, state.player)):
        return index
    log('No legal action at all in state...')
    print(state)
    return -1

  def get_action_index(self, state):
    action_probs = self.get_action_probs(state)
    return self.get_weighted_legal(state, action_probs)
  
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
        actions.append(self.get_weighted_legal(state, action_probs))
    return actions

  def save(self, filename="unnamed.hdf5"):
    self._model.save(os.path.join(LOG_DIR, filename))

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)