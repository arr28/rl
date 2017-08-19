"""Breakthrough CNN."""

# !! ARR Ideas for improvement
#
# - Do the one-hot split up-front, rather than via the graph (interacts with how we deal with the same state but different moves)
# - Measure prediction speed (how would it do in a Monte Carlo rollout?)
# - Add more C-layers to the model
# - Change the number of filters in each layer
# - Add more D-layers to the model?
# - Add reflections to the dataset
# - Deal with different moves made from the same state (and check interaction with reflections)
# - Ensure that states (& reflections) don't appear in both training & validation sets (avoid misleading results due to memoization)
# - Exclude illegal moves when doing evaluation (pick best legal move)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import numpy as np
import os
import re
import sys
import tempfile
import tensorflow as tf

ACTIONS = 8 * 8 * 3 # Not strictly true, but makes the conversion from move to index much simpler
DATA_TYPE = np.float32

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 8, 8, 6])

  # Convolutional Layer #1
  # Computes 64 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Shape: [batch_size, 8, 8, 2]
  # Output Shape: [batch_size, 8, 8, 64]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2
  # Computes 128 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 8, 8, 64]
  # Output Tensor Shape: [batch_size, 8, 8, 128]
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #3
  # Computes 128 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 8, 8, 128]
  # Output Tensor Shape: [batch_size, 8, 8, 128]
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 8, 8, 128]
  # Output Tensor Shape: [batch_size, 8 * 8 * 128]
  flattened = tf.reshape(conv3, [-1, 8 * 8 * 128])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 8 * 8 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=flattened, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, ACTIONS (=192)]
  logits = tf.layers.dense(inputs=dropout, units=ACTIONS)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=tf.argmax(labels, axis=1), predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def decode_move(move):
  (src, dst) = re.split('x|\-', move)
  src_col = ord(src[0]) - ord('a')
  src_row = ord(src[1]) - ord('1')
  dst_col = ord(dst[0]) - ord('a')
  dst_row = ord(dst[1]) - ord('1')
  return (src_row, src_col, dst_row, dst_col)

def convert_move_to_index(move):
  (src_row, src_col, dst_row, dst_col) = move;
  index = ((src_row * 8) + src_col) * 3
  index += 1 + dst_col - src_col # (left, forward, right) => (0, 1, 2)
  return index

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

def convert_state_to_nn_input(state, nn_input=np.empty((8, 8, 6), dtype=DATA_TYPE)):
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

def load_lg_dataset():
  data = {}

  num_matches = 0
  num_moves = 0
  num_duplicate_hits = 0

  # Load all the matches with at least 20 moves each.  Shorter matches are typically test matches or matches played by complete beginners.
  print('Loading data', end='', flush=True)
  raw_lg_data = open('../data/training/breakthrough.txt', 'r', encoding='latin1')
  for line in raw_lg_data:
    if line.startswith('1.') and '20.' in line:
      num_matches += 1
      match = bt.Breakthrough()
      for part in line.split(' '):
        if len(part) == 5:
          num_moves += 1
          if num_moves % 10000 == 0:
              print(".", end='', flush=True)
          move = decode_move(part)

          # Add a training example
          if match in data:
            num_duplicate_hits += 1
          else:
            data[match] = np.zeros((ACTIONS), dtype=DATA_TYPE)
          data[match][convert_move_to_index(move)] += 1

          # Process the move to get the new state
          match = bt.Breakthrough(match, move)

  print('\nLoaded %d moves from %d matches (avg. %d moves/match) with %d duplicate hits' % 
    (num_moves, num_matches, num_moves / num_matches, num_duplicate_hits))
  
  # Normalise the action probabilities
  for action_probs in iter(data.values()):
    total = action_probs.sum()
    for ii in range(ACTIONS):
      action_probs[ii] /= total
      
  return data

def rollout(classifier, state):
  for _ in range(10):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": convert_state_to_nn_input(state)}, shuffle=False)
    predictions = classifier.predict(input_fn=predict_input_fn)
    prediction = next(predictions)
    #for _, prediction in enumerate(predictions):
    index = np.argmax(prediction["probabilities"])
    str_move = convert_index_to_move(index, state.player)
    print(state)
    print("Play %s with probability %f" % (str_move, prediction["probabilities"][index]))
    state = bt.Breakthrough(state, decode_move(str_move))
        
def predict():
  # Create the Estimator
  print('Building model')
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=os.path.join(tempfile.gettempdir(), 'bt', 'current'))
  
  # Advance the game to the desired state
  sys.stderr.flush()
  history = input('Input game history: ')
  state = bt.Breakthrough()
  for part in history.split(' '):
    if len(part) == 5:
      state = bt.Breakthrough(state, decode_move(part))
  
  # Predict the next move
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": convert_state_to_nn_input(state)}, shuffle=False)
  predictions = classifier.predict(input_fn=predict_input_fn)
  for _, prediction in enumerate(predictions):
    sorted_indices = np.argsort(prediction["probabilities"])[::-1][0:5]
    for index in sorted_indices:
      print("Play %s with probability %f" % (convert_index_to_move(index, state.player), prediction["probabilities"][index]))
    _ = input('Press enter to play on')
    rollout(classifier, state)
  
def train():
  # Load the data
  all_data = load_lg_dataset()
  samples = len(all_data);
  states = np.empty((samples, 8, 8, 6), dtype=DATA_TYPE)
  action_probs = np.empty((samples, ACTIONS), dtype=DATA_TYPE)
  ii = 0
  for state, actions in all_data.items():
    convert_state_to_nn_input(state, states[ii:ii+1].reshape((8, 8, 6)))
    np.copyto(action_probs[ii:ii+1].reshape(ACTIONS), actions)
    ii += 1
    
  # Split into training and validation sets.
  print('Shuffling data')
  np.random.seed(0) # Use a fixed seed to get reproducibility over different runs.  This is especially important when resuming training.
  rng_state = np.random.get_state()
  np.random.shuffle(states)
  np.random.set_state(rng_state)
  np.random.shuffle(action_probs)

  print('Splitting data')
  split_point = int(samples * 0.8)
  train_states = states[:split_point]
  train_action_probs = action_probs[:split_point]
  eval_states = states[split_point:]
  eval_labels = action_probs[split_point:] # !! ARR Need to do argmax to produce labels (+possible fix network)
  print('  %d training samples vs %d evaluation samples' % (split_point, samples - split_point))
  
  # Create the Estimator
  print('Building model')
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir=os.path.join(tempfile.gettempdir(), 'bt', 'current'))

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Full run config
  iters = 50
  batch_size = 100
  steps = 2000
  
  if False:
    # Small run config (for testing)
    iters = 1
    batch_size = 10
    steps = 50
  
  for iter in range(iters):
    # Train the model
    print('Training model')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_states},
      y=train_action_probs,
      batch_size=batch_size,
      num_epochs=None,
      shuffle=True)
    classifier.train(
      input_fn=train_input_fn,
      steps=steps,
      hooks=[logging_hook])

    # Evaluate the model and print results
    print('Evaluating model')
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_states},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

def main(argv):
  handled = False;
  while not handled:
    cmd = input("Train (t) or predict (p)? ").lower()
    if cmd == 'train' or cmd == 't':
      handled = True
      train()
    elif cmd == 'predict' or cmd == 'p':
      handled = True
      predict()
  
if __name__ == "__main__":
  tf.app.run()
