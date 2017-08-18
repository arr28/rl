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
LABEL_TYPE = np.int32

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
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=ACTIONS)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

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
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def dbg(string):
  #print(string)
  None

def decode_move(move):
  (src, dst) = move
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

def convert_state_to_nn_input(state):
  nn_input = np.empty((8, 8, 6), dtype=DATA_TYPE)
  if state.player == 0:
    np.copyto(nn_input[:,:,0:1].reshape(8, 8), np.equal(state.grid, np.zeros((8,8))))
    np.copyto(nn_input[:,:,1:2].reshape(8, 8), np.equal(state.grid, np.ones((8,8))))
    np.copyto(nn_input[:,:,2:3].reshape(8, 8), np.zeros((8, 8), dtype=DATA_TYPE))
  else:
    np.copyto(nn_input[:,:,0:1].reshape(8, 8), np.equal(state.grid, np.ones((8,8))))
    np.copyto(nn_input[:,:,1:2].reshape(8, 8), np.equal(state.grid, np.zeros((8,8))))
    np.copyto(nn_input[:,:,2:3].reshape(8, 8), np.ones((8, 8), dtype=DATA_TYPE))

  np.copyto(nn_input[:,:,3:4].reshape(8, 8), np.equal(state.grid, np.full((8,8), 2)))
  np.copyto(nn_input[:,:,4:5].reshape(8, 8), np.zeros((8, 8), dtype=DATA_TYPE))
  np.copyto(nn_input[:,:,5:6].reshape(8, 8), np.ones((8, 8), dtype=DATA_TYPE))

  return nn_input

def load_lg_dataset():
  data = []
  labels = []

  num_matches = 0
  num_moves = 0

  match = bt.Breakthrough()

  # Load all the matches with at least 20 moves each.  Shorter matches are typically test matches or matches played by complete beginners.
  print('Loading data', end='', flush=True)
  raw_lg_data = open('../data/training/breakthrough.txt', 'r', encoding='latin1')
  for line in raw_lg_data:
    if line.startswith('1.') and '20.' in line:
      num_matches += 1
      match.reset()
      dbg('-----------------')
      dbg(match)
      for part in line.split(' '):
        if len(part) == 5:
          num_moves += 1
          if num_moves % 10000 == 0:
              print(".", end='', flush=True)
          move = decode_move(re.split('x|\-', part))

          # Add a training example
          data.append(convert_state_to_nn_input(match))
          labels.append(convert_move_to_index(move))

          # Process the move to get the new state
          match.apply(move)
          dbg(part)
          dbg(match)

  print('\nLoaded %d moves from %d matches (avg. %d moves/match)' % (num_moves, num_matches, num_moves / num_matches))
  return (np.array(data), np.array(labels))

def main(unused_argv):
  # Load the data
  (all_data, all_labels) = load_lg_dataset()
  samples = len(all_data);  

  # Split into training and validation sets.
  print('Shuffling data')
  np.random.seed(0) # Use a fixed seed to get reproducibility over different runs.  This is especially important when resuming training.
  rng_state = np.random.get_state()
  np.random.shuffle(all_data)
  np.random.set_state(rng_state)
  np.random.shuffle(all_labels)

  print('Splitting data')
  split_point = int(samples * 0.8)
  train_data = all_data[:split_point]
  train_labels = all_labels[:split_point]
  eval_data = all_data[split_point:]
  eval_labels = all_labels[split_point:]

  # Create the Estimator
  print('Building model')
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=os.path.join(tempfile.gettempdir(), 'bt', 'current'))

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  for iter in range(50):
    # Train the model
    print('Training model')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    print('Evaluating model')
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()
