"""Breakthrough CNN."""

# !! ARR Ideas for improvement
#
# - Add reflections to the dataset (being careful about interactions with deduplication)
# - Measure prediction speed (how would it do in a Monte Carlo rollout?)
# - Add more C-layers to the model
# - Change the number of filters in each layer
# - Add more D-layers to the model?
# - Exclude illegal moves when doing evaluation (pick best legal move)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import little_golem as lg
import nn
import numpy as np
import os
import re
import sys
import tempfile
import tensorflow as tf
import time

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
  logits = tf.layers.dense(inputs=dropout, units=bt.ACTIONS)

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

def greedy_rollout(classifier, state):
  while not state.terminated:
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": nn.convert_state(state)}, shuffle=False)
    prediction = next(classifier.predict(input_fn=predict_input_fn))
    index = np.argmax(prediction["probabilities"]) # Always pick the best action
    str_move = convert_index_to_move(index, state.player)
    print(state)
    print("Play %s with probability %f" % (str_move, prediction["probabilities"][index]))
    state = bt.Breakthrough(state, lg.decode_move(str_move))
  print("Game complete.  Final state...\n")
  print(state)
  return state.reward
  
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
      state = bt.Breakthrough(state, lg.decode_move(part))
  
  # Predict the next move
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": nn.convert_state(state)}, shuffle=False)
  predictions = classifier.predict(input_fn=predict_input_fn)
  for _, prediction in enumerate(predictions):
    sorted_indices = np.argsort(prediction["probabilities"])[::-1][0:5]
    for index in sorted_indices:
      print("Play %s with probability %f" % (convert_index_to_move(index, state.player), prediction["probabilities"][index]))
    _ = input('Press enter to play on')
    greedy_rollout(classifier, state)

def rollout(classifier, state):
  total_time = -int(round(time.time() * 1000))
  nn_time = 0
  while not state.terminated:
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": nn.convert_state(state)}, shuffle=False)
    nn_time  -= int(round(time.time() * 1000))
    prediction = next(classifier.predict(input_fn=predict_input_fn))
    nn_time  += int(round(time.time() * 1000))
    index = np.random.choice(bt.ACTIONS, p=prediction["probabilities"]) # Weighted sample from action probabilities
    str_move = convert_index_to_move(index, state.player)
    state = bt.Breakthrough(state, lg.decode_move(str_move))
  total_time  += int(round(time.time() * 1000))
  print('Total time %dms of which %dms in NN' % (total_time, nn_time))
  return state.reward
        
def evaluate():
  # Evaluate the model
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=os.path.join(tempfile.gettempdir(), 'bt', 'current'))

  # Run sample games and collect the total reward
  NUM_MATCHES = 20  
  total_reward = 0
  for _ in range(NUM_MATCHES):
    total_reward += rollout(classifier, bt.Breakthrough())
  print('Average reward = %f' % (total_reward / NUM_MATCHES))    
    
def train():
  # Load the data
  all_data = lg.load_data()
  samples = len(all_data);
  print('  Sorting data')
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
  print('  Shuffling data consistently')
  np.random.seed(0)
  rng_state = np.random.get_state()
  np.random.shuffle(nn_states)
  np.random.set_state(rng_state)
  np.random.shuffle(action_probs)

  print('  Splitting data')
  split_point = int(samples * 0.8)
  train_states = nn_states[:split_point]
  train_action_probs = action_probs[:split_point]
  eval_states = nn_states[split_point:]
  eval_labels = action_probs[split_point:]
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
    cmd = input("Train (t), predict (p) or evaluate (e)? ").lower()
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
  tf.app.run()
