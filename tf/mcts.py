from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import little_golem as lg
import math
import nn
import numpy as np

from logger import log

EXPLORATION_FACTOR = 5.0 # c{puct} in the paper.  They don't specify a value.  But the previous AlphaGo paper
                         # used 5.
MCTS_ITERATION_BATCH_SIZE = 8

class MCTSTrainer:
  def __init__(self, policy):
    self.policy = policy

  def self_play(self, num_matches=10):    
    for ii in range(num_matches):
      log('Starting match %d of %d' % (ii + 1, num_matches))
      self.self_play_one_match()
      
  def self_play_one_match(self):
    self.root_node = Node(None)
    self.root_node.evaluate(bt.Breakthrough(), self.policy)
    
    match_states = []
    match_action_probs = []
    match_rewards = []
    
    # Play a match
    match_state = bt.Breakthrough()
    while not self.root_node.terminal:
      print(match_state)

      # Do MCTS iterations from the current root
      self.iterate(state=match_state)

      # Record the stats from the current root node as a training example
      action_probs = self.root_node.get_action_probs(match_state)
      match_states.append(bt.Breakthrough(match_state))
      match_action_probs.append(action_probs)
      match_rewards.append(-self.root_node.total_child_value / self.root_node.total_child_visits)

      # Select a move and re-root the tree.
      edge = self.root_node.best_edge()
      self.root_node = edge.child
      log('Playing %s' % (lg.encode_move(edge.action)))
      match_state.apply(edge.action)
    print(match_state)

    # Do training
    # !! Should really be concurrently on another thread

    # Prepare the data
    samples = len(match_states)
    train_states = np.empty((samples, 8, 8, 6), dtype=nn.DATA_TYPE)
    train_action_probs = np.empty((samples, bt.ACTIONS), dtype=nn.DATA_TYPE)
    train_rewards = np.empty((samples, 1), dtype=nn.DATA_TYPE)
    # reward = float(match_state.reward) * -1.0 # Reward from p.o.v. of player who moved last.  For initial state, this is player 2.
    for ii, state in enumerate(match_states):
      self.policy.convert_state(state, train_states[ii:ii+1].reshape((8, 8, 6)))
      np.copyto(train_action_probs[ii:ii+1].reshape(bt.ACTIONS), match_action_probs[ii])
      train_rewards[ii] = match_rewards[ii]
      # reward *= -1.0

    # Do the training step
    self.policy.train_batch(train_states, train_action_probs, train_rewards)

  def iterate(self, state=bt.Breakthrough(), num_iterations=400):
    
    num_batches = int(num_iterations / MCTS_ITERATION_BATCH_SIZE)
    for _ in range(num_batches):      
      # Perform a batch of MCTS iterations      
      leaves = []
      states = []
      for _ in range(MCTS_ITERATION_BATCH_SIZE):
        
        # Get the root state and node
        match_state = bt.Breakthrough(state)
        node = self.root_node
        
        # Select down to a fresh leaf node
        while (node.have_evaluation and not node.terminal):
          node = node.select_and_expand(match_state)

        if node.terminal:
          # Backup this value, since we already know it.
          value = node.prior
          while (node.parent_edge is not None):
            node.parent_edge.backup(value)
            node = node.parent_edge.parent
            value *= -1.0
        else:        
          # Store the node for batch evaluation later.
          leaves.append(node)
          states.append(match_state)
      
      # Batch-evaluate all the new states
      evaluations = self.policy.evaluate(states)

      for ii in range(len(leaves)):
        node = leaves[ii]
        node.record_evaluation(states[ii], evaluations[0][ii], evaluations[1][ii][0])
        
        # Get the value of the leaf and back it up
        value = node.prior
        while (node.parent_edge is not None):
          node.parent_edge.backup(value)
          node = node.parent_edge.parent
          value *= -1.0
    
    self.root_node.dump_stats(state)
        
class Node:
  
  def __init__(self, parent_edge):
    self.total_child_visits = 0
    self.total_child_value = 0.0
    self.terminal = False
    self.have_evaluation = False
    self.evaluated = False
    self.parent_edge = parent_edge
    self.edges = []

  '''
  Select the most promising child node to explore next, creating one as required.
  
  match_state - MUTABLE match state.  On entry, the match state at the current node.  On output, the match state in
                the selected child node.  
  '''   
  def select_and_expand(self, match_state):
    assert not self.terminal

    # Expansion of leaves delayed until here (i.e. when they're used)
    if not self.evaluated:
      # Create edges for all the legal moves and record the priors.
      assert self.have_evaluation
      for index, prior in enumerate(self.eval_action_priors):
        action = bt.convert_index_to_move(index, match_state.player)
        if match_state.is_legal(action):
          self.edges.append(Edge(self, action, index, prior))
      self.evaluated = True
      self.eval_action_priors = None

    sqrt_visits = math.sqrt(self.total_child_visits)
    score = lambda edge: edge.average_value + (edge.prior * sqrt_visits / (edge.visits_plus_one))
    best_edge = max(self.edges, key=score)
    best_edge.visit()
    match_state.apply(best_edge.action)
        
    if (best_edge.child is None):
      # Create a new child node for the selected edge.
      best_edge.child = Node(best_edge)
      
      # Mark the new node as terminal if necessary.
      if match_state.terminated:
        best_edge.child.have_evaluation = True
        best_edge.child.evaluated = True
        best_edge.child.terminal = True
        # Breakthrough always ends in a win for the player who moved last.
        best_edge.child.prior = 1.0
     
    return best_edge.child

  def best_edge(self):
    num_visits = lambda edge: edge.visits_plus_one
    return max(self.edges, key=num_visits)

  def record_evaluation(self, match_state, action_priors, state_prior):
    if self.have_evaluation:
      return
    self.have_evaluation = True
    self.prior = state_prior

    # Don't expand the edges just yet.  It's really expensive and isn't used for most nodes.     
    self.eval_action_priors = action_priors

  '''
  Evaluate this node.  Better to do batch evaluation where possible and then call record_evaluation.
  '''
  def evaluate(self, match_state, policy):
    # Get the policy's estimate of the value.
    states = []
    states.append(match_state)
    evaluation = policy.evaluate(states)
    self.record_evaluation(match_state, evaluation[0][0], evaluation[1][0][0])
          
  def dump_stats(self, state):
    log('Node prior = % 2.4f and visit-weighted average child value = % 2.4f' % (self.prior, (self.total_child_value / self.total_child_visits)))
    for edge in self.edges:
      edge.dump_stats(state, self.total_child_visits)

  def get_action_probs(self, state):
    action_probs = np.zeros((bt.ACTIONS), dtype=nn.DATA_TYPE)
    for edge in self.edges:
      action_probs[edge.action_index] = (edge.visits_plus_one - 1) / self.total_child_visits
    return action_probs

class Edge:
  
  def __init__(self, parent, action, action_index, prior):
    self.parent = parent
    self.action = action
    self.action_index = action_index
    self.child = None
    self.visits_plus_one = 1
    self.total_value = 0.0
    self.average_value = 0.0
    self.prior = prior * EXPLORATION_FACTOR
    
  def visit(self):
    self.visits_plus_one += 1
    self.parent.total_child_visits += 1
    
  def backup(self, value):
    self.total_value += value
    self.average_value = self.total_value / (self.visits_plus_one - 1)
    self.parent.total_child_value += value
    
  def dump_stats(self, state, total_visits):
    log('%s: N = %4d, V = % 2.4f, Q = % 2.4f, P = %2.4f, pi = %2.4f' % (lg.encode_move(self.action), (self.visits_plus_one - 1), self.child.prior if self.child else 0.0, self.average_value, (self.prior / EXPLORATION_FACTOR), float(self.visits_plus_one - 1) / float(total_visits)))