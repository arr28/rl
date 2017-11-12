from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import little_golem as lg
import math
import nn

from logger import log

EXPLORATION_FACTOR = 5.0 # c{puct} in the paper.  They don't specify a value.  But the previous AlphaGo paper
                         # used 5.
MCTS_ITERATION_BATCH_SIZE = 8

class MCTSTrainer:
  def __init__(self, policy):
    self.policy = policy
    self.root_node = Node(None)
    self.root_node.evaluate(bt.Breakthrough(), policy)
    
  def iterate(self, state=bt.Breakthrough(), num_iterations=1600):
    
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
        while (node.evaluated and not node.terminal):
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
    sqrt_visits = math.sqrt(self.total_child_visits)
    score = lambda edge: edge.average_value + (EXPLORATION_FACTOR * edge.prior * sqrt_visits / (1 + edge.visits))
    best_edge = max(self.edges, key=score)
    best_edge.visit()
    match_state.apply(best_edge.action)
        
    if (best_edge.child is None):
      # Create a new child node for the selected edge.
      best_edge.child = Node(best_edge)
      
      # Mark the new node as terminal if necessary.
      if match_state.terminated:
        self.evaluated = True
        self.terminal = True
        # Breakthrough always ends in a win for the player who moved last.
        self.prior = 1.0
     
    return best_edge.child

  def record_evaluation(self, match_state, action_priors, state_prior):
    self.evaluated = True
    self.prior = state_prior

    # Create edges for all the legal moves and record the priors.
    for index, prior in enumerate(action_priors):
      action = bt.convert_index_to_move(index, match_state.player)
      if match_state.is_legal(action):
        self.edges.append(Edge(self, action, prior))

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

class Edge:
  
  def __init__(self, parent, action, prior):
    self.parent = parent
    self.action = action
    self.child = None
    self.visits = 0
    self.total_value = 0.0
    self.average_value = 0.0
    self.prior = prior
    
  def visit(self):
    self.visits += 1
    self.parent.total_child_visits += 1
    
  def backup(self, value):
    self.total_value += value
    self.average_value = self.total_value / self.visits
    self.parent.total_child_value += value
    
  def dump_stats(self, state, total_visits):
    log('%s: N = %4d, V = % 2.4f, Q = % 2.4f, P = %2.4f, pi = %2.4f' % (lg.encode_move(self.action), self.visits, self.child.prior, self.average_value, self.prior, float(self.visits) / float(total_visits)))