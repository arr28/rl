from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import math
import nn

class MCTSTrainer:
  def __init__(self, policy):
    self.policy = policy
    self.root_node = Node(None)
    self.root_node.evaluate(bt.Breakthrough(), policy)
    
  def iterate(self, state=bt.Breakthrough(), num_iterations=1):
    
    for _ in range(num_iterations):
      # Get the root state and node
      match_state = bt.Breakthrough(state)
      node = self.root_node
      
      # Select down to a fresh leaf node
      while (node.evaluated and not node.terminal):
        node = node.select_and_expand(match_state)
      
      # Evaluate the freshly created leaf node
      if (not node.terminal):
        node.evaluate(match_state, self.policy)
  
      # Get the value of the leaf and back it up
      value = node.value
      while (node.parent_edge is not None):
        parent_edge.backup(value)
        node = parent_edge.parent
        value *= -1.0
        
class Node:
  
  def __init__(self, parent_edge):
    self.total_child_visits = 0
    self.terminal = False
    self.evaluated = False
    self.parent_edge = parent_edge

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
     
    return best_edge.child

  def evaluate(self, match_state, policy):
    self.evaluated = True
    self.terminal = match_state.terminated
    
    # Calculate the value of this node to the player who moved last
    if (self.terminal):
      # Game always ends in a win for the player who moved last
      self.value = 1
    else:
      # Get the policy's estimate of the value.
      # !! Change training data so that state-value is from p.o.v. of player that moved last
      # !! Change training data so that it's in the range [-1, +1].
      self.value = policy.get_state_value(match_state)
      
    # !! Calculate children

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