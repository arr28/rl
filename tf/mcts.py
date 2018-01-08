from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import little_golem as lg
import math
import nn
import numpy as np
import random

from logger import log

EXPLORATION_FACTOR = 5.0 # c{puct} in the paper.  They don't specify a value.  But the previous AlphaGo paper
                         # used 5.
MCTS_ITERATION_BATCH_SIZE = 8

class MCTSTrainer:
  def __init__(self, policy):
    self.policy = policy
    self.training_db = TrainingDB(policy)

  def new_self_play(self, num_matches=10000, num_rollouts=1600):
    # Get a batch of states to evaluate fully.
    log('Getting sample states')
    sampled_states = self.pick_states(num_matches=num_matches)

    for ii, state in enumerate(sampled_states):
      log('Evaluating state %d / %d' % (ii + 1, num_matches))

      # Create a new MCTS tree.
      self.root_node = Node(None)
      self.root_node.evaluate(bt.Breakthrough(state), self.policy)

      # Do PUCT for the chosen state.
      self.iterate(state=state, num_iterations=num_rollouts)

      # Add the sample to the database.      
      action_probs = self.root_node.get_action_probs()
      reward = self.estimate_state_value(state, action_probs, num_matches=10)
      # log('Reward (for previous player) = % 2.4f' % (reward))
      self.training_db.add(state, action_probs, reward)

    # Perform a training cycle
    self.training_db.train()

  '''
    Pick a set of states to evaluate in detail.
  '''
  def pick_states(self, num_matches=100):
    # Run a set of matches, recording all states encountered.
    matches = [bt.Breakthrough() for _ in range(num_matches)]
    all_states = [[] for _ in range(num_matches)]

    move_made = True
    while move_made:
      # Compute the next move for each game in parallel.
      move_made = False    
      for ii, (match_state, action) in enumerate(zip(matches, self.policy.get_action_indicies(matches))):
        if not match_state.terminated:
          all_states[ii].append(bt.Breakthrough(match_state))
          match_state.apply(bt.convert_index_to_move(action, match_state.player))
          move_made = True

    # Choose a single state from each match.    
    return [random.choice(all_states[ii]) for ii in range(num_matches)]

  '''
    Evaluate a state (from the p.o.v. of the player who moved last).
  '''
  def estimate_state_value(self, root_state, root_action_probs, num_matches=100):
    # Do parallel rollouts.
    states = [bt.Breakthrough(root_state) for _ in range(num_matches)]

    # Make the 1st move according to the supplied probabilities.
    for state in states:
      action = np.random.choice(bt.ACTIONS, p=root_action_probs)
      state.apply(bt.convert_index_to_move(action, state.player))

    # Thereafter, use the current policy.
    move_made = True
    while move_made:
      # Compute the next move for each game in parallel.
      move_made = False    
      for (state, action) in zip(states, self.policy.get_action_indicies(states)):
        if not state.terminated:
          state.apply(bt.convert_index_to_move(action, state.player))
          move_made = True

    # Return result from the p.o.v. of the player who played before the root state.
    total = 0
    for state in states:
      if state.is_win_for(root_state.player):
        total -= 1
      else:
        total += 1
    return total / num_matches


  # --------------------------- Old code starts here -------------------------------------
  
  def self_play(self, num_batches=10, batch_size=10):
    # In each batch of play, play some matches and then do some training.
    for ii in range(num_batches):

      # Run a batch of matches
      for jj in range(batch_size):
        log('Starting match %d of %d in batch %d of %d' % (jj + 1, batch_size, ii + 1, num_batches))
        self.self_play_one_match()

      # Perform a training cycle
      self.training_db.train()

  def self_play_one_match(self):
    self.root_node = Node(None)
    self.root_node.evaluate(bt.Breakthrough(), self.policy)
    
    match_states = []
    match_action_probs = []
    
    # Play a match.
    match_state = bt.Breakthrough()
    while not self.root_node.terminal:
      print(match_state)

      # Do MCTS iterations from the current root.
      self.iterate(state=match_state)

      # Record the stats from the current root node as a training example.  The match result (when known) will be used for the reward head
      # because it is an unbiased estimate of the policy.
      action_probs = self.root_node.get_action_probs()
      match_states.append(bt.Breakthrough(match_state))
      match_action_probs.append(action_probs)
      # match_rewards.append(-self.root_node.total_child_value / self.root_node.total_child_visits)

      # Select a move and re-root the tree.
      edge = self.root_node.sample()
      self.root_node = edge.child
      log('Playing %s' % (lg.encode_move(edge.action)))
      match_state.apply(edge.action)
    print(match_state)

    # Add the encountered positions to the training database.
    reward = float(match_state.reward) * -1.0 # Reward from p.o.v. of player who moved last.  For initial state, this is player 2.
    for ii, state in enumerate(match_states):
      self.training_db.add(state, match_action_probs[ii], reward)
      reward *= -1.0

  def iterate(self, state=bt.Breakthrough(), num_iterations=1600):
    
    num_batches = int(num_iterations / MCTS_ITERATION_BATCH_SIZE)
    for _ in range(num_batches):      
      # Perform a batch of MCTS iterations.
      leaves = []
      states = []
      for _ in range(MCTS_ITERATION_BATCH_SIZE):
        
        # Get the root state and node.
        match_state = bt.Breakthrough(state)
        node = self.root_node
        
        # Select down to a fresh leaf node.
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
      
      # Batch-evaluate all the new states.
      evaluations = self.policy.evaluate(states)

      for ii in range(len(leaves)):
        node = leaves[ii]
        node.record_evaluation(states[ii], evaluations[0][ii], evaluations[1][ii][0])
        
        # Get the value of the leaf and back it up.
        value = node.prior
        while (node.parent_edge is not None):
          node.parent_edge.backup(value)
          node = node.parent_edge.parent
          value *= -1.0
    
    # self.root_node.dump_stats(state)

  # Used for direct evaluation outside of a training cycle.
  def prepare_for_eval(self, root_state):
    self.root_node = Node(None)
    self.root_node.evaluate(root_state, self.policy)
              
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

    # Expansion of leaves is delayed until here (i.e. when they're used).
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

  '''
    Sample a child edge in proportion to visit count.
  '''
  def sample(self):
    sample = random.randint(1, self.total_child_visits)
    for edge in self.edges:
      sample -= edge.visits_plus_one - 1
      if sample <= 0:
        return edge
    log('Error: Failed to generate sample')
    
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

  def get_action_probs(self):
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
    log('%s: Visits(N) = %5d, Prior Value(V) = % 2.4f, Value(Q) = % 2.4f, Prior Prob(P) = %2.4f, Prob(pi) = %2.4f' % (lg.encode_move(self.action), (self.visits_plus_one - 1), self.child.prior if self.child else 0.0, self.average_value, (self.prior / EXPLORATION_FACTOR), float(self.visits_plus_one - 1) / float(total_visits)))

'''
 A database of samples for training the network.
'''
class TrainingDB:
  def __init__(self, policy):
    self.policy = policy
    self.states = []
    self.reward = {}
    self.action_probs = {}
    self._num_samples = 0

  def add(self, state, action_probs, reward):
    if not state in self.reward:
      self.states.append(state)
      self._num_samples += 1
    else:
      log('Updating existing sample')

    self.reward[state] = reward
    self.action_probs[state] = action_probs

  def train(self):
    # Create input space in the required format.
    train_states = np.empty((self._num_samples, 8, 8, 6), dtype=nn.DATA_TYPE)
    train_action_probs = np.empty((self._num_samples, bt.ACTIONS), dtype=nn.DATA_TYPE)
    train_rewards = np.empty((self._num_samples, 1), dtype=nn.DATA_TYPE)

    # Copy the data in.
    for ii, state in enumerate(self.states):
      self.policy.convert_state(state, train_states[ii:ii+1].reshape((8, 8, 6)))
      np.copyto(train_action_probs[ii:ii+1].reshape(bt.ACTIONS), self.action_probs[state])
      train_rewards[ii] = self.reward[state]

    # Do the training.
    log('About to train with %d rewards...' % (self._num_samples))
    log(train_rewards)
    self.policy.train(train_states, train_action_probs, train_rewards, None, None, None, epochs=200, lr=0.007)