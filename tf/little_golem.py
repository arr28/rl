from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import breakthrough as bt
import nn
import numpy as np
import re

from logger import log, log_progress

def load_data(min_rounds=20):
  # Starting ELO is 1500
  good_players = {'wanderer_bot', 'Ray Garrison', 'ahhmet', 'edbonnet', 'halladba', 'turab 69', 'David Scott', # 2067
                  'Mojmir Hanes', 'michelwav', 'luffy_bot', 'kingofthebesI', 'hammurabi', 'Stop_Sign', 'smilingface', # 1971
                  'isketzo067', 'Diamante', 'antony', 'ypercube', 'Marius Halsor', 'bennok', 'Tim Shih', # 1878
                  'Ragnar Wikman', 'Micco', 'kyle douglas', 'busybee', 'Zul Nadzri', 'Maciej Celuch', 'mungo', # 1820 
                  'richyfourtytwo', 'Madris', 'MojoRising', 'Reiner Martin', 'Florian Jamain', 'z', 'wallachia', # 1770
                  'Martyn Hamer', 'sZamBa_', 'MRFvR', 'm273cool', 'Chris', 'eaeaeapepe', 'gamesorry', 'Bernard Herwig', # 1747
                  'Maurizio De Leo', 'rafi', 'Willem Gerritsen', 'Mirko Rahn', 'Elsabio', 'kfiecio', 'Nagy Fathy', # 1723
                  'basplund', 'MathPickle', 'Jose M Grau Ribas', 'Matteo A.', 'Arty Sandler', 'dimitris', 'BigChicken', # 1682
                  'Thomas', 'nietsabes', 'Dvd Avins', 'pim', 'Luca Bruzzi', 'Cassiel', 'emilioes', 'vstjrt', # 1653
                  'Christian K', 'diego44', 'steve1964', 'lin1234', 'siroman', 'Tony', 'RoByN', 'slaapgraag', # 1641
                  'Tobias Lang', 'Rex Moore', 'Jonas', 'Richard Malaschitz', 'I R I', 'Peter Koning', 'Ryan'} # 1616
  state_hits = {}
  data = {}
  rewards = {}

  num_matches = 0
  num_moves = 0
  num_duplicate_hits = 0

  white_matcher = re.compile('\[White "(.*)"\]')
  black_matcher = re.compile('\[Black "(.*)"\]')
  result_matcher = re.compile('\[Result "(\d)-."\]')
  
  # Load all the matches with at least 20 moves each.  Shorter matches are typically test matches or matches played by complete beginners.
  log('Loading data', end='')
  raw_lg_data = open('../data/training/breakthrough.txt', 'r', encoding='latin1')
  for line in raw_lg_data:
    
    if 'Event' in line:
      white_good = False
      black_good = False
      result_good = False
      
    match = white_matcher.match(line)
    if match:
      white_good = match.group(1) in good_players

    match = black_matcher.match(line)
    if match:
      black_good = match.group(1) in good_players
    
    match = result_matcher.match(line)
    if match:
      result_good = True
      result = float(match.group(1)) * 2.0 - 1.0 # Scaled to [-1,+1]
      # Results in the match record are always from the p.o.v. of the player who moved first.  For our training example, we want them from the p.o.v. of the player that
      # just played (in any given state).  In the root state, we consider the 2nd player to have just played, therefore switch the result.
      result *= -1.0
      
    round_marker = str(min_rounds) + '.'  
    if line.startswith('1.') and round_marker in line and white_good and black_good and result_good:
      num_matches += 1
      state = bt.Breakthrough()
      for part in line.split(' '):
        if len(part) == 5:
          num_moves += 1
          if num_moves % 10000 == 0:
              log_progress()
          move = decode_move(part)

          # Add a training example
          if state in data:
            num_duplicate_hits += 1
            state_hits[state] += 1
          else:
            state_hits[state] = 1
            data[state] = np.zeros((bt.ACTIONS), dtype=nn.DATA_TYPE)
            rewards[state] = 0.0
          data[state][bt.convert_move_to_index(move)] += 1
          rewards[state] += result

          # Process the move to get the new state
          state = bt.Breakthrough(state, move)
          result *= -1.0

  print('')
  log('  Loaded %d moves from %d matches (avg. %d moves/match) with %d duplicate hits' % 
    (num_moves, num_matches, num_moves / num_matches, num_duplicate_hits))
  
  # Normalise the action probabilities
  log('  Normalising data')
  for state, action_probs in data.items():
    hits = state_hits[state]
    rewards[state] /= hits
    for ii in range(bt.ACTIONS):
      action_probs[ii] /= hits
      
  return data, rewards

def decode_move(move):
  (src, dst) = re.split('x|\-', move)
  src_col = ord(src[0]) - ord('a')
  src_row = ord(src[1]) - ord('1')
  dst_col = ord(dst[0]) - ord('a')
  dst_row = ord(dst[1]) - ord('1')
  return (src_row, src_col, dst_row, dst_col)

def encode_move(move):
  (src_row_ix, src_col_ix, dst_row_ix, dst_col_ix) = move;
  src_col = chr(src_col_ix + ord('a'))
  src_row = chr(src_row_ix + ord('1'))
  dst_col = chr(dst_col_ix + ord('a'))
  dst_row = chr(dst_row_ix + ord('1'))
  return format("%s%s-%s%s" % (src_col, src_row, dst_col, dst_row))

