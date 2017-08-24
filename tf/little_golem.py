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
  data = {}

  num_matches = 0
  num_moves = 0
  num_duplicate_hits = 0

  white_matcher = re.compile('\[White "(.*)"\]')
  black_matcher = re.compile('\[Black "(.*)"\]')
  
  # Load all the matches with at least 20 moves each.  Shorter matches are typically test matches or matches played by complete beginners.
  log('Loading data', end='')
  raw_lg_data = open('../data/training/breakthrough.txt', 'r', encoding='latin1')
  for line in raw_lg_data:
    
    if 'Event' in line:
      white_good = False
      black_good = False
      
    match = white_matcher.match(line)
    if match:
      white_good = match.group(1) in good_players

    match = black_matcher.match(line)
    if match:
      black_good = match.group(1) in good_players
    
    round_marker = str(min_rounds) + '.'  
    if line.startswith('1.') and round_marker in line and white_good and black_good:
      num_matches += 1
      match = bt.Breakthrough()
      for part in line.split(' '):
        if len(part) == 5:
          num_moves += 1
          if num_moves % 10000 == 0:
              log_progress()
          move = decode_move(part)

          # Add a training example
          if match in data:
            num_duplicate_hits += 1
          else:
            data[match] = np.zeros((bt.ACTIONS), dtype=nn.DATA_TYPE)
          data[match][bt.convert_move_to_index(move)] += 1

          # Process the move to get the new state
          match = bt.Breakthrough(match, move)

  print('')
  log('  Loaded %d moves from %d matches (avg. %d moves/match) with %d duplicate hits' % 
    (num_moves, num_matches, num_moves / num_matches, num_duplicate_hits))
  
  # Normalise the action probabilities
  log('  Normalising data')
  for action_probs in iter(data.values()):
    total = action_probs.sum()
    for ii in range(bt.ACTIONS):
      action_probs[ii] /= total
      
  return data

def decode_move(move):
  (src, dst) = re.split('x|\-', move)
  src_col = ord(src[0]) - ord('a')
  src_row = ord(src[1]) - ord('1')
  dst_col = ord(dst[0]) - ord('a')
  dst_row = ord(dst[1]) - ord('1')
  return (src_row, src_col, dst_row, dst_col)
