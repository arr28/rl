"""Breakthrough game state (mutable)"""

import hashlib
import numpy as np

ACTIONS = 8 * 8 * 3 # Not strictly true, but makes the conversion from move to index much simpler
Z_HASHES = np.zeros((8, 8, 3), dtype=np.int64)
ZERO_HASH = 0

def __static_init():
  global Z_HASHES
  global ZERO_HASH
  for row in range(8):
    for col in range(8):
      for val in range(3):
        Z_HASHES[row][col][val] = int(hashlib.sha1(b'r%dc%dv%d' % (row, col, val)).hexdigest()[:15], 16)
        if val == 0:
          ZERO_HASH ^= Z_HASHES[row][col][val]

class Breakthrough:
       
  def __init__(self, parent_state=None, move_to_apply=None):
    if parent_state is None:
      self.grid = np.zeros((8, 8), dtype=np.int8)
      self.zhash = ZERO_HASH
      self.player = 0
      self.terminated = 0
      self.reward = 0
      self.__reset()
    else:
      self.grid = np.copy(parent_state.grid)
      self.zhash = parent_state.zhash
      self.player = parent_state.player
      self.terminated = parent_state.terminated
      self.reward = parent_state.reward
      self.__apply(move_to_apply)
    self.grid.setflags(write = False)

  def __set_cell(self, row, col, val):
    self.zhash ^= Z_HASHES[row][col][self.grid[row][col]]
    self.grid[row][col] = val
    self.zhash ^= Z_HASHES[row][col][val]
    
  def __reset(self):
    # Reset the grid to the starting position.  The first two rows are filled with player 1's pawns.  The last 2 rows are filled with player 2's pawns.
    # All the other cells are empty.
    for row in range(8):
      for col in range(8):
        if (row == 0) or (row == 1):
          self.__set_cell(row, col, 0)
        elif (row == 6) or (row == 7):
          self.__set_cell(row, col, 1)
        else:
          self.__set_cell(row, col, 2)
    self.player = 0

  def __apply(self, move):
    (src_row, src_col, dst_row, dst_col) = move
    # !! Do something about illegal moves
    self.__set_cell(src_row, src_col, 2);                # Vacate source cell
    self.__set_cell(dst_row, dst_col, self.player);      # Occupy target cell
    self.terminated = ((dst_row == 0) or (dst_row == 7)) # Check if the game is over
    self.reward = 1 if (self.player == 0) else -1        # A player 0 win scores +1, otherwise -1 
    self.player = 1 - self.player                        # Other player's turn
    # !! ARR Also need to handle the case where a player's last piece is taken 

  def __str__(self):
    pretty = ''
    for row in reversed(range(8)):
      pretty += str(row + 1) + ' '
      for col in range(8):
        if self.grid[row][col] == 0:
          pretty += '^ '
        elif self.grid[row][col] == 1:
          pretty += 'v '
        else:
          pretty += '  '
      pretty += '\n'
    pretty += "  a b c d e f g h\n"
    pretty += "Player %d to play\n" % (self.player + 1)
    return pretty

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.player == other.player and np.all(self.grid == other.grid)
    else:
      return False
    
  def __ne__(self, other):
    return not self.__eq__(other)
  
  def __lt__(self, other):
    for row in range(8):
      for col in range(8):
        if self.grid[row][col] != other.grid[row][col]:
          return self.grid[row][col] < other.grid[row][col]    
    return self.player < other.player
      
  def __hash__(self):
    return int(self.zhash)

__static_init()

''' ========== Static functions ========== '''

def convert_move_to_index(move):
  (src_row, src_col, dst_row, dst_col) = move;
  index = ((src_row * 8) + src_col) * 3
  index += 1 + dst_col - src_col # (left, forward, right) => (0, 1, 2)
  return index

def convert_index_to_move(index, player):
  dir = (index % 3) - 1
  index = int(index / 3)
  src_col = index % 8
  src_row = int(index / 8)
  dst_col = src_col + dir
  dst_row = src_row + (1 if player == 0 else -1)
  return (src_row, src_col, dst_row, dst_col)
