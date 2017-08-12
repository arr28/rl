"""Breakthrough game state (mutable)"""

import numpy as np

class Breakthrough:
  def __init__(self):
    self.grid = np.zeros((8, 8), dtype=np.float32)
    self.player = 0
    self.reset()

  def reset(self):
    # Reset the grid to the starting position.  The first two rows are filled with player 1's pawns.  The last 2 rows are filled with player 2's pawns.
    # All the other cells are empty.
    for row in range(8):
      for col in range(8):
        if (row == 0) or (row == 1):
          self.grid[row][col] = 0
        elif (row == 6) or (row == 7):
          self.grid[row][col] = 1
        else:
          self.grid[row][col] = 2
    self.player = 0

  def apply(self, move):
    (src_row, src_col, dst_row, dst_col) = move
    self.grid[src_row][src_col] = 2;           # Vacate source cell
    self.grid[dst_row][dst_col] = self.player; # Occupy target cell
    self.player = 1 - self.player              # Other player's turn

  def __str__(self):
    pretty = ''
    for row in range(8):
      for col in range(8):
        if self.grid[row][col] == 0:
          pretty += 'v '
        elif self.grid[row][col] == 1:
          pretty += '^ '
        else:
          pretty += '  '
      pretty += '\n'
    pretty += "Player %d to play\n" % (self.player + 1)
    return pretty
