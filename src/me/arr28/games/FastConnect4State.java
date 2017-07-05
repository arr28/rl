package me.arr28.games;

import java.util.Arrays;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;

/**
 * Connect-4 game state.
 *
 * @author Andrew Rose
 */
public class FastConnect4State implements GameState
{
  /**
   *   6 13 20 27 34 41 48   55 62     Additional row
   * +---------------------+
   * | 5 12 19 26 33 40 47 | 54 61     top row
   * | 4 11 18 25 32 39 46 | 53 60
   * | 3 10 17 24 31 38 45 | 52 59
   * | 2  9 16 23 30 37 44 | 51 58
   * | 1  8 15 22 29 36 43 | 50 57
   * | 0  7 14 21 28 35 42 | 49 56 63  bottom row
   * +---------------------+
   */
  private static final int WIDTH = 6;   // Max supported = 8
  private static final int HEIGHT = 4;  // Max supported = 6

  /**
   * A board for each player, according to the diagram above.  Own counters
   * have the corresponding bit set.  Opponent or blanks have the bit clear.
   */
  private final long[] mBoardForPlayer = new long [2];

  /**
   * The index of the next open cell in a column.
   */
  private int[] mNextFreeCellInColumn = {0, 7, 14, 21, 28, 35, 42};

  /**
   * The open columns.  The first mNumOpenColumns entries in the mOpenColumns
   * array identifies the open columns (in no particular order).
   */
  private final int[] mOpenColumns = new int[WIDTH];
  private int mNumOpenColumns = WIDTH;

  /**
   * Which player is due to play.
   */
  private int mPlayer = 0;

  /**
   * Whether this is a terminal state.
   */
  private boolean mTerminal = false;

  /**
   * The reward for player 0 - only valid for a terminal state.
   */
  private double mReward = 0.0;

  /**
   * Create a new initial state.
   */
  FastConnect4State()
  {
    for (int lCol = 0; lCol < WIDTH; lCol++)
    {
      mOpenColumns[lCol] = lCol;
    }
  }

  @Override
  public void copyTo(GameState xiDestination)
  {
    FastConnect4State xiNew = (FastConnect4State)xiDestination;

    xiNew.mBoardForPlayer[0] = mBoardForPlayer[0];
    xiNew.mBoardForPlayer[1] = mBoardForPlayer[1];
    System.arraycopy(mNextFreeCellInColumn, 0, xiNew.mNextFreeCellInColumn, 0, WIDTH);
    System.arraycopy(mOpenColumns, 0, xiNew.mOpenColumns, 0, mNumOpenColumns);
    xiNew.mNumOpenColumns = mNumOpenColumns;
    xiNew.mPlayer = mPlayer;
    xiNew.mTerminal = mTerminal;
    xiNew.mReward = mReward;
  }

  @Override
  public boolean isTerminal()
  {
    return mTerminal;
  }

  @Override
  public int getPlayer()
  {
    return mPlayer;
  }

  @Override
  public int getNumLegalActions()
  {
    return mNumOpenColumns;
  }

  @Override
  public void applyAction(int xiActionIndex)
  {
    int lColumn = mOpenColumns[xiActionIndex];
    mBoardForPlayer[mPlayer] ^= 1L << mNextFreeCellInColumn[lColumn]++;
    terminalCheck(mBoardForPlayer[mPlayer]);
    mPlayer ^= 1;
    if (mNextFreeCellInColumn[lColumn] % 7 >= HEIGHT)
    {
      // This column is full.  Decrease the number of open columns, swapping one of the other columns into place in
      // the array of open column.
      mOpenColumns[xiActionIndex] = mOpenColumns[--mNumOpenColumns];

      // Whole board is full without a win (calculated above).  It's a draw.
      if ((!mTerminal) && (mNumOpenColumns == 0))
      {
        mTerminal = true;
        mReward = 0.5;
      }
    }
  }

  private void terminalCheck(long xiBoard)
  {
    if (isTerminal(xiBoard))
    {
      // Only the player who has just played (in the specified cell) can have just won.
      mTerminal = true;
      mReward = 1.0 - mPlayer;
    }
  }

  private boolean isTerminal(long xiBoard)
  {
    if ((xiBoard & (xiBoard >> 6) & (xiBoard >> 12) & (xiBoard >> 18)) != 0) return true; // diagonal \
    if ((xiBoard & (xiBoard >> 8) & (xiBoard >> 16) & (xiBoard >> 24)) != 0) return true; // diagonal /
    if ((xiBoard & (xiBoard >> 7) & (xiBoard >> 14) & (xiBoard >> 21)) != 0) return true; // horizontal
    if ((xiBoard & (xiBoard >> 1) & (xiBoard >>  2) & (xiBoard >>  3)) != 0) return true; // vertical
    return false;
  }

  @Override
  public double getReward()
  {
    assert(mTerminal) : "Only call getReward() for terminal positions";
    return mReward;
  }

  @Override
  public boolean equals(Object xiOther)
  {
    // The game state is entirely encoded in the mBoardForPlayer array.
    return Arrays.equals(mBoardForPlayer, ((FastConnect4State)xiOther).mBoardForPlayer);
  }

  @Override
  public int hashCode()
  {
    // Since the state is encoded entirely in mBoardForPlayer (see .equals), just compute the hash code of that.
    return Arrays.hashCode(mBoardForPlayer);
  }

  @Override
  public String toString()
  {
    StringBuilder lResult = new StringBuilder((WIDTH * HEIGHT * 2) + (HEIGHT * 2) + 10);
    for (int lRow = HEIGHT - 1; lRow >= 0; lRow--)
    {
      for (int lCol = 0; lCol < WIDTH; lCol++)
      {
        long lMask = 1L << ((lCol * 7) + lRow);

        if ((mBoardForPlayer[0] & lMask) != 0)
        {
          lResult.append("O ");
        }
        else if ((mBoardForPlayer[1] & lMask) != 0)
        {
          lResult.append("X ");
        }
        else
        {
          lResult.append(". ");
        }
      }
      lResult.append('\n');
    }

    if (mTerminal)
    {
      lResult.append("Game over: Reward for player 0 = ").append(mReward).append('\n');
    }
    else
    {
      lResult.append("Player ").append(mPlayer).append(" to play");
    }
    return lResult.toString();
  }

  /**
   * Factory for creating starting states for C4.
   */
  public static class FastC4GameStateFactory implements GameStateFactory
  {
    @Override
    public GameState createInitialState()
    {
      return new FastConnect4State();
    }
  }
}
