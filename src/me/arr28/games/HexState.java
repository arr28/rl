package me.arr28.games;

import java.util.Arrays;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;

/**
 * Hex game state.
 *
 * @author Andrew Rose
 */
public class HexState implements GameState
{
  private static final int SIZE = 9;
  private static final int HEXES = SIZE * SIZE;
  private static final int MARK = 3;

  /**
   * Which player has played in the hex - 0 for player 0, 1 for player 1, 2 otherwise.
   */
  private final byte mHex[] = new byte[HEXES];

  /**
   * Which player is due to play.
   */
  private byte mPlayer = 0;

  /**
   * Whether this is a terminal state.
   */
  private boolean mTerminal = false;

  /**
   * The reward for player 0 - only valid for a terminal state.
   */
  private double mReward = 0.0;

  /**
   * The legal cells to play in.  The first mNumOpenHexes items in mOpenHexes are (in no particular order) the
   * hex numbers which are vacant.
   */
  private int mNumOpenHexes = HEXES;
  private final int[] mOpenHex = new int[HEXES];

  /**
   * Create a new initial state.
   */
  HexState()
  {
    for (int lHex = 0; lHex < HEXES; lHex++)
    {
      mHex[lHex] = 2;
      mOpenHex[lHex] = lHex;
    }
  }

  @Override
  public void copyTo(GameState xiDestination)
  {
    HexState xiNew = (HexState)xiDestination;

    System.arraycopy(mHex, 0, xiNew.mHex, 0, HEXES);
    xiNew.mPlayer = mPlayer;
    xiNew.mTerminal = mTerminal;
    xiNew.mReward = mReward;
    xiNew.mNumOpenHexes = mNumOpenHexes;
    System.arraycopy(mOpenHex, 0, xiNew.mOpenHex, 0, mNumOpenHexes);
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
    return mNumOpenHexes;
  }

  @Override
  public void applyAction(int xiActionIndex)
  {
    assert(!mTerminal) : "Can't continue to play in a terminal state";
    int lHex = mOpenHex[xiActionIndex];
    assert(mHex[lHex] == 2) : "Can't play in occupied hex: " + lHex + "\n" + this;

    mHex[lHex] = mPlayer;
    terminalCheck();
    if (mPlayer == 0) mPlayer = 1; else mPlayer = 0;

    mOpenHex[xiActionIndex] = mOpenHex[--mNumOpenHexes];
  }

  private void terminalCheck()
  {
    // Check if the player who has just played has made a connection.
    int lStride = (mPlayer == 0) ? 1 : SIZE;
    for (int lii = 0; lii < SIZE; lii++)
    {
      if (isConnected(lii * lStride))
      {
        mTerminal = true;
        mReward = 1 - mPlayer;
      }
    }

    // Reset marked hexes to the current player.
    for (int lHex = 0; lHex < HEXES; lHex++)
    {
      if (mHex[lHex] == MARK)
      {
        mHex[lHex] = mPlayer;
      }
    }
  }

  private boolean isConnected(int xiHex)
  {
    if (xiHex == -1) return false;
    if (mHex[xiHex] != mPlayer) return false;

    int lRow = xiHex / SIZE;
    int lCol = xiHex % SIZE;
    if (isTarget(lRow, lCol)) return true;

    // Connected at least this far.  Check for ongoing connections.
    mHex[xiHex] = MARK;

    // The connected hexes are at: -ROW, -ROW+COL, -COL, +COL, +ROW, +ROW-COL
    // ...provided that none of those go off the board.

    boolean lConnected = false;
    lConnected |= isConnected(checkIndex(lRow - 1, lCol));
    lConnected |= isConnected(checkIndex(lRow - 1, lCol + 1));
    lConnected |= isConnected(checkIndex(lRow, lCol - 1));
    lConnected |= isConnected(checkIndex(lRow, lCol + 1));
    lConnected |= isConnected(checkIndex(lRow + 1, lCol));
    lConnected |= isConnected(checkIndex(lRow + 1, lCol - 1));
    return lConnected;
  }

  private boolean isTarget(int xiRow, int xiCol)
  {
    return (((mPlayer == 0) && (xiRow == SIZE - 1)) ||
            ((mPlayer == 1) && (xiCol == SIZE - 1)));
  }

  private static int checkIndex(int xiRow, int xiCol)
  {
    if ((xiRow < 0) || (xiRow >= SIZE) || (xiCol < 0) || (xiCol >= SIZE)) return -1;
    return (xiRow * SIZE) + xiCol;
  }

  @Override
  public double getReward()
  {
    assert(mTerminal) : "Only call getReward() for terminal positions";
    return mReward;
  }

  @Override
  public String toString()
  {
    StringBuilder lResult = new StringBuilder();
    for (int lRow = 0; lRow < SIZE; lRow++)
    {
      // Indent the row
      for (int lSpace = 0; lSpace < lRow; lSpace++)
      {
        lResult.append(' ');
      }

      // Print the row
      for (int lCol = 0; lCol < SIZE; lCol++)
      {
        int lHex = (lRow * SIZE) + lCol;
        lResult.append(mHex[lHex] == 2 ? "." : mHex[lHex]).append(' ');
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

  @Override
  public boolean equals(Object xiOther)
  {
    // The game state is entirely encoded in mHex.  All other members can be derived from mHex.
    return Arrays.equals(mHex, ((HexState)xiOther).mHex);
  }

  @Override
  public int hashCode()
  {
    // Since the state is encoded entirely in mHex (see .equals), just compute the hash code of mHex.
    return Arrays.hashCode(mHex);
  }

  /**
   * Factory for creating starting states for Hex.
   */
  public static class HexGameStateFactory implements GameStateFactory
  {
    @Override
    public GameState createInitialState()
    {
      return new HexState();
    }
  }
}
