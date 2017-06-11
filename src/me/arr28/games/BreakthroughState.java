package me.arr28.games;

import java.util.Arrays;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;

/**
 * Breakthrough game state.
 *
 * @author Andrew Rose
 */
public class BreakthroughState implements GameState
{
  private static final int WIDTH = 8;
  private static final int HEIGHT = 8;

  /**
   * Whether the cell is occupied by player 0 (0), player 1 (1) or is empty (2).
   */
  private final byte mCell[] = new byte[WIDTH * HEIGHT];

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
   * Both players start with pieces in their nearest two rows.
   */
  private byte[] mPieceCount = {2 * WIDTH, 2 * WIDTH};

  /**
   * The legal moves for each player.
   *
   * The legals are encoded as the lowest 3 bits for the source column, then 3 bits for the source row and finally 2
   * bits for the direction of travel (00 for diagonally left, 01 for straight, 10 for diagonally right - from the
   * p.o.v. of player 0).
   *
   * 8 bits = 256 legals.  This is an upper-bound.  Moves off the edge of the board are never legal and, once a player
   * has reached the final row, the game is over so there are no legals.  All the same, this is a simple encoding.  With
   * only 2 * WIDTH pieces, the upper bound on the number of legals in any one turn is even smaller, but not to worry.
   *
   * As actions become legal and illegal, we swap them (in mLegals) to the left and right of mNumLegals.  Therefore,
   * they quickly become shuffled.  To efficiently locate where the action is currently located, we also maintain
   * mActionIndex to point to the place in mLegals where the action is to be found.
   */
  private int[][] mLegals = {new int[256], new int[256]};
  private byte[] mNumLegals = {0, 0};
  private int[][] mActionIndex = {new int[256], new int[256]};
  private static final int LFT = 0b00000000;
  private static final int FWD = 0b01000000;
  private static final int RGT = 0b10000000;

  /**
   * Create a new initial state.
   */
  BreakthroughState()
  {
    for (int lPlayer = 0; lPlayer < 2; lPlayer++)
    {
      for (int lAction = 0; lAction < 256; lAction++)
      {
        mLegals[lPlayer][lAction] = lAction;
        mActionIndex[lPlayer][lAction] = lAction;
      }
      mNumLegals[lPlayer] = 0;
    }

    for (int lCell = 0; lCell < WIDTH * HEIGHT; lCell++)
    {
      int lRow = lCell / WIDTH;
      if (lRow < 2)
      {
        // Starting cell for player 0.
        mCell[lCell] = 0;

        if (lRow == 1)
        {
          // The second row of pieces can move.
          legal((byte)0, LFT | lCell);
          legal((byte)0, FWD | lCell);
          legal((byte)0, RGT | lCell);
        }
      }
      else if (lRow >= HEIGHT - 2)
      {
        // Starting cell for player 1.
        mCell[lCell] = 1;

        if (lRow == HEIGHT - 2)
        {
          legal((byte)1, LFT | lCell);
          legal((byte)1, FWD | lCell);
          legal((byte)1, RGT | lCell);
        }
      }
      else
      {
        // Blank cell.
        mCell[lCell] = 2;
      }
    }
  }

  @Override
  public void copyTo(GameState xiDestination)
  {
    BreakthroughState xiNew = (BreakthroughState)xiDestination;

    System.arraycopy(mCell, 0, xiNew.mCell, 0, WIDTH * HEIGHT);
    xiNew.mPlayer = mPlayer;
    xiNew.mTerminal = mTerminal;
    xiNew.mReward = mReward;

    for (int lPlayer = 0; lPlayer < 2; lPlayer++)
    {
      xiNew.mPieceCount[lPlayer] = mPieceCount[lPlayer];
      xiNew.mNumLegals[lPlayer] = mNumLegals[lPlayer];
      System.arraycopy(mLegals[lPlayer], 0, xiNew.mLegals[lPlayer], 0, mLegals[lPlayer].length);
      System.arraycopy(mActionIndex[lPlayer], 0, xiNew.mActionIndex[lPlayer], 0, mActionIndex[lPlayer].length);
    }
  }

  private static int index(int xiRow, int xiCol)
  {
    return (xiRow * WIDTH) + xiCol;
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
    return mNumLegals[mPlayer];
  }

  @Override
  public void applyAction(int xiActionIndex)
  {
    int lAction = mLegals[mPlayer][xiActionIndex];
    int lSrc = lAction & 0b111111;
    int lSrcRow = lSrc / WIDTH;
    int lSrcCol = lSrc % WIDTH;
    int lFwd = (mPlayer == 0) ? 1 : -1;
    int lDir = ((lAction >> 6) & 0b11) - 1;
    int lDstRow = lSrcRow + lFwd;
    int lDstCol = lSrcCol + lDir;
    int lDst = index(lDstRow, lDstCol);

    byte lOpponent = (byte)(1 - mPlayer);

    boolean lCapture = (mCell[lDst] != 2);

    // Update the board.
    assert(mCell[lSrc] == mPlayer) : "Player doesn't have a piece in that cell";
    assert(mCell[lDst] != mPlayer) : "Target cell is blocked by player's own piece";
    mCell[lSrc] = 2;
    mCell[lDst] = mPlayer;

    if (lCapture)
    {
      mPieceCount[lOpponent]--;
    }

    // Check whether the game is over.
    terminalCheck(lDst);
    if (!mTerminal)
    {
      // Update the legals.  First for us.

      // The cell we've just moved from is now empty so can't be moved from again.
      notLegal(mPlayer, LFT | lSrc);
      notLegal(mPlayer, FWD | lSrc);
      notLegal(mPlayer, RGT | lSrc);

      // We can (potentially) move on from the cell that we've just arrived at.
      if ((lDstCol > 0) && (mCell[index(lDstRow + lFwd, lDstCol - 1)] != mPlayer))
      {
        // Diagonally left isn't blocked by us, so we can go there.
        legal(mPlayer, LFT | lDst);
      }
      if ((lDstCol < WIDTH - 1) && (mCell[index(lDstRow + lFwd, lDstCol + 1)] != mPlayer))
      {
        // Diagonally right isn't blocked by us, so we can go there.
        legal(mPlayer, RGT | lDst);
      }
      if (mCell[index(lDstRow + lFwd, lDstCol)] == 2)
      {
        // Straight ahead is empty, so we can go there.
        legal(mPlayer, FWD | lDst);
      }

      // We might have moved in front of our own piece, thereby blocking it.
      notLegal(mPlayer, FWD | index(lSrcRow, lDstCol));
      if (lDstCol > 0)
      {
        notLegal(mPlayer, RGT | index(lSrcRow, lDstCol - 1));
      }
      if (lDstCol < WIDTH - 1)
      {
        notLegal(mPlayer, LFT | index(lSrcRow, lDstCol + 1));
      }

      // We might have moved away from being in front of our piece, thereby unblocking it.
      int lRowBehind = lSrcRow - lFwd;
      if (lRowBehind >=0 && lRowBehind < HEIGHT)
      {
        if (mCell[index(lRowBehind, lSrcCol)] == mPlayer)
        {
          legal(mPlayer, FWD | index(lRowBehind, lSrcCol));
        }
        if ((lSrcCol < WIDTH - 1) && (mCell[index(lRowBehind, lSrcCol + 1)] == mPlayer))
        {
          legal(mPlayer, LFT | index(lRowBehind, lSrcCol + 1));
        }
        if ((lSrcCol > 0) && (mCell[index(lRowBehind, lSrcCol - 1)] == mPlayer))
        {
          legal(mPlayer, RGT | index(lRowBehind, lSrcCol - 1));
        }
      }

      // Then for our opponent.
      if (lCapture)
      {
        // If we've captured an opponent piece, that piece can't move now.
        notLegal(lOpponent, LFT | lDst);
        notLegal(lOpponent, FWD | lDst);
        notLegal(lOpponent, RGT | lDst);

      }

      // We might have unblocked an opponent piece (by moving diagonally, but no need to check that here - the fact
      // that we've moved is enough).
      int lStraightAhead = index(lSrcRow + lFwd, lSrcCol);
      if (mCell[lStraightAhead] == lOpponent)
      {
        legal(lOpponent, FWD | lStraightAhead);
      }

      // We might have moved in front of an opponent piece, thereby blocking it.
      notLegal(lOpponent, FWD | index(lDstRow + lFwd, lDstCol));
    }

    // Other player to move.
    mPlayer = lOpponent;
  }

  private void notLegal(byte xiPlayer, int xiAction)
  {
    int lCurrentIndex = mActionIndex[xiPlayer][xiAction];
    if (lCurrentIndex < mNumLegals[xiPlayer])
    {
      swapLegals(xiPlayer, xiAction, lCurrentIndex, --mNumLegals[xiPlayer]);
    }
  }

  private void legal(byte xiPlayer, int xiAction)
  {
    // Check that the move wouldn't leave us out of bounds (and ignore this call if it would).
    int lSrc = xiAction & 0b00111111;
    int lSrcCol = lSrc % WIDTH;
    int lDir = ((xiAction >> 6) & 0b11) - 1;
    int lDstCol = lSrcCol + lDir;
    if ((lDstCol < 0) || (lDstCol >= WIDTH))
    {
      return;
    }

    int lCurrentIndex = mActionIndex[xiPlayer][xiAction];
    if (lCurrentIndex >= mNumLegals[xiPlayer])
    {
      swapLegals(xiPlayer, xiAction, lCurrentIndex, mNumLegals[xiPlayer]++);
    }
  }

  private void swapLegals(byte xiPlayer, int xiAction, int xiCurrentIndex, int xiNewIndex)
  {
    int lTempAction = mLegals[xiPlayer][xiNewIndex];
    mLegals[xiPlayer][xiNewIndex] = mLegals[xiPlayer][xiCurrentIndex];
    mLegals[xiPlayer][xiCurrentIndex] = lTempAction;

    mActionIndex[xiPlayer][xiAction] = xiNewIndex;
    mActionIndex[xiPlayer][lTempAction] = xiCurrentIndex;
  }

  private void terminalCheck(int xiDestination)
  {
    // Check whether the moving player has just won by reaching the back row or by capturing all the opponent's pieces.
    int lDstRow = xiDestination / WIDTH;
    mTerminal = ((lDstRow == 0) && (mPlayer == 1)) ||
                ((lDstRow == HEIGHT - 1) && (mPlayer == 0)) ||
                (mPieceCount[1 - mPlayer] == 0);
    mReward = 1.0 - mPlayer;
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
    StringBuilder lResult = new StringBuilder((WIDTH * HEIGHT * 2) + (HEIGHT * 2) + 10);
    for (int lRow = HEIGHT - 1; lRow >= 0; lRow--)
    {
      for (int lCol = 0; lCol < WIDTH; lCol++)
      {
        int lCell = mCell[index(lRow, lCol)];
        lResult.append(lCell == 2 ? ". " : lCell + " ");
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
    // The game state is entirely encoded in the mCell array + the player whose turn it is.
    BreakthroughState lOther = (BreakthroughState)xiOther;
    return (mPlayer == lOther.mPlayer) && (Arrays.equals(mCell, lOther.mCell));
  }

  @Override
  public int hashCode()
  {
    // See equals().
    return Arrays.hashCode(mCell) | mPlayer;
  }

  /**
   * Factory for creating starting states for Breakthrough.
   */
  public static class BreakthroughGameStateFactory implements GameStateFactory
  {
    @Override
    public GameState createInitialState()
    {
      return new BreakthroughState();
    }
  }
}
