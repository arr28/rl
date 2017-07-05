package me.arr28.mcts.scripts;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;
import me.arr28.games.Connect4State.C4GameStateFactory;
import me.arr28.mcts.zeroalloc.LRUCachable;
import me.arr28.mcts.zeroalloc.LRUCache;

public class Solver
{
  public static final int MAX_DEPTH = 50;
  public static final int MAX_ACTIONS = 7;
  public static final int MAX_CACHE_SIZE = 5_000_000;
  public static final boolean DO_ONE_MOVE_LOOKAHEAD = true;

  public static class CachableState implements LRUCachable<GameState, CachableState>
  {
    private CachableState mPrev;
    private CachableState mNext;
    public GameState mState;
    public double mReward;

    @Override public void setPrev(CachableState xiPrev) {mPrev = xiPrev;}
    @Override public CachableState getPrev() {return mPrev;}
    @Override public void setNext(CachableState xiNext) {mNext = xiNext;}
    @Override public CachableState getNext() {return mNext;}
    @Override public GameState getKey() {return mState;}

    @Override public int hashCode() {return mState.hashCode();}
    @Override public boolean equals(Object xiOther)
    {
      return mState.equals(((CachableState)xiOther).mState);
    }
  }

  private final GameStateFactory mGameStateFactory = new C4GameStateFactory();
  private final LRUCache<GameState, CachableState> mCache = new LRUCache<>(MAX_CACHE_SIZE);

  private long mTerminalStatesVisited;
  private long mWinningPeeks;
  private int mSmallestCompleteDepth;

  private final GameState[][] mLookAhead = new GameState[MAX_DEPTH][MAX_ACTIONS];

  public static void main(String[] xiArgs)
  {
    Solver lSolver = new Solver();
    lSolver.solve();
  }

  private void solve()
  {
    mSmallestCompleteDepth = MAX_DEPTH;

    for (int lDepth = 0; lDepth < MAX_DEPTH; lDepth++)
    {
      for (int lAction = 0; lAction < MAX_ACTIONS; lAction++)
      {
        mLookAhead[lDepth][lAction] = mGameStateFactory.createInitialState();
      }
    }

    double lValue = solve(mGameStateFactory.createInitialState(), 0, 0, 1);
    log("Value of initial state is " + lValue);
  }

  private double solve(GameState xiState, int xiDepth, double xiAlpha, double xiBeta)
  {
    boolean lDoCache = ((xiDepth % 4 == 0) && (xiDepth < 40));
    if (lDoCache && mCache.contains(xiState))
    {
      return mCache.get(xiState).mReward;
    }

    if (xiState.isTerminal())
    {
      if (xiState.getReward() != 0.5)
      {
        throw new RuntimeException("Unexpected score " + xiState.getReward() + " in state:\n" + xiState);
      }
      recordTerminal();
      return xiState.getReward();
    }

    int lNumActions = xiState.getNumLegalActions();
    double lBestReward = 0;

    GameState[] lLookAhead = mLookAhead[xiDepth];
    if (xiState.getPlayer() == 0)
    {
      // Player 0 is trying to maximize the score

      // Look for a 1-move win
      boolean lFoundWin = false;
      for (int lAction = 0; lAction < lNumActions; lAction++)
      {
        GameState lPeekState = lLookAhead[lAction];
        xiState.copyTo(lPeekState);
        lPeekState.applyAction(lAction);
        if ((lPeekState.isTerminal()) && (lPeekState.getReward() == 1))
        {
          mWinningPeeks++;
          recordTerminal();
          lBestReward = 1;
          lFoundWin = true;
          break;
        }
      }

      if (!lFoundWin)
      {
        // No immediate win, so explore all options thoroughly.
        lBestReward = 0;
        for (int lAction = 0; lAction < lNumActions; lAction++)
        {
          lBestReward = Math.max(lBestReward, solve(lLookAhead[lAction], xiDepth + 1, xiAlpha, xiBeta));
          xiAlpha = Math.max(xiAlpha, lBestReward);
          if (xiBeta <= xiAlpha) break;
        }
      }
    }
    else
    {
      // Player 1 is trying to minimize the score

      // Look for a 1-move win
      boolean lFoundWin = false;
      for (int lAction = 0; lAction < lNumActions; lAction++)
      {
        GameState lPeekState = lLookAhead[lAction];
        xiState.copyTo(lPeekState);
        lPeekState.applyAction(lAction);
        if ((lPeekState.isTerminal()) && (lPeekState.getReward() == 0))
        {
          mWinningPeeks++;
          recordTerminal();
          lBestReward = 0;
          lFoundWin = true;
          break;
        }
      }

      if (!lFoundWin)
      {
        // No immediate win, so explore all options thoroughly.
        lBestReward = 1;
        for (int lAction = 0; lAction < lNumActions; lAction++)
        {
          lBestReward = Math.min(lBestReward, solve(lLookAhead[lAction], xiDepth + 1, xiAlpha, xiBeta));
          xiBeta = Math.min(xiBeta, lBestReward);
          if (xiBeta <= xiAlpha) break;
        }
      }
    }

    if (xiDepth < mSmallestCompleteDepth)
    {
      mSmallestCompleteDepth = xiDepth;
      log("Completed a sub-tree at depth: " + mSmallestCompleteDepth);
    }

    if (lDoCache)
    {
      CachableState lCachableState;
      if (mCache.size() >= MAX_CACHE_SIZE)
      {
        lCachableState = mCache.evict();
      }
      else
      {
        lCachableState = new CachableState();
        lCachableState.mState = mGameStateFactory.createInitialState();
      }

      xiState.copyTo(lCachableState.mState);
      lCachableState.mReward = lBestReward;
      mCache.put(lCachableState.mState, lCachableState);
    }

    return lBestReward;
  }

  private void recordTerminal()
  {
    if (++mTerminalStatesVisited % 100_000_000L == 0)
    {
      log("Terminals, Wins, Cache Size, Hits, Evictions = " + mTerminalStatesVisited + ", " + mWinningPeeks + ", " + mCache.size() + ", " + mCache.mHits + ", " + mCache.mEvictions);
    }
  }

  private static void log(String xiMessage)
  {
    System.out.println("[" + ZonedDateTime.now().format( DateTimeFormatter.ISO_INSTANT) + "] " + xiMessage);
  }
}