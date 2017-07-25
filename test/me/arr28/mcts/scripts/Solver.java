package me.arr28.mcts.scripts;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;

import me.arr28.algs.mcts.zeroalloc.LRUCachable;
import me.arr28.algs.mcts.zeroalloc.LRUCache;
import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;
import me.arr28.games.FastConnect4State.FastC4GameStateFactory;

public class Solver
{
  public static final int MAX_DEPTH = 50;
  public static final int MAX_ACTIONS = 7;
  public static final int MAX_CACHE_SIZE = 10_000_000;
  public static final boolean DO_ONE_MOVE_LOOKAHEAD = true;

  public static class CachableState implements LRUCachable<GameState, CachableState>
  {
    private CachableState mPrev;
    private CachableState mNext;
    public GameState mState;
    public double mReward;
    public double mAlpha;
    public double mBeta;

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

  private final GameStateFactory mGameStateFactory = new FastC4GameStateFactory();
  private final LRUCache<GameState, CachableState> mCache = new LRUCache<>(MAX_CACHE_SIZE);

  private long mTerminalStatesVisited;
  private long mPeekResults;
  private int mSmallestCompleteDepth;
  private long mUnusableCacheHits;

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

    GameState lInitialState = mGameStateFactory.createInitialState();
    // --- Column numbering: 012345
    // lInitialState.applyAction(5);
    double lValue = solve(lInitialState, 0, 0, 1);
    log("Value of state is " + lValue + "\n" + lInitialState);
    logStats();
  }

  private double solve(GameState xiState, int xiDepth, double xiAlpha, double xiBeta)
  {
    double lInitialAlpha = xiAlpha;
    double lInitialBeta = xiBeta;

    if (xiState.isTerminal())
    {
      recordTerminal();
      return xiState.getReward();
    }

    boolean lDoCache = ((xiDepth % 4 == 0) && (xiDepth < 28));
    CachableState lCachedState;
    if (lDoCache && ((lCachedState = mCache.get(xiState)) != null))
    {
      // We can only use the cached state if the alpha/beta values it was
      // recorded with an no more restrictive than the current values.
      //if ((xiState.getPlayer() == 0 && xiBeta <= lCachedState.mBeta) ||
      //    (xiState.getPlayer() == 1 && xiAlpha >= lCachedState.mAlpha))
      if ((xiAlpha >= lCachedState.mAlpha) && (xiBeta <= lCachedState.mBeta))
      {
        return lCachedState.mReward;
      }

      // Remove the item from the cache because we're going to replace it.
      mUnusableCacheHits++;
      mCache.remove(lCachedState);
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
        if ((lPeekState.isTerminal()) && (lPeekState.getReward() >= xiBeta))
        {
          mPeekResults++;
          recordTerminal();
          lBestReward = lPeekState.getReward();
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
        if ((lPeekState.isTerminal()) && (lPeekState.getReward() <= xiAlpha))
        {
          mPeekResults++;
          recordTerminal();
          lBestReward = lPeekState.getReward();
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

    if ((xiDepth < mSmallestCompleteDepth) || (xiDepth == 1))
    {
      mSmallestCompleteDepth = xiDepth;
      log("Completed a sub-tree at depth: " + mSmallestCompleteDepth + " with result " + lBestReward);
      if (xiDepth == 1)
      {
        log("\n" + xiState);
      }
    }

    if (lDoCache)
    {
      cache(xiState, lBestReward, lInitialAlpha, lInitialBeta);
    }

    return lBestReward;
  }

  private void cache(GameState xiState, double xiBestReward, double xiAlpha, double xiBeta)
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
    lCachableState.mReward = xiBestReward;
    lCachableState.mAlpha = xiAlpha;
    lCachableState.mBeta = xiBeta;
    mCache.put(lCachableState.mState, lCachableState);
  }

  private void recordTerminal()
  {
    if (++mTerminalStatesVisited % 100_000_000L == 0)
    {
      logStats();
    }
  }

  private static void log(String xiMessage)
  {
    System.out.println("[" + ZonedDateTime.now().format( DateTimeFormatter.ISO_INSTANT) + "] " + xiMessage);
  }

  private void logStats()
  {
    log("Terminals, Peeks, Cache Size, Hits, Unusable, Evictions = " +
        mTerminalStatesVisited + ", " +
        mPeekResults + ", " +
        mCache.size() + ", " +
        mCache.mHits + ", " +
        mUnusableCacheHits + ", " +
        mCache.mEvictions);
  }
}
