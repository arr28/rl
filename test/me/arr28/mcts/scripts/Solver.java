package me.arr28.mcts.scripts;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;
import me.arr28.games.FastConnect4State.FastC4GameStateFactory;
import me.arr28.mcts.zeroalloc.LRUCachable;
import me.arr28.mcts.zeroalloc.LRUCache;

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

    // !! When starting at the real root state, the 1-move scores are reported as
    // !! loss, loss, draw, draw, draw, draw
    // !!
    // !! But when playing the first move manually, the 0-move scores are reported as
    // !! loss, loss, loss, draw, draw, loss
    // !!
    // !! Which makes it look like there's some state corruption happening.
    // !!
    // !! Aha - turning of caching solves it.  I strongly suspect this is a nasty interaction
    // !! between caching & alpha-beta pruning.  Due to a-b pruning, we might choose to stop looking when we find a draw
    // !! even if the position is actually a win (because we know that the previous node can already hold us to a draw).
    // !! That's fine in context.  But we would then cache the state as a draw (when it could still be a win).  Then we
    // !! might use the cached state's value in a different context.
    // !!
    // !! 1 possible solution is to store the relevant a-b parameters and only use the cached value if the new context
    // !! is the same as the context in which it was stored.  Otherwise do the full calculation and updated the cached
    // !! value.  (Reusing cached results from a less constrained run is also okay.  And we only care about 1 of alpha or
    // !! beta.)

    double lValue = solve(lInitialState, 0, 0, 1);
    log("Value of state is " + lValue + "\n" + lInitialState);
    logStats();
  }

  private double solve(GameState xiState, int xiDepth, double xiAlpha, double xiBeta)
  {
    // log("solve(depth=" + xiDepth + ", alpha=" + xiAlpha + ", beta=" + xiBeta + ")...\n" + xiState);

    // boolean lDoCache = ((xiDepth % 4 == 0) && (xiDepth < 28));
    boolean lDoCache = false; // !! ARR Enable caching
    if (lDoCache && mCache.contains(xiState))
    {
      return mCache.get(xiState).mReward;
    }

    if (xiState.isTerminal())
    {
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

    if (xiDepth <= mSmallestCompleteDepth)
    {
      mSmallestCompleteDepth = xiDepth;
      log("Completed a sub-tree at depth: " + mSmallestCompleteDepth + " with result " + lBestReward /*+ "\n" + xiState*/);
      if (xiDepth == 1)
      {
        log("\n" + xiState);
      }
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

    // log("Solved (depth=" + xiDepth + ", alpha=" + xiAlpha + ", beta=" + xiBeta + ") with reward " + lBestReward + " for\n" + xiState);

    return lBestReward;
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
    log("Terminals, Peeks, Cache Size, Hits, Evictions = " + mTerminalStatesVisited + ", " + mPeekResults + ", " + mCache.size() + ", " + mCache.mHits + ", " + mCache.mEvictions);
  }
}
