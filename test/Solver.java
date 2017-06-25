import java.util.HashMap;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;
import me.arr28.games.Connect4State.C4GameStateFactory;
import me.arr28.mcts.zeroalloc.Linkable;
import me.arr28.mcts.zeroalloc.ZeroAllocLinkedList;

public class Solver
{
  private static int MAX_CACHE_SIZE = 10_000_000;

  private static class CachableState implements Linkable<CachableState>
  {
    private CachableState mPrev;
    private CachableState mNext;
    private final GameState mState;
    public final double mReward;

    public CachableState(GameState xiState, double xiReward)
    {
      mState = xiState;
      mReward = xiReward;
    }

    @Override public void setPrev(CachableState xiPrev) {mPrev = xiPrev;}
    @Override public CachableState getPrev() {return mPrev;}
    @Override public void setNext(CachableState xiNext) {mNext = xiNext;}
    @Override public CachableState getNext() {return mNext;}

    public GameState getState() {return mState;}

    @Override public int hashCode() {return mState.hashCode();}
    @Override public boolean equals(Object xiOther)
    {
      return mState.equals(((CachableState)xiOther).mState);
    }
  }

  private final GameStateFactory mGameStateFactory = new C4GameStateFactory();

  private final HashMap<GameState, CachableState> mCache = new HashMap<>(MAX_CACHE_SIZE + 100);
  private final ZeroAllocLinkedList<CachableState> mRUOrder = new ZeroAllocLinkedList<>();
  private long mCacheHits;
  private long mCacheEvictions;

  private long mTerminalStatesVisited;
  private int mSmallestCompleteDepth;

  public static void main(String[] xiArgs)
  {
    Solver lSolver = new Solver();
    lSolver.solve();
  }

  private void solve()
  {
    final int MAX_DEPTH = 50;
    mSmallestCompleteDepth = MAX_DEPTH;

    GameState[] lStateStack = new GameState[MAX_DEPTH];
    for (int lii = 0; lii < MAX_DEPTH; lii++)
    {
      lStateStack[lii] = mGameStateFactory.createInitialState();
    }

    double lValue = solve(lStateStack, 0, 0, 1);
    System.out.println("Value of initial state is " + lValue);
  }

  private double solve(GameState[] xiStateStack, int xiDepth, double xiAlpha, double xiBeta)
  {
    GameState lCurrentState = xiStateStack[xiDepth];
    boolean lDoCache = ((xiDepth % 8 == 0) && (xiDepth < 40));
    if (lDoCache && mCache.containsKey(lCurrentState))
    {
      mCacheHits++;
      CachableState lCached = mCache.get(lCurrentState);
      mRUOrder.remove(lCached);
      mRUOrder.add(lCached);
      return lCached.mReward;
    }

    if (lCurrentState.isTerminal())
    {
      if (++mTerminalStatesVisited % 100_000_000L == 0)
      {
        System.out.println("Terminals, Cache Size, Hits, Evictions = " + mTerminalStatesVisited + ", " + mCache.size() + ", " + mCacheHits + ", " + mCacheEvictions);
      }
      return lCurrentState.getReward();
    }

    GameState lNextState = xiStateStack[xiDepth + 1];
    int lNumActions = lCurrentState.getNumLegalActions();
    double lBestReward;

    if (lCurrentState.getPlayer() == 0)
    {
      // Player 0 is trying to maximize the score
      lBestReward = 0;
      for (int lAction = 0; lAction < lNumActions; lAction++)
      {
        lCurrentState.copyTo(lNextState);
        lNextState.applyAction(lAction);
        lBestReward = Math.max(lBestReward, solve(xiStateStack, xiDepth + 1, xiAlpha, xiBeta));
        xiAlpha = Math.max(xiAlpha, lBestReward);
        if (xiBeta <= xiAlpha) break;
      }
    }
    else
    {
      // Player 1 is trying to minimize the score
      lBestReward = 1;
      for (int lAction = 0; lAction < lNumActions; lAction++)
      {
        lCurrentState.copyTo(lNextState);
        lNextState.applyAction(lAction);
        lBestReward = Math.min(lBestReward, solve(xiStateStack, xiDepth + 1, xiAlpha, xiBeta));
        xiBeta = Math.min(xiBeta, lBestReward);
        if (xiBeta <= xiAlpha) break;
      }
    }

    if (xiDepth < mSmallestCompleteDepth)
    {
      mSmallestCompleteDepth = xiDepth;
      System.out.println("Completed a sub-tree at depth: " + mSmallestCompleteDepth);
    }

    if (lDoCache)
    {
      GameState lState = mGameStateFactory.createInitialState();
      lCurrentState.copyTo(lState);
      CachableState lCachableState = new CachableState(lState, lBestReward);
      mCache.put(lState, lCachableState);
      mRUOrder.add(lCachableState);

      if (mCache.size() > MAX_CACHE_SIZE)
      {
        mCacheEvictions++;
        CachableState lEvict = mRUOrder.getFirst();
        mRUOrder.remove(lEvict);
        mCache.remove(lEvict.getState());
      }
    }

    return lBestReward;
  }
}
