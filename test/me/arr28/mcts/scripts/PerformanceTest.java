package me.arr28.mcts.scripts;
import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;
import me.arr28.games.BreakthroughState.BreakthroughGameStateFactory;
import me.arr28.games.Connect4State.C4GameStateFactory;
import me.arr28.games.DummyState.DummyGameStateFactory;
import me.arr28.games.FastConnect4State.FastC4GameStateFactory;
import me.arr28.games.HexState.HexGameStateFactory;
import me.arr28.mcts.MCTSTree;
import me.arr28.mcts.ScoreBoard;
import me.arr28.mcts.ScoreBoard.PlainScoreBoardFactory;
import me.arr28.mcts.ScoreBoardFactory;
import me.arr28.mcts.TreeNode;
import me.arr28.mcts.policy.ExpandPolicy;
import me.arr28.mcts.policy.RolloutPolicy;
import me.arr28.mcts.policy.SelectPolicy;
import me.arr28.mcts.policy.UpdatePolicy;
import me.arr28.mcts.policy.rollout.DummyRolloutPolicy;
import me.arr28.mcts.policy.rollout.SimpleRolloutPolicy;
import me.arr28.mcts.policy.tree.Exp3Policy;
import me.arr28.mcts.policy.tree.Exp3Policy.Exp3ScoreBoardFactory;
import me.arr28.mcts.policy.tree.NoExpandPolicy;
import me.arr28.mcts.policy.tree.RandomSelectPolicy;
import me.arr28.mcts.policy.tree.SimpleExpandPolicy;
import me.arr28.mcts.policy.tree.SimpleUpdatePolicy;
import me.arr28.mcts.policy.tree.UCB1ApproxSelectPolicy;
import me.arr28.mcts.policy.tree.UCB1SelectPolicy;

public class PerformanceTest
{
  /**
   * Configuration.
   */
  private static final int MAX_THREADS = 4;
  private static final int NUM_REPEATS = 3;
  private static final int NUM_PURE_ROLLOUTS = 1_000_000;
  private static final int NUM_MCTS_ITERATIONS = 0; // MCTSTree.NODE_POOL_SIZE - 1;
  private static final Policies POLICY_SET = Policies.UCT;
  private static final Game GAME = Game.CONNECT4FAST;

  private static enum Policies
  {
    SUPER_SIMPLE, UCT, UCT_APPROX, EXP3, TREE_SPEED;
  }

  private static enum Game
  {
    CONNECT4, CONNECT4FAST, HEX, BREAKTHROUGH;
  }

  private final SelectPolicy lSelectPolicy;
  private final ExpandPolicy lExpandPolicy;
  private final RolloutPolicy lRolloutPolicy;
  private final UpdatePolicy lUpdatePolicy;
  private final GameStateFactory mGameStateFactory;
  private final ScoreBoardFactory lScoreBoardFactory;

  public static void main(String[] xiArgs)
  {
    PerformanceTest lTest = new PerformanceTest();
    lTest.testAll();
  }

  /**
   * Configure the performance test.
   */
  public PerformanceTest()
  {
    GameStateFactory lGameStateFactory = null;

    switch (POLICY_SET)
    {
      case SUPER_SIMPLE:
        lSelectPolicy = new RandomSelectPolicy();
        lExpandPolicy = new NoExpandPolicy();
        lRolloutPolicy = new SimpleRolloutPolicy();
        lUpdatePolicy = new SimpleUpdatePolicy();
        lScoreBoardFactory = new PlainScoreBoardFactory();
        break;

      case UCT:
        lSelectPolicy = new UCB1SelectPolicy();
        lExpandPolicy = new SimpleExpandPolicy();
        lRolloutPolicy = new SimpleRolloutPolicy();
        lUpdatePolicy = new SimpleUpdatePolicy();
        lScoreBoardFactory = new PlainScoreBoardFactory();
        break;

      case UCT_APPROX:
        lSelectPolicy = new UCB1ApproxSelectPolicy();
        lExpandPolicy = new SimpleExpandPolicy();
        lRolloutPolicy = new SimpleRolloutPolicy();
        lUpdatePolicy = new SimpleUpdatePolicy();
        lScoreBoardFactory = new PlainScoreBoardFactory();
        break;

      case EXP3:
        lSelectPolicy = new Exp3Policy();
        lExpandPolicy = new SimpleExpandPolicy();
        lRolloutPolicy = new SimpleRolloutPolicy();
        lUpdatePolicy = (UpdatePolicy)lSelectPolicy;
        lScoreBoardFactory = new Exp3ScoreBoardFactory();
        break;

      case TREE_SPEED:
        lSelectPolicy = new UCB1ApproxSelectPolicy();
        lExpandPolicy = new SimpleExpandPolicy();
        lRolloutPolicy = new DummyRolloutPolicy();
        lUpdatePolicy = new SimpleUpdatePolicy();
        lScoreBoardFactory = new PlainScoreBoardFactory();
        lGameStateFactory = new DummyGameStateFactory();
        break;

      default:
        throw new IllegalStateException("Invalid configuration");
    }

    if (lGameStateFactory == null)
    {
      switch (GAME)
      {
        case CONNECT4: lGameStateFactory = new C4GameStateFactory(); break;
        case CONNECT4FAST: lGameStateFactory = new FastC4GameStateFactory(); break;
        case HEX: lGameStateFactory = new HexGameStateFactory(); break;
        case BREAKTHROUGH: lGameStateFactory = new BreakthroughGameStateFactory(); break;
        default: throw new IllegalStateException("Invalid game selected");
      }
    }

    mGameStateFactory = lGameStateFactory;
  }

  private void testAll()
  {
    testPureRolloutSpeed();
    testFullMCTSSpeed();
  }

  private void testPureRolloutSpeed()
  {
    if (NUM_PURE_ROLLOUTS > 0)
    {
      System.out.println("Running " + NUM_PURE_ROLLOUTS + " pure rollouts (single-threaded)");
      GameState lInitialState = mGameStateFactory.createInitialState();
      GameState lRolloutState = mGameStateFactory.createInitialState();
      long lStartTime = System.nanoTime();
      for (int lii = 0; lii < NUM_PURE_ROLLOUTS; lii++)
      {
        lInitialState.copyTo(lRolloutState);
        lRolloutPolicy.rollout(lRolloutState);
      }
      long lRolloutTime = System.nanoTime() - lStartTime;
      System.out.print("  Took " + (lRolloutTime / 1_000_000) + "ms");
      System.out.print(" = " + (lRolloutTime  / NUM_PURE_ROLLOUTS) + "ns per rollout");
      System.out.println(" = " + (long)(NUM_PURE_ROLLOUTS * ((double)1_000_000_000 / lRolloutTime)) + " rollouts/s");
    }
  }

  private void testFullMCTSSpeed()
  {
    if (NUM_MCTS_ITERATIONS == 0) return;

    long[] lSpeed = new long[MAX_THREADS];
    for (int lThreads = 1; lThreads <= MAX_THREADS; lThreads++)
    {
      System.out.println("=== Testing with " + lThreads + " thread(s) ===");
      lSpeed[lThreads - 1] = testFullMCTSSpeedNThreads(lThreads);
      System.out.println();
    }

    System.out.println("====================");
    for (int lThreads = 1; lThreads <= MAX_THREADS; lThreads++)
    {
      long lThisSpeed = lSpeed[lThreads - 1];
      System.out.println(lThreads + " thread(s) performed " +
                         lThisSpeed + " iterations/s (" +
                         lThisSpeed * 100 / lSpeed[0] + "%)");
    }
  }

  private long testFullMCTSSpeedNThreads(int xiNumThreads)
  {
    int lIterationsPerThread = NUM_MCTS_ITERATIONS / xiNumThreads;

    // Test tree building
    long lBestSpeed = 0;
    for (int lTestRun = 0; lTestRun < NUM_REPEATS; lTestRun++)
    {
      System.gc();
      MCTSTree lMCTS = new MCTSTree(lSelectPolicy,
                                    lExpandPolicy,
                                    lRolloutPolicy,
                                    lUpdatePolicy,
                                    mGameStateFactory,
                                    lScoreBoardFactory);

      // Test full MCTS iteration speed.
      MCTSWorker[] lWorker = new MCTSWorker[xiNumThreads];
      Thread[] lWorkerThread = new Thread[xiNumThreads];
      for (int lii = 0; lii < xiNumThreads; lii++)
      {
        lWorker[lii] = new MCTSWorker(lMCTS, lIterationsPerThread);
        lWorkerThread[lii] = new Thread(lWorker[lii], "MCTS Worker " + lii);
      }

      System.out.println("Starting MCTS iterations on " + xiNumThreads + " threads");
      for (int lii = 0; lii < xiNumThreads; lii++)
      {
        lWorkerThread[lii].start();
      }
      long lTotalIterationTime = 0;
      long lTotalIterationsPerSecond = 0;
      for (int lii = 0; lii < xiNumThreads; lii++)
      {
        try
        {
          lWorkerThread[lii].join();
        }
        catch (InterruptedException ex)
        {
          throw new RuntimeException("Failed to join worker", ex);
        }

        long lIterationTime = lWorker[lii].getElapsedTime();
        long lIterationsPerSecond = (long)(lIterationsPerThread * ((double)1_000_000_000 / lIterationTime));
        lTotalIterationTime += lIterationTime;
        lTotalIterationsPerSecond += lIterationsPerSecond;
        System.out.println("  Worker " + lii + " performed " + lIterationsPerSecond + " iterations/s");
      }
      System.out.println("  Took " + (lTotalIterationTime / 1_000_000) + "ms total CPU time");
      System.out.println("    = " + (lTotalIterationTime  / NUM_MCTS_ITERATIONS) + "ns per iteration");
      System.out.println("    = " + lTotalIterationsPerSecond + " parallel iterations/s");
      System.out.println("Average reward: " + lMCTS.getRoot().mScoreBoard.getAverageReward());
      dumpRootStats(NUM_MCTS_ITERATIONS, lMCTS);
      lBestSpeed = Math.max(lBestSpeed, lTotalIterationsPerSecond);
    }

    return lBestSpeed;
  }

  private static void dumpRootStats(int xiIteration, MCTSTree xiTree)
  {
    System.out.print(xiIteration);
    for (TreeNode lChild : xiTree.getRoot().getChildren())
    {
      if (lChild != null)
      {
        ScoreBoard lSB = lChild.mScoreBoard;
        System.out.print("," + String.format("%.3g", lSB.getAverageReward()));
      }
    }
    System.out.println();
    System.out.println(xiTree.getRoot().mScoreBoard.getSelectCount() + " iterations recorded");
  }

  private static class MCTSWorker implements Runnable
  {
    private final MCTSTree mTree;
    private final int mIterations;
    private long mTime;

    public MCTSWorker(MCTSTree xiTree, int xiIterations)
    {
      mTree = xiTree;
      mIterations = xiIterations;
    }

    @Override
    public void run()
    {
      mTree.warm(mIterations);
      long lStart = System.nanoTime();
      mTree.iterate(mIterations);
      mTime = System.nanoTime() - lStart;
    }

    public long getElapsedTime()
    {
      return mTime;
    }
  }
}
