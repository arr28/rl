import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;
import me.arr28.games.BreakthroughState.BreakthroughGameStateFactory;
import me.arr28.games.Connect4State.C4GameStateFactory;
import me.arr28.games.DummyState.DummyGameStateFactory;
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

public class TestHarness
{
  private static enum Policies
  {
    SUPER_SIMPLE, UCT, UCT_APPROX, EXP3, TREE_SPEED;
  }

  private static enum Game
  {
    CONNECT4, HEX, BREAKTHROUGH;
  }

  private static final int NUM_THREADS = 4;
  private static final int NUM_PURE_ROLLOUTS = 0; //10_000_000;
  private static final int NUM_MCTS_ITERATIONS = MCTSTree.NODE_POOL_SIZE - 1;
  private static final Policies POLICY_SET = Policies.UCT_APPROX;
  private static final Game GAME = Game.CONNECT4;

  public static void main(String[] xiArgs) throws InterruptedException
  {
    SelectPolicy lSelectPolicy;
    ExpandPolicy lExpandPolicy;
    RolloutPolicy lRolloutPolicy;
    UpdatePolicy lUpdatePolicy;
    GameStateFactory lGameStateFactory = null;
    ScoreBoardFactory lScoreBoardFactory;

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
        lSelectPolicy = new UCB1SelectPolicy();
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
        case HEX: lGameStateFactory = new HexGameStateFactory(); break;
        case BREAKTHROUGH: lGameStateFactory = new BreakthroughGameStateFactory(); break;
        default: throw new IllegalStateException("Invalid game selected");
      }
    }

    MCTSTree lMCTS = new MCTSTree(lSelectPolicy,
                                  lExpandPolicy,
                                  lRolloutPolicy,
                                  lUpdatePolicy,
                                  lGameStateFactory,
                                  lScoreBoardFactory);

    // Test full MCTS iteration speed.
    Thread[] lWorker = new Thread[NUM_THREADS];
    for (int lii = 0; lii < NUM_THREADS; lii++)
    {
      lWorker[lii] = new Thread(new MCTSWorker(lMCTS, NUM_MCTS_ITERATIONS / NUM_THREADS), "MCTS Worker " + lii);
    }

    if (NUM_PURE_ROLLOUTS > 0)
    {
      System.out.println("Running " + NUM_PURE_ROLLOUTS + " pure rollouts (single-threaded)");
      GameState lInitialState = lGameStateFactory.createInitialState();
      GameState lRolloutState = lGameStateFactory.createInitialState();
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

    System.out.println("Starting MCTS iterations on " + NUM_THREADS + " threads");
    long lStartTime = System.nanoTime();
    for (int lii = 0; lii < NUM_THREADS; lii++)
    {
      lWorker[lii].start();
    }
    for (int lii = 0; lii < NUM_THREADS; lii++)
    {
      lWorker[lii].join();
    }
    long lIterationTime = System.nanoTime() - lStartTime;
    System.out.print("  Took " + (lIterationTime / 1_000_000) + "ms");
    System.out.print(" = " + (lIterationTime  / NUM_MCTS_ITERATIONS) + "ns per iteration");
    System.out.println(" = " + (long)(NUM_MCTS_ITERATIONS * ((double)1_000_000_000 / lIterationTime)) + " iterations/s");
    System.out.println("Average reward: " + lMCTS.getRoot().mScoreBoard.getAverageReward());
    dumpRootStats(NUM_MCTS_ITERATIONS, lMCTS);
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

    public MCTSWorker(MCTSTree xiTree, int xiIterations)
    {
      mTree = xiTree;
      mIterations = xiIterations;
    }

    @Override
    public void run()
    {
      mTree.warm(mIterations);
      mTree.iterate(mIterations);
    }
  }
}
