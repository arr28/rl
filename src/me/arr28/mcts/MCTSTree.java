package me.arr28.mcts;

import java.util.concurrent.ConcurrentHashMap;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;
import me.arr28.mcts.TreeNode.TreeNodeAllocator;
import me.arr28.mcts.policy.ExpandPolicy;
import me.arr28.mcts.policy.RolloutPolicy;
import me.arr28.mcts.policy.SelectPolicy;
import me.arr28.mcts.policy.UpdatePolicy;
import me.arr28.pool.CappedPool;
import me.arr28.pool.Pool;

/**
 * A Monte Carlo Search Tree.
 *
 * @author Andrew Rose
 */
public class MCTSTree
{
  private static final int MAX_PATH_LENGTH = 1024;
  public static final int NODE_POOL_SIZE = 3_000_001;

  private final ThreadLocal<CappedPool<TreeNode>> tlNodePool;

  private final SelectPolicy mSelectPolicy;
  private final ExpandPolicy mExpandPolicy;
  private final RolloutPolicy mRolloutPolicy;
  private final UpdatePolicy mUpdatePolicy;
  private final GameStateFactory mGameStateFactory;

  private final ConcurrentHashMap<GameState, TreeNode> mTranpositionTable = new ConcurrentHashMap<>(NODE_POOL_SIZE);

  private TreeNode mRoot;

  /**
   * Create an Monte Carlo Tree Searcher.
   *
   * @param xiSelectPolicy - the policy to use when selecting a node for rollouts.
   * @param xiExpandPolicy - the policy to use when expanding a leaf node.
   * @param xiRolloutPolicy - the policy to use during the rollout.
   * @param xiUpdatePolicy - the policy to use when updating node values.
   * @param xiGameStateFactory - a factory for creation initial game states.
   * @param xiScoreBoardFactory - a factory for creating score boards for the nodes.
   */
  public MCTSTree(SelectPolicy xiSelectPolicy,
                  ExpandPolicy xiExpandPolicy,
                  RolloutPolicy xiRolloutPolicy,
                  UpdatePolicy xiUpdatePolicy,
                  GameStateFactory xiGameStateFactory,
                  ScoreBoardFactory xiScoreBoardFactory)
  {
    mSelectPolicy = xiSelectPolicy;
    mExpandPolicy = xiExpandPolicy;
    mRolloutPolicy = xiRolloutPolicy;
    mUpdatePolicy = xiUpdatePolicy;
    mGameStateFactory = xiGameStateFactory;

    MCTSTree thisTree = this;

    tlNodePool = new ThreadLocal<CappedPool<TreeNode>>() {
      @Override
      protected CappedPool<TreeNode> initialValue()
      {
        return new CappedPool<>(new TreeNodeAllocator(thisTree, xiGameStateFactory, xiScoreBoardFactory), NODE_POOL_SIZE);
      }
    };

    mRoot = new TreeNodeAllocator(this, xiGameStateFactory, xiScoreBoardFactory).newObject();
  }

  /**
   * Perform the specified number of MCTS iterations.
   *
   * @param xiNumIterations - the number of iterations.
   */
  public void iterate(int xiNumIterations)
  {
    // Create resources for this run.
    GameState mState = mGameStateFactory.createInitialState();
    TreeNode[] mPath = new TreeNode[MAX_PATH_LENGTH];

    for (int lii = 0; lii < xiNumIterations; lii++)
    {
      iterate(mState, mPath);
    }
  }

  /**
   * Perform a single MCTS iteration.
   *
   * @param xiDummyState - a state for copying into (a performance optimization).
   * @param xiPath - an array for storing the iteration path.
   */
  private void iterate(GameState xiDummyState, TreeNode[] xiPath)
  {
    // Keep track of the path from root to leaf.
    int lPathLength = 0;

    TreeNode lNode = mRoot;
    lNode.mScoreBoard.selected();
    xiPath[lPathLength++] = lNode;

    boolean lExpanded = false;
    while (!lExpanded)
    {
      // SELECT
      while (!lNode.shouldStopSelection())
      {
        lNode = mSelectPolicy.select(lNode);
        lNode.mScoreBoard.selected();
        xiPath[lPathLength++] = lNode;
      }

      if (lNode.isTerminal())
      {
        // Can't select or expand from a terminal node.
        break;
      }

      // EXPAND
      TreeNode lNewNode = mExpandPolicy.expand(lNode);
      if (lNewNode != null)
      {
        lExpanded = true;
        lNode = lNewNode;
        if (lNode != xiPath[lPathLength - 1])
        {
          lNode.mScoreBoard.selected();
          xiPath[lPathLength++] = lNode;
        }
      }
    }

    // ROLLOUT
    double lResult;
    if (lNode.isTerminal())
    {
      // The game is already terminal here - there's no rollout to do.
      lResult = lNode.getGameState().getReward();
    }
    else
    {
      // Do the rollout, recording the result.
      lNode.getGameState().copyTo(xiDummyState);
      lResult = mRolloutPolicy.rollout(xiDummyState);
    }

    // UPDATE
    mUpdatePolicy.update(lResult, xiPath, lPathLength);
  }

  /**
   * @return the root node of the tree.
   */
  public TreeNode getRoot()
  {
    return mRoot;
  }

  /**
   * @return the node pool from which nodes in this tree are allocated.
   */
  public Pool<TreeNode> getNodePool()
  {
    return tlNodePool.get();
  }

  /**
   * Pre-warm the node pool.
   */
  public void warm(int xiCapacity)
  {

    Pool<TreeNode> lPool = getNodePool();
    for (int lii = 0; lii < xiCapacity; lii++)
    {
      lPool.allocate();
    }
    lPool.clear();
  }

  public ConcurrentHashMap<GameState, TreeNode> getTranspositionTable()
  {
    return mTranpositionTable;
  }
}
