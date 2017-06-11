package me.arr28.mcts;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;
import me.arr28.pool.Pool.ObjectAllocator;

/**
 * A node in an MCTS tree.
 */
public class TreeNode
{
  private static final int MAX_BRANCHING_FACTOR = 100; // !! ARR Branching factor hack.

  private final MCTSTree mTree;
  private final TreeNode[] mChildren = new TreeNode[MAX_BRANCHING_FACTOR];
  private final GameState mGameState;

  /**
   * A record of the scores when rollouts have played through this node.
   */
  public final ScoreBoard mScoreBoard;

  private boolean mTerminal;
  private int mUnexpandedChildCount;

  /**
   * Utility class for allocating tree nodes from a Pool.
   */
  public static class TreeNodeAllocator implements ObjectAllocator<TreeNode>
  {
    private final MCTSTree mTree;
    private final GameStateFactory mGameStateFactory;
    private final ScoreBoardFactory mScoreBoardFactory;

    /**
     * Create a new tree node allocator.
     *
     * @param xiTree - the tree in which nodes are allocated.
     * @param xiGameStateFactory - a factory from creating new game states.
     * @param xiScoreBoardFactory - a factory for creating score boards.
     */
    public TreeNodeAllocator(MCTSTree xiTree,
                             GameStateFactory xiGameStateFactory,
                             ScoreBoardFactory xiScoreBoardFactory)
    {
      mTree = xiTree;
      mGameStateFactory = xiGameStateFactory;
      mScoreBoardFactory = xiScoreBoardFactory;
    }

    @Override
    public TreeNode newObject()
    {
      return new TreeNode(mTree, mGameStateFactory.createInitialState(), mScoreBoardFactory.createScoreBoard());
    }

    @Override
    public void resetObject(TreeNode xiNode)
    {
      xiNode.reset();
    }
  }

  /**
   * Create a tree node.  This should only be called from the allocator.
   */
  TreeNode(MCTSTree xiTree, GameState xiGameState, ScoreBoard xiScoreBoard)
  {
    mTree = xiTree;
    mGameState = xiGameState;
    mScoreBoard = xiScoreBoard;
    reinitialise();
  }

  /**
   * Clear references to other objects (to avoid unwanted retention).
   */
  public void reset()
  {
    synchronized (mChildren)
    {
      for (int lii = 0; lii < mChildren.length; lii++)
      {
        mChildren[lii] = null;
      }
    }
    mScoreBoard.reset();
  }

  /**
   * @return the game state in this node.
   */
  public GameState getGameState()
  {
    return mGameState;
  }

  /**
   * @return whether the game state is terminal.
   */
  public boolean isTerminal()
  {
    return mTerminal;
  }

  /**
   * Add a child of the current node.
   *
   * @param xiAction - the action to perform in this node to reach the child.
   *
   * @return the newly added child node or null if no child was created (because the node had already been expanded).
   */
  public TreeNode addChild(int xiAction)
  {
    // Create a child node.
    TreeNode lChild = mTree.getNodePool().allocate();

    // Give the child our state and then apply the specified action to it.
    GameState lChildState = lChild.getGameState();
    mGameState.copyTo(lChildState);
    lChildState.applyAction(xiAction);
    lChild.reinitialise();

    TreeNode lSavedChild = mTree.getTranspositionTable().putIfAbsent(lChildState, lChild);
    if (lSavedChild == null)
    {
      synchronized (mChildren)
      {
        if (mChildren[xiAction] == null)
        {
          mChildren[xiAction] = lChild;
          mUnexpandedChildCount--;
        }
      }
    }
    else
    {
      // 2 possibilities
      //
      // - Another thread simultaneously expanded this child.
      // - This is a transposition into an existing node elsewhere in the tree.
      synchronized (mChildren)
      {
        if (mChildren[xiAction] == null)
        {
          mChildren[xiAction] = lSavedChild;
          mUnexpandedChildCount--;
        }
      }
      mTree.getNodePool().free(lChild);
      lChild = null;
    }

    return lChild;
  }

  private void reinitialise()
  {
    mTerminal = mGameState.isTerminal();
    if (mTerminal)
    {
      mUnexpandedChildCount = 0;
    }
    else
    {
      mUnexpandedChildCount = mGameState.getNumLegalActions();
    }
  }

  /**
   * @return all the child nodes, ordered by action index.  The array should be treated as read-only.
   */
  public TreeNode[] getChildren()
  {
    return mChildren;
  }

  /**
   * @return whether MCTS selection should stop at this node.  MCTS selection should stop at terminal nodes and nodes
   * with any unexpanded children.
   */
  public boolean shouldStopSelection()
  {
    return mTerminal || (mUnexpandedChildCount > 0);
  }
}
