package me.arr28.algs.mcts.policy.tree;

import java.util.concurrent.ThreadLocalRandom;

import me.arr28.algs.mcts.TreeNode;
import me.arr28.algs.mcts.policy.ExpandPolicy;

/**
 * Expands the root node but then performs no further expansion - i.e. turns MCTS into plain Monte Carlo search.
 *
 * @author Andrew Rose
 */
public class NoExpandPolicy implements ExpandPolicy
{
  private boolean mExpanded = false;

  @Override
  public TreeNode expand(TreeNode xiLeaf)
  {
    if (mExpanded) return xiLeaf;
    mExpanded = true;

    int lNumActions = xiLeaf.getGameState().getNumLegalActions();
    for (int lAction = 0; lAction < lNumActions; lAction++)
    {
      xiLeaf.addChild(lAction);
    }

    return xiLeaf.getChildren()[ThreadLocalRandom.current().nextInt(lNumActions)];
  }
}
