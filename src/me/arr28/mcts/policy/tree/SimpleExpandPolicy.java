package me.arr28.mcts.policy.tree;

import me.arr28.mcts.TreeNode;
import me.arr28.mcts.policy.ExpandPolicy;

/**
 * A simple expansion policy that expands a single child at random.
 *
 * @author Andrew Rose
 */
public class SimpleExpandPolicy implements ExpandPolicy
{
  @Override
  public TreeNode expand(TreeNode xiLeaf)
  {
    // Expand an unexpanded child.
    TreeNode lChild = null;
    TreeNode[] lChildren = xiLeaf.getChildren();
    int lNumActions = xiLeaf.getGameState().getNumLegalActions();
    for (int lAction = 0; lAction < lNumActions; lAction++)
    {
      if (lChildren[lAction] == null)
      {
        lChild = xiLeaf.addChild(lAction);
        if (lChild != null)
        {
          break;
        }
      }
    }
    return lChild;
  }
}
