package me.arr28.mcts.policy;

import me.arr28.mcts.TreeNode;

/**
 * An MCTS node expansion policy.
 *
 * @author Andrew Rose
 */
public interface ExpandPolicy
{
  /**
   * Expand a single child of the specified parent node (currently a leaf).
   *
   * @param xiLeaf - the leaf node to expand.
   *
   * @return the child to use next (which might be the leaf or a new node).
   */
  public TreeNode expand(TreeNode xiLeaf);
}
