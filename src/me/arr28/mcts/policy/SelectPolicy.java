package me.arr28.mcts.policy;

import me.arr28.mcts.TreeNode;

/**
 * An MCTS leaf selection policy.
 *
 * @author Andrew Rose
 */
public interface SelectPolicy
{
  /**
   * @return the best child of the specified node (for the player whose turn it is).
   *
   * @param xiNode - the parent node, guaranteed not to be a leaf node.
   */
  public TreeNode select(TreeNode xiNode);
}
