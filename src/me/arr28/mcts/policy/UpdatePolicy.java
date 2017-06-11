package me.arr28.mcts.policy;

import me.arr28.mcts.TreeNode;

/**
 * An MCTS update policy.
 *
 * @author Andrew Rose
 */
public interface UpdatePolicy
{
  /**
   * Update a tree with the result of an MCTS rollout.
   *
   * @param xiResult - the result, for player 0.
   * @param xiPath - the selected path, from root to leaf.
   * @param xiPathLength - the length of the selected path.  (The path array may be longer than this, but any other
   * elements will be garbage.)
   */
  public void update(double xiResult, TreeNode[] xiPath, int xiPathLength);
}
