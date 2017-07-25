package me.arr28.algs.mcts.policy.tree;

import java.util.concurrent.ThreadLocalRandom;

import me.arr28.algs.mcts.TreeNode;
import me.arr28.algs.mcts.policy.SelectPolicy;

/**
 * Select policy that picks at (uniform) random from all actions.  Assumes that all children have already been expanded.
 *
 * @author Andrew Rose
 */
public class RandomSelectPolicy implements SelectPolicy
{
  @Override
  public TreeNode select(TreeNode xiNode)
  {
    int lAction = ThreadLocalRandom.current().nextInt(xiNode.getGameState().getNumLegalActions());
    return xiNode.getChildren()[lAction];
  }
}
