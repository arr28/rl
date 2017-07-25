package me.arr28.algs.mcts.policy.tree;

import me.arr28.algs.mcts.ScoreBoard;
import me.arr28.algs.mcts.TreeNode;
import me.arr28.algs.mcts.policy.UpdatePolicy;

public class SimpleUpdatePolicy implements UpdatePolicy
{

  @Override
  public void update(double xiResult, TreeNode[] xiPath, int xiPathLength)
  {
    double lPlayer0Reward = xiResult;
    double lPlayer1Reward = 1.0 - xiResult;

    // For each node in the path (with the exception of the root), starting with the leaf, update the weights.
    for (int lii = xiPathLength - 1; lii > 0; lii--)
    {
      TreeNode lChild = xiPath[lii];
      TreeNode lParent = xiPath[lii - 1];

      // The reward is the reward for the player whose turn it was in the *parent* state - i.e. the player who chose to
      // get to this point.
      double lReward = lParent.getGameState().getPlayer() == 0 ? lPlayer0Reward : lPlayer1Reward;
      ScoreBoard lChildScoreBoard = lChild.mScoreBoard;
      lChildScoreBoard.reward(lReward);
    }
    xiPath[0].mScoreBoard.reward(xiPath[0].getGameState().getPlayer() == 0 ? lPlayer0Reward : lPlayer1Reward);
  }
}
