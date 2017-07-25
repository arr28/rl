package me.arr28.algs.mcts.policy.tree;

import me.arr28.algs.mcts.ScoreBoard;
import me.arr28.algs.mcts.TreeNode;
import me.arr28.algs.mcts.policy.SelectPolicy;

/**
 * Select the child node with the maximum UCB1 score.
 *
 * @author Andrew Rose
 */
public class UCB1SelectPolicy implements SelectPolicy
{
  @Override
  public TreeNode select(TreeNode xiNode)
  {
    // Scan the children and record the best.
    double lBestScore = -1;
    TreeNode[] lChildren = xiNode.getChildren();
    TreeNode lBestChild = null;
    int lNumActions = xiNode.getGameState().getNumLegalActions();
    for (int lAction = 0; lAction < lNumActions; lAction++)
    {
      TreeNode lChild = lChildren[lAction];
      double lScore = calculateUCB1(xiNode, lChild);
      if (lScore > lBestScore)
      {
        lBestChild = lChild;
        lBestScore = lScore;
      }
    }
    return lBestChild;
  }

  private static double calculateUCB1(TreeNode xiParent, TreeNode xiChild)
  {
    ScoreBoard lParentScores = xiParent.mScoreBoard;
    ScoreBoard lChildScores = xiChild.mScoreBoard;
    return lChildScores.getAverageReward() +
           Math.sqrt(2.0 * Math.log(lParentScores.getSelectCount()) / lChildScores.getSelectCount());
  }
}
