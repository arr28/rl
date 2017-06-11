package me.arr28.mcts.policy.tree;

import java.util.concurrent.ThreadLocalRandom;

import me.arr28.mcts.ScoreBoard;
import me.arr28.mcts.ScoreBoardFactory;
import me.arr28.mcts.TreeNode;
import me.arr28.mcts.policy.SelectPolicy;
import me.arr28.mcts.policy.UpdatePolicy;

/**
 * Exponential-weight algorithm for Exploration and Exploitation (Exp3).
 *
 * @see <a href="https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/">Exp3 Algorithm</a>
 *
 * @author Andrew Rose
 */
public class Exp3Policy implements SelectPolicy,
                                   UpdatePolicy
{
  private static final double GAMMA = 0.1;

  /**
   * Weights used to track the score of the node.
   */
  private static class Exp3ScoreBoard extends ScoreBoard
  {
    /**
     * The current weight of the node for the player who has just played.
     */
    double mWeight;

    /**
     * Create a score board for the Exp3 algorithm.
     *
     * @param xiNumPlayers - the number of players in the game.
     */
    public Exp3ScoreBoard()
    {
      mWeight = 1.0;
    }

    @Override
    public double getSelectionWeight()
    {
      // !! ARR Need to include GAMMA in this.
      return mWeight;
    }
  }

  /**
   * Factory for these ScoreBoards.
   */
  public static class Exp3ScoreBoardFactory implements ScoreBoardFactory
  {
    @Override
    public ScoreBoard createScoreBoard()
    {
      return new Exp3ScoreBoard();
    }
  }

  @Override
  public TreeNode select(TreeNode xiNode)
  {
    // This implementation is O(N) in the number of children.  It's possible to do log2(N) selections and updates.

    double lSelectedWeight = ThreadLocalRandom.current().nextDouble() * getTotalChildWeight(xiNode);

    double lRunningWeight = 0;
    TreeNode[] lChildren = xiNode.getChildren();
    int lNumActions = xiNode.getGameState().getNumLegalActions();
    int lAction;
    for (lAction = 0; lAction < lNumActions; lAction++)
    {
      TreeNode lChild = lChildren[lAction];
      if (lChild == null)
      {
        lRunningWeight += 1.0;
      }
      else
      {
        lRunningWeight += lChild.mScoreBoard.getSelectionWeight();
      }
      if (lRunningWeight > lSelectedWeight) break;
    }

    // Rounding error.  Should be choosing the last node.
    if (lAction == lNumActions) lAction--;

    // Create the child if the edge doesn't exist yet.
    if (lChildren[lAction] == null)
    {
      xiNode.addChild(lAction);
    }

    return xiNode.getChildren()[lAction];
  }

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
      double lReward = lParent.getGameState().getPlayer() == 0 ? lPlayer0Reward : lPlayer1Reward;
      Exp3ScoreBoard lChildScoreBoard = (Exp3ScoreBoard)lChild.mScoreBoard;
      lChildScoreBoard.reward(lReward);

      double lProbability = lChildScoreBoard.mWeight / getTotalChildWeight(lParent);
      double lEstimatedReward = lReward / lProbability;
      lChildScoreBoard.mWeight *= Math.exp(lEstimatedReward * GAMMA / lParent.getGameState().getNumLegalActions());
    }
    xiPath[0].mScoreBoard.reward(xiPath[0].getGameState().getPlayer() == 0 ? lPlayer0Reward : lPlayer1Reward);
  }

  private static double getTotalChildWeight(TreeNode xiNode)
  {
    double lTotalWeight = 0;
    TreeNode[] lChildren = xiNode.getChildren();
    int lNumActions = xiNode.getGameState().getNumLegalActions();
    for (int lAction = 0; lAction < lNumActions; lAction++)
    {
      TreeNode lChild = lChildren[lAction];
      if (lChild == null)
      {
        lTotalWeight += 1.0;
      }
      else
      {
        lTotalWeight += lChild.mScoreBoard.getSelectionWeight();
      }
    }
    return lTotalWeight;
  }
}
