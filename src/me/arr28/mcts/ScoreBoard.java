package me.arr28.mcts;

import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;

/**
 * A score board keeping track of the results of performing a particular action in a particular state.
 *
 * @author Andrew Rose
 */
public class ScoreBoard
{
  /**
   * The number of times that the node associated with this score board has been selected.
   */
  private final LongAdder mSelectCount = new LongAdder();

  /**
   * The total score (a double) from all rollouts through this node, encoded
   * as a long for use with AtomicLongFieldUpdater.
   */
  private final DoubleAdder mTotalReward = new DoubleAdder();

  /**
   * Factory for these ScoreBoards.
   */
  public static class PlainScoreBoardFactory implements ScoreBoardFactory
  {
    @Override
    public ScoreBoard createScoreBoard()
    {
      return new ScoreBoard();
    }
  }

  /**
   * Create a score board.  Only to be used by the factory or by sub-class factories.
   */
  protected ScoreBoard()
  {
    // Default initialisation.
  }

  /**
   * Record that the node to which the score board is attached has been selected.
   *
   * If a sub-class overrides this method, it is expected to call this version before doing its own processing.
   */
  public void selected()
  {
    // Record the selection now, which makes this node look like a poorer choice until the result is recorded.  In a
    // multi-threaded environment, this tends to make different threads explore different parts of the tree.
    mSelectCount.increment();
  }

  /**
   * @return the number of times that this node has been selected.
   */
  public long getSelectCount()
  {
    return mSelectCount.sum();
  }

  /**
   * Update the score board with the result of a rollout.
   *
   * If a sub-class overrides this method, it is expected to call this version before doing its own processing.
   *
   * @param xiReward - the observed reward.
   */
  public void reward(double xiReward)
  {
    mTotalReward.add(xiReward);
  }

  /**
   * @return the weight associated with the node.  Sub-classes are expected to override this method.
   */
  public double getSelectionWeight()
  {
    return mTotalReward.sum() / mSelectCount.sum();
  }

  /**
   * @return the average score for this node (for the "other" player).
   */
  public final double getAverageReward()
  {
    return mTotalReward.sum() / mSelectCount.sum();
  }

  /**
   * Reset the score board as if created from scratch.
   */
  public void reset()
  {
    mSelectCount.reset();
    mTotalReward.reset();
  }
}
