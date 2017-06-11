package me.arr28.mcts.policy.rollout;

import me.arr28.game.GameState;
import me.arr28.mcts.policy.RolloutPolicy;

/**
 * A dummy rollout policy that immediately returns 0.  Useful for performance testing of the tree.
 *
 * @author Andrew Rose
 */
public class DummyRolloutPolicy implements RolloutPolicy
{
  @Override
  public double rollout(GameState xiState)
  {
    return 0;
  }
}
