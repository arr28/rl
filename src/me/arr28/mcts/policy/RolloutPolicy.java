package me.arr28.mcts.policy;

import me.arr28.game.GameState;

/**
 * An MCTS rollout policy.
 *
 * @author Andrew Rose
 */
public interface RolloutPolicy
{
  /**
   * Perform a rollout.
   *
   * @param xiState - the game state to rollout from.
   *
   * @return the score from the rollout, for player 0.
   */
  public double rollout(GameState xiState);
}
