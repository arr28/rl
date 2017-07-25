package me.arr28.algs.mcts.policy.rollout;

import java.util.concurrent.ThreadLocalRandom;

import me.arr28.algs.mcts.policy.RolloutPolicy;
import me.arr28.game.GameState;

/**
 * A simple random rollout policy.
 *
 * @author Andrew Rose
 */
public class SimpleRolloutPolicy implements RolloutPolicy
{
  @Override
  public double rollout(GameState xiState)
  {
    while (!xiState.isTerminal())
    {
      xiState.applyAction(ThreadLocalRandom.current().nextInt(xiState.getNumLegalActions()));
    }
    return xiState.getReward();
  }
}
