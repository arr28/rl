package me.arr28.mcts;

/**
 * Factory for score boards.
 *
 * @author Andrew Rose
 */
public interface ScoreBoardFactory
{
  /**
   * @return a new scoreboard.
   */
  public ScoreBoard createScoreBoard();
}
