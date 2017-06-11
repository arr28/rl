package me.arr28.game;

/**
 * Factory for game states.
 *
 * @author Andrew Rose
 */
public interface GameStateFactory
{
  /**
   * @return a new game state at the start of the game.
   */
  public GameState createInitialState();
}
