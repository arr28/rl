package me.arr28.game;

/**
 * Factory for MDP states.
 *
 * @author Andrew Rose
 */
public interface MDPStateFactory
{
  /**
   * @return a new initial MDP state.
   */
  public MDPState createInitialState();
}
