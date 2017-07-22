package me.arr28.game;


/**
 * Interface for all MDP state representations.
 *
 * @author Andrew Rose
 */
public interface MDPState
{
  /**
   * @return whether the state is terminal.
   */
  public boolean isTerminal();

  /**
   * @return the player to play.
   *
   * Only valid for non-terminal states.
   */
  public int getPlayer();

  /**
   * @return the number of legal actions.
   *
   * Only valid for non-terminal states.
   */
  public int getNumLegalActions();

  /**
   * Perform the specified action.
   *
   * Only valid for non-terminal states.
   *
   * @param xiAction - the action to be performed in the given state.
   *
   * @return the reward for playing the specified action.
   */
  public MDPStateAndReward perform(int xiAction);
}
