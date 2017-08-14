package me.arr28.game;


/**
 * Interface for all game state representations.
 *
 * @author Andrew Rose
 */
public interface GameState
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
   * Mutate this game state by playing the specified action.
   *
   * Only valid for non-terminal states.
   *
   * @param xiAction - the action to be played in the given state.
   */
  public void applyAction(int xiAction);

  /**
   * @return the reward for player 0 in the given state.
   *
   * Only valid for terminal states.
   */
  public double getReward();

  /**
   * Clone this state into the specified (unused) state.
   *
   * @param xiDestination - the state to copy into.
   */
  public void copyTo(GameState xiDestination);
}
