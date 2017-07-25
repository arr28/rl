package me.arr28.game;

import me.arr28.util.MutableDouble;

/**
 * Interface for all MDP state representations.
 *
 * @author Andrew Rose
 */
public interface MDPState {
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
     * @param xoReward - wrapper for the reward, which is valid on exit.
     */
    public MDPState perform(int xiAction, MutableDouble xoReward);
}
