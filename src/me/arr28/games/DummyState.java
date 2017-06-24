package me.arr28.games;

import java.util.concurrent.ThreadLocalRandom;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;

/**
 * Dummy game state, used for testing performance of the tree.
 *
 * @author Andrew Rose
 */
public class DummyState implements GameState
{
  private static final int BRANCHING_FACTOR = 7;
  private final int mHash = ThreadLocalRandom.current().nextInt();
  private int mPlayer = 0;

  @Override
  public boolean isTerminal()
  {
    return false;
  }

  @Override
  public int getPlayer()
  {
    return mPlayer;
  }

  @Override
  public int getNumLegalActions()
  {
    return BRANCHING_FACTOR;
  }

  @Override
  public void applyAction(@SuppressWarnings("unused") int xiAction)
  {
    mPlayer = 1 - mPlayer;
  }

  @Override
  public double getReward()
  {
    return 0;
  }

  @Override
  public void copyTo(GameState xiDestination)
  {
    DummyState xiNew = (DummyState)xiDestination;
    xiNew.mPlayer = mPlayer;
  }

  @Override
  public int hashCode()
  {
    return mHash;
  }

  public static class DummyGameStateFactory implements GameStateFactory
  {
    /**
     * Factory for creating starting states for the Dummy game.
     */
    @Override
    public GameState createInitialState()
    {
      return new DummyState();
    }
  }
}
