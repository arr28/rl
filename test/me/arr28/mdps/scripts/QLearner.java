package me.arr28.mdps.scripts;

import java.util.HashMap;
import java.util.Random;

import me.arr28.game.MDPState;
import me.arr28.game.MDPStateAndReward;
import me.arr28.games.mdps.CliffWalkingState.CliffWalkingStateFactory;

public class QLearner
{
  private static final double ALPHA = 0.1;
  private static final double EPSILON = 0.1;
  private static final Random RANDOM = new Random();

  private final HashMap<MDPState, StateActionValues> mStates = new HashMap<>();

  public static void main(String[] xiArgs)
  {
    QLearner lQLearning = new QLearner();
    lQLearning.run();
  }

  private static class StateActionValues
  {
    private final MDPState mState;
    private final double[] mActionValues;

    public StateActionValues(MDPState xiState)
    {
      mState = xiState;
      mActionValues = new double[mState.getNumLegalActions()];
    }

    public double getBestValue()
    {
      double lBestValue = Double.NEGATIVE_INFINITY;
      for (double lActionValue : mActionValues)
      {
        lBestValue = Math.max(lBestValue, lActionValue);
      }
      return lBestValue;
    }

    public int getBestAction()
    {
      double lBestValue = Double.NEGATIVE_INFINITY;
      int lBestAction = -1;
      for (int lAction = 0; lAction < mActionValues.length; lAction++)
      {
        if (mActionValues[lAction] > lBestValue)
        {
          lBestValue = mActionValues[lAction];
          lBestAction = lAction;
        }
      }
      return lBestAction;
    }
  }

  private void run()
  {
    MDPState lInitialState = new CliffWalkingStateFactory().createInitialState();
    seen(lInitialState);
    for (int lEpisode = 0; lEpisode < 500; lEpisode++)
    {
      MDPState lState = lInitialState;
      int lTotalReward = 0;
      while (!lState.isTerminal())
      {
        int lAction = chooseAction(lState);
        MDPStateAndReward lResult = lState.perform(lAction);
        seen(lResult.mState);
        lTotalReward += lResult.mReward;
        learn(lState, lAction, lResult);
        lState = lResult.mState;
      }
      System.out.print(lTotalReward + ", ");

      if ((lEpisode + 1) % 20 == 0)
      {
        int lSteps = 0;
        lState = lInitialState;
        lTotalReward = 0;
        while (!lState.isTerminal())
        {
          int lAction = mStates.get(lState).getBestAction();
          MDPStateAndReward lResult = lState.perform(lAction);
          lTotalReward += lResult.mReward;
          lState = lResult.mState;
          if (++lSteps == 100)
          {
            break;
          }
        }
        System.out.println("\nGreedy reward at episode " + (lEpisode + 1) + " = " + lTotalReward);
      }
    }
  }

  private void seen(MDPState xiState)
  {
    if (!mStates.containsKey(xiState))
    {
      mStates.put(xiState, new StateActionValues(xiState));
    }
  }

  private int chooseAction(MDPState xiState)
  {
    // Epsilon-greedy
    if (RANDOM.nextDouble() < EPSILON)
    {
      return RANDOM.nextInt(xiState.getNumLegalActions());
    }
    return mStates.get(xiState).getBestAction();
  }

  private void learn(MDPState xiState, int lAction, MDPStateAndReward xiResult)
  {
    StateActionValues lOld = mStates.get(xiState);
    StateActionValues lNew = mStates.get(xiResult.mState);

    // Q-learning update
    lOld.mActionValues[lAction] += ALPHA * (xiResult.mReward + lNew.getBestValue() - lOld.mActionValues[lAction]);
  }
}
