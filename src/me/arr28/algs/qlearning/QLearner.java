package me.arr28.algs.qlearning;

import java.util.Random;

import me.arr28.game.DoubleQLearnableState;
import me.arr28.util.MutableDouble;

public class QLearner {
    protected static final Random RANDOM = new Random();

    protected final double mAlpha;
    protected final double mEpsilon;

    public QLearner(double xiAlpha, double xiEpsilon) {
        mAlpha = xiAlpha;
        mEpsilon = xiEpsilon;
    }

    /**
     * Perform Q-learning over a single episode.
     */
    public double iterate(DoubleQLearnableState xiInitialState) {
        double lTotalReward = 0;
        MutableDouble lReward = new MutableDouble();
        DoubleQLearnableState lNewState;
        for (DoubleQLearnableState lState = xiInitialState; !lState.isTerminal(); lState = lNewState) {
            int lAction = chooseAction(lState);
            lNewState = lState.perform(lAction, lReward);
            learn(lState, lAction, lNewState, lReward.mValue);
            lTotalReward += lReward.mValue;
        }
        return lTotalReward;
    }

    private int chooseAction(DoubleQLearnableState xiState) {
        // Epsilon-greedy
        if (RANDOM.nextDouble() < mEpsilon) {
            return RANDOM.nextInt(xiState.getNumLegalActions());
        }
        return xiState.getBestAction(true);
    }

    private void learn(DoubleQLearnableState xiOld, int xiAction, DoubleQLearnableState xiNew, double xiReward) {
        // Q-learning update
        double lOldQ = xiOld.getActionQ(true, xiAction);
        double lNewQ = lOldQ + mAlpha * (xiReward + xiNew.getBestActionValue(true) - lOldQ);
        xiOld.setActionQ(true, xiAction, lNewQ);
    }
}
