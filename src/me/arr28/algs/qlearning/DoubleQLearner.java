package me.arr28.algs.qlearning;

import java.util.Random;

import me.arr28.game.DoubleQLearnableState;

public class DoubleQLearner extends QLearner {
    private static final Random RANDOM = new Random();

    public DoubleQLearner(double xiAlpha, double xiEpsilon) {
        super(xiAlpha, xiEpsilon);
    }

    @Override
    protected void learn(DoubleQLearnableState xiOld, int xiAction, DoubleQLearnableState xiNew, double xiReward) {
        // Double Q-learning update
        boolean lPrimary = RANDOM.nextBoolean();
        double lOldQ = xiOld.getActionQ(lPrimary, xiAction);
        double lNewQ = lOldQ + mAlpha * (xiReward + xiNew.getActionQ(!lPrimary, xiNew.getBestAction(lPrimary)) - lOldQ);
        xiOld.setActionQ(lPrimary, xiAction, lNewQ);
    }
}
