package me.arr28.game;

import java.util.Random;

import me.arr28.util.MutableDouble;

/**
 * Interface state representations for "small" MDPs (where the whole state space fits in memory).
 *
 * @author Andrew Rose
 */
public interface DoubleQLearnableState extends MDPState {

    @Override
    public DoubleQLearnableState perform(int xiAction, MutableDouble xoReward);

    /**
     * @return the current Q-value for the specified action.
     *
     * @param xiUsePrimaryQ - whether to use the primary (or secondary) Q-value.
     * @param xiAction - the action.
     */
    public double getActionQ(boolean xiUsePrimaryQ, int xiAction);

    /**
     * Set the Q-value for the specified action.
     *
     * @param xiUsePrimaryQ - whether to use the primary (or secondary) Q-value.
     * @param xiAction - the action.
     * @param xiQ - the new Q value.
     */
    public void setActionQ(boolean xiUsePrimaryQ, int xiAction, double xiQ);

    /**
     * @param xiUsePrimaryQ - whether to use the primary (or secondary) Q-value.
     *
     * @return the action with the highest Q-value.
     */
    public int getBestAction(boolean xiUsePrimaryQ);

    /**
     * @param xiUsePrimaryQ - whether to use the primary (or secondary) Q-value.
     *
     * @return the value of the action with the highest Q-value.
     */
    public double getBestActionValue(boolean xiUsePrimaryQ);

    /**
     * A base-class that implements the methods of this interface.
     */
    public static abstract class QLearnableBaseState implements DoubleQLearnableState {
        private static final Random RANDOM = new Random(); // Small random values used for tie-breaking
        private static final double EPSILON = 1e-50;

        protected final double[][] mQValues;

        protected QLearnableBaseState(int xiNumActions) {
            mQValues = new double[2][xiNumActions];
        }

        @Override
        public double getActionQ(boolean xiUsePrimaryQ, int xiAction) {
            return mQValues[xiUsePrimaryQ ? 0 : 1][xiAction];
        }

        @Override
        public void setActionQ(boolean xiUsePrimaryQ, int xiAction, double xiQ) {
            mQValues[xiUsePrimaryQ ? 0 : 1][xiAction] = xiQ;
        }

        @Override
        public int getBestAction(boolean xiUsePrimaryQ) {
            double lBestValue = Double.NEGATIVE_INFINITY;
            int lBestAction = -1;
            for (int lAction = 0; lAction < mQValues[xiUsePrimaryQ ? 0 : 1].length; lAction++) {
                double lValue = mQValues[xiUsePrimaryQ ? 0 : 1][lAction] + ((RANDOM.nextDouble() - 0.5) * EPSILON);
                if (lValue > lBestValue) {
                    lBestValue = lValue;
                    lBestAction = lAction;
                }
            }
            return lBestAction;
        }

        @Override
        public double getBestActionValue(boolean xiUsePrimaryQ) {
            double lBestValue = Double.NEGATIVE_INFINITY;
            for (double lValue : mQValues[xiUsePrimaryQ ? 0 : 1]) {
                lBestValue = Math.max(lBestValue, lValue);
            }
            return lBestValue;
        }
    }
}
