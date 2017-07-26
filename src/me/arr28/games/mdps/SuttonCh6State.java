package me.arr28.games.mdps;

import java.util.Random;

import me.arr28.game.DoubleQLearnableState;
import me.arr28.game.DoubleQLearnableState.QLearnableBaseState;
import me.arr28.game.MDPStateFactory;
import me.arr28.util.MutableDouble;

public class SuttonCh6State extends QLearnableBaseState {

    private static final Random RANDOM = new Random();

    private static final int B_ACTIONS = 100000;

    private static SuttonCh6State A = new SuttonCh6State(2);
    private static SuttonCh6State B = new SuttonCh6State(B_ACTIONS);
    private static SuttonCh6State T = new SuttonCh6State(1);

    private SuttonCh6State(int xiNumActions) {
        super(xiNumActions);
    }

    @Override
    public DoubleQLearnableState perform(int xiAction, MutableDouble xoReward) {
        if (this == A) {
            xoReward.mValue = 0;
            if (xiAction == 0) {
                return B;
            }
            return T;
        }
        assert (this == B);
        xoReward.mValue = RANDOM.nextGaussian() - 0.1;
        return T;
    }

    @Override
    public boolean isTerminal() {
        return this == T;
    }

    @Override
    public int getPlayer() {
        return 0;
    }

    @Override
    public int getNumLegalActions() {
        if (this == A) {
            return 2;
        }
        assert (this == B);
        return B_ACTIONS;
    }

    @Override
    public String toString() {
        if (this == A)
            return "A";
        if (this == B)
            return "B";
        assert (this == T);
        return "T";
    }

    /**
     * Factory for creating starting states for C4.
     */
    public static class SuttonCh6StateFactory implements MDPStateFactory {
        @Override
        public DoubleQLearnableState createInitialState() {
            return A;
        }
    }

    public static void reset() {
        for (boolean lPrimary : new boolean[] { true, false }) {
            A.setActionQ(lPrimary, 0, 0);
            A.setActionQ(lPrimary, 1, 0);
            B.setActionQ(lPrimary, 0, 0);
            T.setActionQ(lPrimary, 0, 0);
        }
    }
}
