package me.arr28.games.mdps;

import me.arr28.game.DoubleQLearnableState;
import me.arr28.game.DoubleQLearnableState.QLearnableBaseState;
import me.arr28.game.MDPStateFactory;
import me.arr28.util.MutableDouble;

/**
 * Cliff Walking puzzle state.
 *
 * @author Andrew Rose
 */
public class CliffWalkingState extends QLearnableBaseState {
    public static final int WIDTH = 48;
    public static final int HEIGHT = 16;

    private static final CliffWalkingState[] STATES = new CliffWalkingState[WIDTH * HEIGHT];
    static {
        for (int xx = 0; xx < WIDTH; xx++) {
            for (int yy = 0; yy < HEIGHT; yy++) {
                int index = index(xx, yy);
                STATES[index] = new CliffWalkingState(xx, yy);
            }
        }
    }

    private static int index(int xiX, int xiY) {
        return xiY * WIDTH + xiX;
    }

    public static void draw() {
        for (int yy = CliffWalkingState.HEIGHT - 1; yy >= 0; yy--) {
            for (int xx = 0; xx < CliffWalkingState.WIDTH; xx++) {
                if (yy != 0 || xx == 0) {
                    switch (STATES[index(xx, yy)].getBestAction(true)) {
                        case 0:
                            System.out.print("<");
                            break;
                        case 1:
                            System.out.print(">");
                            break;
                        case 2:
                            System.out.print("v");
                            break;
                        case 3:
                            System.out.print("^");
                            break;
                    }
                }
            }
            System.out.println();
        }
    }

    private final int mX;
    private final int mY;

    /**
     * Create a new initial state.
     */
    public CliffWalkingState(int xiX, int xiY) {
        super(4);
        mX = xiX;
        mY = xiY;
    }

    @Override
    public boolean isTerminal() {
        // Game is terminal if we reach the goal.
        return (mY == 0) && (mX == WIDTH - 1);
    }

    @Override
    public int getPlayer() {
        return 0;
    }

    @Override
    public int getNumLegalActions() {
        // Can always move in any of the 4 directions. If attempting to move off
        // the edge of the world, nothing happens.
        return 4;
    }

    @Override
    public DoubleQLearnableState perform(int xiAction, MutableDouble xoReward) {
        int lX = mX;
        int lY = mY;

        switch (xiAction) {
            case 0:
                lX = Math.max(mX - 1, 0);
                break;
            case 1:
                lX = Math.min(mX + 1, WIDTH - 1);
                break;
            case 2:
                lY = Math.max(mY - 1, 0);
                break;
            case 3:
                lY = Math.min(mY + 1, HEIGHT - 1);
                break;
        }

        if ((lY == 0) && (lX > 0) && (lX < WIDTH - 1)) {
            // We've fallen off the cliff, so reset the position to the start and
            // return a reward of -100.
            lX = 0;
            xoReward.mValue = -100;
        } else {
            // In all other cases (including a transition to the goal), return -1.
            xoReward.mValue = -1;
        }

        return STATES[index(lX, lY)];
    }

    @Override
    public String toString() {
        return "(" + mX + "," + mY + "), LRDU=(" + mQValues[0][0] + "," + mQValues[0][1] + "," + mQValues[0][2] + ","
                + mQValues[0][3] + ")";
    }

    @Override
    public boolean equals(Object xiOther) {
        CliffWalkingState lOther = (CliffWalkingState) xiOther;
        return ((lOther.mX == mX) && (lOther.mY == mY));
    }

    @Override
    public int hashCode() {
        return Integer.hashCode(mX) ^ Integer.hashCode(mY);
    }

    /**
     * Factory for creating starting states for C4.
     */
    public static class CliffWalkingStateFactory implements MDPStateFactory {
        @Override
        public DoubleQLearnableState createInitialState() {
            return STATES[index(0, 0)];
        }
    }
}
