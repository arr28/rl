package me.arr28.games;

import java.util.Arrays;

import me.arr28.game.GameState;
import me.arr28.game.GameStateFactory;

/**
 * Connect-4 game state.
 *
 * @author Andrew Rose
 */
public class Connect4State implements GameState {
    private static final int WIDTH = 8;
    private static final int HEIGHT = 6;

    /**
     * The first free row in each of the columns.
     */
    private final int mFirstFreeRowInCol[] = new int[WIDTH];

    /**
     * Which player has played in the cell - 0 for player 0, 1 for player 1, 2 otherwise.  Cells are indexed by
     * index(row, col).
     */
    private final int mCell[] = new int[WIDTH * HEIGHT];

    /**
     * Which player is due to play.
     */
    private int mPlayer = 0;

    /**
     * Whether this is a terminal state.
     */
    private boolean mTerminal = false;

    /**
     * The reward for player 0 - only valid for a terminal state.
     */
    private double mReward = 0.0;

    /**
     * The legal columns to play in.  The first mNumOpenColumns items in mOpenColumns are (in no particular order) the
     * column numbers which still have vacant spaces.
     */
    private int mNumOpenColumns = WIDTH;
    private final int[] mOpenColumns = new int[WIDTH];

    /**
     * Create a new initial state.
     */
    Connect4State() {
        for (int lCell = 0; lCell < WIDTH * HEIGHT; lCell++) {
            mCell[lCell] = 2;
        }
        for (int lCol = 0; lCol < WIDTH; lCol++) {
            mOpenColumns[lCol] = lCol;
        }
    }

    @Override
    public void copyTo(GameState xiDestination) {
        Connect4State xiNew = (Connect4State) xiDestination;

        for (int lii = 0; lii < mFirstFreeRowInCol.length; lii++) {
            xiNew.mFirstFreeRowInCol[lii] = mFirstFreeRowInCol[lii];
        }

        for (int lii = 0; lii < mCell.length; lii++) {
            xiNew.mCell[lii] = mCell[lii];
        }

        xiNew.mPlayer = mPlayer;
        xiNew.mTerminal = mTerminal;
        xiNew.mReward = mReward;
        xiNew.mNumOpenColumns = mNumOpenColumns;
        System.arraycopy(mOpenColumns, 0, xiNew.mOpenColumns, 0, mNumOpenColumns);
    }

    private static int index(int xiRow, int xiCol) {
        return (xiRow * WIDTH) + xiCol;
    }

    @Override
    public boolean isTerminal() {
        return mTerminal;
    }

    @Override
    public int getPlayer() {
        return mPlayer;
    }

    @Override
    public int getNumLegalActions() {
        return mNumOpenColumns;
    }

    @Override
    public void applyAction(int xiActionIndex) {
        assert (!mTerminal) : "Can't continue to play in a terminal state";
        int lColumn = mOpenColumns[xiActionIndex];

        assert (mFirstFreeRowInCol[lColumn] < HEIGHT) : "Can't play in full column " + lColumn;
        int lCellIndex = index(mFirstFreeRowInCol[lColumn], lColumn);
        mCell[lCellIndex] = mPlayer;

        if (++mFirstFreeRowInCol[lColumn] == HEIGHT) {
            // This column is full. Decrease the number of open columns, swapping one of the other columns into place in
            // the array of open column.
            mOpenColumns[xiActionIndex] = mOpenColumns[--mNumOpenColumns];
        }

        terminalCheck(mFirstFreeRowInCol[lColumn] - 1, lColumn);

        mPlayer = 1 - mPlayer;
    }

    private void terminalCheck(int xiRow, int xiCol) {
        int lPlayedCell = index(xiRow, xiCol);

        // Only the player who has just played (in the specified cell) can have just won.
        int lTargetValue = mCell[lPlayedCell];
        double lReward = 1.0 - mPlayer;

        // Search downwards for a line of 4. (The easy case!)
        if (xiRow >= 3) {
            if ((mCell[lPlayedCell - (1 * WIDTH)] == lTargetValue) && (mCell[lPlayedCell - (2 * WIDTH)] == lTargetValue)
                    && (mCell[lPlayedCell - (3 * WIDTH)] == lTargetValue)) {
                mTerminal = true;
                mReward = lReward;
                return;
            }
        }

        // Search for lines through the played cell.
        boolean lLOpen = true;
        boolean lROpen = true;
        boolean lTLOpen = true;
        boolean lBROpen = true;
        boolean lTROpen = true;
        boolean lBLOpen = true;
        int lLRCount = 1;
        int lTLBRCount = 1;
        int lTRBLCount = 1;
        for (int lOffset = 1; lOffset <= 3; lOffset++) {
            // Horizontals
            if (lLOpen && (xiCol - lOffset >= 0) && (mCell[lPlayedCell - lOffset] == lTargetValue)) {
                lLRCount++;
            } else {
                lLOpen = false;
            }

            if (lROpen && (xiCol + lOffset < WIDTH) && (mCell[lPlayedCell + lOffset] == lTargetValue)) {
                lLRCount++;
            } else {
                lROpen = false;
            }

            // Top-left to bottom-right diagonals
            if ((lTLOpen) && (xiCol - lOffset >= 0) && (xiRow + lOffset < HEIGHT)
                    && (mCell[index(xiRow + lOffset, xiCol - lOffset)] == lTargetValue)) {
                lTLBRCount++;
            } else {
                lTLOpen = false;
            }

            if ((lBROpen) && (xiCol + lOffset < WIDTH) && (xiRow - lOffset >= 0)
                    && (mCell[index(xiRow - lOffset, xiCol + lOffset)] == lTargetValue)) {
                lTLBRCount++;
            } else {
                lBROpen = false;
            }

            // Top-right to bottom-left diagonals
            if ((lTROpen) && (xiCol + lOffset < WIDTH) && (xiRow + lOffset < HEIGHT)
                    && (mCell[index(xiRow + lOffset, xiCol + lOffset)] == lTargetValue)) {
                lTRBLCount++;
            } else {
                lTROpen = false;
            }

            if ((lBLOpen) && (xiCol - lOffset >= 0) && (xiRow - lOffset >= 0)
                    && (mCell[index(xiRow - lOffset, xiCol - lOffset)] == lTargetValue)) {
                lTRBLCount++;
            } else {
                lBLOpen = false;
            }

            // Do we have a win now?
            if ((lLRCount >= 4) || (lTLBRCount >= 4) || (lTRBLCount >= 4)) {
                mTerminal = true;
                mReward = lReward;
                return;
            }
        }

        // Nowhere left to play - it's a draw.
        if (mNumOpenColumns == 0) {
            mTerminal = true;
            mReward = 0.5;
            return;
        }
    }

    @Override
    public double getReward() {
        assert (mTerminal) : "Only call getReward() for terminal positions";
        return mReward;
    }

    @Override
    public String toString() {
        StringBuilder lResult = new StringBuilder((WIDTH * HEIGHT * 2) + (HEIGHT * 2) + 10);
        for (int lRow = HEIGHT - 1; lRow >= 0; lRow--) {
            for (int lCol = 0; lCol < WIDTH; lCol++) {
                int lCell = mCell[index(lRow, lCol)];
                lResult.append(lCell == 2 ? ". " : lCell + " ");
            }
            lResult.append('\n');
        }

        if (mTerminal) {
            lResult.append("Game over: Reward for player 0 = ").append(mReward).append('\n');
        } else {
            lResult.append("Player ").append(mPlayer).append(" to play");
        }
        return lResult.toString();
    }

    @Override
    public boolean equals(Object xiOther) {
        // The game state is entirely encoded in the mCell array. All other members can be derived from mCell.
        return Arrays.equals(mCell, ((Connect4State) xiOther).mCell);
    }

    @Override
    public int hashCode() {
        // Since the state is encoded entirely in mCell (see .equals), just compute the hash code of mCell.
        return Arrays.hashCode(mCell);
    }

    /**
     * Factory for creating starting states for C4.
     */
    public static class C4GameStateFactory implements GameStateFactory {
        @Override
        public GameState createInitialState() {
            return new Connect4State();
        }
    }
}
