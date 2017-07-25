package me.arr28.mdps.scripts;

import me.arr28.algs.qlearning.DoubleQLearner;
import me.arr28.game.DoubleQLearnableState;
import me.arr28.games.mdps.CliffWalkingState;
import me.arr28.games.mdps.SuttonCh6State.SuttonCh6StateFactory;
import me.arr28.util.MutableDouble;

public class QLearningTest {
    private static final double ALPHA = 0.1;
    private static final double EPSILON = 0.1;
    private static final int EPISODES = 300;
    private static final int ITERATIONS = 10_000;

    public static void main(String[] xiArgs) {
        QLearningTest lQLearning = new QLearningTest();
        lQLearning.run();
    }

    private void run() {
        // QLearner lQLearner = new QLearner(ALPHA, EPSILON);
        DoubleQLearner lQLearner = new DoubleQLearner(ALPHA, EPSILON);

        int[] lLeftCount = new int[ITERATIONS];
        for (int lIteration = 0; lIteration < ITERATIONS; lIteration++) {
            DoubleQLearnableState lInitialState = new SuttonCh6StateFactory().createInitialState();
            for (int lEpisode = 0; lEpisode < EPISODES; lEpisode++) {
                if (lQLearner.iterate(lInitialState) != 0) {
                    lLeftCount[lEpisode]++;
                }

                // Do an occasional debugging evaluation.
                if ((lEpisode + 1) % 10 == 0) {
                    // greedyEvaluation(lInitialState, lEpisode);
                }
            }
        }

        for (int lii = 0; lii < EPISODES; lii++) {
            System.out.print((lLeftCount[lii] - 500) + ", ");
        }
        System.out.println();
    }

    private void greedyEvaluation(DoubleQLearnableState xiInitialState, int xiEpisode) {
        int lSteps = 0;
        DoubleQLearnableState lState = xiInitialState;
        double lTotalReward = 0;
        MutableDouble lReward = new MutableDouble();
        while (!lState.isTerminal()) {
            int lAction = lState.getBestAction(true);
            DoubleQLearnableState lNewState = lState.perform(lAction, lReward);
            lTotalReward += lReward.mValue;
            lState = lNewState;
            if (++lSteps == 100) {
                break;
            }
        }
        CliffWalkingState.draw();
        System.out
                .println("Greedy reward at episode " + (xiEpisode + 1) + " = " + lTotalReward + " / " + xiInitialState);
    }
}
