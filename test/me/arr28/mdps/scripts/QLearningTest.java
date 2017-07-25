package me.arr28.mdps.scripts;

import me.arr28.algs.qlearning.DoubleQLearner;
import me.arr28.game.DoubleQLearnableState;
import me.arr28.games.mdps.CliffWalkingState.CliffWalkingStateFactory;
import me.arr28.util.MutableInt;

public class QLearningTest {
    private static final double ALPHA = 0.1;
    private static final double EPSILON = 0.1;

    public static void main(String[] xiArgs) {
        QLearningTest lQLearning = new QLearningTest();
        lQLearning.run();
    }

    private void run() {
        // QLearner lQLearner = new QLearner(ALPHA, EPSILON);
        DoubleQLearner lQLearner = new DoubleQLearner(ALPHA, EPSILON);

        MutableInt lReward = new MutableInt();
        DoubleQLearnableState lInitialState = new CliffWalkingStateFactory().createInitialState();
        for (int lEpisode = 0; lEpisode < 5000; lEpisode++) {
            lQLearner.iterate(lInitialState);

            // Do an occasional debugging evaluation.
            if ((lEpisode + 1) % 500 == 0) {
                int lSteps = 0;
                DoubleQLearnableState lState = lInitialState;
                int lTotalReward = 0;
                while (!lState.isTerminal()) {
                    int lAction = lState.getBestAction(true);
                    DoubleQLearnableState lNewState = lState.perform(lAction, lReward);
                    lTotalReward += lReward.mValue;
                    lState = lNewState;
                    if (++lSteps == 100) {
                        break;
                    }
                }
                // CliffWalkingState.draw();
                System.out.println(
                        "Greedy reward at episode " + (lEpisode + 1) + " = " + lTotalReward + " / " + lInitialState);
            }
        }
    }
}
