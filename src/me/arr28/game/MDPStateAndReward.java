package me.arr28.game;

public class MDPStateAndReward
{
  public final MDPState mState;
  public final int mReward;

  public MDPStateAndReward(MDPState xiState, int xiReward)
  {
    mState = xiState;
    mReward = xiReward;
  }
}
