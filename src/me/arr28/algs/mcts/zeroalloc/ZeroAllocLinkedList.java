package me.arr28.algs.mcts.zeroalloc;

public class ZeroAllocLinkedList<T extends Linkable<T>>
{
  private T mHead;
  private T mTail;
  private int mSize;

  public void add(T xiItem)
  {
    assert(xiItem.getPrev() == null);
    assert(xiItem.getNext() == null);

    mSize++;

    if (mTail == null)
    {
      // No items in the list.
      assert(mHead == null);
      mHead = xiItem;
      mTail = xiItem;
    }
    else
    {
      // Add to the end of the list
      mTail.setNext(xiItem);
      xiItem.setPrev(mTail);
      mTail = xiItem;
    }
  }

  public void remove (T xiItem)
  {
    T lPrev = xiItem.getPrev();
    T lNext = xiItem.getNext();

    mSize--;

    if (lPrev == null)
    {
      // Item was at the head of the list
      assert(mHead == xiItem);
      mHead = xiItem.getNext();
    }
    else
    {
      lPrev.setNext(lNext);
    }

    if (lNext == null)
    {
      // Item was at the tail of the list
      assert(mTail == xiItem);
      mTail = xiItem.getPrev();
    }
    else
    {
      lNext.setPrev(lPrev);
    }

    xiItem.setPrev(null);
    xiItem.setNext(null);
  }

  public T getFirst()
  {
    return mHead;
  }

  public int size()
  {
    return mSize;
  }
}
