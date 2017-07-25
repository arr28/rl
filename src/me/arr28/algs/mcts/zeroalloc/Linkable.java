package me.arr28.algs.mcts.zeroalloc;

/**
 * Interface for objects that can be added to a (single) zero-alloc LinkedList.
 *
 * @param <T> - the type of object that can be added to the list.
 */
public interface Linkable<T>
{
  public void setPrev(T xiPrev);
  public T getPrev();
  public void setNext(T xiNext);
  public T getNext();
}
