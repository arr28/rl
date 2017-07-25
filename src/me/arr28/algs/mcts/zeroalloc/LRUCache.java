package me.arr28.algs.mcts.zeroalloc;

import java.util.HashMap;

public class LRUCache<K, V extends LRUCachable<K, V>>
{
  private final HashMap<K, V> mMap;
  private final ZeroAllocLinkedList<V> mRUOrderedList;

  /**
   * Read-only statistics.
   */
  public long mHits;
  public long mEvictions;

  public LRUCache(int xiSizeHint)
  {
    mMap = new HashMap<>(xiSizeHint);
    mRUOrderedList = new ZeroAllocLinkedList<>();
  }

  public V get(K xiKey)
  {
    V lCached = mMap.get(xiKey);
    if (lCached != null)
    {
      // Record the hit and move the item to the end of the LRU list.
      mHits++;
      mRUOrderedList.remove(lCached);
      mRUOrderedList.add(lCached);
    }
    return lCached;
  }

  public void put(K xiKey, V xiValue)
  {
    mMap.put(xiKey, xiValue);
    mRUOrderedList.add(xiValue);
  }

  public V evict()
  {
    mEvictions++;
    V lEvicted = mRUOrderedList.getFirst();
    mRUOrderedList.remove(lEvicted);
    mMap.remove(lEvicted.getKey());
    return lEvicted;
  }

  public void remove(V xiValue)
  {
    mRUOrderedList.remove(xiValue);
    mMap.remove(xiValue.getKey());
  }

  public int size()
  {
    return mMap.size();
  }
}
