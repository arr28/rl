package me.arr28.mcts.zeroalloc;

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

  public boolean contains(K xiKey)
  {
    if (mMap.containsKey(xiKey))
    {
      // Record the hit and move the item to the end of the LRU list.
      mHits++;
      V lCached = mMap.get(xiKey);
      mRUOrderedList.remove(lCached);
      mRUOrderedList.add(lCached);
      return true;
    }

    return false;
  }

  public V get(K xiKey)
  {
    return mMap.get(xiKey);
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

  public int size()
  {
    return mMap.size();
  }
}
