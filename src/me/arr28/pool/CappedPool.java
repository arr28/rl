package me.arr28.pool;


/**
 * A pool with a fixed maximum size.
 *
 * Freed items are kept in the pool to avoid excessive object allocation.
 *
 * @param <ItemType> the type of item to be kept in the pool.
 */
public class CappedPool<ItemType> implements Pool<ItemType>
{
  // Object allocator for creating new objects in this pool.
  private final ObjectAllocator<ItemType> mAllocator;

  // Maximum number of items to allocate.
  private final int                                    mPoolSize;

  // Number of free entries required for isFull() to return false
  private int                                          mFreeThresholdForNonFull;

  // The pool of items.
  private final ItemType[]                             mItems;

  // Items that are available for re-use.  This is a circular buffer so that it has a LIFO access pattern.
  private final ItemType[]                             mFreeItems;
  private int                                          mNumFreeItems;
  private int                                          mFirstFreeitem;

  // Array index of the largest allocated item.  Used to track whether an attempt to allocate a new item should really
  // allocate a new item (if we're not yet at the maximum) or re-use and existing item.  This can never exceed
  // mPoolSize.
  private int                                          mLargestUsedIndex = -1;

  // Statistical information about pool usage.
  //
  // - The number of items currently is use.
  private int                                          mNumItemsInUse = 0;

  /**
   * Create a new pool of the specified maximum size.
   *
   * @param xiAllocator - an object allocator for creating new objects in this pool.
   * @param xiPoolSize - the pool size.
   */
  @SuppressWarnings("unchecked")
  public CappedPool(ObjectAllocator<ItemType> xiAllocator, int xiPoolSize)
  {
    mAllocator = xiAllocator;
    mPoolSize  = xiPoolSize;
    mItems     = (ItemType[])(new Object[xiPoolSize]);
    mFreeItems = (ItemType[])(new Object[xiPoolSize]);

    mFreeThresholdForNonFull = xiPoolSize / 100;  // Default to 1% free
  }

  @Override
  public void setNonFreeThreshold(int xiThreshold)
  {
    if (mFreeThresholdForNonFull < xiThreshold)
    {
      mFreeThresholdForNonFull = xiThreshold;
    }
  }

  /**
   * @return the table of items that backs this pool.
   *
   * This is a hack which is only used for MCTSTree validation.
   */
  public ItemType[] getItemTable()
  {
    return mItems;
  }

  @Override
  public int getCapacity()
  {
    return mPoolSize;
  }

  @Override
  public int getNumItemsInUse()
  {
    return mNumItemsInUse;
  }

  @Override
  public int getPoolUsage()
  {
    return mNumItemsInUse * 100 / mPoolSize;
  }

  @Override
  public ItemType allocate()
  {
    ItemType lAllocatedItem;

    // Prefer to re-use a node because it avoids memory allocation which is (a) slow, (b) liable to provoke GC and
    // (c) makes GC slower because there's a bigger heap to inspect.
    if (mNumFreeItems != 0)
    {
      lAllocatedItem = mFreeItems[(mFirstFreeitem + --mNumFreeItems) % mPoolSize];

      // Reset the item so that it's ready for re-use.
      mAllocator.resetObject(lAllocatedItem);
    }
    else
    {
      // No free items so allocate another one.
      assert(mLargestUsedIndex < mPoolSize - 1) :
        "Unexpectedly full pool (" + mLargestUsedIndex + " >= " + (mPoolSize - 1) + ")";
      mLargestUsedIndex++;
      lAllocatedItem = mAllocator.newObject();
      mItems[mLargestUsedIndex] = lAllocatedItem;
    }

    mNumItemsInUse++;
    return lAllocatedItem;
  }

  @Override
  public ItemType get(int xiIndex)
  {
    return mItems[xiIndex];
  }

  @Override
  public void free(ItemType xiItem)
  {
    assert(mNumItemsInUse > 0);
    mNumItemsInUse--;
    mFreeItems[(mFirstFreeitem + mNumFreeItems++) % mPoolSize] = xiItem;
  }

  @Override
  public boolean isFull()
  {
    return (mNumItemsInUse > mPoolSize - mFreeThresholdForNonFull);
  }

  @Override
  public void clear()
  {
    // Reset every allocated object and add it to the free list.
    for (int lii = 0; lii <= mLargestUsedIndex; lii++)
    {
      mAllocator.resetObject(mItems[lii]);
      mFreeItems[lii] = mItems[lii];
    }

    mNumFreeItems = mLargestUsedIndex + 1;
    mNumItemsInUse = 0;
    mFirstFreeitem = 0;
  }
}
