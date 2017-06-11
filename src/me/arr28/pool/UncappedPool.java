package me.arr28.pool;


/**
 * A pool with no maximum size.
 *
 * Freed items are kept in the pool to avoid excessive object allocation.
 *
 * @param <ItemType> the type of item to be kept in the pool.
 */
public class UncappedPool<ItemType> implements Pool<ItemType>
{
  // Object allocator for creating new objects in this pool.
  private final ObjectAllocator<ItemType> mAllocator;

  // Items that are available for re-use.
  private final int                                    mMaxFreeItems;
  private final ItemType[]                             mFreeItems;
  private int                                          mNumFreeItems;

  // Statistical information about pool usage.
  //
  // - The number of items currently is use.
  private int                                          mNumItemsInUse = 0;

  /**
   * Create an pool with no maximum size.  The pool keeps freed objects for re-use, up to the specified maximum.
   *
   * @param xiAllocator - an object allocator for creating new objects in this pool.
   * @param xiMaxFreeItems - the number of freed items to keep for re-use.
   */
  @SuppressWarnings("unchecked")
  public UncappedPool(ObjectAllocator<ItemType> xiAllocator, int xiMaxFreeItems)
  {
    mAllocator = xiAllocator;
    mMaxFreeItems = xiMaxFreeItems;
    mFreeItems = (ItemType[])(new Object[xiMaxFreeItems]);
  }

  @Override
  public ItemType allocate()
  {
    ItemType lAllocatedItem;

    if (mNumFreeItems > 0)
    {
      // Re-use an item from the free list.
      lAllocatedItem = mFreeItems[--mNumFreeItems];
      mAllocator.resetObject(lAllocatedItem);
    }
    else
    {
      // Allocate a new item.
      lAllocatedItem = mAllocator.newObject();
    }

    mNumItemsInUse++;
    return lAllocatedItem;
  }

  @Override
  public void free(ItemType xiItem)
  {
    mNumItemsInUse--;
    if (mNumFreeItems < mMaxFreeItems)
    {
      mFreeItems[mNumFreeItems++] = xiItem;
    }
  }

  @Override
  public boolean isFull()
  {
    return false;
  }

  @Override
  public int getPoolUsage()
  {
    // This pool has no maximum size, so is always 0% full.
    return 0;
  }

  @Override
  public void clear()
  {
    // This pool doesn't keep references to the items it has allocated, so it can't free them all.
    assert(false) : "Don't call clear() on UncappedPool";
  }

  @Override
  public ItemType get(@SuppressWarnings("unused") int xiIndex)
  {
    throw new RuntimeException("get(int) not supported by UncappedPool");
  }

  /**
   * @return the current capacity of the pool (in-use items + free items).  This is an uncapped pool, so the pool will
   * be increased as required.
   */
  @Override
  public int getCapacity()
  {
    return 0;
  }

  @Override
  public int getNumItemsInUse()
  {
    return mNumItemsInUse;
  }

  @Override
  public void setNonFreeThreshold(@SuppressWarnings("unused") int xiThreshold)
  {
    // Nothing to do.
  }
}
