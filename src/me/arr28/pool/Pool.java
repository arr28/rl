package me.arr28.pool;

/**
 * An pool of items.
 *
 * @param <ItemType> - the type of items stored in this pool.
 */
public interface Pool<ItemType>
{
  /**
   * Interface to be implemented by classes capable of allocating (and resetting) objects in a pool.
   *
   * @param <ItemType> the type of item to be allocated.
   */
  public interface ObjectAllocator<ItemType>
  {
    /**
     * @return a newly allocated object.
     */
    public ItemType newObject();

    /**
     * Reset an object, ready for re-use.
     *
     * @param xiObject - the object to reset.
     */
    public void resetObject(ItemType xiObject);
  }

  /**
   * Allocate a new item from the pool.
   *
   * @return the new item.
   */
  public ItemType allocate();

  /**
   * Return an item to the pool.
   *
   * The pool promises to call resetObject() for any freed items before re-use.
   *
   * @param xiItem  - the item.
   */
  public void free(ItemType xiItem);

 /**
   * Clear the pool - resetting all items that are still allocated.
   */
  public void clear();

  /**
   * Optional method to get the pool capacity.
   *
   * @return the capacity of the pool.
   */
  public int getCapacity();

  /**
   * @return whether the pool is (nearly) full.
   *
   * When full, the caller needs to free() some items to ensure that subsequently allocations will continue to succeed.
   */
  public boolean isFull();

  /**
   * Set a minimum free node requirement to report non-full.
   *
   * @param xiThreshold - the threshold.
   */
  public void setNonFreeThreshold(int xiThreshold);

  /**
   * @return the number of items currently in use.
   */
  public int getNumItemsInUse();

  /**
   * @return the percentage of this pool that is in use.
   */
  public int getPoolUsage();

  /**
   * Optional method to retrieve by index an item that has already been allocated.
   *
   * @param xiIndex - the index of the item to retrieve.
   *
   * @return the item.
   *
   */
  public ItemType get(int xiIndex);
}
