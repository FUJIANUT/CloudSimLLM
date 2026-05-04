package org.cloudsimplus.llm.workload;

/**
 * One PagedAttention block (β tokens). Used by {@link KvCacheProvisioner} to
 * track HBM occupancy. Reference counting permits prefix sharing across
 * requests in future extensions; the baseline scheduler uses refCount ∈ {0,1}.
 */
public final class KvCacheBlock {
    private final long id;
    private final int blockSizeTokens;   // β
    private final long bytesPerToken;    // m_kv (Eq. 5)
    private int refCount;
    private boolean allocated;

    public KvCacheBlock(long id, int blockSizeTokens, long bytesPerToken) {
        this.id = id;
        this.blockSizeTokens = blockSizeTokens;
        this.bytesPerToken = bytesPerToken;
    }

    public long bytes()           { return (long) blockSizeTokens * bytesPerToken; }
    public boolean allocated()    { return allocated; }
    public int refCount()         { return refCount; }
    public long id()              { return id; }
    public int blockSizeTokens()  { return blockSizeTokens; }
    public long bytesPerToken()   { return bytesPerToken; }

    public void acquire()         { refCount++; allocated = true; }
    public void release()         { if (--refCount <= 0) { refCount = 0; allocated = false; } }
}
