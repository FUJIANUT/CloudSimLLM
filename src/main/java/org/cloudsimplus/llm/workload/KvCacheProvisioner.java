package org.cloudsimplus.llm.workload;

import org.cloudsimplus.llm.core.GpuPe;
import org.cloudsimplus.llm.core.LlmCloudlet;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

/**
 * HBM-aware KV cache pool implementing the capacity constraint of Eq. (7):
 * <pre>M_w + M_kv^paged(R_active) ≤ C_HBM</pre>
 *
 * <p>Conceptually a {@code ResourceProvisioner}, but it does not extend
 * {@link org.cloudsimplus.provisioners.ResourceProvisionerAbstract} because
 * KV blocks are <em>per-request</em> rather than <em>per-VM</em>. The LLM
 * scheduler queries this class directly each tick.</p>
 *
 * <p>Lifecycle:
 * <ol>
 *   <li>{@link #setModelLoaded(LlmModelSpec)} reserves M_w once per (GPU, model).</li>
 *   <li>{@link #tryAdmit(LlmCloudlet)} grows blocks as the request lengthens.</li>
 *   <li>{@link #evict(LlmCloudlet)} releases all blocks owned by a request.</li>
 * </ol>
 * </p>
 */
public class KvCacheProvisioner {
    private final GpuPe gpu;
    private final int blockSizeTokens;          // β

    private long weightBytes = 0;               // M_w  for the loaded model
    private long allocatedKvBytes = 0;
    private final Deque<KvCacheBlock> freeBlocks = new ArrayDeque<>();
    private final Map<Long, Integer> blocksPerRequest = new HashMap<>();
    private long nextBlockId = 0;

    private LlmModelSpec model;

    public KvCacheProvisioner(GpuPe gpu, int blockSizeTokens) {
        this.gpu = gpu;
        this.blockSizeTokens = blockSizeTokens;
    }

    public void setModelLoaded(LlmModelSpec spec) {
        this.model = spec;
        this.weightBytes = spec.weightBytes();
    }

    /** Bytes still free for KV after weights and currently allocated KV. */
    public long freeBytes() {
        return Math.max(0L, gpu.hbmBytes() - weightBytes - allocatedKvBytes);
    }

    /** Aggregate active KV bytes — the M_kv^paged(R_active) term in Eq. (7). */
    public long allocatedKvBytes() { return allocatedKvBytes; }

    /**
     * Try to grow allocation to cover the request's current length. Returns
     * true if successful, false if HBM is exhausted (caller decides whether
     * to queue, evict, or migrate).
     */
    public boolean tryAdmit(LlmCloudlet r) {
        if (model == null) {
            throw new IllegalStateException("KvCacheProvisioner: model not loaded");
        }
        final long curLen = (long) r.inputTokens() + Math.max(0, r.generated());
        final int wantedBlocks = (int) ((curLen + blockSizeTokens - 1) / blockSizeTokens);
        final int currentBlocks = blocksPerRequest.getOrDefault(r.getId(), 0);
        final int delta = wantedBlocks - currentBlocks;
        if (delta <= 0) return true;

        final long blockBytes = (long) blockSizeTokens * model.kvBytesPerToken();
        final long needed = (long) delta * blockBytes;
        if (needed > freeBytes()) return false;

        for (int i = 0; i < delta; i++) {
            KvCacheBlock blk = freeBlocks.isEmpty()
                ? new KvCacheBlock(nextBlockId++, blockSizeTokens, model.kvBytesPerToken())
                : freeBlocks.pop();
            blk.acquire();
            allocatedKvBytes += blockBytes;
        }
        blocksPerRequest.put(r.getId(), wantedBlocks);
        return true;
    }

    /** Release all blocks owned by a finished or preempted request. */
    public void evict(LlmCloudlet r) {
        Integer n = blocksPerRequest.remove(r.getId());
        if (n == null) return;
        final long blockBytes = (long) blockSizeTokens * model.kvBytesPerToken();
        allocatedKvBytes -= (long) n * blockBytes;
        if (allocatedKvBytes < 0) allocatedKvBytes = 0;
    }

    public int blockSizeTokens() { return blockSizeTokens; }
    public LlmModelSpec model()  { return model; }
    public GpuPe gpu()           { return gpu; }
}
