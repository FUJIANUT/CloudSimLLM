package org.cloudsimplus.llm.scheduler;

import org.cloudsimplus.llm.core.GpuPe;
import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.llm.workload.KvCacheProvisioner;
import org.cloudsimplus.llm.workload.LlmModelSpec;

import java.util.List;

/**
 * Splitwise-style scheduler. Each instance is dedicated to either prefill or
 * decode. Used by Case Study 1 in §6.3 to evaluate prefill/decode disaggregation
 * at datacenter scale. Pair two instances per {@code GpuHost}: one PREFILL,
 * one DECODE; the broker routes by phase and migrates KV state across them via
 * the inter-host fabric (see Eq. 11 with k = 1 + RDMA cost added in subclass).
 */
public class PrefillDecodeDisaggScheduler extends ContinuousBatchScheduler {

    public enum Role { PREFILL_ONLY, DECODE_ONLY }

    private final Role role;
    /** RDMA bytes/s used to ship KV between prefill and decode workers. */
    private double kvTransferBwGbs = 200.0; // typical InfiniBand HDR

    public PrefillDecodeDisaggScheduler(GpuPe gpu, KvCacheProvisioner kv, LlmModelSpec model, Role role) {
        super(gpu, kv, model);
        this.role = role;
    }

    public PrefillDecodeDisaggScheduler setKvTransferBwGbs(double v) { this.kvTransferBwGbs = v; return this; }
    public Role role() { return role; }

    @Override
    protected double pickNextPhase(List<LlmCloudlet> prefillSet,
                                   List<LlmCloudlet> decodeSet,
                                   double currentTime) {
        return switch (role) {
            case PREFILL_ONLY -> {
                if (prefillSet.isEmpty()) yield 0.0;
                double dt = runPrefillBatch(prefillSet, currentTime);
                // Mark each request finished so broker hand-off fires.
                // Decode phase will run on a sibling DECODE_ONLY worker.
                for (LlmCloudlet r : prefillSet) {
                    r.markFinished(currentTime + dt);
                    kv().evict(r);
                }
                yield dt;
            }
            case DECODE_ONLY  -> decodeSet.isEmpty()  ? 0.0 : runDecodeStep(decodeSet, currentTime);
        };
    }

    /** Estimated time to ship one request's KV across the fabric (called by broker on hand-off). */
    public double kvTransferTimeSec(LlmCloudlet r) {
        return r.currentKvBytes(kv().blockSizeTokens()) / (kvTransferBwGbs * 1e9);
    }
}
