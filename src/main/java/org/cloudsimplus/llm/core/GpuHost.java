package org.cloudsimplus.llm.core;

import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.HarddriveStorage;
import org.cloudsimplus.resources.Pe;

import java.util.List;

/**
 * Host whose PEs are {@link GpuPe}s. Adds NVLink/NVSwitch fabric properties for
 * Eq. (11) (tensor-parallel AllReduce) and a tensor-parallel degree hint that
 * the LLM scheduler reads when sizing batches.
 *
 * <p>Memory accounting for VM RAM remains in {@link HostSimple}; HBM is tracked
 * separately by {@link org.cloudsimplus.llm.workload.KvCacheProvisioner} per
 * {@link GpuPe} since HBM is the binding capacity for KV cache (Eq. 7).</p>
 */
public class GpuHost extends HostSimple {
    /** Single-direction NVLink/NVSwitch bandwidth between GPUs on this host (GB/s). */
    private double intraNvlinkGbs;
    /** Inter-host fabric bandwidth (GB/s) — InfiniBand / RoCE; lower than intra. */
    private double interFabricGbs;
    /** Tensor parallel degree this host can support natively (== local GPU count by default). */
    private int defaultTpDegree;

    public GpuHost(long ram, long bw, long storage, List<Pe> peList) {
        super(ram, bw, storage, peList);
        this.defaultTpDegree = peList.size();
    }

    public GpuHost(long ram, long bw, HarddriveStorage storage, List<Pe> peList) {
        super(ram, bw, storage, peList);
        this.defaultTpDegree = peList.size();
    }

    public GpuHost setIntraNvlinkGbs(double v)   { this.intraNvlinkGbs = v; return this; }
    public GpuHost setInterFabricGbs(double v)   { this.interFabricGbs = v; return this; }
    public GpuHost setDefaultTpDegree(int v)     { this.defaultTpDegree = v; return this; }

    public double intraNvlinkGbs()   { return intraNvlinkGbs; }
    public double interFabricGbs()   { return interFabricGbs; }
    public int defaultTpDegree()     { return defaultTpDegree; }

    /** GPU PEs only (filters out non-GPU PEs in heterogeneous hosts). */
    public List<GpuPe> gpuPes() {
        return getPeList().stream()
            .filter(GpuPe.class::isInstance)
            .map(GpuPe.class::cast)
            .toList();
    }

    /** Aggregate HBM across all GPU PEs (bytes). */
    public long aggregateHbmBytes() {
        return gpuPes().stream().mapToLong(GpuPe::hbmBytes).sum();
    }
}
