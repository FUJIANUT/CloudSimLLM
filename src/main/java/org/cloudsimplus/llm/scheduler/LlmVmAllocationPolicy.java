package org.cloudsimplus.llm.scheduler;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.llm.core.GpuHost;
import org.cloudsimplus.llm.core.GpuPe;
import org.cloudsimplus.llm.workload.LlmModelSpec;
import org.cloudsimplus.vms.Vm;

import java.util.Comparator;
import java.util.Optional;

/**
 * GPU-aware VM placement. Filters to {@link GpuHost}s that can fit weights
 * for the requested model (M_w from Eq. 5/spec) and ranks by free HBM and
 * estimated tail latency.
 *
 * <p>Used in Case Study 2 (heterogeneous A100/H100/L40S mixing). For multi-DC
 * placement (Case Study 3), set a {@code DatacenterFilter} via the
 * {@link #setDatacenterFilter} hook.</p>
 */
public class LlmVmAllocationPolicy extends VmAllocationPolicyAbstract {

    public enum RankBy { FREE_HBM, EST_TTFT, COST_PER_TOKEN, CARBON_PER_TOKEN }

    private LlmModelSpec targetModel;
    private RankBy rankBy = RankBy.FREE_HBM;

    public LlmVmAllocationPolicy setTargetModel(LlmModelSpec spec) { this.targetModel = spec; return this; }
    public LlmVmAllocationPolicy setRankBy(RankBy r)               { this.rankBy = r; return this; }

    @Override
    protected Optional<Host> defaultFindHostForVm(final Vm vm) {
        if (targetModel == null) {
            return getHostList().stream()
                .filter(h -> h.isSuitableForVm(vm))
                .findFirst()
                .map(h -> (Host) h);
        }
        return getHostList().stream()
            .filter(GpuHost.class::isInstance)
            .map(GpuHost.class::cast)
            .filter(h -> hostCanLoadModel(h, targetModel))
            .filter(h -> h.isSuitableForVm(vm))
            .min(Comparator.comparingDouble(this::cost))
            .map(h -> (Host) h);
    }

    /** Returns true if at least one GpuPe on the host has HBM ≥ M_w. */
    public static boolean hostCanLoadModel(GpuHost h, LlmModelSpec spec) {
        final long mw = spec.weightBytes();
        return h.gpuPes().stream().anyMatch(g -> g.hbmBytes() >= mw);
    }

    /** Smaller is better. */
    private double cost(GpuHost h) {
        return switch (rankBy) {
            case FREE_HBM         -> -freeHbm(h);                 // maximize free HBM
            case EST_TTFT         -> estimatedTtftPenalty(h);
            case COST_PER_TOKEN   -> 0.0;                         // hook for §6.4 study
            case CARBON_PER_TOKEN -> 0.0;                         // hook for §6.5 study
        };
    }

    private long freeHbm(GpuHost h) {
        return h.gpuPes().stream().mapToLong(GpuPe::hbmBytes).sum();
    }

    /** Coarse penalty: hosts at high utilization rank worse. */
    private double estimatedTtftPenalty(GpuHost h) {
        return h.getCpuPercentUtilization();
    }
}
