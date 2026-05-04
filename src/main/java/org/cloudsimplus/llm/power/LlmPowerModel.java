package org.cloudsimplus.llm.power;

import org.cloudsimplus.llm.core.GpuHost;
import org.cloudsimplus.llm.core.GpuPe;
import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.llm.scheduler.ContinuousBatchScheduler;
import org.cloudsimplus.power.PowerMeasurement;
import org.cloudsimplus.power.models.PowerModelHostAbstract;
import org.cloudsimplus.vms.Vm;

/**
 * Phase-aware GPU power model implementing Eq. (8). The host's instantaneous
 * power equals Σ over GPUs, where each GPU's power depends on the dominant
 * phase of its current batch:
 * <ul>
 *   <li>prefill (compute-bound): full TDP scaled by SM utilization</li>
 *   <li>decode  (memory-bound):  TDP scaled by ρ_dec ∈ [0.6, 0.75]</li>
 *   <li>idle: P_idle</li>
 * </ul>
 *
 * <p>For energy accounting per request (Eq. 9), {@link org.cloudsimplus.llm.metrics.LlmStatistics}
 * integrates these instantaneous values amortized by batch size {@code |B(t)|}.</p>
 */
public class LlmPowerModel extends PowerModelHostAbstract {

    /** ρ_dec — decode power discount factor (POLCA-calibrated). */
    private double decodeDiscount = 0.7;

    public LlmPowerModel setDecodeDiscount(double v) { this.decodeDiscount = v; return this; }
    public double decodeDiscount() { return decodeDiscount; }

    @Override
    protected double getPowerInternal(double utilizationFraction) {
        // Backwards-compat path: assume mixed-phase utilization → blend.
        if (!(getHost() instanceof GpuHost gh)) return 0.0;
        double total = 0.0;
        for (GpuPe pe : gh.gpuPes()) {
            total += pe.idleWatts() + (pe.tdpWatts() - pe.idleWatts())
                * utilizationFraction * 0.85; // blended ρ
        }
        return total;
    }

    @Override
    public PowerMeasurement getPowerMeasurement() {
        if (!(getHost() instanceof GpuHost gh)) {
            return new PowerMeasurement(0.0, 0.0);
        }
        double staticW = 0.0;
        double dynamicW = 0.0;
        for (GpuPe pe : gh.gpuPes()) {
            staticW += pe.idleWatts();
            dynamicW += dynamicWattsFor(gh, pe);
        }
        return new PowerMeasurement(staticW, dynamicW);
    }

    /** Per-GPU dynamic power inferred from its VM's scheduler phase. */
    private double dynamicWattsFor(GpuHost h, GpuPe pe) {
        double sumDyn = 0.0;
        int gpuCount = Math.max(1, h.gpuPes().size());
        for (Vm vm : h.getVmList()) {
            if (!(vm.getCloudletScheduler() instanceof ContinuousBatchScheduler s)) continue;
            if (s.gpu() != pe) continue;
            double phaseFactor = phaseFactorFor(s);
            double smUtil = approxSmUtil(s);
            sumDyn += (pe.tdpWatts() - pe.idleWatts()) * smUtil * phaseFactor;
        }
        // If no VM was scheduled on this GPU, contribute 0 (idle handled in static term).
        return sumDyn / gpuCount;
    }

    /** Returns 1.0 in prefill, ρ_dec in decode, 0 in idle. */
    private double phaseFactorFor(ContinuousBatchScheduler s) {
        boolean anyPrefill = false, anyDecode = false;
        for (var cle : s.getCloudletExecList()) {
            if (cle.getCloudlet() instanceof LlmCloudlet r) {
                if (r.phase() == LlmCloudlet.Phase.PREFILL) anyPrefill = true;
                else if (r.phase() == LlmCloudlet.Phase.DECODE) anyDecode = true;
            }
        }
        if (anyPrefill) return 1.0;
        if (anyDecode)  return decodeDiscount;
        return 0.0;
    }

    /** Cheap approximation: utilization ∝ batch fill ratio. */
    private double approxSmUtil(ContinuousBatchScheduler s) {
        int active = s.getCloudletExecList().size();
        if (active == 0) return 0.0;
        return Math.min(1.0, active / 64.0); // saturates around batch=64; refined in §6.1
    }
}
