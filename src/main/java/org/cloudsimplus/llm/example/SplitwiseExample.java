package org.cloudsimplus.llm.example;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.llm.core.GpuHost;
import org.cloudsimplus.llm.core.GpuPe;
import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.llm.metrics.LlmStatistics;
import org.cloudsimplus.llm.power.LlmPowerModel;
import org.cloudsimplus.llm.scheduler.ContinuousBatchScheduler;
import org.cloudsimplus.llm.scheduler.PrefillDecodeBroker;
import org.cloudsimplus.llm.scheduler.PrefillDecodeDisaggScheduler;
import org.cloudsimplus.llm.workload.KvCacheProvisioner;
import org.cloudsimplus.llm.workload.LlmModelSpec;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * §6.3 Case Study 1 — Prefill/decode disaggregation vs co-located baseline.
 *
 * <p>Runs the same workload twice:
 * <ul>
 *   <li><b>CO_LOCATED</b>: 8 GPUs each running a {@link ContinuousBatchScheduler}
 *       (vLLM-style, prefill and decode share one engine)</li>
 *   <li><b>SPLITWISE</b>: 4 GPUs as prefill-only workers + 4 GPUs as decode-only,
 *       routed by {@link PrefillDecodeBroker} with KV transfer over a 200 GB/s
 *       inter-host fabric</li>
 * </ul>
 *
 * <p>Outputs the head-to-head comparison table that drives the §6.3 figure.
 * Calibration values are placeholders; replace with §6.1 fit before publishing.</p>
 */
public final class SplitwiseExample {

    private static final int    NUM_REQUESTS    = 500;
    private static final int    PREFILL_GPUS    = 4;
    private static final int    DECODE_GPUS     = 4;
    private static final int    COLOCATED_GPUS  = PREFILL_GPUS + DECODE_GPUS;     // fair compute budget
    private static final long   RNG_SEED        = 42L;
    private static final double PUE             = 1.15;
    private static final double CARBON_GPER_KWH = 380.0;
    private static final double KV_TRANSFER_GBS = 200.0;                          // InfiniBand HDR

    private SplitwiseExample() { }

    public static void main(String[] args) {
        Result colocated = runCoLocated();
        Result splitwise = runSplitwise();
        printComparison(colocated, splitwise);
    }

    /* ============================================================
       Co-located baseline: COLOCATED_GPUS workers, each handles
       prefill+decode in the same continuous-batch engine.
       ============================================================ */
    private static Result runCoLocated() {
        var simulation = new CloudSimPlus();
        var broker = new DatacenterBrokerSimple(simulation);
        var model = LlmModelSpec.llama3_8B_fp16();

        List<GpuHost> hosts = new ArrayList<>();
        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < COLOCATED_GPUS; i++) {
            GpuPe gpu = calibratedA100();
            GpuHost host = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
            host.setIntraNvlinkGbs(600.0).setInterFabricGbs(KV_TRANSFER_GBS);
            host.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));
            hosts.add(host);

            var kv = new KvCacheProvisioner(gpu, 16);
            var sched = new ContinuousBatchScheduler(gpu, kv, model).setMaxBatchSize(64);
            Vm vm = new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000);
            vms.add(vm);
        }
        new DatacenterSimple(simulation, hosts, new VmAllocationPolicySimple());

        broker.submitVmList(vms);
        broker.submitCloudletList(generateRequests(model, NUM_REQUESTS, RNG_SEED));
        simulation.start();

        return collect("CO-LOCATED", broker, hosts);
    }

    /* ============================================================
       Splitwise: PREFILL_GPUS prefill-only workers + DECODE_GPUS
       decode-only, routed by PrefillDecodeBroker.
       ============================================================ */
    private static Result runSplitwise() {
        var simulation = new CloudSimPlus();
        var broker = new PrefillDecodeBroker(simulation).setKvTransferBwGbs(KV_TRANSFER_GBS);
        var model = LlmModelSpec.llama3_8B_fp16();

        List<GpuHost> hosts = new ArrayList<>();
        List<Vm> prefillVms = new ArrayList<>();
        List<Vm> decodeVms = new ArrayList<>();

        for (int i = 0; i < PREFILL_GPUS; i++) {
            GpuPe gpu = calibratedA100();
            GpuHost host = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
            host.setIntraNvlinkGbs(600.0).setInterFabricGbs(KV_TRANSFER_GBS);
            host.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));
            hosts.add(host);

            var kv = new KvCacheProvisioner(gpu, 16);
            var sched = new PrefillDecodeDisaggScheduler(gpu, kv, model,
                PrefillDecodeDisaggScheduler.Role.PREFILL_ONLY).setMaxBatchSize(64);
            Vm vm = new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000);
            prefillVms.add(vm);
        }
        for (int i = 0; i < DECODE_GPUS; i++) {
            GpuPe gpu = calibratedA100();
            GpuHost host = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
            host.setIntraNvlinkGbs(600.0).setInterFabricGbs(KV_TRANSFER_GBS);
            host.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));
            hosts.add(host);

            var kv = new KvCacheProvisioner(gpu, 16);
            var sched = new PrefillDecodeDisaggScheduler(gpu, kv, model,
                PrefillDecodeDisaggScheduler.Role.DECODE_ONLY).setMaxBatchSize(64);
            Vm vm = new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000);
            decodeVms.add(vm);
        }

        new DatacenterSimple(simulation, hosts, new VmAllocationPolicySimple());

        broker.setPrefillVms(prefillVms).setDecodeVms(decodeVms);
        var allVms = new ArrayList<Vm>();
        allVms.addAll(prefillVms);
        allVms.addAll(decodeVms);
        broker.submitVmList(allVms);
        broker.submitCloudletList(generateRequests(model, NUM_REQUESTS, RNG_SEED));
        simulation.start();

        return collect("SPLITWISE", broker, hosts);
    }

    /* ============================================================
       Stats collection — uses LlmStatistics with batch-amortized
       energy approximation. For paper-quality numbers attach a
       per-tick PowerMeter; here we use a coarse average.
       ============================================================ */
    private static Result collect(String label, DatacenterBrokerSimple broker, List<GpuHost> hosts) {
        var stats = new LlmStatistics();
        var finished = broker.getCloudletFinishedList().stream()
            .filter(LlmCloudlet.class::isInstance)
            .map(LlmCloudlet.class::cast)
            // Splitwise broker exposes both originals and decode-shadows; use shadows
            // (which carry the true end-to-end finish time inherited from arrival).
            .toList();

        // For Splitwise, the "originals" (prefill cloudlets) finish early; we want the
        // shadow's finish time but the original's arrival time. The shadow constructor
        // already inherits arrivalSimTime, so we just filter to shadows when present.
        boolean hasShadows = finished.stream().anyMatch(LlmCloudlet::isDecodeShadow);
        var toReport = hasShadows
            ? finished.stream().filter(LlmCloudlet::isDecodeShadow).toList()
            : finished;

        double avgPowerW = hosts.stream()
            .mapToDouble(h -> h.getPowerModel().getPower(0.6)).average().orElse(0.0);
        for (LlmCloudlet r : toReport) {
            Vm vm = r.getVm();
            if (vm == null || vm == Vm.NULL) continue;
            if (!(vm.getCloudletScheduler() instanceof ContinuousBatchScheduler s)) continue;
            stats.onRequestFinished(r, s, PUE, CARBON_GPER_KWH, 16.0, avgPowerW);
        }
        return new Result(label, toReport.size(), stats);
    }

    private static GpuPe calibratedA100() {
        return GpuPe.a100_80gb()
            .setEffFp16TflopsPrefill(180.0)
            .setEffFp16TflopsDecode(60.0)
            .setEffHbmBwGbs(1500.0)
            .setAlphaPrefillSec(0.005)
            .setAlphaDecodeSec(0.001);
    }

    private static List<Cloudlet> generateRequests(LlmModelSpec model, int n, long seed) {
        var rng = new Random(seed);
        var out = new ArrayList<Cloudlet>(n);
        // Poisson-ish arrivals at 50 req/s; longer prompts than LlmExample to stress prefill.
        double lambda = 50.0;
        double simT = 0.0;
        for (int i = 0; i < n; i++) {
            simT += -Math.log(1 - rng.nextDouble()) / lambda;
            int sIn  = 512  + rng.nextInt(1537); // 512–2048 prompt tokens
            int sOut = 128  + rng.nextInt(385);  // 128–512 response
            var c = new LlmCloudlet(i, model, sIn, sOut, LlmCloudlet.SloClass.INTERACTIVE);
            c.onArrival(simT);
            out.add(c);
        }
        return out;
    }

    private record Result(String label, int finished, LlmStatistics stats) {}

    private static void printComparison(Result a, Result b) {
        System.out.println();
        System.out.println("=========================== Splitwise vs Co-located ===========================");
        System.out.printf("%-22s %15s %15s %12s%n", "metric", a.label, b.label, "Δ (B vs A)");
        row("requests finished",    a.finished,            b.finished, "n");
        row("Mean TTFT (s)",        a.stats.meanTtft(),    b.stats.meanTtft(),    "s");
        row("P99  TTFT (s)",        a.stats.p99Ttft(),     b.stats.p99Ttft(),     "s");
        row("Mean TPOT (ms)",       a.stats.meanTpot()*1000, b.stats.meanTpot()*1000, "ms");
        row("P99  TPOT (ms)",       a.stats.p99Tpot()*1000,  b.stats.p99Tpot()*1000,  "ms");
        row("Mean E2E  (s)",        a.stats.meanE2e(),     b.stats.meanE2e(),     "s");
        row("SLO attainment (%)",   100*a.stats.sloAttainment(), 100*b.stats.sloAttainment(), "%");
        row("Total energy (kWh)",   a.stats.totalEnergyJoules()/3.6e6, b.stats.totalEnergyJoules()/3.6e6, "kWh");
        row("Total carbon (kg)",    a.stats.totalCarbonGrams()/1000,    b.stats.totalCarbonGrams()/1000,    "kg");
        System.out.println("================================================================================");
    }

    private static void row(String name, double a, double b, String unit) {
        double delta = (a == 0 || Double.isNaN(a)) ? Double.NaN : 100.0 * (b - a) / a;
        if (Double.isNaN(delta)) {
            System.out.printf("%-22s %15.4f %15.4f %12s%n", name, a, b, "—");
        } else {
            System.out.printf("%-22s %15.4f %15.4f %+11.1f%%%n", name, a, b, delta);
        }
    }
    private static void row(String name, int a, int b, String unit) {
        System.out.printf("%-22s %15d %15d %12s%n", name, a, b, unit);
    }
}
