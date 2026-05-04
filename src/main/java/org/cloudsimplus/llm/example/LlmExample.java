package org.cloudsimplus.llm.example;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.builders.tables.CloudletsTableBuilder;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.llm.core.GpuHost;
import org.cloudsimplus.llm.core.GpuPe;
import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.llm.metrics.LlmStatistics;
import org.cloudsimplus.llm.power.LlmPowerModel;
import org.cloudsimplus.llm.scheduler.ContinuousBatchScheduler;
import org.cloudsimplus.llm.workload.KvCacheProvisioner;
import org.cloudsimplus.llm.workload.LlmModelSpec;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Smallest end-to-end example of CloudSimLLM. One A100 host, one VM running
 * Llama-3-8B fp16, 200 synthetic requests. Used as a smoke test that all
 * pieces wire together; calibration values are placeholders. After running
 * it once, replace the {@code .setEff*} numbers with §6.1 calibration output.
 *
 * <p>Run from project root:
 * <pre>./mvnw exec:java -Dexec.mainClass=org.cloudsimplus.llm.example.LlmExample</pre>
 * </p>
 */
public final class LlmExample {

    private static final int    NUM_REQUESTS    = 200;
    private static final long   RNG_SEED        = 42L;
    private static final double PUE             = 1.15;
    private static final double CARBON_GPER_KWH = 380.0; // ElectricityMap-style placeholder

    private LlmExample() { }

    public static void main(String[] args) {
        final var simulation = new CloudSimPlus();
        final var broker = new DatacenterBrokerSimple(simulation);

        // ---- 1. Build one A100-80GB GPU host with placeholder calibration ----
        final GpuPe gpu = GpuPe.a100_80gb()
            .setEffFp16TflopsPrefill(180.0)   // 58% of peak — typical vLLM
            .setEffFp16TflopsDecode(60.0)
            .setEffHbmBwGbs(1500.0)            // 74% of peak HBM BW
            .setAlphaPrefillSec(0.005)
            .setAlphaDecodeSec(0.001);

        // RAM/BW/storage in MB — placeholders; HBM accounted in GpuPe.
        final GpuHost host = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
        host.setIntraNvlinkGbs(600.0).setInterFabricGbs(200.0);
        host.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));

        new DatacenterSimple(simulation, List.of(host), new VmAllocationPolicySimple());

        // ---- 2. Build the VM with continuous-batching scheduler ----
        final LlmModelSpec model = LlmModelSpec.llama3_8B_fp16();
        final KvCacheProvisioner kv = new KvCacheProvisioner(gpu, 16);
        final ContinuousBatchScheduler scheduler = new ContinuousBatchScheduler(gpu, kv, model)
            .setMaxBatchSize(64);

        final Vm vm = new VmSimple((double) gpu.getCapacity(), 1L, scheduler)
            .setRam(8_000).setBw(1_000).setSize(10_000);

        // ---- 3. Generate synthetic LLM requests ----
        final List<Cloudlet> requests = generateRequests(model, NUM_REQUESTS, RNG_SEED);

        broker.submitVmList(List.of(vm));
        broker.submitCloudletList(requests);

        // ---- 4. Run ----
        simulation.start();

        // ---- 5. Aggregate LLM-specific statistics from finished cloudlets ----
        final LlmStatistics stats = new LlmStatistics();
        final List<LlmCloudlet> finished = broker.getCloudletFinishedList().stream()
            .filter(LlmCloudlet.class::isInstance)
            .map(LlmCloudlet.class::cast)
            .toList();


        // Coarse fleet-level energy/batch-size approximations for the example.
        // In production we attach a Listener that integrates per-tick.
        final double avgBatchSize = Math.max(1.0,
            finished.stream().mapToInt(LlmCloudlet::outputTokens).average().orElse(1.0) / 32.0);
        final double avgPowerW = host.getPowerModel().getPower(0.6);
        for (LlmCloudlet r : finished) {
            stats.onRequestFinished(r, scheduler, PUE, CARBON_GPER_KWH, avgBatchSize, avgPowerW);
        }

        // ---- 6. Print human-readable summary ----
        new CloudletsTableBuilder(broker.getCloudletFinishedList()).build();

        System.out.println();
        System.out.println("===== CloudSimLLM summary =====");
        System.out.printf ("Requests finished       : %d%n",     finished.size());
        System.out.printf ("Mean TTFT (s)           : %.4f%n",   stats.meanTtft());
        System.out.printf ("P99 TTFT (s)            : %.4f%n",   stats.p99Ttft());
        System.out.printf ("Mean TPOT (s)           : %.5f%n",   stats.meanTpot());
        System.out.printf ("P99 TPOT (s)            : %.5f%n",   stats.p99Tpot());
        System.out.printf ("Mean E2E (s)            : %.4f%n",   stats.meanE2e());
        System.out.printf ("SLO attainment          : %.2f%%%n", 100 * stats.sloAttainment());
        System.out.printf ("Total energy (kWh)      : %.4f%n",   stats.totalEnergyJoules() / 3.6e6);
        System.out.printf ("Total carbon (kgCO2eq)  : %.4f%n",   stats.totalCarbonGrams() / 1000.0);
    }

    private static List<Cloudlet> generateRequests(LlmModelSpec model, int n, long seed) {
        final var rng = new Random(seed);
        final List<Cloudlet> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            // ShareGPT-like distribution: prompt 256–1024, response 64–256.
            int sIn  = 256 + rng.nextInt(769);
            int sOut = 64  + rng.nextInt(193);
            var c = new LlmCloudlet(i, model, sIn, sOut, LlmCloudlet.SloClass.INTERACTIVE);
            c.onArrival(0.0); // example uses simultaneous arrival; replace with trace timestamps
            out.add(c);
        }
        return out;
    }
}
