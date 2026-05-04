package org.cloudsimplus.llm.example;

import ch.qos.logback.classic.Level;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.llm.core.GpuHost;
import org.cloudsimplus.llm.core.GpuPe;
import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.llm.geo.CarbonAwareBroker;
import org.cloudsimplus.llm.geo.GeoRegion;
import org.cloudsimplus.llm.metrics.LlmStatistics;
import org.cloudsimplus.llm.power.LlmPowerModel;
import org.cloudsimplus.llm.scheduler.ContinuousBatchScheduler;
import org.cloudsimplus.llm.workload.KvCacheProvisioner;
import org.cloudsimplus.llm.workload.LlmModelSpec;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.util.Log;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * §6.5 Case Study 3 — Carbon-aware geo-distributed routing.
 *
 * <p>Wires up 3 regional datacenters, each with 4 A100 GPUs (12 total GPU
 * budget), and routes incoming LLM requests via {@link CarbonAwareBroker}.
 * Sweeps routing policy and time-of-day; reports (P99 TTFT, total carbon,
 * total energy).</p>
 *
 * <p>CLI:
 * <pre>--policy=LATENCY_GREEDY|CARBON_AWARE|BLENDED
 * --hour=0..23      (sim start hour for the carbon profile)
 * --workload=short|medium|long
 * --requests=500
 * --lambda=0.005    (carbon-latency exchange, BLENDED only)
 * --output=geo_results.csv</pre>
 * </p>
 */
public final class GeoDistributedRunner {

    private static final double PUE_DEFAULT      = 1.20;   // overridden by region
    private GeoDistributedRunner() { }

    public static void main(String[] argv) throws IOException {
        Log.setLevel(Level.WARN);
        Map<String,String> args = parseArgs(argv);

        String policyStr = args.getOrDefault("policy", "LATENCY_GREEDY");
        int hour         = Integer.parseInt(args.getOrDefault("hour", "12"));
        int numRequests  = Integer.parseInt(args.getOrDefault("requests", "500"));
        String workload  = args.getOrDefault("workload", "medium");
        long seed        = Long.parseLong(args.getOrDefault("seed", "42"));
        double lambda    = Double.parseDouble(args.getOrDefault("lambda", "0.005"));
        String label     = args.getOrDefault("label", "default");
        Path output      = Path.of(args.getOrDefault("output", "geo_results.csv"));

        var policy = CarbonAwareBroker.Policy.valueOf(policyStr);

        long t0 = System.currentTimeMillis();
        Result r = run(policy, hour, numRequests, workload, seed, lambda);
        long wallMs = System.currentTimeMillis() - t0;

        appendCsvRow(output, label, policyStr, hour, numRequests, workload, seed, lambda, wallMs, r);
        System.out.printf(
            "[done] policy=%s hour=%d workload=%s finished=%d ttft50=%.2f ttft99=%.2f e2e=%.2f carbon=%.3fkg energy=%.4fkWh wall=%dms%n",
            policyStr, hour, workload, r.finished, r.stats.meanTtft(), r.stats.p99Ttft(),
            r.stats.meanE2e(),
            r.totalCarbonGrams / 1000.0, r.stats.totalEnergyJoules() / 3.6e6, wallMs);
    }

    private static Result run(CarbonAwareBroker.Policy policy, int hour, int n,
                              String workload, long seed, double lambda) {
        var simulation = new CloudSimPlus(0.0001);
        var broker = new CarbonAwareBroker(simulation, seed)
            .setPolicy(policy)
            .setLambda(lambda);

        var model = LlmModelSpec.llama3_8B_fp16();
        var regions = GeoRegion.threeRegionWorld();
        // Inject the "hour-of-day" by shifting each region's carbon profile phase
        // rather than starting cloudlets at hour*3600 (which would race CloudSim's
        // event loop into a million-tick idle).
        for (var r : regions.values()) r.setPhaseOffsetHours(hour);

        // 4 GPU/VM per region × 3 regions = 12 GPUs total
        for (var entry : regions.entrySet()) {
            String regName = entry.getKey();
            var region     = entry.getValue();

            List<GpuHost> hosts = new ArrayList<>();
            List<Vm> vms = new ArrayList<>();
            for (int i = 0; i < 4; i++) {
                GpuPe gpu = calibratedA100();
                GpuHost host = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
                host.setIntraNvlinkGbs(600.0).setInterFabricGbs(200.0);
                host.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));
                hosts.add(host);

                var kv = new KvCacheProvisioner(gpu, 16);
                var sched = new ContinuousBatchScheduler(gpu, kv, model).setMaxBatchSize(64);
                vms.add(new VmSimple((double) gpu.getCapacity(), 1L, sched)
                    .setRam(8_000).setBw(1_000).setSize(10_000));
            }
            new DatacenterSimple(simulation, hosts, new VmAllocationPolicySimple());
            broker.register(region, vms);

            var allVms = new ArrayList<Vm>(vms);
            broker.submitVmList(allVms);
        }

        // Submit cloudlets, with arrival timestamps offset by `hour`*3600 so the
        // carbon profile is sampled at the right phase of the daily cycle.
        var cloudlets = generateRequests(model, n, workload, seed, hour);
        for (Cloudlet c : cloudlets) {
            if (c instanceof LlmCloudlet llm) broker.assignHome(llm);
        }
        broker.submitCloudletList(cloudlets);

        simulation.start();

        return collect(broker);
    }

    private static Result collect(CarbonAwareBroker broker) {
        var stats = new LlmStatistics();
        var finished = broker.getCloudletFinishedList().stream()
            .filter(LlmCloudlet.class::isInstance)
            .map(LlmCloudlet.class::cast)
            .toList();
        if (Boolean.parseBoolean(System.getenv().getOrDefault("LLM_DEBUG_TIMESTAMPS", "false"))) {
            int n = Math.min(finished.size(), 10);
            for (int i = 0; i < n; i++) {
                LlmCloudlet r = finished.get(i);
                System.err.printf("[ts] id=%d home=%s arr=%.4f firstTok=%.4f finish=%.4f gen=%d/o=%d ttft=%.4f%n",
                    r.getId(), broker.homeOf(r), r.arrivalSimTime(), r.firstTokenSimTime(),
                    r.finishSimTime(), r.generated(), r.outputTokens(),
                    r.firstTokenSimTime() - r.arrivalSimTime());
            }
        }

        // Use a representative GPU power for batch-amortized per-request energy.
        double avgPowerW = 240.0;
        double totalCarbonG = 0.0;
        for (LlmCloudlet r : finished) {
            Vm vm = r.getVm();
            if (vm == null || vm == Vm.NULL) continue;
            if (!(vm.getCloudletScheduler() instanceof ContinuousBatchScheduler s)) continue;
            // Use the per-request carbon from the broker (region-aware), not the
            // single-region default in LlmStatistics.
            double pCarbon = broker.estimatedCarbonGrams(r, avgPowerW);
            totalCarbonG += pCarbon;
            stats.onRequestFinished(r, s, 1.20, 0.0,    /*pue placeholder, no fleet carbon*/
                16.0, avgPowerW);
        }
        return new Result(finished.size(), stats, totalCarbonG);
    }

    /* ---------------------------------------------------------------- */

    private static GpuPe calibratedA100() {
        return GpuPe.a100_80gb()
            .setEffFp16TflopsPrefill(180.0).setEffFp16TflopsDecode(60.0)
            .setEffHbmBwGbs(1500.0).setAlphaPrefillSec(0.005).setAlphaDecodeSec(0.001);
    }

    private static List<Cloudlet> generateRequests(LlmModelSpec model, int n, String workload,
                                                    long seed, int hour) {
        var rng = new Random(seed);
        int[] shape = switch (workload) {
            case "short"  -> new int[] { 128,  385,   64,  129 };
            case "long"   -> new int[] {2048, 6145,  128,  385 };
            default       -> new int[] { 512, 1537,  256,  385 };
        };
        var out = new ArrayList<Cloudlet>(n);
        // Cloudlets always arrive starting at sim t=0; hour-of-day is injected via
        // the carbon profile phase offset on each region (see run() above).
        double simT = 0.0;
        for (int i = 0; i < n; i++) {
            simT += -Math.log(1 - rng.nextDouble()) / 50.0;
            int sIn  = shape[0] + rng.nextInt(shape[1]);
            int sOut = shape[2] + rng.nextInt(shape[3]);
            var c = new LlmCloudlet(i, model, sIn, sOut, LlmCloudlet.SloClass.INTERACTIVE);
            c.onArrival(simT);
            out.add(c);
        }
        return out;
    }

    private record Result(int finished, LlmStatistics stats, double totalCarbonGrams) {}

    /* ------------- output ------------- */

    private static void appendCsvRow(Path path, String label, String policy, int hour,
                                     int n, String workload, long seed, double lambda,
                                     long wallMs, Result r) throws IOException {
        boolean writeHeader = !Files.exists(path);
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(path,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
            if (writeHeader) {
                pw.println("label,policy,hour,requests,workload,seed,lambda,n_finished,"
                    + "mean_ttft_s,p99_ttft_s,mean_tpot_s,p99_tpot_s,mean_e2e_s,"
                    + "slo_attainment,total_energy_kwh,total_carbon_kg,wall_ms");
            }
            pw.printf("%s,%s,%d,%d,%s,%d,%.4f,%d,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%.6f,%.6f,%d%n",
                label, policy, hour, n, workload, seed, lambda, r.finished,
                r.stats.meanTtft(), r.stats.p99Ttft(),
                r.stats.meanTpot(), r.stats.p99Tpot(),
                r.stats.meanE2e(), r.stats.sloAttainment(),
                r.stats.totalEnergyJoules() / 3.6e6,
                r.totalCarbonGrams / 1000.0,
                wallMs);
        }
    }

    private static Map<String,String> parseArgs(String[] argv) {
        Map<String,String> m = new HashMap<>();
        for (String a : argv) {
            if (!a.startsWith("--")) continue;
            int eq = a.indexOf('=');
            if (eq < 0) m.put(a.substring(2), "true");
            else        m.put(a.substring(2, eq), a.substring(eq + 1));
        }
        return m;
    }
}
