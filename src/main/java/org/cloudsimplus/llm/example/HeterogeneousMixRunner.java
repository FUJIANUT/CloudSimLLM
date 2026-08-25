package org.cloudsimplus.llm.example;

import ch.qos.logback.classic.Level;
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
import org.cloudsimplus.llm.scheduler.LlmVmAllocationPolicy;
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
 * §6.4 Case Study 2 — Heterogeneous GPU mix at constant TFLOPS budget.
 *
 * <p>Mixes A100, H100, and L40S devices and routes via {@link LlmVmAllocationPolicy}.
 * Scans the placement policy axis ({@code FREE_HBM} / {@code EST_TTFT}) at
 * a fixed total compute budget, and reports the (cost, energy, latency)
 * Pareto surface.</p>
 *
 * <p>CLI mirrors {@code SplitwiseSweepRunner} for easy integration with the
 * Python sweep harness:
 * <pre>--mix=a100x4_h100x2_l40sx2  --policy=FREE_HBM  --requests=500</pre>
 * </p>
 */
public final class HeterogeneousMixRunner {

    private static final double PUE             = 1.15;
    private static final double CARBON_GPER_KWH = 380.0;

    private HeterogeneousMixRunner() { }

    public static void main(String[] argv) throws IOException {
        Log.setLevel(Level.WARN);
        Map<String,String> args = parseArgs(argv);

        String mix       = args.getOrDefault("mix", "a100x8");
        String policyStr = args.getOrDefault("policy", "FREE_HBM");
        int numRequests  = Integer.parseInt(args.getOrDefault("requests", "500"));
        String workload  = args.getOrDefault("workload", "medium");
        long seed        = Long.parseLong(args.getOrDefault("seed", "42"));
        String label     = args.getOrDefault("label", "default");
        Path output      = Path.of(args.getOrDefault("output", "het_results.csv"));
        double rate      = Double.parseDouble(args.getOrDefault("rate", "50"));

        var policy = LlmVmAllocationPolicy.RankBy.valueOf(policyStr);

        long t0 = System.currentTimeMillis();
        Result r = run(mix, policy, numRequests, workload, seed, rate);
        long wallMs = System.currentTimeMillis() - t0;

        appendCsvRow(output, label, mix, policyStr, numRequests, workload, seed, wallMs, r);
        System.out.printf(
            "[done] mix=%s policy=%s workload=%s finished=%d ttft50=%.2f ttft99=%.2f e2e=%.2f slo=%.0f%% wall=%dms%n",
            mix, policyStr, workload, r.finished, r.stats.meanTtft(), r.stats.p99Ttft(),
            r.stats.meanE2e(), 100*r.stats.sloAttainment(), wallMs);
    }

    /**
     * Parse a mix string like "a100x4_h100x2_l40sx2" into a list of factories.
     */
    private static List<java.util.function.Supplier<GpuPe>> parseMix(String mix) {
        List<java.util.function.Supplier<GpuPe>> out = new ArrayList<>();
        for (String tok : mix.split("_")) {
            String[] parts = tok.split("x");
            if (parts.length != 2) throw new IllegalArgumentException("bad mix token: " + tok);
            int n = Integer.parseInt(parts[1]);
            java.util.function.Supplier<GpuPe> factory = switch (parts[0].toLowerCase()) {
                case "a100"  -> () -> calibratedA100();
                case "h100"  -> () -> calibratedH100();
                case "l40s"  -> () -> calibratedL40S();
                default       -> throw new IllegalArgumentException("unknown sku: " + parts[0]);
            };
            for (int i = 0; i < n; i++) out.add(factory);
        }
        return out;
    }

    private static Result run(String mix, LlmVmAllocationPolicy.RankBy policy,
                               int n, String workload, long seed, double rate) {
        var simulation = new CloudSimPlus(0.0001);
        var broker = new DatacenterBrokerSimple(simulation);
        var model = LlmModelSpec.llama3_8B_fp16();

        List<GpuHost> hosts = new ArrayList<>();
        List<Vm> vms = new ArrayList<>();
        for (var factory : parseMix(mix)) {
            GpuPe gpu = factory.get();
            GpuHost host = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
            host.setIntraNvlinkGbs(600.0).setInterFabricGbs(200.0);
            host.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));
            hosts.add(host);

            var kv = new KvCacheProvisioner(gpu, 16);
            var sched = new ContinuousBatchScheduler(gpu, kv, model).setMaxBatchSize(64);
            vms.add(new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000));
        }

        var alloc = new LlmVmAllocationPolicy().setTargetModel(model).setRankBy(policy);
        new DatacenterSimple(simulation, hosts, alloc);

        broker.submitVmList(vms);
        broker.submitCloudletList(generateRequests(model, n, workload, seed, rate));
        simulation.start();

        return collect(broker, hosts);
    }

    private static Result collect(DatacenterBrokerSimple broker, List<GpuHost> hosts) {
        var stats = new LlmStatistics();
        var finished = broker.getCloudletFinishedList().stream()
            .filter(LlmCloudlet.class::isInstance)
            .map(LlmCloudlet.class::cast)
            .toList();

        double avgPowerW = hosts.stream()
            .mapToDouble(h -> h.getPowerModel().getPower(0.6)).average().orElse(0.0);
        for (LlmCloudlet r : finished) {
            Vm vm = r.getVm();
            if (vm == null || vm == Vm.NULL) continue;
            if (!(vm.getCloudletScheduler() instanceof ContinuousBatchScheduler s)) continue;
            stats.onRequestFinished(r, s, PUE, CARBON_GPER_KWH, 16.0, avgPowerW);
        }
        return new Result(finished.size(), stats);
    }

    /* ---------------------------------------------------------------- */

    private static GpuPe calibratedA100() {
        return GpuPe.a100_80gb()
            .setEffFp16TflopsPrefill(180.0).setEffFp16TflopsDecode(60.0)
            .setEffHbmBwGbs(1500.0).setAlphaPrefillSec(0.005).setAlphaDecodeSec(0.001);
    }
    private static GpuPe calibratedH100() {
        // Calibrated to peak ratios from POLCA/Splitwise reports; ~2.5x A100 prefill.
        return GpuPe.h100_80gb()
            .setEffFp16TflopsPrefill(450.0).setEffFp16TflopsDecode(150.0)
            .setEffHbmBwGbs(2500.0).setAlphaPrefillSec(0.004).setAlphaDecodeSec(0.0008);
    }
    private static GpuPe calibratedL40S() {
        // Inference-tier card; bandwidth-starved relative to A100/H100.
        return GpuPe.l40s_48gb()
            .setEffFp16TflopsPrefill(220.0).setEffFp16TflopsDecode(45.0)
            .setEffHbmBwGbs(650.0).setAlphaPrefillSec(0.006).setAlphaDecodeSec(0.0015);
    }

    private static List<Cloudlet> generateRequests(LlmModelSpec model, int n, String workload, long seed, double rate) {
        var rng = new Random(seed);
        int[] shape = switch (workload) {
            case "short"  -> new int[] { 128,  385,   64,  129 };
            case "long"   -> new int[] {2048, 6145,  128,  385 };
            default       -> new int[] { 512, 1537,  256,  385 };
        };
        var out = new ArrayList<Cloudlet>(n);
        double simT = 0.0;
        for (int i = 0; i < n; i++) {
            simT += -Math.log(1 - rng.nextDouble()) / rate;
            int sIn  = shape[0] + rng.nextInt(shape[1]);
            int sOut = shape[2] + rng.nextInt(shape[3]);
            var c = new LlmCloudlet(i, model, sIn, sOut, LlmCloudlet.SloClass.INTERACTIVE);
            // Paper thresholds (Azure LLM-trace guidance): TTFT <= 1/5/10 s
            // for short/medium/long prompt classes.
            c.setSloTtft(switch (workload) {
                case "short" -> 1.0; case "long" -> 10.0; default -> 5.0; });
            c.onArrival(simT);
            out.add(c);
        }
        return out;
    }

    private record Result(int finished, LlmStatistics stats) {}

    /* ------------- output ------------- */

    private static void appendCsvRow(Path path, String label, String mix, String policy,
                                     int n, String workload, long seed,
                                     long wallMs, Result r) throws IOException {
        boolean writeHeader = !Files.exists(path);
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(path,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
            if (writeHeader) {
                pw.println("label,mix,policy,requests,workload,seed,n_finished,"
                    + "mean_ttft_s,p99_ttft_s,mean_tpot_s,p99_tpot_s,mean_e2e_s,"
                    + "slo_attainment,total_energy_kwh,total_carbon_kg,wall_ms");
            }
            pw.printf("%s,%s,%s,%d,%s,%d,%d,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%.6f,%.6f,%d%n",
                label, mix, policy, n, workload, seed, r.finished,
                r.stats.meanTtft(), r.stats.p99Ttft(),
                r.stats.meanTpot(), r.stats.p99Tpot(),
                r.stats.meanE2e(), r.stats.sloAttainment(),
                r.stats.totalEnergyJoules() / 3.6e6,
                r.stats.totalCarbonGrams() / 1000.0,
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
