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
import org.cloudsimplus.llm.scheduler.PrefillDecodeBroker;
import org.cloudsimplus.llm.scheduler.PrefillDecodeDisaggScheduler;
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
 * CLI-driven single-configuration runner for the §6.3 case study sweep.
 * Reads parameters via {@code --key=value} CLI args, runs <em>one</em>
 * (mode, P:D, workload, KV bandwidth) cell, and appends one CSV row.
 *
 * <p>Driven by {@code tools/experiments/run_splitwise_sweep.py} which iterates
 * over the experimental grid and aggregates results.</p>
 *
 * <p>Example:
 * <pre>
 * ./mvnw exec:java -q -Dexec.mainClass=org.cloudsimplus.llm.example.SplitwiseSweepRunner \
 *   -Dexec.args="--mode=splitwise --prefill-gpus=4 --decode-gpus=4 \
 *                --requests=500 --workload=medium --kv-bw-gbs=200 \
 *                --seed=42 --label=default --output=results.csv"
 * </pre>
 * </p>
 */
public final class SplitwiseSweepRunner {

    private static final double PUE             = 1.15;
    private static final double CARBON_GPER_KWH = 380.0;

    private SplitwiseSweepRunner() { }

    public static void main(String[] argv) throws IOException {
        Log.setLevel(Level.WARN); // keep stdout clean for sweep aggregation

        Map<String,String> args = parseArgs(argv);
        String mode      = args.getOrDefault("mode", "splitwise");        // colocated | splitwise
        int prefillGpus  = Integer.parseInt(args.getOrDefault("prefill-gpus", "4"));
        int decodeGpus   = Integer.parseInt(args.getOrDefault("decode-gpus", "4"));
        int numRequests  = Integer.parseInt(args.getOrDefault("requests", "500"));
        String workload  = args.getOrDefault("workload", "medium");       // short | medium | long
        double kvBwGbs   = Double.parseDouble(args.getOrDefault("kv-bw-gbs", "200"));
        long seed        = Long.parseLong(args.getOrDefault("seed", "42"));
        String label     = args.getOrDefault("label", "default");
        Path output      = Path.of(args.getOrDefault("output", "results.csv"));

        long t0 = System.currentTimeMillis();
        Result r = "colocated".equals(mode)
            ? runCoLocated(prefillGpus + decodeGpus, numRequests, workload, seed)
            : runSplitwise(prefillGpus, decodeGpus, numRequests, workload, kvBwGbs, seed);
        long wallMs = System.currentTimeMillis() - t0;

        appendCsvRow(output, label, mode, prefillGpus, decodeGpus, numRequests,
            workload, kvBwGbs, seed, wallMs, r);

        // Concise stdout summary so the sweep driver can show progress.
        System.out.printf(
            "[done] mode=%s P:D=%d:%d workload=%s kvBw=%.0f finished=%d ttft50=%.2f ttft99=%.2f e2e=%.2f slo=%.0f%% wall=%dms%n",
            mode, prefillGpus, decodeGpus, workload, kvBwGbs,
            r.finished, r.stats.meanTtft(), r.stats.p99Ttft(), r.stats.meanE2e(),
            100*r.stats.sloAttainment(), wallMs);

    }

    /* ---------------------------------------------------------------- */

    private static Result runCoLocated(int totalGpus, int n, String workload, long seed) {
        // 0.1 ms event granularity: KV transfer delays and decode steps are
        // sub-millisecond, so the default 0.1 s would batch shadow arrivals
        // into the same tick and skip {@code updateCloudletProcessing}.
        var simulation = new CloudSimPlus(0.0001);
        var broker = new DatacenterBrokerSimple(simulation);
        var model = LlmModelSpec.llama3_8B_fp16();

        List<GpuHost> hosts = new ArrayList<>();
        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < totalGpus; i++) {
            GpuPe gpu = calibratedA100();
            GpuHost host = newHost(gpu, 200);
            hosts.add(host);
            var kv = new KvCacheProvisioner(gpu, 16);
            var sched = new ContinuousBatchScheduler(gpu, kv, model).setMaxBatchSize(64);
            vms.add(new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000));
        }
        new DatacenterSimple(simulation, hosts, new VmAllocationPolicySimple());
        broker.submitVmList(vms);
        broker.submitCloudletList(generateRequests(model, n, workload, seed));
        simulation.start();

        return collect(broker, hosts, /*hasShadows*/ false);
    }

    private static Result runSplitwise(int p, int d, int n, String workload,
                                       double kvBwGbs, long seed) {
        // 0.1 ms event granularity: KV transfer delays and decode steps are
        // sub-millisecond, so the default 0.1 s would batch shadow arrivals
        // into the same tick and skip {@code updateCloudletProcessing}.
        var simulation = new CloudSimPlus(0.0001);
        var broker = new PrefillDecodeBroker(simulation).setKvTransferBwGbs(kvBwGbs);
        var model = LlmModelSpec.llama3_8B_fp16();

        List<GpuHost> hosts = new ArrayList<>();
        List<Vm> prefillVms = new ArrayList<>();
        List<Vm> decodeVms = new ArrayList<>();

        for (int i = 0; i < p; i++) {
            GpuPe gpu = calibratedA100();
            GpuHost host = newHost(gpu, kvBwGbs);
            hosts.add(host);
            var kv = new KvCacheProvisioner(gpu, 16);
            var sched = new PrefillDecodeDisaggScheduler(gpu, kv, model,
                PrefillDecodeDisaggScheduler.Role.PREFILL_ONLY).setMaxBatchSize(64);
            prefillVms.add(new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000));
        }
        for (int i = 0; i < d; i++) {
            GpuPe gpu = calibratedA100();
            GpuHost host = newHost(gpu, kvBwGbs);
            hosts.add(host);
            var kv = new KvCacheProvisioner(gpu, 16);
            var sched = new PrefillDecodeDisaggScheduler(gpu, kv, model,
                PrefillDecodeDisaggScheduler.Role.DECODE_ONLY).setMaxBatchSize(64);
            decodeVms.add(new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000));
        }
        new DatacenterSimple(simulation, hosts, new VmAllocationPolicySimple());

        broker.setPrefillVms(prefillVms).setDecodeVms(decodeVms);
        var allVms = new ArrayList<Vm>();
        allVms.addAll(prefillVms);
        allVms.addAll(decodeVms);
        broker.submitVmList(allVms);
        broker.submitCloudletList(generateRequests(model, n, workload, seed));
        simulation.start();

        return collect(broker, hosts, /*hasShadows*/ true);
    }

    private static Result collect(DatacenterBrokerSimple broker, List<GpuHost> hosts, boolean hasShadows) {
        var stats = new LlmStatistics();
        var finished = broker.getCloudletFinishedList().stream()
            .filter(LlmCloudlet.class::isInstance)
            .map(LlmCloudlet.class::cast)
            .toList();
        var toReport = hasShadows
            ? finished.stream().filter(LlmCloudlet::isDecodeShadow).toList()
            : finished;

        if (Boolean.parseBoolean(System.getenv().getOrDefault("LLM_DEBUG_TIMESTAMPS", "false"))) {
            System.err.printf("[ts] all %d cloudlets:%n", toReport.size());
            for (LlmCloudlet r : toReport) {
                double e2e = r.finishSimTime() - r.arrivalSimTime();
                String mark = e2e > 1e6 ? " <-- HUGE" : "";
                System.err.printf("  id=%d arr=%.4f firstTok=%.4f finish=%.4f gen=%d/o=%d e2e=%.4f%s%n",
                    r.getId(), r.arrivalSimTime(), r.firstTokenSimTime(), r.finishSimTime(),
                    r.generated(), r.outputTokens(), e2e, mark);
            }
        }

        double avgPowerW = hosts.stream()
            .mapToDouble(h -> h.getPowerModel().getPower(0.6)).average().orElse(0.0);
        for (LlmCloudlet r : toReport) {
            Vm vm = r.getVm();
            if (vm == null || vm == Vm.NULL) continue;
            if (!(vm.getCloudletScheduler() instanceof ContinuousBatchScheduler s)) continue;
            stats.onRequestFinished(r, s, PUE, CARBON_GPER_KWH, 16.0, avgPowerW);
        }
        return new Result(toReport.size(), stats);
    }

    /* ---------------------------------------------------------------- */

    private static GpuHost newHost(GpuPe gpu, double interFabricGbs) {
        GpuHost h = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
        h.setIntraNvlinkGbs(600.0).setInterFabricGbs(interFabricGbs);
        h.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));
        return h;
    }

    private static GpuPe calibratedA100() {
        return GpuPe.a100_80gb()
            .setEffFp16TflopsPrefill(180.0)
            .setEffFp16TflopsDecode(60.0)
            .setEffHbmBwGbs(1500.0)
            .setAlphaPrefillSec(0.005)
            .setAlphaDecodeSec(0.001);
    }

    private static List<Cloudlet> generateRequests(LlmModelSpec model, int n, String workload, long seed) {
        var rng = new Random(seed);
        // (minIn, rangeIn, minOut, rangeOut)
        int[] shape = switch (workload) {
            case "short"  -> new int[] { 128,  385,   64,  129 };  // 128–512 in,  64–192 out
            case "long"   -> new int[] {2048, 6145,  128,  385 };  // 2K–8K in,   128–512 out
            default       -> new int[] { 512, 1537,  256,  385 };  // 512–2048,   256–640 out  (medium)
        };
        var out = new ArrayList<Cloudlet>(n);
        double lambda = 50.0;       // 50 req/s Poisson arrivals
        double simT = 0.0;
        for (int i = 0; i < n; i++) {
            simT += -Math.log(1 - rng.nextDouble()) / lambda;
            int sIn  = shape[0] + rng.nextInt(shape[1]);
            int sOut = shape[2] + rng.nextInt(shape[3]);
            var c = new LlmCloudlet(i, model, sIn, sOut, LlmCloudlet.SloClass.INTERACTIVE);
            c.onArrival(simT);
            out.add(c);
        }
        return out;
    }

    private record Result(int finished, LlmStatistics stats) {}

    /* ------------- output ------------- */

    private static void appendCsvRow(Path path, String label, String mode, int p, int d,
                                     int n, String workload, double kvBw, long seed,
                                     long wallMs, Result r) throws IOException {
        boolean writeHeader = !Files.exists(path);
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(path,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
            if (writeHeader) {
                pw.println("label,mode,prefill_gpus,decode_gpus,requests,workload,kv_bw_gbs,seed,"
                    + "n_finished,mean_ttft_s,p99_ttft_s,mean_tpot_s,p99_tpot_s,mean_e2e_s,"
                    + "slo_attainment,total_energy_kwh,total_carbon_kg,wall_ms");
            }
            pw.printf("%s,%s,%d,%d,%d,%s,%.1f,%d,%d,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%.6f,%.6f,%d%n",
                label, mode, p, d, n, workload, kvBw, seed,
                r.finished,
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
