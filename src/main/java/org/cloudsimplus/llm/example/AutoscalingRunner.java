package org.cloudsimplus.llm.example;

import ch.qos.logback.classic.Level;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.llm.autoscale.AutoscalingBroker;
import org.cloudsimplus.llm.autoscale.WarmPoolAutoscaler;
import org.cloudsimplus.llm.core.GpuHost;
import org.cloudsimplus.llm.core.GpuPe;
import org.cloudsimplus.llm.core.LlmCloudlet;
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
 * §6.6 Case Study 4 — Bursty workload auto-scaling.
 *
 * <p>Generates a bursty Poisson trace (base rate plus periodic bursts) and
 * routes it through one of three {@link WarmPoolAutoscaler.Policy} variants
 * managed by {@link AutoscalingBroker}. Reports SLO violations, VM-hours,
 * cost, and energy.</p>
 *
 * <p>CLI:
 * <pre>--policy=STATIC|REACTIVE|PREDICTIVE
 * --burst=low|med|high
 * --startup-sec=15
 * --max-pool=8 --min-pool=2 --initial-pool=4
 * --requests=2000 --workload=medium
 * --output=autoscale_results.csv</pre>
 * </p>
 */
public final class AutoscalingRunner {

    private static final double PUE             = 1.15;
    private static final double CARBON_GPER_KWH = 380.0;
    private static final double VM_COST_PER_HR  = 1.85;     // A100 list price

    private AutoscalingRunner() { }

    public static void main(String[] argv) throws IOException {
        Log.setLevel(Level.WARN);
        Map<String,String> args = parseArgs(argv);

        String policyStr = args.getOrDefault("policy", "REACTIVE");
        String burstLevel = args.getOrDefault("burst", "med");
        double startupSec = Double.parseDouble(args.getOrDefault("startup-sec", "15"));
        int maxPool      = Integer.parseInt(args.getOrDefault("max-pool", "8"));
        int minPool      = Integer.parseInt(args.getOrDefault("min-pool", "2"));
        int initialPool  = Integer.parseInt(args.getOrDefault("initial-pool", "4"));
        int numRequests  = Integer.parseInt(args.getOrDefault("requests", "2000"));
        String workload  = args.getOrDefault("workload", "medium");
        long seed        = Long.parseLong(args.getOrDefault("seed", "42"));
        String label     = args.getOrDefault("label", "default");
        Path output      = Path.of(args.getOrDefault("output", "autoscale_results.csv"));
        double baseRate  = Double.parseDouble(args.getOrDefault("base-rate", "8.0"));
        // Override per-cloudlet TTFT SLO. Default tracks the medium-workload
        // production guidance (5 s); reviewer M3 motivated this knob.
        double ttftSlo   = Double.parseDouble(args.getOrDefault("ttft-slo", "5.0"));

        var policy = WarmPoolAutoscaler.Policy.valueOf(policyStr);

        long t0 = System.currentTimeMillis();
        Result r = run(policy, burstLevel, startupSec, maxPool, minPool, initialPool,
                       numRequests, workload, seed, baseRate, ttftSlo);
        long wallMs = System.currentTimeMillis() - t0;

        appendCsvRow(output, label, policyStr, burstLevel, startupSec, maxPool, minPool,
                     initialPool, numRequests, workload, seed, wallMs, r);
        System.out.printf(
            "[done] policy=%s burst=%s startup=%.0fs maxPool=%d finished=%d ttft50=%.2f ttft99=%.2f " +
            "slo=%.0f%% coldStart=%d activeVmH=%.2f $%.2f wall=%dms%n",
            policyStr, burstLevel, startupSec, maxPool, r.finished, r.stats.meanTtft(),
            r.stats.p99Ttft(), 100*r.stats.sloAttainment(), r.coldStartViolations,
            r.activeVmHours, r.cost, wallMs);
    }

    private static Result run(WarmPoolAutoscaler.Policy policy, String burstLevel,
                               double startupSec, int maxPool, int minPool, int initialPool,
                               int n, String workload, long seed, double baseRate,
                               double ttftSlo) {
        var simulation = new CloudSimPlus(0.0001);

        var autoscaler = new WarmPoolAutoscaler(policy, initialPool, minPool, maxPool, 4.0)
            .setHorizonSec(startupSec)            // PREDICTIVE looks ahead one cold-start
            .setEvaluationIntervalSec(2.0)
            .setScaleDownCooldownSec(30.0);
        var broker = new AutoscalingBroker(simulation, autoscaler).setStartupDelaySec(startupSec);
        var model = LlmModelSpec.llama3_8B_fp16();

        // Build maxPool VMs (one host each).
        List<GpuHost> hosts = new ArrayList<>();
        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < maxPool; i++) {
            GpuPe gpu = calibratedA100();
            GpuHost host = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
            host.setIntraNvlinkGbs(600.0).setInterFabricGbs(200.0);
            host.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));
            hosts.add(host);

            var kv = new KvCacheProvisioner(gpu, 16);
            var sched = new ContinuousBatchScheduler(gpu, kv, model)
                .setMaxBatchSize(64)
                // Cloudlets are delivered in 5-s arrival buckets; cap the
                // drain-ahead at that granularity so the internal clock never
                // races past undelivered work (the AUTOSCALE_TICK events
                // resume the drain every 2 s).
                .setMaxDrainAheadSec(5.0);
            vms.add(new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000));
        }
        new DatacenterSimple(simulation, hosts, new VmAllocationPolicySimple());

        broker.submitVmList(vms);
        broker.registerPool(vms, initialPool);
        // Single submission; the LLM scheduler gates by arrivalSimTime so VMs
        // see realistic queue build-up. Each cloudlet's finish listener pings
        // the autoscaler so scaling reacts as load evolves over the run.
        var trace = generateBurstyTrace(model, n, workload, seed, burstLevel, baseRate);
        // Apply the runner-supplied TTFT SLO to every cloudlet so the
        // metric reflects the regime the operator declares (M3).
        for (var c : trace) ((LlmCloudlet) c).setSloTtft(ttftSlo);
        // Group cloudlets by 5-second arrival buckets and submit each bucket
        // AT its arrival time (not merely with a delivery delay). The
        // vmMapper then runs when the bucket actually arrives, so cloudlets
        // landing after a scale-up are routed to the newly warm VMs — the
        // prerequisite for autoscaling decisions to influence latency.
        Map<Long, List<Cloudlet>> buckets = new HashMap<>();
        for (Cloudlet c : trace) {
            long bucket = (long)(((LlmCloudlet) c).arrivalSimTime() / 5.0);
            buckets.computeIfAbsent(bucket, b -> new ArrayList<>()).add(c);
        }
        var sortedKeys = new ArrayList<>(buckets.keySet());
        sortedKeys.sort(Long::compareTo);
        for (long bucket : sortedKeys) {
            broker.submitCloudletBucketAt(buckets.get(bucket), bucket * 5.0);
        }

        simulation.start();
        return collect(broker);
    }

    private static Result collect(AutoscalingBroker broker) {
        var stats = new LlmStatistics();
        var finished = broker.getCloudletFinishedList().stream()
            .filter(LlmCloudlet.class::isInstance)
            .map(LlmCloudlet.class::cast)
            .toList();

        double avgPowerW = 240.0;
        for (LlmCloudlet r : finished) {
            Vm vm = r.getVm();
            if (vm == null || vm == Vm.NULL) continue;
            if (!(vm.getCloudletScheduler() instanceof ContinuousBatchScheduler s)) continue;
            stats.onRequestFinished(r, s, PUE, CARBON_GPER_KWH, 16.0, avgPowerW);
        }
        double activeVmH = broker.totalActiveVmSeconds() / 3600.0;
        double idleVmH   = broker.totalIdleVmSeconds() / 3600.0;
        double cost      = activeVmH * VM_COST_PER_HR;
        return new Result(finished.size(), stats, broker.coldStartViolations(), activeVmH, idleVmH, cost);
    }

    /* ---------------------------------------------------------------- */

    private static GpuPe calibratedA100() {
        return GpuPe.a100_80gb()
            .setEffFp16TflopsPrefill(180.0).setEffFp16TflopsDecode(60.0)
            .setEffHbmBwGbs(1500.0).setAlphaPrefillSec(0.005).setAlphaDecodeSec(0.001);
    }

    /**
     * Modulated Poisson trace: base rate plus periodic bursts. The "low"
     * burst level multiplies base by 1.5×; "med" 3×; "high" 5×. Each burst
     * lasts {@code burstSec} seconds and recurs every {@code periodSec}.
     */
    private static List<Cloudlet> generateBurstyTrace(LlmModelSpec model, int n, String workload,
                                                      long seed, String burstLevel, double baseRate) {
        var rng = new Random(seed);
        int[] shape = switch (workload) {
            case "short"  -> new int[] { 128,  385,   64,  129 };
            case "long"   -> new int[] {2048, 6145,  128,  385 };
            default       -> new int[] { 512, 1537,  256,  385 };
        };
        double burstMult = switch (burstLevel) {
            case "low"  -> 1.5;
            case "high" -> 5.0;
            default     -> 3.0;       // med
        };
        double burstSec  = 20.0;       // duration of each burst
        double periodSec = 120.0;      // distance between bursts

        var out = new ArrayList<Cloudlet>(n);
        double simT = 0.0;
        for (int i = 0; i < n; i++) {
            // Determine burst state at simT:
            double phase = simT % periodSec;
            double rate = baseRate;
            if (phase < burstSec) rate = baseRate * burstMult;
            simT += -Math.log(1 - rng.nextDouble()) / rate;

            int sIn  = shape[0] + rng.nextInt(shape[1]);
            int sOut = shape[2] + rng.nextInt(shape[3]);
            var c = new LlmCloudlet(i, model, sIn, sOut, LlmCloudlet.SloClass.INTERACTIVE);
            c.onArrival(simT);
            out.add(c);
        }
        return out;
    }

    private record Result(int finished, LlmStatistics stats, int coldStartViolations,
                          double activeVmHours, double idleVmHours, double cost) {}

    /* ------------- output ------------- */

    private static void appendCsvRow(Path path, String label, String policy, String burst,
                                     double startupSec, int maxPool, int minPool,
                                     int initialPool, int n, String workload, long seed,
                                     long wallMs, Result r) throws IOException {
        boolean writeHeader = !Files.exists(path);
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(path,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
            if (writeHeader) {
                pw.println("label,policy,burst,startup_sec,max_pool,min_pool,initial_pool,"
                    + "requests,workload,seed,n_finished,mean_ttft_s,p99_ttft_s,"
                    + "mean_tpot_s,p99_tpot_s,mean_e2e_s,slo_attainment,"
                    + "cold_start_violations,active_vm_hours,idle_vm_hours,cost_usd,"
                    + "total_energy_kwh,wall_ms");
            }
            pw.printf("%s,%s,%s,%.1f,%d,%d,%d,%d,%s,%d,%d,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,"
                    + "%d,%.4f,%.4f,%.4f,%.6f,%d%n",
                label, policy, burst, startupSec, maxPool, minPool, initialPool, n, workload, seed,
                r.finished,
                r.stats.meanTtft(), r.stats.p99Ttft(),
                r.stats.meanTpot(), r.stats.p99Tpot(),
                r.stats.meanE2e(), r.stats.sloAttainment(),
                r.coldStartViolations, r.activeVmHours, r.idleVmHours, r.cost,
                r.stats.totalEnergyJoules() / 3.6e6,
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
