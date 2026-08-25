package org.cloudsimplus.llm.example;

import ch.qos.logback.classic.Level;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
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
 * §6.7 Case Study 5 — DynamoLLM-inspired instance-pool routing at a fixed
 * same-GPU budget (8 × A100-80GB).
 *
 * <p>DynamoLLM (Stojkovic et al., HPCA'25) reconfigures LLM inference
 * clusters into pools of differently-configured instances (tensor
 * parallelism, frequency, instance count) and routes queries by their
 * token lengths to the pool that meets the SLO at least energy. We
 * emulate the pool + size-aware-routing concept (without DVFS, which
 * CloudSimLLM does not model) at a fixed 8-GPU budget:</p>
 *
 * <ul>
 *   <li><b>tp1x8</b> — eight TP=1 instances, round-robin routing.</li>
 *   <li><b>tp4x2</b> — two TP=4 instances (latency-optimal), round-robin.</li>
 *   <li><b>mixed</b> — pools of two TP=1, one TP=2, and one TP=4 instance
 *       (2+2+4 = 8 GPUs) with either SIZE_AWARE routing (input &lt; 512 →
 *       TP1, &lt; 2048 → TP2, else TP4; the DynamoLLM-style policy) or
 *       ROUND_ROBIN (routing-oblivious baseline).</li>
 * </ul>
 *
 * <p>TP-k instances are emulated as aggregate {@link GpuPe}s with
 * k-scaled effective FLOPS/bandwidth de-rated by a tensor-parallel
 * efficiency factor (eta_2 = 0.95, eta_4 = 0.90, consistent with the
 * ≤5%-of-TPOT AllReduce overhead bound of Eq. 10) and k-scaled power.</p>
 *
 * <pre>--config=mixed --routing=SIZE_AWARE --requests=1000 --seed=42</pre>
 */
public final class PoolRoutingRunner {

    private static final double PUE             = 1.15;
    private static final double CARBON_GPER_KWH = 380.0;
    /** Default mixed-trace arrival rate (req/s); override with --rate. */
    private static final double ARRIVAL_RATE    = 30.0;

    private PoolRoutingRunner() { }

    /** One instance pool: identical TP-k instances + its own broker. */
    private record Pool(String name, int tpDegree, int instances,
                        DatacenterBrokerSimple broker, List<GpuHost> hosts) {}

    public static void main(String[] argv) throws IOException {
        Log.setLevel(Level.WARN);
        Map<String,String> args = parseArgs(argv);

        String config   = args.getOrDefault("config", "mixed");        // tp1x8 | tp4x2 | mixed
        String routing  = args.getOrDefault("routing", "SIZE_AWARE");  // SIZE_AWARE | ROUND_ROBIN
        double rate     = Double.parseDouble(args.getOrDefault("rate", String.valueOf(ARRIVAL_RATE)));
        // Fixed observation horizon (not fixed request count) so different
        // arrival rates are compared over the same steady-state window.
        double horizon  = Double.parseDouble(args.getOrDefault("horizon-sec", "150"));
        int numRequests = args.containsKey("requests")
            ? Integer.parseInt(args.get("requests"))
            : (int) Math.ceil(rate * horizon);
        long seed       = Long.parseLong(args.getOrDefault("seed", "42"));
        String label    = args.getOrDefault("label", "default");
        Path output     = Path.of(args.getOrDefault("output", "pool_results.csv"));

        long t0 = System.currentTimeMillis();
        Result r = run(config, routing, numRequests, seed, rate);
        long wallMs = System.currentTimeMillis() - t0;

        appendCsvRow(output, label, config, routing, numRequests, seed, wallMs, r);
        System.out.printf(
            "[done] config=%s routing=%s finished=%d ttft99=%.2f slo=%.0f%% energy=%.4fkWh J/tok=%.2f wall=%dms%n",
            config, routing, r.finished, r.stats.p99Ttft(), 100*r.stats.sloAttainment(),
            r.stats.totalEnergyJoules()/3.6e6, r.joulesPerToken, wallMs);
    }

    private static Result run(String config, String routing, int n, long seed, double rate) {
        var simulation = new CloudSimPlus(0.0001);
        var model = LlmModelSpec.llama3_8B_fp16();

        // ---- Build pools per config (8-GPU budget) ----
        List<Pool> pools = new ArrayList<>();
        switch (config) {
            case "tp1x8" -> pools.add(buildPool(simulation, model, "tp1", 1, 8));
            case "tp4x2" -> pools.add(buildPool(simulation, model, "tp4", 4, 2));
            case "mixed" -> {
                pools.add(buildPool(simulation, model, "tp1", 1, 2));
                pools.add(buildPool(simulation, model, "tp2", 2, 1));
                pools.add(buildPool(simulation, model, "tp4", 4, 1));
            }
            default -> throw new IllegalArgumentException("unknown config: " + config);
        }

        // ---- Generate the mixed trace and route to pools ----
        List<Cloudlet> trace = generateMixedTrace(model, n, seed, rate);
        long rr = 0;
        // Weight round-robin slots by instance count so ROUND_ROBIN is not
        // penalised for pool-count asymmetry.
        List<Pool> rrSlots = new ArrayList<>();
        for (Pool p : pools) for (int i = 0; i < p.instances(); i++) rrSlots.add(p);
        for (Cloudlet c : trace) {
            LlmCloudlet llm = (LlmCloudlet) c;
            Pool target;
            if (pools.size() == 1) {
                target = pools.get(0);
            } else if (routing.equals("SIZE_AWARE")) {
                // DynamoLLM-style: long inputs need TP4 prefill speed for
                // their TTFT SLO; short inputs meet SLO on the cheapest pool.
                int sIn = llm.inputTokens();
                target = sIn < 512 ? pools.get(0) : sIn < 2048 ? pools.get(1) : pools.get(2);
            } else { // ROUND_ROBIN
                target = rrSlots.get((int) (rr++ % rrSlots.size()));
            }
            target.broker().submitCloudletList(List.of(llm));
        }

        simulation.start();
        return collect(pools);
    }

    /** Build one pool of {@code instances} aggregate TP-{@code k} A100 instances. */
    private static Pool buildPool(CloudSimPlus sim, LlmModelSpec model,
                                  String name, int k, int instances) {
        double eta = k == 1 ? 1.0 : k == 2 ? 0.95 : 0.90;   // TP efficiency (Eq. 10 bound)
        var broker = new DatacenterBrokerSimple(sim);
        List<GpuHost> hosts = new ArrayList<>();
        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < instances; i++) {
            GpuPe gpu = new GpuPe("A100x" + k + "-TP", 312.0 * k, 2039.0 * k,
                                  80L * 1024 * 1024 * 1024 * k, 400.0 * k, 50.0 * k)
                .setEffFp16TflopsPrefill(180.0 * k * eta)
                .setEffFp16TflopsDecode(12.0 * k * eta)
                .setEffHbmBwGbs(1500.0 * k * eta)
                .setAlphaPrefillSec(0.005).setAlphaDecodeSec(0.001);
            GpuHost host = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
            host.setIntraNvlinkGbs(600.0).setInterFabricGbs(200.0);
            host.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));
            hosts.add(host);

            var kv = new KvCacheProvisioner(gpu, 16);
            var sched = new ContinuousBatchScheduler(gpu, kv, model).setMaxBatchSize(64);
            vms.add(new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000));
        }
        Datacenter dc = new DatacenterSimple(sim, hosts, new VmAllocationPolicySimple());
        // Pin this pool's broker to its own datacenter so a small-TP VM is
        // never placed onto another pool's (larger) hosts.
        broker.setDatacenterMapper((last, vm) -> dc);
        broker.submitVmList(vms);
        return new Pool(name, k, instances, broker, hosts);
    }

    /**
     * Mixed trace: 1/3 short, 1/3 medium, 1/3 long request shapes with
     * per-class TTFT SLOs (1 s / 5 s / 10 s), Poisson arrivals.
     */
    private static List<Cloudlet> generateMixedTrace(LlmModelSpec model, int n, long seed, double rate) {
        var rng = new Random(seed);
        int[][] shapes = {
            { 128,  385,   64,  129 },   // short
            { 512, 1537,  256,  385 },   // medium
            {2048, 6145,  128,  385 },   // long
        };
        double[] ttftSlo = { 1.0, 5.0, 10.0 };
        var out = new ArrayList<Cloudlet>(n);
        double simT = 0.0;
        for (int i = 0; i < n; i++) {
            simT += -Math.log(1 - rng.nextDouble()) / rate;
            int cls = rng.nextInt(3);
            int[] shape = shapes[cls];
            int sIn  = shape[0] + rng.nextInt(shape[1]);
            int sOut = shape[2] + rng.nextInt(shape[3]);
            var c = new LlmCloudlet(i, model, sIn, sOut, LlmCloudlet.SloClass.INTERACTIVE);
            c.setSloTtft(ttftSlo[cls]);
            c.onArrival(simT);
            out.add(c);
        }
        return out;
    }

    /**
     * Per-pool energy accounting: instance power is k-scaled and the
     * batch-amortisation term uses the pool's Little's-law mean
     * concurrency per instance, so TP-k pools amortise weight reads over
     * their (larger) shared batches.
     */
    private static Result collect(List<Pool> pools) {
        var stats = new LlmStatistics();
        int finished = 0;
        long totalTokens = 0;

        for (Pool p : pools) {
            var done = p.broker().getCloudletFinishedList().stream()
                .filter(LlmCloudlet.class::isInstance)
                .map(LlmCloudlet.class::cast)
                .toList();
            if (done.isEmpty()) continue;

            // Little's law: mean concurrent requests per instance in this pool.
            double horizon = done.stream().mapToDouble(LlmCloudlet::finishSimTime).max().orElse(1.0);
            double activeSum = done.stream()
                .mapToDouble(r -> Math.max(0, r.finishSimTime() - r.arrivalSimTime())).sum();
            double meanConcurrencyPerInstance =
                Math.max(1.0, activeSum / Math.max(1e-6, horizon) / p.instances());

            double instancePowerW = p.hosts().get(0).getPowerModel().getPower(0.6);

            for (LlmCloudlet r : done) {
                Vm vm = r.getVm();
                if (vm == null || vm == Vm.NULL) continue;
                if (!(vm.getCloudletScheduler() instanceof ContinuousBatchScheduler s)) continue;
                stats.onRequestFinished(r, s, PUE, CARBON_GPER_KWH,
                    meanConcurrencyPerInstance, instancePowerW);
                totalTokens += r.outputTokens();
            }
            finished += done.size();
        }
        double jPerTok = totalTokens == 0 ? Double.NaN
            : stats.totalEnergyJoules() / totalTokens;
        return new Result(finished, stats, jPerTok);
    }

    private record Result(int finished, LlmStatistics stats, double joulesPerToken) {}

    private static void appendCsvRow(Path path, String label, String config, String routing,
                                     int n, long seed, long wallMs, Result r) throws IOException {
        boolean writeHeader = !Files.exists(path);
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(path,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
            if (writeHeader) {
                pw.println("label,config,routing,requests,seed,n_finished,"
                    + "mean_ttft_s,p99_ttft_s,mean_tpot_s,p99_tpot_s,mean_e2e_s,"
                    + "slo_attainment,total_energy_kwh,joules_per_token,wall_ms");
            }
            pw.printf("%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f,%.6f,%.4f,%d%n",
                label, config, routing, n, seed, r.finished,
                r.stats.meanTtft(), r.stats.p99Ttft(),
                r.stats.meanTpot(), r.stats.p99Tpot(),
                r.stats.meanE2e(), r.stats.sloAttainment(),
                r.stats.totalEnergyJoules() / 3.6e6,
                r.joulesPerToken, wallMs);
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
