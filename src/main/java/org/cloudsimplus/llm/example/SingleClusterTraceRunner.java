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
import org.cloudsimplus.llm.workload.KvCacheProvisioner;
import org.cloudsimplus.llm.workload.LlmModelSpec;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.util.Log;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * §6.2 cross-simulator comparison — replays an externally generated trace
 * (CSV: {@code arrival_s,input_tokens,output_tokens}) on a single-cluster
 * A100 deployment and dumps per-request records for comparison against
 * Vidur's per-request output on the identical trace.
 *
 * <pre>
 * --trace=trace.csv --gpus=1 --out-records=records.csv
 * </pre>
 */
public final class SingleClusterTraceRunner {

    private SingleClusterTraceRunner() { }

    public static void main(String[] argv) throws IOException {
        Log.setLevel(Level.WARN);
        Map<String,String> args = parseArgs(argv);

        Path trace   = Path.of(args.get("trace"));
        int gpus     = Integer.parseInt(args.getOrDefault("gpus", "1"));
        Path outRec  = Path.of(args.getOrDefault("out-records", "records.csv"));

        var simulation = new CloudSimPlus(0.0001);
        var broker = new DatacenterBrokerSimple(simulation);
        var model = LlmModelSpec.llama3_8B_fp16();

        List<GpuHost> hosts = new ArrayList<>();
        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < gpus; i++) {
            GpuPe gpu = GpuPe.a100_80gb()
                .setEffFp16TflopsPrefill(180.0).setEffFp16TflopsDecode(12.0)
                .setEffHbmBwGbs(1500.0).setAlphaPrefillSec(0.005).setAlphaDecodeSec(0.001);
            GpuHost host = new GpuHost(64_000L, 100_000L, 2_000_000L, List.<Pe>of(gpu));
            host.setIntraNvlinkGbs(600.0).setInterFabricGbs(200.0);
            host.setPowerModel(new LlmPowerModel().setDecodeDiscount(0.7));
            hosts.add(host);
            var kv = new KvCacheProvisioner(gpu, 16);
            // Arrival-bounded stepping (unconditional): requests arriving
            // mid-batch are admitted into the running batch at their exact
            // arrival time (continuous batching semantics, matching vLLM/Vidur).
            var sched = new ContinuousBatchScheduler(gpu, kv, model)
                .setMaxBatchSize(128);
            vms.add(new VmSimple((double) gpu.getCapacity(), 1L, sched)
                .setRam(8_000).setBw(1_000).setSize(10_000));
        }
        new DatacenterSimple(simulation, hosts, new VmAllocationPolicySimple());
        broker.submitVmList(vms);

        // ---- Load the shared trace ----
        List<Cloudlet> cloudlets = new ArrayList<>();
        int id = 0;
        for (String line : Files.readAllLines(trace)) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#") || line.startsWith("arrival")) continue;
            String[] tok = line.split(",");
            double arrival = Double.parseDouble(tok[0]);
            int sIn  = Integer.parseInt(tok[1].trim());
            int sOut = Integer.parseInt(tok[2].trim());
            var c = new LlmCloudlet(id++, model, sIn, sOut, LlmCloudlet.SloClass.INTERACTIVE);
            c.setSloTtft(5.0);
            c.onArrival(arrival);
            cloudlets.add(c);
        }
        // Bulk submission: all cloudlets (with their arrival timestamps) are
        // visible in the exec list from t=0, so the scheduler's
        // arrival-bounded drain admits each request into the running batch
        // at exactly its arrival time — vLLM-style continuous batching.
        broker.submitCloudletList(cloudlets);

        long t0 = System.currentTimeMillis();
        simulation.start();
        long wallMs = System.currentTimeMillis() - t0;

        var stats = new LlmStatistics();
        int finished = 0;
        for (Cloudlet c : broker.getCloudletFinishedList()) {
            if (!(c instanceof LlmCloudlet r)) continue;
            Vm vm = r.getVm();
            if (vm == null || vm == Vm.NULL) continue;
            if (!(vm.getCloudletScheduler() instanceof ContinuousBatchScheduler s)) continue;
            stats.onRequestFinished(r, s, 1.15, 380.0, 16.0,
                hosts.get(0).getPowerModel().getPower(0.6));
            finished++;
        }
        stats.writeRecordsCsv(outRec);
        System.out.printf("[done] trace=%s gpus=%d finished=%d/%d wall=%dms "
            + "ttft50=%.3f ttft99=%.3f tpot50=%.4f%n",
            trace, gpus, finished, cloudlets.size(), wallMs,
            stats.records().stream().mapToDouble(rr -> rr.ttftSec()).sorted()
                .skip(finished/2).findFirst().orElse(Double.NaN),
            stats.p99Ttft(), stats.meanTpot());
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
