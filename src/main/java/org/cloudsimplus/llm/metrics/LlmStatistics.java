package org.cloudsimplus.llm.metrics;

import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.llm.scheduler.ContinuousBatchScheduler;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Aggregates per-request and fleet-level metrics. Implements Eqs. (3), (4),
 * (9), (10), (12). One instance per simulation; receives finished cloudlets
 * via {@link #onRequestFinished} and per-tick power samples via
 * {@link #recordPowerSample}.
 */
public class LlmStatistics {

    public record RequestRecord(
        long id,
        int inputTokens,
        int outputTokens,
        double ttftSec,         // Eq. (3)
        double tpotSec,         // Eq. (4)
        double e2eSec,
        double energyJoules,    // Eq. (9)
        double carbonGrams,     // Eq. (10)
        boolean ttftOk,
        boolean tpotOk
    ) {}

    private final List<RequestRecord> records = new ArrayList<>();
    private double totalEnergyJoules = 0.0;
    private double totalCarbonGrams = 0.0;
    private double simHorizonSec = 0.0;

    /** Hook the scheduler into a request-finish callback. */
    public void onRequestFinished(LlmCloudlet r,
                                  ContinuousBatchScheduler scheduler,
                                  double pue,
                                  double carbonIntensityGramsPerKwh,
                                  double averageBatchSizeWhileActive,
                                  double averagePowerWattsWhileActive) {
        final double ttft = r.ttftSec();
        final double tpot = r.tpotSec();
        final double e2e = (r.finishSimTime() < 0 || r.arrivalSimTime() < 0)
            ? Double.NaN : (r.finishSimTime() - r.arrivalSimTime());

        // Eq. (9): batch-amortized energy
        final double activeSec = (r.finishSimTime() - r.firstTokenSimTime());
        final double energyJ = averagePowerWattsWhileActive
            / Math.max(1.0, averageBatchSizeWhileActive)
            * Math.max(0.0, activeSec);

        // Eq. (10): operational carbon
        final double energyKwh = energyJ / 3.6e6;
        final double carbonG = energyKwh * pue * carbonIntensityGramsPerKwh;

        boolean ttftOk = ttft <= scheduler.effectiveSloTtft(r);
        boolean tpotOk = tpot <= scheduler.effectiveSloTpot(r);

        records.add(new RequestRecord(r.getId(), r.inputTokens(), r.outputTokens(),
            ttft, tpot, e2e, energyJ, carbonG, ttftOk, tpotOk));

        totalEnergyJoules += energyJ;
        totalCarbonGrams += carbonG;
        simHorizonSec = Math.max(simHorizonSec, r.finishSimTime());
    }

    /** Allows external power meters (e.g., {@code PowerMeter}) to push samples. */
    public void recordPowerSample(double watts, double dtSec) {
        totalEnergyJoules += watts * dtSec;
    }

    /** Eq. (12) — Goodput over a window [t0, t1]. */
    public double goodputTokensPerSec(double t0, double t1) {
        double tokens = 0.0;
        for (RequestRecord r : records) {
            // boundary check against e2e finish time embedded in record creation order
            if (r.ttftOk && r.tpotOk) {
                tokens += r.outputTokens;
            }
        }
        double window = Math.max(1e-6, t1 - t0);
        return tokens / window;
    }

    public double meanTtft()        { return mean(r -> r.ttftSec); }
    public double p99Ttft()         { return percentile(99, r -> r.ttftSec); }
    public double meanTpot()        { return mean(r -> r.tpotSec); }
    public double p99Tpot()         { return percentile(99, r -> r.tpotSec); }
    public double meanE2e()         { return mean(r -> r.e2eSec); }
    public double sloAttainment()   {
        if (records.isEmpty()) return Double.NaN;
        long ok = records.stream().filter(r -> r.ttftOk && r.tpotOk).count();
        return ok / (double) records.size();
    }
    public double totalEnergyJoules() { return totalEnergyJoules; }
    public double totalCarbonGrams()  { return totalCarbonGrams; }
    public List<RequestRecord> records() { return records; }

    /**
     * Dump per-request records as CSV. Header columns line up with
     * {@code tools/analysis/validate_cli.py}'s expected per-record schema.
     * Aggregation to per-cell summaries is done in the Python side.
     */
    public void writeRecordsCsv(Path out) throws IOException {
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out))) {
            pw.println("id,input_len,output_len,ttft_sec,tpot_sec,e2e_sec,energy_j,carbon_g,ttft_ok,tpot_ok");
            for (RequestRecord r : records) {
                pw.printf("%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%b,%b%n",
                    r.id, r.inputTokens, r.outputTokens,
                    r.ttftSec, r.tpotSec, r.e2eSec, r.energyJoules, r.carbonGrams,
                    r.ttftOk, r.tpotOk);
            }
        }
    }

    private double mean(java.util.function.ToDoubleFunction<RequestRecord> f) {
        return records.stream().mapToDouble(f).filter(d -> !Double.isNaN(d)).average().orElse(Double.NaN);
    }
    private double percentile(int p, java.util.function.ToDoubleFunction<RequestRecord> f) {
        var sorted = records.stream().mapToDouble(f).filter(d -> !Double.isNaN(d)).sorted().toArray();
        if (sorted.length == 0) return Double.NaN;
        int idx = (int) Math.ceil(p / 100.0 * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(idx, sorted.length - 1))];
    }
}
