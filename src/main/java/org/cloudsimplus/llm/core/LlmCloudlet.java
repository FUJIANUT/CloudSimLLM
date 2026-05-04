package org.cloudsimplus.llm.core;

import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.llm.workload.LlmModelSpec;
import org.cloudsimplus.utilizationmodels.UtilizationModel;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

/**
 * One LLM inference request. Repurposes {@link CloudletSimple} so existing
 * brokers, statistics, and listeners still work; LLM-specific fields are
 * added below and consumed by {@link org.cloudsimplus.llm.scheduler.ContinuousBatchScheduler}.
 *
 * <p><b>Length semantics:</b> the inherited {@code length} (in MI) is set to a
 * sentinel derived from {@code (inputTokens + outputTokens)}, only so that the
 * default Cloudlet termination logic still fires. The real progress is driven
 * by the LLM scheduler stepping {@link #generated()} until {@code outputTokens}.
 * </p>
 */
public class LlmCloudlet extends CloudletSimple {

    public enum Phase { WAITING, PREFILL, DECODE, DONE }
    public enum SloClass { INTERACTIVE, BATCH, BACKGROUND }

    private final LlmModelSpec model;
    private final int inputTokens;       // s_r
    private final int outputTokens;      // o_r (target length)
    private final SloClass sloClass;     // c_r

    /** Per-class SLO thresholds; null falls back to scheduler defaults. */
    private Double sloTtftSec;           // tau_T^{c_r}
    private Double sloTpotSec;           // tau_P^{c_r}

    /* ------- runtime state (mutated by scheduler) ------- */
    private Phase phase = Phase.WAITING;
    private int generated = 0;            // ℓ_r(t) − 1 once decode starts
    private double arrivalSimTime = -1.0;
    private double prefillStartSimTime = -1.0;
    private double firstTokenSimTime = -1.0;
    private double finishSimTime = -1.0;
    private double accumulatedDecodeSec = 0.0;

    public LlmCloudlet(long id, LlmModelSpec model, int inputTokens, int outputTokens, SloClass sloClass) {
        super(id, syntheticLength(inputTokens, outputTokens), 1L);
        this.model = model;
        this.inputTokens = inputTokens;
        this.outputTokens = outputTokens;
        this.sloClass = sloClass;
        // CPU/RAM/BW utilization are scheduler-driven for LLM cloudlets.
        final UtilizationModel full = new UtilizationModelFull();
        setUtilizationModelCpu(full);
    }

    private static long syntheticLength(int sIn, int oOut) {
        return Math.max(1L, (long) sIn + (long) oOut);
    }

    /* ------- accessors ------- */
    public LlmModelSpec model()       { return model; }
    public int inputTokens()          { return inputTokens; }
    public int outputTokens()         { return outputTokens; }
    public SloClass sloClass()        { return sloClass; }

    public LlmCloudlet setSloTtft(Double v) { this.sloTtftSec = v; return this; }
    public LlmCloudlet setSloTpot(Double v) { this.sloTpotSec = v; return this; }
    public Double sloTtftSec()        { return sloTtftSec; }
    public Double sloTpotSec()        { return sloTpotSec; }

    /* ------- runtime mutators (called by scheduler) ------- */
    public Phase phase()              { return phase; }
    public LlmCloudlet setPhase(Phase p) { this.phase = p; return this; }

    public int generated()            { return generated; }
    public void incrementGenerated()  { this.generated++; }

    public double arrivalSimTime()        { return arrivalSimTime; }
    public double prefillStartSimTime()   { return prefillStartSimTime; }
    public double firstTokenSimTime()     { return firstTokenSimTime; }
    public double finishSimTime()         { return finishSimTime; }
    public double accumulatedDecodeSec()  { return accumulatedDecodeSec; }

    public void onArrival(double now)      { this.arrivalSimTime = now; }
    public void onPrefillStart(double now) { this.prefillStartSimTime = now; this.phase = Phase.PREFILL; }
    public void onFirstToken(double now)   { this.firstTokenSimTime = now; this.phase = Phase.DECODE; }
    public void onDecodeStep(double dtSec) { this.accumulatedDecodeSec += dtSec; }
    /**
     * Records LLM-level completion time and transitions phase to DONE. Renamed
     * from {@code onFinish} because that name collides with
     * {@link org.cloudsimplus.core.StartableAbstract#onFinish(double)}, which
     * is invoked indirectly by CloudSim Plus on every status transition and
     * would otherwise overwrite our LLM-level finish timestamp.
     */
    public void markFinished(double now)   { this.finishSimTime = now; this.phase = Phase.DONE; }

    /** Eq. (3) realized: TTFT = firstToken − arrival + network RTT. NaN if undefined. */
    public double ttftSec() {
        return (firstTokenSimTime < 0 || arrivalSimTime < 0)
            ? Double.NaN : (firstTokenSimTime - arrivalSimTime + networkRttSec);
    }

    /** Eq. (4) realized: mean per-token decode latency. */
    public double tpotSec() {
        return generated <= 1 ? Double.NaN : accumulatedDecodeSec / (generated - 1);
    }

    /** Total active KV bytes for this request given current generated length. */
    public long currentKvBytes(int blockSize) {
        final long curLen = (long) inputTokens + Math.max(0, generated);
        final long blocks = (curLen + blockSize - 1) / blockSize;            // ⌈ℓ_r/β⌉
        return model.kvBytesPerToken() * (long) blockSize * blocks;          // Eq. (7) per-request term
    }

    /** Cross-region network round-trip latency added to user-perceived TTFT. */
    private double networkRttSec = 0.0;
    public double networkRttSec()                  { return networkRttSec; }
    public LlmCloudlet setNetworkRttSec(double s)  { this.networkRttSec = s; return this; }

    /* ---------- Splitwise hand-off support ---------- */
    private Long parentId;
    public Long parentId()                       { return parentId; }
    public LlmCloudlet setParentId(Long id)      { this.parentId = id; return this; }
    public boolean isDecodeShadow()              { return parentId != null; }

    /**
     * Build a decode-shadow of this request to run on a DECODE_ONLY worker
     * (Splitwise-style hand-off, §6.3). The shadow inherits arrival and
     * first-token timestamps so TTFT reported by the original is preserved
     * end-to-end. The shadow's input_len is set to the post-prefill length
     * so its KV footprint matches reality.
     */
    public LlmCloudlet newDecodeShadow(long shadowId) {
        final int postPrefillLen = inputTokens + 1;
        final int remainingOut   = Math.max(1, outputTokens - 1);
        LlmCloudlet shadow = new LlmCloudlet(shadowId, model, postPrefillLen, remainingOut, sloClass);
        shadow.setSloTtft(sloTtftSec).setSloTpot(sloTpotSec);
        shadow.parentId = this.getId();
        shadow.arrivalSimTime = this.arrivalSimTime;
        shadow.firstTokenSimTime = this.firstTokenSimTime;
        shadow.phase = Phase.DECODE;
        return shadow;
    }
}
