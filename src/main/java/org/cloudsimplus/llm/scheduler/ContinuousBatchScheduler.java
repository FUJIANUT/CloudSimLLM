package org.cloudsimplus.llm.scheduler;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.llm.core.GpuPe;
import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.llm.workload.KvCacheProvisioner;
import org.cloudsimplus.llm.workload.LlmModelSpec;
import org.cloudsimplus.schedulers.MipsShare;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerAbstract;

import java.util.ArrayList;
import java.util.List;

/**
 * vLLM-style continuous batching scheduler. One instance lives inside one VM
 * pinned to one (or TP-group of) {@link GpuPe}. Drives Algorithm 1 each
 * simulation tick by overriding {@link #updateProcessing(double, MipsShare)}.
 *
 * <p>Computes:
 * <ul>
 *   <li>Prefill latency from Eq. (1)</li>
 *   <li>Decode-step latency from Eq. (2)</li>
 *   <li>Capacity-aware admission via {@link KvCacheProvisioner} (Eq. 7)</li>
 *   <li>Records TTFT (Eq. 3) and TPOT (Eq. 4) onto each cloudlet</li>
 * </ul>
 * </p>
 *
 * <p><b>Phase mode:</b> the default policy interleaves prefill steps with
 * decode (chunked-prefill) by yielding to the next decode tick if any prefill
 * batch is pending. Subclasses may override {@link #pickNextPhase} for
 * disaggregated (Splitwise) or strict-priority variants.</p>
 */
public class ContinuousBatchScheduler extends CloudletSchedulerAbstract {

    public enum BatchPolicy { CONTIGUOUS_BATCH, CHUNKED_PREFILL, FIRST_COME_FIRST_SERVED }

    private final GpuPe gpu;
    private final KvCacheProvisioner kv;
    private final LlmModelSpec model;

    private int maxBatchSize = 256;
    private BatchPolicy policy = BatchPolicy.CONTIGUOUS_BATCH;

    /** Default SLOs by class (sec) — overridden per-cloudlet when set. */
    private double defaultTtftInteractive = 0.5;
    private double defaultTpotInteractive = 0.05;
    private double defaultTtftBatch       = 5.0;
    private double defaultTpotBatch       = 0.20;

    public ContinuousBatchScheduler(GpuPe gpu, KvCacheProvisioner kv, LlmModelSpec model) {
        this.gpu = gpu;
        this.kv = kv;
        this.model = model;
        this.kv.setModelLoaded(model);
    }

    public ContinuousBatchScheduler setMaxBatchSize(int v)  { this.maxBatchSize = v; return this; }
    public ContinuousBatchScheduler setPolicy(BatchPolicy p) { this.policy = p; return this; }

    /**
     * Accept LLM cloudlets immediately into the parent's exec list so that
     * CloudSim Plus continues to schedule {@code updateProcessing} ticks for
     * this VM. Arrival gating is enforced inside {@code updateProcessing} by
     * filtering on {@link LlmCloudlet#arrivalSimTime()}.
     */
    @Override
    protected boolean canExecuteCloudletInternal(CloudletExecution cle) {
        return cle.getCloudlet() instanceof LlmCloudlet;
    }

    /**
     * LLM continuous batching does not model classical pause/resume — each
     * tick is a fresh batch decision. Returning 0 says "no expected finish
     * delta from a resume call".
     */
    @Override
    public double cloudletResume(final org.cloudsimplus.cloudlets.Cloudlet cloudlet) {
        return 0.0;
    }

    /** Internal LLM clock — advances as we run batches; never goes backward. */
    private double internalClock = -1.0;
    /**
     * How far (sim-seconds) the internal clock may drain ahead of the last
     * datacenter event. Infinite for bulk-submitted traces (all future
     * arrivals visible in the exec list, so the arrival bound is exact).
     * Runners that deliver cloudlets dynamically (e.g. autoscaling buckets)
     * should set this to their delivery granularity so the drain never
     * races past work that has not been delivered yet; periodic broker
     * events then resume the drain.
     */
    private double maxDrainAheadSec = Double.MAX_VALUE;
    public ContinuousBatchScheduler setMaxDrainAheadSec(double v) { this.maxDrainAheadSec = v; return this; }

    /**
     * <b>Important:</b> we bypass parent's MI-based completion logic (LLM
     * completion is phase-driven, Eqs. 1–4) and we maintain an internal clock
     * because CloudSim Plus may invoke {@code updateProcessing} multiple times
     * at the same simulation timestamp (one call per cloudlet-submission event).
     * Without an internal clock, all batches would record the same finish time.
     *
     * <p>On each call we:
     * <ol>
     *   <li>Advance {@link #internalClock} forward to {@code currentTime} if
     *       CloudSim's clock has progressed.</li>
     *   <li>Drain as many batches as possible while requests remain (the
     *       LLM workload is naturally batch-greedy: a single Vm wants to run
     *       all admitted requests through their phases without yielding).</li>
     *   <li>Return the simulation time at which we expect to be re-invoked.</li>
     * </ol>
     * </p>
     */
    @Override
    public double updateProcessing(final double currentTime, final MipsShare mipsShare) {
        setPreviousTime(currentTime);
        setCurrentMipsShare(mipsShare);

        if (internalClock < currentTime) internalClock = currentTime;
        if (Boolean.parseBoolean(System.getenv().getOrDefault("LLM_DEBUG_TICK", "false"))) {
            System.err.printf("[tick] sim=%.6f intClock=%.6f exec=%d wait=%d%n",
                currentTime, internalClock, getCloudletExecList().size(),
                getCloudletWaitingList().size());
        }

        // Arrival-bounded greedy drain: process admittable work forward in
        // (internal) time, but never advance past the next future arrival in
        // the exec list without first admitting it. This gives vLLM-style
        // continuous-batching semantics (new requests join the running batch
        // at their exact arrival time) while remaining event-driven: every
        // arrival among bulk-submitted cloudlets is an internal idle-jump
        // target, and dynamically delivered cloudlets re-trigger this method
        // via their own datacenter events. Decode jumps are budget-bounded by
        // the next arrival (see runDecodeStep), so no request can starve
        // inside a long non-preemptible jump.
        final double drainLimit = maxDrainAheadSec == Double.MAX_VALUE
            ? Double.MAX_VALUE : currentTime + maxDrainAheadSec;
        final int safety = 1_000_000;
        for (int iter = 0; iter < safety; iter++) {
            if (internalClock >= drainLimit) break;   // resume at next event
            final List<LlmCloudlet> arrived = arrivedCloudlets(internalClock);
            if (arrived.isEmpty()) {
                // Nothing admittable *now* — idle-jump to the next future
                // arrival already present in the exec list, if any.
                final double nextArr = nextFutureArrival(internalClock);
                if (nextArr == Double.MAX_VALUE || nextArr > drainLimit) break;
                internalClock = nextArr;
                continue;
            }

            // KV admission per request (best-effort, idempotent for already-admitted).
            for (LlmCloudlet r : arrived) {
                if (r.arrivalSimTime() < 0) r.onArrival(internalClock);
                kv.tryAdmit(r);
            }
            // Bound to maxBatchSize.
            final List<LlmCloudlet> batch = arrived.size() <= maxBatchSize
                ? arrived : arrived.subList(0, maxBatchSize);

            final List<LlmCloudlet> prefillSet = new ArrayList<>();
            final List<LlmCloudlet> decodeSet = new ArrayList<>();
            for (LlmCloudlet r : batch) {
                if (r.phase() == LlmCloudlet.Phase.WAITING || r.phase() == LlmCloudlet.Phase.PREFILL) prefillSet.add(r);
                else if (r.phase() == LlmCloudlet.Phase.DECODE) decodeSet.add(r);
            }
            final double nextArr = Math.min(nextFutureArrival(internalClock), drainLimit);
            final double budget = nextArr == Double.MAX_VALUE
                ? Double.MAX_VALUE : Math.max(0.0, nextArr - internalClock);
            double dt = pickNextPhase(prefillSet, decodeSet, internalClock, budget);
            if (dt <= 0) break;
            internalClock += dt;
            finalizeDoneRequests();
        }

        if (getCloudletWaitingList().isEmpty() && getCloudletExecList().isEmpty()) {
            return Double.MAX_VALUE;
        }
        // If we did real work this call, wake up at internalClock (advanced by dt).
        // Otherwise, if everything is arrival-gated, jump to the earliest upcoming
        // arrival rather than busy-polling 1e-6 ticks (which never reaches hour 12).
        double nextArrival = activeLlmCloudlets().stream()
            .mapToDouble(LlmCloudlet::arrivalSimTime)
            .filter(t -> t > internalClock)
            .min().orElse(Double.MAX_VALUE);
        if (internalClock > currentTime) {
            return Math.min(internalClock, nextArrival);
        }
        return nextArrival == Double.MAX_VALUE ? currentTime + 1e-6 : nextArrival;
    }

    /** Earliest arrival strictly after {@code now} among exec-list cloudlets. */
    private double nextFutureArrival(double now) {
        return activeLlmCloudlets().stream()
            .mapToDouble(LlmCloudlet::arrivalSimTime)
            .filter(t -> t > now)
            .min().orElse(Double.MAX_VALUE);
    }

    /** Cloudlets in execList whose Poisson arrival time has elapsed. */
    private List<LlmCloudlet> arrivedCloudlets(double now) {
        return getCloudletExecList().stream()
            .map(CloudletExecution::getCloudlet)
            .filter(LlmCloudlet.class::isInstance)
            .map(LlmCloudlet.class::cast)
            .filter(r -> r.arrivalSimTime() < 0 || r.arrivalSimTime() <= now)
            .toList();
    }

    /**
     * Move any phase=DONE LlmCloudlet from execList → finishedList so
     * {@link org.cloudsimplus.cloudlets.Cloudlet#addOnFinishListener listeners}
     * fire. We consume the cloudlet's remaining MI to satisfy parent invariants
     * (e.g. accounting code that reads {@link CloudletExecution#getRemainingCloudletLength()}),
     * then remove from execList so we do not enqueue it again next tick.
     */
    private void finalizeDoneRequests() {
        for (CloudletExecution cle : new ArrayList<>(getCloudletExecList())) {
            if (cle.getCloudlet() instanceof LlmCloudlet r && r.phase() == LlmCloudlet.Phase.DONE) {
                long remaining = cle.getRemainingCloudletLength();
                if (remaining > 0) cle.updateProcessing(remaining);
                removeCloudletFromExecList(cle);
                cloudletFinish(cle);
            }
        }
    }

    /**
     * Decides which phase to run this tick and returns the elapsed time. Default
     * implementation: run prefill if any are pending (Eq. 1); otherwise advance
     * decode (Eq. 2) for at most {@code budgetSec} of wall-clock so the next
     * arriving request can join the batch on time. Override for
     * Splitwise-style splitting.
     */
    protected double pickNextPhase(List<LlmCloudlet> prefillSet,
                                   List<LlmCloudlet> decodeSet,
                                   double currentTime,
                                   double budgetSec) {
        if (!prefillSet.isEmpty() && policy != BatchPolicy.FIRST_COME_FIRST_SERVED) {
            return runPrefillBatch(prefillSet, currentTime);
        }
        if (!decodeSet.isEmpty()) {
            return runDecodeStep(decodeSet, currentTime, budgetSec);
        }
        if (!prefillSet.isEmpty()) {
            return runPrefillBatch(prefillSet, currentTime);
        }
        return 0.0;
    }

    /** Eq. (1) — prefill latency for the chosen batch. */
    protected double runPrefillBatch(List<LlmCloudlet> batch, double now) {
        long sumS = 0;
        for (LlmCloudlet r : batch) {
            if (r.prefillStartSimTime() < 0) r.onPrefillStart(now);
            sumS += r.inputTokens();
        }
        final double flopsPerToken = 2.0 * model.parameters();              // 2·P_m FLOPs/token
        final double effFlops = gpu.effFp16TflopsPrefill() * 1e12;
        final double t = (flopsPerToken * sumS) / effFlops + gpu.alphaPrefillSec();
        for (LlmCloudlet r : batch) {
            r.onFirstToken(now + t);
            r.incrementGenerated();
        }
        return t;
    }

    /**
     * Eq. (2) — advance the decode batch by K steps at once, where K is the
     * smallest number of remaining tokens across the batch (i.e., until the
     * first request finishes). Returns total wall-clock time consumed.
     *
     * <p>This is the event-driven formulation: we jump straight to the next
     * "interesting" event (a request finishing) rather than ticking 1 token at
     * a time, which fights CloudSim's default {@code minTimeBetweenEvents}.</p>
     */
    protected double runDecodeStep(List<LlmCloudlet> batch, double now) {
        return runDecodeStep(batch, now, Double.MAX_VALUE);
    }

    /**
     * Budget-bounded variant: advances at most
     * {@code max(1, floor(budgetSec / tStep))} steps so a request arriving at
     * {@code now + budgetSec} is admitted into the very next batch rather than
     * waiting out a long non-preemptible jump.
     */
    protected double runDecodeStep(List<LlmCloudlet> batch, double now, double budgetSec) {
        if (batch.isEmpty()) return 0.0;

        final double flopsBound = (2.0 * model.parameters() * batch.size())
            / (gpu.effFp16TflopsDecode() * 1e12);
        final long mw = model.weightBytes();
        final long mkv = kv.allocatedKvBytes();
        final double bwBound = (mw + mkv) / (gpu.effHbmBwGbs() * 1e9);
        final double tStep = Math.max(flopsBound, bwBound) + gpu.alphaDecodeSec();

        int kSteps = batch.stream()
            .mapToInt(r -> Math.max(1, r.outputTokens() - r.generated()))
            .min().orElse(1);
        if (budgetSec != Double.MAX_VALUE) {
            final int kBudget = Math.max(1, (int) Math.floor(budgetSec / tStep));
            kSteps = Math.min(kSteps, kBudget);
        }

        for (LlmCloudlet r : batch) {
            int actualK = Math.min(kSteps, r.outputTokens() - r.generated());
            for (int i = 0; i < actualK; i++) {
                r.onDecodeStep(tStep);
                r.incrementGenerated();
            }
            if (r.generated() >= r.outputTokens()) {
                r.markFinished(now + actualK * tStep);
                kv.evict(r);
            }
        }
        return kSteps * tStep;
    }

    /**
     * Pull from waiting queue while batch and HBM allow (Algorithm 1, lines 2–3).
     * Uses {@link CloudletSchedulerAbstract#addWaitingCloudletToExecList} to
     * properly transfer the cloudlet from waiting → exec list, which is
     * required for {@link #finalizeDoneRequests} to find them in execList later.
     *
     * <p><b>Arrival gating:</b> if {@link LlmCloudlet#arrivalSimTime()} is set to a
     * future time (e.g. Poisson trace timestamps pre-loaded at submission), the
     * cloudlet is held until {@code now} reaches that arrival. This makes TTFT
     * (= firstToken − arrival) non-negative regardless of broker timing.</p>
     */
    /** @deprecated retained for subclasses; logic moved into {@link #updateProcessing}. */
    @Deprecated
    protected void admitWaitingFromQueue(List<LlmCloudlet> active, double now) { /* no-op */ }

    private List<LlmCloudlet> activeLlmCloudlets() {
        return getCloudletExecList().stream()
            .map(CloudletExecution::getCloudlet)
            .filter(LlmCloudlet.class::isInstance)
            .map(LlmCloudlet.class::cast)
            .toList();
    }

    /** SLO threshold lookup with class fallback. */
    public double effectiveSloTtft(LlmCloudlet r) {
        if (r.sloTtftSec() != null) return r.sloTtftSec();
        return switch (r.sloClass()) {
            case INTERACTIVE -> defaultTtftInteractive;
            case BATCH       -> defaultTtftBatch;
            case BACKGROUND  -> Double.POSITIVE_INFINITY;
        };
    }
    public double effectiveSloTpot(LlmCloudlet r) {
        if (r.sloTpotSec() != null) return r.sloTpotSec();
        return switch (r.sloClass()) {
            case INTERACTIVE -> defaultTpotInteractive;
            case BATCH       -> defaultTpotBatch;
            case BACKGROUND  -> Double.POSITIVE_INFINITY;
        };
    }

    public KvCacheProvisioner kv()  { return kv; }
    public GpuPe gpu()              { return gpu; }
    public LlmModelSpec model()     { return model; }
}
