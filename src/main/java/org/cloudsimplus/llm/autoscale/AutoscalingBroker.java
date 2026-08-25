package org.cloudsimplus.llm.autoscale;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.core.events.SimEvent;
import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.vms.Vm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pre-allocates a fixed maximum-size warm pool of VMs and routes incoming
 * {@link LlmCloudlet}s only to the {@link #activePoolSize active subset}
 * dictated by a {@link WarmPoolAutoscaler}. New VMs entering the active
 * pool become available only after a configurable {@code startupDelaySec}
 * (the cold-start cost). VMs leaving the active pool stop receiving new
 * traffic immediately but continue draining their in-flight cloudlets.
 *
 * <p>Cost / energy accounting tracks each VM's lifetime in three buckets:
 * <ul>
 *   <li><b>active-warm</b>: VM is in the active pool and has fully started.</li>
 *   <li><b>active-warming</b>: VM has been added but is still in startup; bills
 *       at full rate but cannot serve traffic (this is the cold-start tax).</li>
 *   <li><b>idle (out-of-pool)</b>: VM is in the warm pool's max-size buffer
 *       but not active; bills at idle rate (≈ 0 in our model).</li>
 * </ul>
 * </p>
 */
public class AutoscalingBroker extends DatacenterBrokerSimple {

    private final List<Vm>           pool         = new ArrayList<>();   // size == maxPool
    private final Map<Long, Double>  vmReadyAt    = new HashMap<>();      // VM-id → time it becomes warm
    private final Map<Long, Double>  vmActivatedAt = new HashMap<>();     // first time a VM became active
    private final Map<Long, Double>  vmActiveSec   = new HashMap<>();     // accumulated active-warm seconds
    private final WarmPoolAutoscaler autoscaler;
    private double startupDelaySec = 15.0;
    private int activePoolSize;
    private double lastDecisionTime = 0.0;
    private int coldStartViolations = 0;
    private long roundRobinCounter = 0;
    /** Cloudlets routed since the last autoscaler tick — reactive metric. */
    private int sinceLastTickCount = 0;

    /** Custom event tag for the periodic autoscaler tick. */
    private static final int AUTOSCALE_TICK = CloudSimTag.NONE - 9999;
    /** Custom event tag for arrival-time bucket submission. */
    private static final int SUBMIT_BUCKET = CloudSimTag.NONE - 9998;
    /** How often we poll the autoscaler (sim seconds). */
    private double tickIntervalSec = 2.0;
    private boolean tickerStarted = false;
    /** Buckets registered before simulation start, flushed in startInternal. */
    private final List<Object[]> pendingBuckets = new ArrayList<>();
    private int bucketsOutstanding = 0;

    public AutoscalingBroker(CloudSimPlus simulation, WarmPoolAutoscaler autoscaler) {
        super(simulation);
        this.autoscaler = autoscaler;
        setVmDestructionDelay(60.0);
    }

    public AutoscalingBroker setTickIntervalSec(double v) { this.tickIntervalSec = v; return this; }

    @Override
    public void startInternal() {
        super.startInternal();
        // Kick off the first periodic autoscaler tick.
        if (!tickerStarted) {
            schedule(tickIntervalSec, AUTOSCALE_TICK);
            tickerStarted = true;
        }
        // Flush buckets registered before simulation start: each bucket is
        // delivered (and therefore VM-mapped) at its arrival time, so scaling
        // decisions made before that time genuinely change request routing.
        for (Object[] entry : pendingBuckets) {
            schedule((Double) entry[0], SUBMIT_BUCKET, entry[1]);
        }
        pendingBuckets.clear();
    }

    /**
     * Register a bucket of cloudlets to be submitted (and VM-mapped) at
     * {@code atTime} sim seconds. Unlike {@code submitCloudletList(list,
     * delay)} — which maps every cloudlet to a VM immediately at t=0 and only
     * delays datacenter delivery — this defers the {@code vmMapper} call
     * itself, so cloudlets arriving after a scale-up are actually routed to
     * the newly warmed VMs. This is the mechanism that lets autoscaling
     * decisions affect request latency.
     */
    public AutoscalingBroker submitCloudletBucketAt(List<? extends Cloudlet> bucket, double atTime) {
        bucketsOutstanding++;
        if (isStarted()) {
            schedule(Math.max(0, atTime - getSimulation().clock()), SUBMIT_BUCKET, bucket);
        } else {
            pendingBuckets.add(new Object[]{ atTime, bucket });
        }
        return this;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void processEvent(final SimEvent evt) {
        if (evt.getTag() == AUTOSCALE_TICK) {
            onProgressTick();
            // Only reschedule if work is still pending; otherwise let sim end.
            int finished = getCloudletFinishedList().size();
            if (bucketsOutstanding > 0 || finished < getCloudletSubmittedList().size() || finished == 0) {
                schedule(tickIntervalSec, AUTOSCALE_TICK);
            }
            return;
        }
        if (evt.getTag() == SUBMIT_BUCKET) {
            bucketsOutstanding--;
            submitCloudletList((List<Cloudlet>) evt.getData());
            return;
        }
        super.processEvent(evt);
    }

    public AutoscalingBroker setStartupDelaySec(double v) { this.startupDelaySec = v; return this; }
    public double startupDelaySec()       { return startupDelaySec; }
    public int activePoolSize()           { return activePoolSize; }
    public int coldStartViolations()      { return coldStartViolations; }
    public WarmPoolAutoscaler autoscaler(){ return autoscaler; }

    /**
     * Register a pre-built pool. The first {@code initialActive} VMs are
     * marked as warm-from-start; the remainder are idle.
     */
    public AutoscalingBroker registerPool(List<Vm> vms, int initialActive) {
        pool.clear();
        pool.addAll(vms);
        activePoolSize = Math.min(initialActive, pool.size());
        for (int i = 0; i < activePoolSize; i++) {
            vmReadyAt.put(pool.get(i).getId(), 0.0);
            vmActivatedAt.put(pool.get(i).getId(), 0.0);
        }
        return this;
    }

    /**
     * Drives the autoscaler from a periodic tick. The broker's vmMapper is
     * only called at submission time (often a single t=0 burst), so we cannot
     * rely on it for ongoing scaling decisions. Call this from a per-cloudlet
     * finish listener installed by the runner.
     */
    public void onProgressTick() {
        if (pool.isEmpty()) return;
        final double now = getSimulation().clock();
        // The metric we hand the autoscaler is the *arrival rate* over the last
        // tickIntervalSec window (cloudlets routed per VM-equivalent), not the
        // instantaneous VM queue depth. Reason: with greedy in-call batching by
        // the LLM scheduler (§4 design choice), the per-VM exec list is empty
        // immediately after each updateProcessing — but the broker still
        // observed bursty arrivals that motivate scaling.
        double demandPerVmEquivalent = sinceLastTickCount;   // cloudlets arrived this window
        sinceLastTickCount = 0;
        int newTarget = autoscaler.decide(now, (int) demandPerVmEquivalent);
        if (newTarget != activePoolSize) {
            if (Boolean.parseBoolean(System.getenv().getOrDefault("LLM_DEBUG_AUTOSCALE", "false"))) {
                System.err.printf("[scale] t=%.2f demand=%.0f target %d -> %d%n",
                    now, demandPerVmEquivalent, activePoolSize, newTarget);
            }
            applyTarget(newTarget, now);
        }
    }

    @Override
    protected Vm defaultVmMapper(final Cloudlet cloudlet) {
        if (!(cloudlet instanceof LlmCloudlet llm)) return super.defaultVmMapper(cloudlet);
        if (pool.isEmpty()) return Vm.NULL;

        final double now = getSimulation().clock();

        // (We do not defer mapping by arrivalSimTime here because CloudSim's
        // CLOUDLET_CREATION event only fires once per submitCloudletList call;
        // returning Vm.NULL for future arrivals would strand them. The LLM
        // scheduler enforces arrival-time gating, which produces the correct
        // VM-side queue dynamics for the autoscaler to observe.)

        // Queue depth reflects what is *currently* in-flight on the active VMs.
        // We deliberately do NOT include broker.cloudletWaitingList because that
        // list holds the entire trace from sim start onward (cloudlets are added
        // to it at submission, not at their arrival time), which would make the
        // autoscaler immediately scale to maxPool.
        int queueDepth = 0;
        for (int i = 0; i < activePoolSize; i++) {
            Vm v = pool.get(i);
            queueDepth += v.getCloudletScheduler().getCloudletExecList().size()
                        + v.getCloudletScheduler().getCloudletWaitingList().size();
        }
        int newTarget = autoscaler.decide(now, queueDepth);
        if (newTarget != activePoolSize) {
            if (Boolean.parseBoolean(System.getenv().getOrDefault("LLM_DEBUG_AUTOSCALE", "false"))) {
                System.err.printf("[scale] t=%.2f queue=%d target %d -> %d%n",
                    now, queueDepth, activePoolSize, newTarget);
            }
            applyTarget(newTarget, now);
        }

        // Build the live warm subset (size ≤ activePoolSize). Round-robin
        // assigns load uniformly across warm VMs; cold (warming) VMs do not
        // receive traffic until their readyAt time has elapsed.
        List<Vm> warmVms = new ArrayList<>();
        for (int i = 0; i < activePoolSize; i++) {
            Vm v = pool.get(i);
            Double readyAt = vmReadyAt.get(v.getId());
            if (readyAt != null && now >= readyAt) warmVms.add(v);
        }
        if (warmVms.isEmpty()) {
            // No warm VM at all (e.g. mid-cold-start with min pool already drained);
            // fall back to the first VM and record a cold-start violation.
            coldStartViolations++;
            return pool.get(0);
        }
        Vm chosen = warmVms.get((int) (roundRobinCounter++ % warmVms.size()));
        sinceLastTickCount++;
        // If cloudlet was supposed to go to a not-yet-warm VM (any VM in the
        // active pool whose readyAt > now), record a cold-start delay penalty.
        if (warmVms.size() < activePoolSize) coldStartViolations++;
        return chosen;
    }

    /**
     * Adjust the active pool size, accounting for cold-start delays for new
     * VMs and immediate drain (no termination) for removed VMs.
     */
    private void applyTarget(int newTarget, double now) {
        newTarget = Math.max(0, Math.min(newTarget, pool.size()));
        if (newTarget == activePoolSize) return;

        if (newTarget > activePoolSize) {
            // Activate VMs with cold-start delay.
            for (int i = activePoolSize; i < newTarget; i++) {
                Vm v = pool.get(i);
                double readyAt = now + startupDelaySec;
                vmReadyAt.put(v.getId(), readyAt);
                vmActivatedAt.putIfAbsent(v.getId(), now);
            }
        } else {
            // Drain VMs (instant removal from active pool).
            for (int i = newTarget; i < activePoolSize; i++) {
                Vm v = pool.get(i);
                Double activatedAt = vmActivatedAt.remove(v.getId());
                Double readyAt = vmReadyAt.remove(v.getId());
                if (activatedAt != null && readyAt != null) {
                    double activeSec = Math.max(0, now - readyAt);
                    vmActiveSec.merge(v.getId(), activeSec, Double::sum);
                }
            }
        }
        activePoolSize = newTarget;
        lastDecisionTime = now;
    }

    /**
     * Wall-clock end of the *useful* work — the last cloudlet's finish time.
     * Using this rather than {@code simulation.clock()} avoids inflating
     * cost integrals by the broker's vm-destruction-delay tail.
     */
    public double effectiveDurationSec() {
        double maxFinish = 0.0;
        for (var c : getCloudletFinishedList()) {
            if (c instanceof LlmCloudlet llm && llm.finishSimTime() > maxFinish) {
                maxFinish = llm.finishSimTime();
            }
        }
        return Math.max(maxFinish, getSimulation().clock() / 2);   // safety lower bound
    }

    /**
     * Total VM-active-warm seconds across the simulation. Cap each VM's
     * accumulator at {@link #effectiveDurationSec()}.
     */
    public double totalActiveVmSeconds() {
        final double duration = effectiveDurationSec();
        double total = 0;
        for (Vm v : pool) {
            Double activatedAt = vmActivatedAt.get(v.getId());
            Double readyAt     = vmReadyAt.get(v.getId());
            if (activatedAt != null && readyAt != null) {
                total += Math.max(0, Math.min(duration, readyAt + (duration - readyAt)) - readyAt);
            }
            total += vmActiveSec.getOrDefault(v.getId(), 0.0);
        }
        return total;
    }

    public double totalIdleVmSeconds() {
        double duration = effectiveDurationSec();
        double maxSeconds = pool.size() * duration;
        return Math.max(0, maxSeconds - totalActiveVmSeconds());
    }
}
