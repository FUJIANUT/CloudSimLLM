package org.cloudsimplus.llm.autoscale;

/**
 * Decides how many VMs the warm pool should hold given the current queue
 * depth and a short history of recent depths. Three policies model the
 * canonical industry choices for §6.6:
 *
 * <ul>
 *   <li><b>STATIC</b> — fixed pool size; never scales. Over-provisions to
 *       absorb peak load; pays idle cost off-peak.</li>
 *   <li><b>REACTIVE</b> — scale up when queue exceeds a threshold for
 *       {@code triggerWindowSec}; scale down when consistently under-utilized.
 *       Suffers cold-start during sudden bursts because new VMs need
 *       {@code startupDelaySec} to come online.</li>
 *   <li><b>PREDICTIVE</b> — EWMA of queue depth + look-ahead of one
 *       startup-delay window. Pre-warms before bursts hit, at the cost of
 *       slightly higher idle hours when the predictor over-shoots.</li>
 * </ul>
 *
 * <p>This is a pure decision component — it neither owns VMs nor schedules
 * events. {@link AutoscalingBroker} consumes its decisions.</p>
 */
public final class WarmPoolAutoscaler {

    public enum Policy { STATIC, REACTIVE, PREDICTIVE }

    private final Policy policy;
    private final int minPool;
    private final int maxPool;
    private final double targetCloudletsPerVm;
    private double evaluationIntervalSec = 2.0;
    private double scaleDownCooldownSec  = 30.0;
    private double ewmaAlpha             = 0.30;

    /** Look-ahead horizon (sec) for PREDICTIVE; should match cold-start delay. */
    private double horizonSec = 15.0;

    private double ewma = 0.0;
    private double lastEvalTime = -1.0;
    /** Initialised to 0 so the cooldown blocks scale-down during sim start (when
     *  the queue is trivially empty before the first cloudlet arrives). */
    private double lastScaleDownTime = 0.0;
    private int    consecutiveLowEvals = 0;
    private int    target;

    public WarmPoolAutoscaler(Policy policy, int initialPool, int minPool, int maxPool,
                              double targetCloudletsPerVm) {
        this.policy = policy;
        this.minPool = Math.max(0, minPool);
        this.maxPool = Math.max(this.minPool, maxPool);
        this.target  = Math.min(Math.max(initialPool, this.minPool), this.maxPool);
        this.targetCloudletsPerVm = Math.max(1.0, targetCloudletsPerVm);
    }

    public WarmPoolAutoscaler setHorizonSec(double v)        { this.horizonSec = v; return this; }
    public WarmPoolAutoscaler setEwmaAlpha(double v)         { this.ewmaAlpha = v; return this; }
    public WarmPoolAutoscaler setEvaluationIntervalSec(double v) { this.evaluationIntervalSec = v; return this; }
    public WarmPoolAutoscaler setScaleDownCooldownSec(double v)  { this.scaleDownCooldownSec = v; return this; }

    public Policy policy()             { return policy; }
    public int minPool()               { return minPool; }
    public int maxPool()               { return maxPool; }
    public int currentTarget()         { return target; }
    public double ewma()               { return ewma; }

    /**
     * Compute the desired pool size at time {@code now}. The autoscaler
     * rate-limits internally; {@link AutoscalingBroker} can call this on
     * every cloudlet without thrashing.
     *
     * @return the new desired pool size; equals {@link #currentTarget} if
     *         no change was made this call.
     */
    public int decide(double now, int currentQueueDepth) {
        if (lastEvalTime > 0 && now - lastEvalTime < evaluationIntervalSec) {
            return target;
        }
        lastEvalTime = now;
        if (Boolean.parseBoolean(System.getenv().getOrDefault("LLM_DEBUG_AUTOSCALE", "false"))) {
            System.err.printf("[decide] t=%.2f queue=%d ewma=%.2f target=%d%n", now, currentQueueDepth, ewma, target);
        }

        // EWMA update on every evaluation.
        ewma = ewmaAlpha * currentQueueDepth + (1 - ewmaAlpha) * ewma;

        // Sustained-low gate: require N consecutive low-queue evaluations before
        // any scale-down to filter out brief inter-burst lulls.
        final int lowGateEvals = 3;

        int desired = target;
        switch (policy) {
            case STATIC -> { /* no change */ }
            case REACTIVE -> {
                int wantUp = (int) Math.ceil(currentQueueDepth / targetCloudletsPerVm);
                if (wantUp > target) {
                    desired = Math.min(maxPool, wantUp);
                    consecutiveLowEvals = 0;
                } else if (wantUp <= target / 2) {
                    consecutiveLowEvals++;
                    if (consecutiveLowEvals >= lowGateEvals
                            && now - lastScaleDownTime > scaleDownCooldownSec) {
                        desired = Math.max(minPool, target - 1);
                        lastScaleDownTime = now;
                        consecutiveLowEvals = 0;
                    }
                } else {
                    consecutiveLowEvals = 0;
                }
            }
            case PREDICTIVE -> {
                double trend = currentQueueDepth - ewma;
                double projected = ewma + Math.max(0, trend) * (horizonSec / evaluationIntervalSec);
                int wantUp = (int) Math.ceil(projected / targetCloudletsPerVm);
                if (wantUp > target) {
                    desired = Math.min(maxPool, wantUp);
                    consecutiveLowEvals = 0;
                } else if (wantUp <= target / 2) {
                    consecutiveLowEvals++;
                    if (consecutiveLowEvals >= lowGateEvals
                            && now - lastScaleDownTime > scaleDownCooldownSec) {
                        desired = Math.max(minPool, target - 1);
                        lastScaleDownTime = now;
                        consecutiveLowEvals = 0;
                    }
                } else {
                    consecutiveLowEvals = 0;
                }
            }
        }
        target = desired;
        return target;
    }
}
