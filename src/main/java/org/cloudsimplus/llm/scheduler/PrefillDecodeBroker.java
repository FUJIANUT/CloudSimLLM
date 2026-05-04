package org.cloudsimplus.llm.scheduler;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.listeners.CloudletVmEventInfo;
import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.vms.Vm;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Splitwise-style broker. Routes new {@link LlmCloudlet}s to a pool of
 * <b>prefill-only</b> VMs. When the prefill finishes (signalled by the
 * {@link PrefillDecodeDisaggScheduler#PREFILL_ONLY} role marking the cloudlet
 * done), the broker:
 * <ol>
 *   <li>Computes KV transfer time over the inter-host fabric (Eq. 11 with k=1
 *       plus the actual KV bytes), then</li>
 *   <li>Builds a {@link LlmCloudlet#newDecodeShadow decode shadow} and submits
 *       it to a <b>decode-only</b> VM with that delay.</li>
 * </ol>
 *
 * <p>End-to-end statistics use the <em>shadow</em>'s finish time but the
 * <em>original</em>'s arrival/first-token times, so TTFT reflects the prefill
 * worker's queue + execution and TPOT reflects the decode worker's behaviour.</p>
 *
 * <p>Use this broker only with disaggregated VMs: see {@link #setPrefillVms}
 * and {@link #setDecodeVms}.</p>
 */
public class PrefillDecodeBroker extends DatacenterBrokerSimple {

    private List<Vm> prefillVms = new ArrayList<>();
    private List<Vm> decodeVms = new ArrayList<>();
    private double kvTransferBwGbs = 200.0;
    private final AtomicLong shadowIdGen = new AtomicLong(1_000_000_000L);
    private final ConcurrentHashMap<Long, LlmCloudlet> originalsByShadowId = new ConcurrentHashMap<>();
    private int rrPrefill = 0;
    private int rrDecode = 0;

    public PrefillDecodeBroker(CloudSimPlus simulation) {
        super(simulation);
        // Splitwise dynamically materializes decode-shadow cloudlets *after* the
        // initial cloudlet list is exhausted; without a keep-alive, the broker
        // destroys decode VMs the moment their initial queue is empty. 1000s of
        // sim time covers the longest LLM trace we plan to run while staying
        // far below values that poison the scheduler's internal clock.
        setVmDestructionDelay(1000.0);
    }

    public PrefillDecodeBroker setPrefillVms(List<Vm> vms) { this.prefillVms = new ArrayList<>(vms); return this; }
    public PrefillDecodeBroker setDecodeVms(List<Vm> vms)  { this.decodeVms = new ArrayList<>(vms); return this; }
    public PrefillDecodeBroker setKvTransferBwGbs(double v) { this.kvTransferBwGbs = v; return this; }

    /** Look up the original cloudlet for a finished decode shadow. */
    public LlmCloudlet originalOf(LlmCloudlet shadow) {
        return originalsByShadowId.get(shadow.getId());
    }

    /**
     * Route prefills to prefill VMs and decode shadows to decode VMs. Round-robin
     * is the baseline; subclass and override for cost/SLO-aware variants.
     */
    @Override
    protected Vm defaultVmMapper(final Cloudlet cloudlet) {
        if (!(cloudlet instanceof LlmCloudlet llm)) {
            return super.defaultVmMapper(cloudlet);
        }
        if (llm.isDecodeShadow()) {
            if (decodeVms.isEmpty()) return Vm.NULL;
            Vm v = decodeVms.get(rrDecode % decodeVms.size());
            rrDecode++;
            return v;
        }
        if (prefillVms.isEmpty()) return Vm.NULL;
        Vm v = prefillVms.get(rrPrefill % prefillVms.size());
        rrPrefill++;
        // Attach the hand-off listener exactly once per original.
        llm.addOnFinishListener(this::onPrefillFinished);
        return v;
    }

    /** Fired by the prefill VM when {@link PrefillDecodeDisaggScheduler#PREFILL_ONLY} marks it done. */
    private void onPrefillFinished(CloudletVmEventInfo info) {
        if (!(info.getCloudlet() instanceof LlmCloudlet original)) return;
        if (original.isDecodeShadow()) return;     // safety: never re-shadow a shadow

        final long shadowId = shadowIdGen.getAndIncrement();
        final LlmCloudlet shadow = original.newDecodeShadow(shadowId);
        originalsByShadowId.put(shadowId, original);

        final double kvBytes = original.currentKvBytes(16);          // β = 16 default
        final double transferDelaySec = kvBytes / (kvTransferBwGbs * 1e9);

        // Submit shadow with the KV transfer delay; vmMapper will pick a decode VM.
        submitCloudletList(List.<Cloudlet>of(shadow), transferDelaySec);
    }

    public List<Vm> prefillVms()       { return prefillVms; }
    public List<Vm> decodeVms()        { return decodeVms; }
    public double kvTransferBwGbs()    { return kvTransferBwGbs; }
}
