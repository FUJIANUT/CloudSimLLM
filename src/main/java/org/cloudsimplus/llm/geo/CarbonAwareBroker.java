package org.cloudsimplus.llm.geo;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.vms.Vm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Routes incoming {@link LlmCloudlet}s across geographically-distributed
 * regions per a chosen {@link Policy}. Encapsulates §6.5's three policies:
 *
 * <ul>
 *   <li><b>LATENCY_GREEDY</b> — always route to the request's home region.</li>
 *   <li><b>CARBON_AWARE</b> — pick the region with the lowest current
 *     carbon intensity, ignoring latency. Lower bound on emissions.</li>
 *   <li><b>BLENDED</b> — minimize {@code rtt + λ · CI(region, t)}, where
 *     {@code λ} is the carbon-vs-latency exchange rate (s per gCO2eq/kWh).</li>
 * </ul>
 *
 * <p>The home region of a request is round-robin assigned at submission to
 * approximate a globally-distributed user base.</p>
 */
public class CarbonAwareBroker extends DatacenterBrokerSimple {

    public enum Policy { LATENCY_GREEDY, CARBON_AWARE, BLENDED }

    private final Map<String, GeoRegion> regions = new HashMap<>();
    private final Map<String, List<Vm>>  vmsByRegion = new HashMap<>();
    private final Map<Long, String>      homeOf = new HashMap<>();
    private final Map<Long, String>      chosenOf = new HashMap<>();
    private final List<String>           regionOrder = new ArrayList<>();
    private Policy policy = Policy.LATENCY_GREEDY;
    /** Carbon→latency exchange rate (seconds per (gCO2/kWh)). */
    private double lambdaSecPerGramKwh = 0.005;
    private final Random rng;
    private int rrAssign = 0;

    public CarbonAwareBroker(CloudSimPlus simulation, long seed) {
        super(simulation);
        this.rng = new Random(seed);
        // Carbon-aware routing must keep VMs alive long enough that late requests
        // (Poisson tail at high hour-of-day shifts) finish before destruction.
        setVmDestructionDelay(100_000.0);
    }

    public CarbonAwareBroker register(GeoRegion region, List<Vm> vms) {
        regions.put(region.name(), region);
        vmsByRegion.put(region.name(), new ArrayList<>(vms));
        if (!regionOrder.contains(region.name())) regionOrder.add(region.name());
        return this;
    }

    public CarbonAwareBroker setPolicy(Policy p)  { this.policy = p; return this; }
    public CarbonAwareBroker setLambda(double v)  { this.lambdaSecPerGramKwh = v; return this; }

    public Map<String, GeoRegion> regions() { return regions; }
    public String homeOf(LlmCloudlet r)     { return homeOf.get(r.getId()); }

    /** Assign a home region to a freshly-arrived request (round-robin across regions). */
    public CarbonAwareBroker assignHome(LlmCloudlet r) {
        String home = regionOrder.get(rrAssign % regionOrder.size());
        rrAssign++;
        homeOf.put(r.getId(), home);
        return this;
    }

    @Override
    protected Vm defaultVmMapper(final Cloudlet cloudlet) {
        if (!(cloudlet instanceof LlmCloudlet llm)) return super.defaultVmMapper(cloudlet);
        if (regionOrder.isEmpty())                  return Vm.NULL;

        final String home = homeOf.computeIfAbsent(llm.getId(), id -> {
            String h = regionOrder.get(rrAssign % regionOrder.size()); rrAssign++;
            return h;
        });
        final GeoRegion homeRegion = regions.get(home);
        final double now = getSimulation().clock();

        final String chosen = switch (policy) {
            case LATENCY_GREEDY -> home;
            case CARBON_AWARE -> regions.values().stream()
                .min(Comparator.comparingDouble(g -> g.carbonIntensityGramsPerKwh(now)))
                .map(GeoRegion::name).orElse(home);
            case BLENDED -> regions.values().stream()
                .min(Comparator.comparingDouble(g ->
                    g.rttTo(homeRegion) + lambdaSecPerGramKwh * g.carbonIntensityGramsPerKwh(now)))
                .map(GeoRegion::name).orElse(home);
        };

        // Charge user-perceived TTFT for cross-region network RTT by recording
        // it on the cloudlet; LlmCloudlet.ttftSec() will add it to the
        // server-side (firstToken − arrival) duration.
        if (!chosen.equals(home)) {
            llm.setNetworkRttSec(homeRegion.rttTo(regions.get(chosen)));
        }
        chosenOf.put(llm.getId(), chosen);

        // Pick a created VM in the chosen region (round-robin).
        var vms = vmsByRegion.get(chosen);
        if (vms == null || vms.isEmpty()) return Vm.NULL;
        for (int i = 0; i < vms.size(); i++) {
            Vm v = vms.get((int) ((cloudlet.getId() + i) % vms.size()));
            if (v.isCreated()) return v;
        }
        return Vm.NULL;
    }

    /** Estimate per-request carbon based on the *chosen* (compute-host) region. */
    public double estimatedCarbonGrams(LlmCloudlet r, double avgPowerW) {
        // Carbon is incurred where compute happens, not where the user is.
        String chosen = chosenOf.getOrDefault(r.getId(), homeOf(r));
        if (chosen == null) return 0;
        GeoRegion g = regions.get(chosen);
        double e2eSec = Math.max(0, r.finishSimTime() - r.arrivalSimTime());
        double energyKwh = avgPowerW * e2eSec / 3.6e6;
        return energyKwh * g.pue() * g.carbonIntensityGramsPerKwh(r.arrivalSimTime());
    }

    public String chosenOf(LlmCloudlet r) { return chosenOf.get(r.getId()); }
}
