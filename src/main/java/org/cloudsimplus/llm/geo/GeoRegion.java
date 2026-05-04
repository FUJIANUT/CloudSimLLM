package org.cloudsimplus.llm.geo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Geographic region with carbon-intensity time series and inter-region latency
 * (Eq. 10 / Eq. 11 in §4). Carbon profiles approximate real grid mixes for
 * 24h cycles using simple analytic curves; replace with ElectricityMap
 * historical data for a paper-grade evaluation.
 *
 * <p>Latency to peer regions is round-trip, in seconds. Distances reflect
 * publicly observed AWS / Azure / GCP inter-region latencies (early 2026).</p>
 */
public final class GeoRegion {

    public enum Profile {
        /** California — gas peakers + solar dip; 100–400 gCO2/kWh. */
        US_WEST_GAS_SOLAR,
        /** Texas — wind variability; 50–600 gCO2/kWh stochastic. */
        US_MID_WIND,
        /** Iceland — geothermal+hydro; ≈ 25 gCO2/kWh constant. */
        EU_NORTH_RENEWABLE,
        /** Germany — solar+coal mix; 200–500 gCO2/kWh. */
        EU_CENTRAL_MIX,
        /** India — coal-heavy baseline; 700–800 gCO2/kWh. */
        AP_SOUTH_COAL,
    }

    private final String name;
    private final Profile profile;
    private final double pue;
    private final Map<String, Double> rttSecToPeer = new HashMap<>();
    /** Phase offset (hours) applied to the 24h carbon cycle. */
    private double phaseOffsetHours = 0.0;
    /**
     * If non-null, overrides the analytic profile with a real-data 24-element
     * lookup of carbon intensity (gCO2eq/kWh) per hour. Loaded via
     * {@link #setHourlyProfile(double[])} or
     * {@link #loadHourlyProfilesFromCsv}.
     */
    private double[] hourlyOverride = null;

    public GeoRegion(String name, Profile profile, double pue) {
        this.name = name;
        this.profile = profile;
        this.pue = pue;
    }

    public GeoRegion withRttTo(String peerName, double rttSec) {
        rttSecToPeer.put(peerName, rttSec);
        return this;
    }

    public String name()      { return name; }
    public Profile profile()  { return profile; }
    public double pue()       { return pue; }
    public double phaseOffsetHours() { return phaseOffsetHours; }
    public GeoRegion setPhaseOffsetHours(double h) { this.phaseOffsetHours = h; return this; }

    /** Round-trip latency in seconds to a peer region (0 if same region). */
    public double rttTo(GeoRegion peer) {
        if (peer == this) return 0.0;
        return rttSecToPeer.getOrDefault(peer.name, 0.300);   // unknown → 300 ms
    }

    /**
     * Carbon intensity (gCO2eq / kWh) at simulation time {@code now}, modulo
     * a 24-hour cycle. {@code now} is in seconds; one full day = 86400 s.
     */
    public double carbonIntensityGramsPerKwh(double now) {
        final double h = ((now / 3600.0) + phaseOffsetHours) % 24.0;
        if (hourlyOverride != null) {
            int hi = (int) Math.floor(h) % 24;
            int hi2 = (hi + 1) % 24;
            double frac = h - Math.floor(h);
            return (1 - frac) * hourlyOverride[hi] + frac * hourlyOverride[hi2];
        }
        final double phase = 2 * Math.PI * h / 24.0;
        return switch (profile) {
            // Solar dip at noon (h=12), evening peak (h=18) from gas peakers.
            case US_WEST_GAS_SOLAR ->
                250 + 150 * Math.sin(phase - Math.PI/2)  // -150..+150
                    + 100 * Math.max(0, Math.sin(phase - Math.PI*5/4));
            // Stochastic-ish wind: large amplitude, multiple harmonics.
            case US_MID_WIND ->
                300 + 250 * Math.sin(phase * 2 + 1.0)
                    +  50 * Math.sin(phase * 5 + 2.5);
            // Almost constant.
            case EU_NORTH_RENEWABLE -> 25.0;
            // Solar dip noon, coal at night.
            case EU_CENTRAL_MIX ->
                350 - 150 * Math.max(0, Math.sin(phase - Math.PI/2));
            // High flat coal.
            case AP_SOUTH_COAL ->
                720 + 30 * Math.cos(phase);
        };
    }

    /* ----- Convenience factories with realistic inter-region RTTs ----- */

    /** Build a 3-region world: US-West, EU-North, AP-South. */
    public static Map<String, GeoRegion> threeRegionWorld() {
        var us = new GeoRegion("us-west", Profile.US_WEST_GAS_SOLAR,  1.20);
        var eu = new GeoRegion("eu-north", Profile.EU_NORTH_RENEWABLE, 1.10);
        var ap = new GeoRegion("ap-south", Profile.AP_SOUTH_COAL,      1.40);
        us.withRttTo("eu-north", 0.100).withRttTo("ap-south", 0.250);
        eu.withRttTo("us-west", 0.100).withRttTo("ap-south", 0.200);
        ap.withRttTo("us-west", 0.250).withRttTo("eu-north", 0.200);
        var m = new HashMap<String, GeoRegion>();
        m.put(us.name, us); m.put(eu.name, eu); m.put(ap.name, ap);
        return m;
    }

    /**
     * Override the analytic carbon profile with a 24-element hourly array
     * (gCO2eq/kWh per hour, index 0..23). Linear interpolation between
     * hours. Pass {@code null} to revert to the analytic profile.
     */
    public GeoRegion setHourlyProfile(double[] hourly) {
        if (hourly != null && hourly.length != 24)
            throw new IllegalArgumentException("hourly profile must have exactly 24 entries");
        this.hourlyOverride = hourly == null ? null : hourly.clone();
        return this;
    }

    /**
     * Load hourly carbon profiles from a CSV (e.g.\ ElectricityMap export)
     * with header "hour,&lt;col&gt;,..." and 24 data rows. Each named column
     * is matched to the corresponding region in {@code regions} (by
     * substring of the region's {@link #name()}). Unmatched columns are
     * ignored; unmatched regions keep their analytic profile.
     */
    public static void loadHourlyProfilesFromCsv(Map<String, GeoRegion> regions, Path csv) throws IOException {
        var lines = Files.readAllLines(csv).stream()
            .map(String::trim)
            .filter(s -> !s.isEmpty() && !s.startsWith("#"))
            .toList();
        if (lines.size() < 25)
            throw new IOException("CSV must have header + 24 data rows; got " + lines.size());
        String[] header = lines.get(0).split(",");
        Map<String, double[]> cols = new HashMap<>();
        for (int c = 1; c < header.length; c++) cols.put(header[c].trim(), new double[24]);
        for (int h = 0; h < 24; h++) {
            String[] tok = lines.get(h + 1).split(",");
            for (int c = 1; c < header.length; c++) {
                cols.get(header[c].trim())[h] = Double.parseDouble(tok[c].trim());
            }
        }
        for (var entry : regions.entrySet()) {
            String regName = entry.getKey().replace("-", "_");
            for (var col : cols.entrySet()) {
                if (regName.contains(col.getKey()) || col.getKey().contains(regName)) {
                    entry.getValue().setHourlyProfile(col.getValue());
                    break;
                }
            }
        }
    }

    /** Build a 5-region world adding US-Mid (wind) and EU-Central. */
    public static Map<String, GeoRegion> fiveRegionWorld() {
        var m = threeRegionWorld();
        var usMid = new GeoRegion("us-mid",     Profile.US_MID_WIND,    1.15);
        var euCen = new GeoRegion("eu-central", Profile.EU_CENTRAL_MIX, 1.25);
        usMid.withRttTo("us-west", 0.040).withRttTo("eu-north", 0.090)
             .withRttTo("eu-central", 0.110).withRttTo("ap-south", 0.220);
        euCen.withRttTo("us-west", 0.140).withRttTo("us-mid", 0.110)
             .withRttTo("eu-north", 0.025).withRttTo("ap-south", 0.180);
        // Symmetric updates for existing regions.
        m.get("us-west").withRttTo("us-mid", 0.040).withRttTo("eu-central", 0.140);
        m.get("eu-north").withRttTo("us-mid", 0.090).withRttTo("eu-central", 0.025);
        m.get("ap-south").withRttTo("us-mid", 0.220).withRttTo("eu-central", 0.180);
        m.put(usMid.name, usMid);
        m.put(euCen.name, euCen);
        return m;
    }
}
