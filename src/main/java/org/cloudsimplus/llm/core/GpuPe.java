package org.cloudsimplus.llm.core;

import org.cloudsimplus.resources.PeSimple;

/**
 * GPU Processing Element. Extends {@link PeSimple} to carry GPU-specific peak
 * and effective throughputs that drive Eqs. (1)–(2) and (8). The "MIPS" field
 * inherited from {@link PeSimple} is repurposed to represent peak FP16 TFLOPS
 * scaled into MIPS-equivalents only for back-compat with existing schedulers
 * that still query MIPS; LLM scheduler uses the GPU-specific accessors below.
 *
 * <p>Calibration parameters (effective FLOPS, memory bandwidth, kernel
 * overheads) are stored here so each GPU instance can carry per-SKU values.</p>
 */
public class GpuPe extends PeSimple {
    /** Vendor / SKU descriptor — e.g., "A100-80GB", "H100-SXM5-80GB". */
    private final String sku;

    /** Peak FP16 throughput (TFLOPS). */
    private final double peakFp16Tflops;
    /** Peak HBM bandwidth (GB/s). */
    private final double peakHbmBwGbs;
    /** HBM capacity (bytes). C_HBM in Eq. (7). */
    private final long hbmBytes;
    /** TDP in Watts — P_tdp in Eq. (8). */
    private final double tdpWatts;
    /** Idle power in Watts — P_idle in Eq. (8). */
    private final double idleWatts;

    /* ----- Calibrated effective values (filled in §6.1) ----- */
    /** F_eff^pre — TFLOPS effectively achieved during prefill. */
    private double effFp16TflopsPrefill;
    /** F_eff^dec — TFLOPS effectively achieved during decode. */
    private double effFp16TflopsDecode;
    /** B_mem^eff — effective HBM bandwidth (GB/s). */
    private double effHbmBwGbs;
    /** Kernel launch overheads (seconds). */
    private double alphaPrefillSec;
    private double alphaDecodeSec;

    public GpuPe(String sku,
                 double peakFp16Tflops,
                 double peakHbmBwGbs,
                 long hbmBytes,
                 double tdpWatts,
                 double idleWatts) {
        super(peakFp16Tflops * 1_000_000.0); // TFLOPS → MIPS-equivalent for legacy paths
        this.sku = sku;
        this.peakFp16Tflops = peakFp16Tflops;
        this.peakHbmBwGbs = peakHbmBwGbs;
        this.hbmBytes = hbmBytes;
        this.tdpWatts = tdpWatts;
        this.idleWatts = idleWatts;
    }

    /* ----- Calibration setters (used by §6.1 calibration harness) ----- */
    public GpuPe setEffFp16TflopsPrefill(double v) { this.effFp16TflopsPrefill = v; return this; }
    public GpuPe setEffFp16TflopsDecode(double v)  { this.effFp16TflopsDecode = v; return this; }
    public GpuPe setEffHbmBwGbs(double v)          { this.effHbmBwGbs = v; return this; }
    public GpuPe setAlphaPrefillSec(double v)      { this.alphaPrefillSec = v; return this; }
    public GpuPe setAlphaDecodeSec(double v)       { this.alphaDecodeSec = v; return this; }

    public String sku()                  { return sku; }
    public double peakFp16Tflops()       { return peakFp16Tflops; }
    public double peakHbmBwGbs()         { return peakHbmBwGbs; }
    public long hbmBytes()               { return hbmBytes; }
    public double tdpWatts()             { return tdpWatts; }
    public double idleWatts()            { return idleWatts; }
    public double effFp16TflopsPrefill() { return effFp16TflopsPrefill; }
    public double effFp16TflopsDecode()  { return effFp16TflopsDecode; }
    public double effHbmBwGbs()          { return effHbmBwGbs; }
    public double alphaPrefillSec()      { return alphaPrefillSec; }
    public double alphaDecodeSec()       { return alphaDecodeSec; }

    /** Convenience factory for A100-80GB SXM4. */
    public static GpuPe a100_80gb() {
        return new GpuPe("A100-SXM4-80GB", 312.0, 2039.0, 80L * 1024 * 1024 * 1024, 400.0, 50.0);
    }
    /** Convenience factory for H100-80GB SXM5. */
    public static GpuPe h100_80gb() {
        return new GpuPe("H100-SXM5-80GB", 989.0, 3350.0, 80L * 1024 * 1024 * 1024, 700.0, 75.0);
    }

    /** Convenience factory for L40S 48GB (lower tier inference card). */
    public static GpuPe l40s_48gb() {
        return new GpuPe("L40S-48GB", 362.0, 864.0, 48L * 1024 * 1024 * 1024, 350.0, 40.0);
    }
}
