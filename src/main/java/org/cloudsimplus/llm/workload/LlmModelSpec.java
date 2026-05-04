package org.cloudsimplus.llm.workload;

/**
 * Static description of an LLM (architecture + dtype). Used by latency, memory,
 * and energy models defined in §4. All fields are independent of any deployment;
 * runtime parameters (batch, sequence length) live elsewhere.
 *
 * <p>Defaults to FP16 (b = 2 bytes/elem). For FP8/INT4 quantization, set
 * {@code bytesPerElement} accordingly and ensure calibrated FLOPS in
 * {@link org.cloudsimplus.llm.core.GpuPe} match.</p>
 */
public final class LlmModelSpec {
    private final String name;
    private final long parameters;       // P_m
    private final int numLayers;         // L
    private final int numQueryHeads;     // H
    private final int numKvHeads;        // H_kv  (== H if no GQA/MQA)
    private final int headDim;           // d_h
    private final int bytesPerElement;   // b

    public LlmModelSpec(final String name,
                        final long parameters,
                        final int numLayers,
                        final int numQueryHeads,
                        final int numKvHeads,
                        final int headDim,
                        final int bytesPerElement) {
        this.name = name;
        this.parameters = parameters;
        this.numLayers = numLayers;
        this.numQueryHeads = numQueryHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.bytesPerElement = bytesPerElement;
    }

    /** Hidden dim d = H · d_h. */
    public int hiddenDim() { return numQueryHeads * headDim; }

    /** Per-token KV bytes — Eq. (5): m_kv = 2 · L · H_kv · d_h · b. */
    public long kvBytesPerToken() {
        return 2L * numLayers * numKvHeads * headDim * bytesPerElement;
    }

    /** Weight bytes M_w = 2 · P_m · b (factor 2 covers fwd activations padding). */
    public long weightBytes() {
        return 2L * parameters * bytesPerElement;
    }

    public String name()           { return name; }
    public long parameters()       { return parameters; }
    public int numLayers()         { return numLayers; }
    public int numQueryHeads()     { return numQueryHeads; }
    public int numKvHeads()        { return numKvHeads; }
    public int headDim()           { return headDim; }
    public int bytesPerElement()   { return bytesPerElement; }

    public static LlmModelSpec llama3_8B_fp16() {
        return new LlmModelSpec("llama-3-8b", 8_030_000_000L, 32, 32, 8, 128, 2);
    }
    public static LlmModelSpec llama3_70B_fp16() {
        return new LlmModelSpec("llama-3-70b", 70_600_000_000L, 80, 64, 8, 128, 2);
    }
    public static LlmModelSpec mistral7B_fp16() {
        return new LlmModelSpec("mistral-7b", 7_240_000_000L, 32, 32, 8, 128, 2);
    }
}
