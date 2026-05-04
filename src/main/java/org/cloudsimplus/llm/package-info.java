/**
 * <h2>CloudSimLLM — datacenter-scale LLM inference serving extension</h2>
 *
 * Non-invasive extension to CloudSim Plus enabling simulation of large language
 * model inference workloads with first-class modeling of:
 * <ul>
 *   <li>Two-phase execution (prefill / decode) — Eqs. (1), (2)</li>
 *   <li>PagedAttention KV cache and HBM capacity constraint — Eqs. (5)–(7)</li>
 *   <li>Continuous batching scheduling — Algorithm 1</li>
 *   <li>Phase-aware GPU power and per-request energy/carbon — Eqs. (8)–(10)</li>
 *   <li>Tensor-parallel AllReduce overhead — Eq. (11)</li>
 *   <li>SLO-attainment-weighted Goodput — Eq. (12)</li>
 * </ul>
 *
 * <h3>Design principle: Non-invasive</h3>
 * Every class in this package extends {@code non-sealed abstract} types from
 * upstream CloudSim Plus. No upstream source is modified. Listeners are used
 * for hook-points; the package can be deleted with no impact on the framework.
 *
 * <h3>Package layout</h3>
 * <ul>
 *   <li>{@link org.cloudsimplus.llm.core} — {@code GpuPe}, {@code GpuHost}, {@code LlmCloudlet}</li>
 *   <li>{@link org.cloudsimplus.llm.workload} — {@code LlmModelSpec}, {@code KvCacheBlock}, {@code KvCacheProvisioner}</li>
 *   <li>{@link org.cloudsimplus.llm.scheduler} — continuous-batch and Splitwise schedulers, GPU-aware allocation policy</li>
 *   <li>{@link org.cloudsimplus.llm.power} — {@code LlmPowerModel}</li>
 *   <li>{@link org.cloudsimplus.llm.trace} — Azure LLM and BurstGPT trace readers</li>
 *   <li>{@link org.cloudsimplus.llm.metrics} — TTFT/TPOT/Goodput/energy/carbon aggregator</li>
 * </ul>
 *
 * <h3>Calibration</h3>
 * Effective throughputs and overheads on each {@link org.cloudsimplus.llm.core.GpuPe}
 * are filled from §6.1 calibration (vLLM/TensorRT-LLM measurements on real A100/H100
 * hardware). Defaults must not be used without calibration; the simulator logs a
 * warning if any required calibration parameter is left at zero.
 *
 * @since 1.0
 */
package org.cloudsimplus.llm;
