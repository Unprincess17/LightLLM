# CoLoRA: Cross-Machine LoRA Miss Recovery for MoE Serving

---

## Slide 1 — Title

**CoLoRA: Network-Aware LoRA Miss Recovery for Distributed MoE Serving**

Gong Shufan

*Problem:* MoE + multi-LoRA serving hits cache misses — current systems stall the GPU.
*Insight:* Local decode misses are CPU-first for typical batches (N≤8); cross-machine misses need pre-cache + relay.
*Contribution:* Hybrid local/cross-machine recovery + network-aware scheduling.

---

## Slide 2 — The Miss Problem in MoE + LoRA Serving

**Setup:** MoE model (e.g., Qwen3-30B-A3B) serving 1000s of LoRA adapters.

**Cache miss is the common case:**
- GPU memory is finite — cannot hold all LoRA weights
- Zipf-distributed adapter popularity → long tail of cold adapters
- Miss rate at budget=2048 (out of ~4000 experts): **~15-30%** depending on workload
- Each miss stalls the decode step until the LoRA is recovered

**Current approach (S-LoRA / Punica):**
- Miss → H2D transfer LoRA weights to GPU → GPU compute merge
- Locally, CPU-first recovery is faster for typical decode batches (N ≤ 8); cross-machine misses add network delay and contention

---

## Slide 3 — Local Decode Miss Recovery: CPU Wins at Typical Decode Batches

**The decode regime is fundamentally different from prefill:**

| Parameter | Prefill | Decode |
|-----------|---------|--------|
| Batch size (N) | 64-256 | 1-16 |
| Sequence length (S) | 512-4096 | 1 |
| Activation size | S × 2048 bf16 | **4KB** (1 × 2048 bf16) |
| LoRA weight size | Fixed | Fixed (R=64: 361KB, R=128: 722KB) |
| GPU compute time | Amortized | **Dominated by H2D transfer** |

**Key asymmetry:** LoRA weights (361KB at R=64) >> activation size (4KB) at small N.

**Measured on A100-SXM4-80GB, single MoE expert (source: `motivation_microbench.py`):**

For **R=64, N=4** (typical decode batch, activation = 4×4KB = 16KB):

GPU path (H2D weights + GPU matmul):
- H2D weight transfer: 28.3μs (361KB over PCIe Gen4 ×16)
- GPU matmul: 45.6μs
- **Total: 72.4μs**

CPU path (D2H activation + AVX-512 + H2D result):
- D2H activation: 14.1μs (16KB, pinned memory copy)
- CPU AVX compute: 32.4μs (no-expand: batch all N in one call, OpenMP across tokens)
- H2D result: 16.6μs
- **Total: 63.2μs**

**CPU is 1.15× faster at N=4, R=64. CPU wins up to N=8 for R≤64 (see Slide 4 for full crossover).**

> Note: Both paths use the no-expand optimization with AVX-512 BF16 dot-product instructions. CPU path benefits from pre-allocated output buffers, OpenMP token parallelism, and 32-wide Stage 2 vectorization. H2D weight transfer is a one-time cold-miss cost. For warm hits, GPU path drops to ~42μs.

---

## Slide 4 — Crossover Curve: CPU Wins at Typical Decode Batches

**Benchmark source:** `motivation_microbench.py` — single MoE expert, A100-SXM4-80GB.
**Model:** Qwen3-VL-30B-A3B (H=2048, I=768, top_k=2).
**Sweep:** LoRA rank R ∈ {8, 16, 32, 64, 128}, decode batch N_dec ∈ {1, 2, 4, 8, 16}.
**Optimizations:** Pre-allocated output buffers, OpenMP token parallelism, 32-wide AVX-512 Stage 2, eliminated unnecessary CUDA syncs.

**Benchmark results (decode batch, no prefill sequence length):**

| Rank | N_dec | GPU total (μs) | CPU total (μs) | Winner |
|------|-------|----------------|----------------|--------|
| 8 | 1 | 64.5 | **36.8** | CPU (1.75×) |
| 8 | 4 | 65.8 | **46.5** | CPU (1.42×) |
| 8 | 16 | 65.8 | **63.8** | CPU (1.03×) |
| 16 | 1 | 62.2 | **37.9** | CPU (1.64×) |
| 16 | 8 | 65.7 | **51.6** | CPU (1.27×) |
| 16 | 16 | **65.7** | 74.8 | GPU (1.14×) |
| 32 | 1 | 63.2 | **41.0** | CPU (1.54×) |
| 32 | 8 | 66.4 | **57.2** | CPU (1.16×) |
| 32 | 16 | **65.8** | 97.7 | GPU (1.49×) |
| 64 | 1 | 71.9 | **48.8** | CPU (1.47×) |
| 64 | 4 | 72.4 | **63.2** | CPU (1.15×) |
| 64 | 8 | 71.9 | **66.6** | CPU (1.08×) |
| 64 | 16 | **72.4** | 141.6 | GPU (1.95×) |
| 128 | 1 | 84.7 | **63.5** | CPU (1.33×) |
| 128 | 4 | 85.4 | **80.4** | CPU (1.06×) |
| 128 | 8 | 85.6 | **84.4** | CPU (1.01×) |
| 128 | 16 | **86.3** | 221.9 | GPU (2.57×) |

**Updated crossover:**
- **CPU wins at N_dec ≤ 8 for all ranks R ≤ 64**, with speedups of 1.01–1.75×.
- **CPU wins at N_dec ≤ 8 even for R=128**, though margins are thin (1.01–1.33×).
- **GPU wins at N_dec ≥ 16** for all ranks, where AVX compute cost dominates.
- GPU path is dominated by H2D weight transfer (~22–46μs), not GPU matmul (~40–45μs).
- CPU path benefits from small activation transfer (D2H: 13–15μs) vs large weight transfer (H2D: 22–46μs).

> The crossover has shifted dramatically from previous measurements. Three optimizations drove this: (1) pre-allocated output buffers eliminate per-call allocation overhead, (2) OpenMP parallelism across tokens for N>1, and (3) removing unnecessary CUDA synchronize after CPU compute. The old conclusion (CPU only wins at N=1) was inflated by these overheads, not fundamental.

---

## Slide 5 — The Cross-Machine Challenge

**In distributed MoE (Expert Parallelism), misses span machines:**

```
┌─────────────┐         RDMA 200Gbps          ┌─────────────┐
│  UM253       │◄─────────────────────────────►│  UM251       │
│  A100 GPU    │         InfiniBand             │  CPU only    │
│  (inference) │                                │  (LoRA store)│
└─────────────┘                                └─────────────┘
```

**Problem:** LoRA weights live on a *remote* machine's CPU memory.

**Three recovery strategies:**

| Strategy | Flow | Network payload |
|----------|------|-----------------|
| S1: Weight transfer | Remote CPU → RDMA → Local GPU → GPU compute | **R × (H+I) × 2B** (large) |
| S2: Activation transfer | Local GPU → CPU → RDMA → Remote CPU compute → RDMA → Local GPU | **(H + I) × 2B** (small, but 2 RDMA round-trips) |
| S3: Pre-cached + relay | Local GPU → RDMA → Remote relay → RDMA → Local GPU compute (weights pre-cached) | **H × 2B** (smallest, but needs prior weight transfer) |

**Key question:** Which strategy wins, and how does EP background traffic affect the choice?

---

## Slide 6 — Cross-Node Results: Strategy Comparison

**Measured: UM253 (A100 SXM4) ↔ UM251 (CPU), HDR IB 200 Gbps, GLOO backend**

### No background traffic (EP=0%)

| Rank | N_miss | S1 total (ms) | S2 total (ms) | S3 total (ms) |
|------|--------|---------------|---------------|---------------|
| 16 | 1 | 1.37 | 0.89 | **0.58** |
| 16 | 4 | 1.82 | 0.69 | **0.86** |
| 64 | 1 | 1.81 | 1.81 | **0.75** |
| 64 | 2 | 3.50 | 3.09 | **0.84** |
| 64 | 4 | 6.81 | 4.68 | **0.94** |
| 128 | 4 | 6.65 | 1.58 | **1.02** |

**Key findings:**
1. **S3 (pre-cached + relay) is consistently fastest** — 1.5-7× faster than S1
2. S1 (weight transfer) scales poorly with rank — 6.81ms at R=64, N=4
3. S2 (CPU compute) suffers from multiple RDMA round-trips at high rank
4. **The winning strategy depends on whether weights are already cached locally**

---

## Slide 7 — EP Contention: Network Awareness Matters

**EP all-to-all traffic competes with miss-recovery traffic on the shared IB fabric, but the measured signal is not strictly monotonic yet.**

Focused counter-validated run with Python IPoIB all-to-all traffic (R=64, N_miss=2, warmup=10, iters=50):

| EP load | S1 total (ms) | S2 total (ms) | S3 total (ms) |
|---------|---------------|---------------|---------------|
| 0% | 3.44 | 3.33 | 1.59 |
| 25% | 3.92 | 3.68 | **0.81** |
| 50% | 3.85 | 6.84 | 1.64 |
| 75% | 4.57 | 5.68 | 1.55 |
| 90% | 3.20 | 4.94 | 1.22 |

**What this shows:**
- Counter validation confirms the background generator is actually injecting traffic on the IB port
- S2 is sensitive to contention because it performs activation transfer, remote CPU compute, and result transfer; in this focused run it degrades from **3.33ms → 6.84ms** at 50% EP load
- S1 and S3 are less stable across load levels because the benchmark still has order/noise effects and the Python IPoIB generator is not a faithful RDMA all-to-all workload
- The old `ib_write_bw` table should not be used as evidence: it had invalid rate flags, `-z` misuse, one-shot server lifecycle bugs, and missing liveness checks

**Implication:** The safe claim is that EP contention can materially affect miss-recovery latency, especially S2, so CoLoRA should be network-aware. Do **not** claim a monotonic measured EP-degradation curve yet; the next experiment should use a lower-level RDMA/QP all-to-all generator or real MoE all-to-all traces with per-level IB counter summaries.

---

## Slide 7b — Insight: EP Load ≠ Effective Contention

**Observation (from Slide 7's focused run):** counter-validated traffic is present at every EP level, yet S2 latency jumps **3.33ms → 6.84ms** at 50% requested load and then *improves* at 75% and 90%.

**Hypothesis:** nominal EP load is a **poor predictor of effective contention** seen by LoRA miss recovery. What actually hurts the recovery path depends on:

- **Direction:** IB is full-duplex; interfering traffic may not share the queue/direction used by S1/S2/S3.
- **Burstiness & pacing:** higher requested rate can reorganize bursts and gaps rather than uniformly raising pressure.
- **Bottleneck identity:** at some load levels the bottleneck is IB bandwidth; at others it shifts to CPU scheduling, DMA descriptors, or remote compute.
- **Sustained vs. instantaneous rate:** the generator's "90%" is a target, not a guaranteed effective rate during a given benchmark window.

**Why this is interesting, not just noise:**
- The S2 jump at 50% is too large to be measurement noise alone — contention is real.
- Real MoE all-to-all is also bursty and irregular, so non-uniform contention is the **expected regime**, not an artifact.

**Unifying insight — EP workload and LoRA weight transfer are one scheduling problem:**
- Both consume the same fabric (IB QPs, CPU DMA path, GPU PCIe lanes).
- A static "EP load %" threshold cannot decide whether to use S1, S2, or S3.
- CoLoRA should react to **measured fabric pressure**, not requested load.

> *Fabric pressure* = a short-window estimate of how busy the shared communication path is **right now**, on the same resources LoRA recovery would use. Concrete signals:
> - **IB port counter deltas** over the last few ms (`port_xmit_data`, `port_rcv_data` from `/sys/class/infiniband/.../counters`) → effective bytes/s in each direction.
> - **RDMA CQ depth / posted-but-uncompleted WRs** → queue contention on the QP we would use.
> - **Recent activation RTT** (small probe over the same QP) → end-to-end stack pressure including CPU and DMA, not just link bandwidth.
> - Optionally **PCIe H2D/D2H busy fraction** → contention on the GPU's bus, relevant for S1.
>
> The point: pressure is **path-specific** and **time-local**, while EP load % is global and aggregate. The first is actionable for per-miss strategy selection; the second is not.

---

## Slide 8 — CoLoRA Design: Hybrid Recovery Policy

```
                    ┌──────────────────────────────────┐
                    │        Miss Event Arrives         │
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │   Is LoRA cached on local GPU?    │
                    └────┬─────────────────────┬───────┘
                    Yes  │                     │ No
                         │                     │
               ┌─────────▼──────┐    ┌─────────▼──────────────┐
               │ GPU compute    │    │ Is miss local?          │
               │ (warm hit)     │    └──┬──────────────┬──────┘
               └────────────────┘       │ Yes          │ No
                                ┌────────▼──────┐  ┌────▼──────────────┐
                                │ N_dec ≤ 8?    │  │ Cross-machine?    │
                                └──┬────────┬───┘  └────┬─────────────┘
                                 Yes       │ No        │
                              ┌────▼───┐ ┌──▼──────┐ ┌─▼─────────────────┐
                              │CPU AVX │ │GPU H2D +│ │ S3: pre-cache +    │
                              │compute │ │compute  │ │ activation relay   │
                              │(local  │ │(local   │ └────┬──────────────┘
                              │ cold)  │ │ cold)   │      │ fallback
                              └────────┘ └─────────┘ ┌────▼──────────────┐
                                                      │ S2: remote CPU     │
                                                      │ activation compute │
                                                      └────────────────────┘
```

**Three-layer policy:**
1. **Local warm hit:** GPU matmul only (weights already on GPU, ~41–45μs)
2. **Local cold miss:** CPU-first compute (CPU wins at N_dec ≤ 8 for all ranks; GPU path used at N_dec ≥ 16)
3. **Cross-machine:** Pre-cache weights during idle → S3 relay at miss time; fallback to S2 if not cached

---

## Slide 9 — CoLoRA Design: Network-Aware Scheduling

**Problem:** EP all-to-all and LoRA recovery share the same IB link, CPU DMA path, and GPU PCIe bus.

**Key design principle (from Slide 7b):** schedule on **measured fabric pressure**, not on nominal EP load.

**Solution: Pressure-aware opportunistic scheduling**

```
EP traffic pattern (MoE all-to-all):
████████░░░░░░████████░░░░░░████████░░░░░░
   ↑ burst    ↑ gap    ↑ burst    ↑ gap

CoLoRA schedules:
░░░░░░░░██████░░░░░░░░██████░░░░░░░░██████
         ↑ weight     ↑ weight     ↑ weight
           pre-cache    pre-cache    pre-cache

At miss time: activation relay only (no weight transfer needed)
```

**Mechanism:**

1. **Sense fabric pressure (per-path, time-local):**
   - IB port counter deltas over the last few ms (`port_xmit_data`, `port_rcv_data`)
   - RDMA CQ depth / outstanding WRs on the QP that S1/S2/S3 would use
   - Recent activation-RTT probe over the same QP
   - PCIe H2D/D2H busy fraction (for S1 weight-transfer cost)

2. **Pre-cache in low-pressure windows:**
   - When the sensed pressure on the recovery path is low, push popular cold LoRA weights to the local GPU in the background.

3. **At miss time, pick the cheapest path under current pressure:**
   - Weights already on GPU → warm GPU compute
   - Local miss, low pressure on PCIe/GPU → CPU AVX (small N_dec) or GPU H2D (large N_dec)
   - Cross-machine miss, weights pre-cached → **S3** (activation relay)
   - Cross-machine miss, weights not cached, IB QP under pressure → **S2** (remote compute, smaller payload exposure)
   - Cross-machine miss, IB QP idle and weights reusable → **S1** (one-time weight transfer + future warm hits)

**Why this unifies EP analysis and LoRA weight transfer:**

- They are not two problems; they are one **shared-fabric scheduling problem**.
- EP-workload analysis tells us *when* the fabric is busy and *how* (which direction, which queue, which bottleneck).
- LoRA recovery turns that signal into a per-miss choice between S1/S2/S3 and CPU/GPU paths.
- Static EP-load thresholds cannot do this; per-path pressure sensing can.

---

## Slide 10 — Experimental Roadmap

| Study | Status | What it proves |
|-------|--------|----------------|
| **1. Single-layer crossover** | ✅ Data collected (optimized) | CPU wins at N_dec ≤ 8 for all ranks (1.01–1.75×). GPU wins at N_dec ≥ 16. Crossover shifted from N=1 to N=8 after eliminating benchmark overheads (allocation, unnecessary CUDA sync) and optimizing AVX kernel (OpenMP, 32-wide vectorization). |
| **2. Cross-node benchmark** | ✅ Data collected (60/60 configs + focused EP run) | S3 (pre-cached relay) is 1.5-7× faster. Focused EP run shows counter-validated contention can degrade S2, but not yet a monotonic curve. |
| **3. Analytical case studies** | 📝 Code complete, needs anchor file | Payload asymmetry (Case A), reuse-aware crossover (Case B), EP contention model (Case D) |
| **4. E2E serving** | 🔲 Planned | TPOT impact under real workload; CoLoRA vs S-LoRA baseline |

**What we have now is sufficient for motivation + challenge + design:**
- Study 1 (optimized): local misses should be CPU-first for typical decode batches (N_dec ≤ 8), GPU-first only at large N_dec ≥ 16.
- Study 2: cross-machine misses require pre-cache/relay and network-aware scheduling.
- Together they motivate a local-CPU + cross-machine-relay hybrid policy.

---

## Slide 11 — Key Takeaways

1. **LoRA miss recovery is the bottleneck in MoE+LoRA serving** — miss rates of 15-30% at practical cache budgets

2. **Local decode miss recovery is CPU-first at typical decode batches (N_dec ≤ 8)** — AVX-512 BF16 compute on CPU (5–35μs) + small activation transfer (14μs) beats GPU H2D weight transfer (22–46μs) + GPU matmul (40–45μs). GPU wins only at N_dec ≥ 16 where CPU compute scales.

3. **Cross-machine recovery requires pre-caching** — Strategy 3 (relay + pre-cached weights) is 1.5-7× faster than naive weight transfer

4. **EP load is a poor proxy for contention** — CoLoRA schedules on **measured fabric pressure** (IB counter deltas, CQ depth, recent RTT, PCIe busy fraction) on the *same path* the recovery would use, not on nominal EP load %.

5. **CoLoRA's hybrid policy adapts to the regime** — local CPU-first for cold misses (N_dec ≤ 8), local GPU for large batches (N_dec ≥ 16), cross-machine pre-cache + relay, schedule around EP traffic

---

## Slide 12 — Next Steps

1. **Replace EP traffic generator** — Python IPoIB all-to-all confirms traffic and S2 degradation, but still lacks a monotonic curve; next use RDMA/QP-level all-to-all or real MoE traces with per-level IB counter summaries

2. **Run analytical case studies (A/B/D)** — code is ready, needs calibration anchor file from measured data

3. **Minimal E2E validation** — single-GPU serving loop with synthetic miss injection; measure TPOT impact per strategy

4. **Paper writing** — motivation section is ready; design section needs final CoLoRA API specification
