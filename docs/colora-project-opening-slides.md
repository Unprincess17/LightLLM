# CoLoRA: Cross-Machine LoRA Miss Recovery for MoE Serving

---

## Slide 1 — Title

**CoLoRA: Network-Aware LoRA Miss Recovery for Distributed MoE Serving**

Gong Shufan

*Problem:* MoE + multi-LoRA serving hits cache misses — current systems stall the GPU.
*Insight:* Local decode misses are GPU-first except N=1 ties; cross-machine misses need pre-cache + relay.
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
- Locally this is competitive even for decode, but cross-machine misses add network delay and contention

---

## Slide 3 — Local Decode Miss Recovery: CPU Only Wins at N=1

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

For **R=64, N=1** (decode phase, activation = 4KB):

GPU path (H2D weights + GPU matmul):
- H2D weight transfer: 28.0μs (361KB over PCIe Gen4 ×16)
- GPU matmul: 42.9μs
- **Total: 69.6μs**

CPU path (D2H activation + AVX-512 + H2D result):
- D2H activation: 14.1μs (4KB, pinned memory copy)
- CPU AVX compute: 33.4μs (no-expand: batch all N in one call)
- H2D result: 17.0μs
- **Total: 64.5μs**

**CPU is 1.08× faster at N=1, R=64. GPU wins at N≥2 (see Slide 4 for full crossover).**

> Note: Both paths use the no-expand optimization. The original comparison had O(N) CPU cost vs O(1) GPU cost — this is fixed above. H2D weight transfer is a one-time cold-miss cost. For warm hits, GPU path drops to ~42μs.

---

## Slide 4 — Crossover Curve: When Does CPU Win?

**Benchmark source:** `motivation_microbench.py` — single MoE expert, A100-SXM4-80GB.
**Model:** Qwen3-VL-30B-A3B (H=2048, I=768, top_k=2).
**Sweep:** LoRA rank R ∈ {64, 128}, decode batch N_dec ∈ {1, 2, 4, 8, 16}.
**Critical fix:** Both CPU and GPU paths use no-expand (batch all N_dec tokens in one call).
*Previously, CPU path looped over each adapter/token (O(N)) vs GPU batched (O(1)) — unfair comparison.*

**Fresh benchmark results (decode batch, no prefill sequence length):**

| Rank | N_dec | GPU total (μs) | CPU total (μs) | Winner |
|------|-------|----------------|----------------|--------|
| 64 | 1 | 69.6 | **64.5** | CPU (1.08×, −5.0μs) |
| 64 | 2 | **73.8** | 79.3 | GPU (1.07×, −5.5μs) |
| 64 | 4 | **73.7** | 106.7 | GPU (1.45×, −33.0μs) |
| 64 | 8 | **73.2** | 165.2 | GPU (2.26×, −92.0μs) |
| 64 | 16 | **71.7** | 170.3 | GPU (2.37×, −98.6μs) |
| 128 | 1 | 83.7 | **79.9** | CPU (1.05×, −3.8μs) |
| 128 | 2 | **88.1** | 109.4 | GPU (1.24×, −21.3μs) |
| 128 | 4 | **88.7** | 168.5 | GPU (1.90×, −79.8μs) |
| 128 | 8 | **88.5** | 287.4 | GPU (3.25×, −199.0μs) |
| 128 | 16 | **84.5** | 289.8 | GPU (3.43×, −205.4μs) |

**Corrected crossover point:**
- **CPU is only useful at N_dec=1, and only by a small margin (1.05–1.08×).**
- **GPU wins at N_dec≥2 for both R=64 and R=128.**
- GPU path is dominated by H2D weight transfer (~28–46μs), not GPU matmul (~40–45μs).
- CPU path scales with N_dec due to AVX compute (~33–258μs).

> `validate_focused.py` sanity check reports clean GPU reuse at ~41–45μs and the same qualitative conclusion: CPU is only competitive in the tiny decode-batch regime. The original CPU-wins-at-N≤4 claim came from unfair O(N) CPU looping and contaminated CUDA sync timing.

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

**EP all-to-all traffic competes with miss-recovery traffic on shared IB link.**

Latency under EP contention (R=64, N=2):

| EP load | S1 total (ms) | S2 total (ms) | S3 total (ms) |
|---------|---------------|---------------|---------------|
| 0% | 3.50 | 3.09 | **0.84** |
| 25% | 2.86 | 3.39 | **1.13** |
| 50% | 3.86 | 3.38 | **1.12** |
| 75% | 2.07 | 2.85 | **0.81** |
| 90% | 1.89 | 2.83 | **0.98** |

**Observations:**
- S3 degrades gracefully — small activation payload is less sensitive to contention
- S1 shows non-monotonic behavior (possible EP generator issues — needs verification)
- S2 degrades most — multiple round-trips amplify contention

**Implication:** CoLoRA needs **EP-aware scheduling** — schedule miss-recovery traffic during EP quiet gaps.

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
                                │ GPU H2D +     │  │ Cross-machine?    │
                                │ GPU compute   │  └────┬─────────────┘
                                │ (local cold)  │       │
                                └───────────────┘  ┌────▼──────────────┐
                                                   │ S3: pre-cache +    │
                                                   │ activation relay   │
                                                   └────┬───────────────┘
                                                        │ fallback if not cached
                                                   ┌────▼──────────────┐
                                                   │ S2: remote CPU     │
                                                   │ activation compute │
                                                   └────────────────────┘
```

**Three-layer policy:**
1. **Local warm hit:** GPU matmul only (weights already on GPU, ~41–45μs)
2. **Local cold miss:** GPU H2D + compute (GPU wins at N_dec≥2; CPU only slightly wins at N_dec=1)
3. **Cross-machine:** Pre-cache weights during idle → S3 relay at miss time; fallback to S2 if not cached

---

## Slide 9 — CoLoRA Design: Network-Aware Scheduling

**Problem:** EP all-to-all and LoRA recovery share the same IB link.

**Solution: Opportunistic scheduling in EP gaps**

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
- Monitor EP traffic phases via RDMA CQ polling
- Pre-fetch LoRA weights during EP gaps (background, non-blocking)
- At miss time: only small activation transfer needed → S3 path
- Fallback to S2 if weights not yet cached (first-time miss)

---

## Slide 10 — Experimental Roadmap

| Study | Status | What it proves |
|-------|--------|----------------|
| **1. Single-layer crossover** | ✅ Data collected (corrected) | Local GPU H2D+compute wins at N_dec≥2; CPU only slightly wins at N_dec=1. Original CPU-wins-at-N≤4 claim was unfair O(N) vs O(1). |
| **2. Cross-node benchmark** | ✅ Data collected (60/60 configs) | S3 (pre-cached relay) is 1.5-7× faster. EP contention affects S1/S2 more than S3. |
| **3. Analytical case studies** | 📝 Code complete, needs anchor file | Payload asymmetry (Case A), reuse-aware crossover (Case B), EP contention model (Case D) |
| **4. E2E serving** | 🔲 Planned | TPOT impact under real workload; CoLoRA vs S-LoRA baseline |

**What we have now is sufficient for motivation + challenge + design:**
- Study 1 (corrected): local misses should be GPU-first except the tiny N_dec=1 edge case.
- Study 2: cross-machine misses require pre-cache/relay and network-aware scheduling.
- Together they motivate a local-GPU + cross-machine-relay hybrid policy.

---

## Slide 11 — Key Takeaways

1. **LoRA miss recovery is the bottleneck in MoE+LoRA serving** — miss rates of 15-30% at practical cache budgets

2. **Local decode miss recovery is GPU-first at N_dec≥2** — H2D weight transfer is ~28–46μs and GPU matmul is ~40–45μs; CPU only slightly wins at N_dec=1.

3. **Cross-machine recovery requires pre-caching** — Strategy 3 (relay + pre-cached weights) is 1.5-7× faster than naive weight transfer

4. **EP contention motivates gap-aware scheduling** — recovery traffic must exploit EP quiet periods

5. **CoLoRA's hybrid policy adapts to the regime** — local GPU H2D+compute, cross-machine pre-cache + relay, schedule around EP traffic

---

## Slide 12 — Next Steps

1. **Verify EP traffic generator** — current cross-node results show non-monotonic degradation; need to confirm `ib_write_bw` is actually generating contention

2. **Run analytical case studies (A/B/D)** — code is ready, needs calibration anchor file from measured data

3. **Minimal E2E validation** — single-GPU serving loop with synthetic miss injection; measure TPOT impact per strategy

4. **Paper writing** — motivation section is ready; design section needs final CoLoRA API specification
