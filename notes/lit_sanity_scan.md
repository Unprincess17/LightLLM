# Literature Sanity Scan

Scope: narrow scan for hypotheses, metrics, and reviewer framing. This is not intended to become a survey section.

## Routing and Expert Skew

- Sparse MoE papers such as GShard and Switch Transformer motivate explicit load-balancing losses because unconstrained routing naturally becomes skewed.
- Later MoE systems papers and model reports keep returning to the same tension: skew improves specialization but hurts balanced capacity and stable throughput.
- Training-side "expert load balance" does not imply serving-side locality. Even when marginal expert popularity is skewed, online arrivals still produce bursty, layer-dependent access patterns.

## Serving and Locality

- Multi-tenant LoRA serving work such as S-LoRA and Punica treats adapter activation as a high-cardinality working-set problem; hot adapters help, but tail adapters dominate cache pressure.
- This case study extends that serving argument into sparse MoE: the relevant cacheable object is not only an expert, but the expert-adapter combination once LoRA weights are adapter-specific.
- The key novelty is the joint access stream. Existing MoE load-balance discussions usually stop at `P(expert)` rather than `P(expert, adapter)`.

## Tail Latency Framing

- Tail amplification is usually attributed to misses, fallback paths, and queueing, not just mean utilization.
- For this case study the paper claim should stay precise: LoRA does not need to change the router to worsen tail latency; it is sufficient that the serving system keys residency on both expert and adapter.
- Mean latency is secondary evidence. P95 and P99 are the headline metrics because the workload mechanism is inherently bursty and miss-concentrated.

## Standard Metrics To Keep

- Expert popularity rank and top-k coverage.
- Entropy or Gini of the access distribution.
- Effective working-set size.
- Reuse distance or stack distance.
- Cache hit rate and miss rate under fixed budgets.
- Per-request miss count.
- Mean, P50, P95, P99 latency or latency proxy.

## Confounders To Control

- Same router trace across conditions.
- Same total request count and ordering across conditions.
- Same cache policy and budget across conditions.
- Only adapter assignment policy changes between expert-only and expert x LoRA conditions.
- Distinguish larger aggregate memory demand from finer-grained fragmentation.
