# Metrics Checklist

Use this as the checklist for B7-BB9 outputs and figure generation.

## Workload Characterization

- Expert popularity rank curve.
- Joint expert x adapter popularity rank curve.
- Top-10 / Top-50 coverage.
- Entropy.
- Gini coefficient.
- Effective working-set size.
- Per-request unique object count.
- Reuse-distance CDF.

## Cache Layer

- Hit rate.
- Miss rate.
- Cold miss count.
- Capacity miss count.
- Eviction count.
- Per-request miss count.
- Miss count by phase if phase labels are available.

## Latency Layer

- Mean latency.
- P50 latency.
- P95 latency.
- P99 latency.
- Tail-request miss breakdown.
- Prefill vs decode breakdown when trace supports it.

## Logging Requirements

- Every run records cache policy.
- Every run records cache budget grid.
- Every run records active seed values.
- Every figureable metric family is exported as CSV in addition to JSON summaries.
