# Likely Reviewer Attack Points

1. "LoRA does not alter routing, so why should routing-locality claims change?"
   Answer path: the router can stay fixed while the cache key expands from `expert` to `expert x adapter`.

2. "Azure Functions is not a real LoRA serving trace."
   Answer path: present it as a semi-realistic tenant-arrival proxy, not as literal production adapter logs.

3. "The result may just reflect larger total memory demand."
   Answer path: keep the same router trace and request rate; only vary adapter assignment and cache key granularity.

4. "Zipf popularity should help caching, so why is the joint workload still bad?"
   Answer path: compare marginal skew with joint skew and show top-k coverage collapses after key-space expansion.

5. "Synthetic mapping may create artificial correlation."
   Answer path: keep independent and correlated variants, document both, and show the mechanism already appears in the independent mode.

6. "Latency proxy results may not reflect real serving behavior."
   Answer path: include small live validation with the current 10 dummy LoRAs and clearly scope it as sanity validation.

7. "The prompt corpus may bias routing toward a narrow expert subset."
   Answer path: publish prompt-length statistics, use a deterministic ShareGPT-derived corpus, and report per-request expert diversity.
