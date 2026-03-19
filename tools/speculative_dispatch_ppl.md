# Speculative Dispatch PPL Probe

This document explains what [speculative_dispatch_ppl.py](/home/shufan/LightLLM-integrate-to-SLoRA/tools/speculative_dispatch_ppl.py) does, how to run it, and what results we have already validated.

## Goal

The script is a standalone offline probe for the COLoRA-style speculative dispatch idea on Hugging Face MoE checkpoints.

It evaluates this partial-overlap approximation during teacher-forced causal LM scoring:

1. At layer `L`, compute the current token routing `E_L^t`.
2. Compare it against the previous token routing at the same layer, `E_L^{t-1}`.
3. For the routed experts in the intersection `E_L^t ∩ E_L^{t-1}`, use the stale activation from the previous layer, `X_{L-1}^t`.
4. For the newly routed experts in `E_L^t \ E_L^{t-1}`, keep the exact current activation `X_L^t`.

So the trigger is in the time dimension, but the approximation payload is now mixed per expert within the same token.

The script also has a `router_trace.jsonl` analysis mode so you can verify whether your online PPL experiment is using the same locality definition as your offline traces.

## What The Script Measures

For the teacher-forced PPL path, it prints:

- `Clean PPL`
- `Speculative PPL`
- `Delta PPL`
- `Matched-Token Rate`
- `Partial-Swap Token Rate`
- `Swapped-Expert Rate`
- `Average Expert Overlap`

For the trace-analysis path, it prints two different locality notions:

- `Same-Layer Previous-Token`
  This is the metric that matches the current speculative-dispatch trigger logic.
- `Adjacent-Layer Same-Token`
  This is the older cross-layer metric that turned out not to explain the 72% ShareGPT locality number.

## Measurement Definitions

### Teacher-Forced PPL Output

- `Evaluated Tokens`
  The number of input tokens kept after tokenization and `--max_eval_tokens` truncation.
- `Predicted Tokens`
  The number of causal next-token targets that were actually scored in the NLL computation after window masking.
- `Seq Len`
  The maximum token span per teacher-forced scoring window.
- `Stride`
  The step size between scoring windows.

- `Clean PPL`
  Perplexity from the exact model with no speculative swap.
- `Speculative PPL`
  Perplexity from the patched model using the partial-overlap hybrid approximation.
- `Delta PPL`
  `Speculative PPL - Clean PPL`. Positive means the approximation hurt perplexity.

- `Matched-Token Rate`
  Fraction of candidate token-layer positions where `E_L^t` and `E_L^{t-1}` match exactly under `same_set`. This is now a diagnostic exact-match statistic, not the actual partial-reuse trigger.
- `Partial-Swap Token Rate`
  Fraction of candidate token-layer positions where at least one routed expert reused stale `X_{L-1}^t`.
- `Swapped-Expert Rate`
  Fraction of candidate expert assignments that reused stale `X_{L-1}^t`. This is the online metric that corresponds most closely to overlap hit rate.
- `Average Expert Overlap`
  The mean number of shared experts between `E_L^t` and `E_L^{t-1}` across all candidate token-layer positions.

### Trace Analysis Output

- `Trace Phase`
  Which phase from `router_trace.jsonl` is being analyzed, usually `decode`.
- `Trace Events`
  Number of trace rows that survived the phase filter.
- `Trace Top-K`
  Number of routed experts per token in the trace.
- `Same-Set Rate`
  Fraction of transitions whose expert sets are exactly equal after sorting the top-k ids.
- `Ordered Rate`
  Fraction of transitions whose routed expert lists match in exact order.
- `Top-1 Rate`
  Fraction of transitions whose first-ranked expert id matches.
- `Any-Overlap Rate`
  Fraction of transitions that share at least one expert.
- `Average Expert Overlap`
  Mean number of overlapping experts per transition.
- `Mean Overlap Hit Rate`
  Mean `|intersection| / top_k` across transitions. This is the metric that produced the earlier `72%` ShareGPT locality result.
- `Prefix set match rates`
  Exact-set match rates computed only on the first `k` routed experts for `k in {1, 2, 4, 8}` when available.

## Current Semantics

The current patch logic in [speculative_dispatch_ppl.py](/home/shufan/LightLLM-integrate-to-SLoRA/tools/speculative_dispatch_ppl.py) is:

- Layer `0` never swaps, because there is no `X_{L-1}`.
- Token `t=0` only becomes a candidate if the previous global token is available from the previous scoring window.
- Tokens `t>0` compare their routing against token `t-1` at the same layer.
- For each routed expert in token `t`, if that expert also appeared in token `t-1` at the same layer, that expert branch uses stale `prev_hidden_flat[t]`.
- For each newly routed expert not present in token `t-1`, that expert branch uses the true current hidden state.
- The `same_set` rule is retained only for exact-match diagnostics and reporting.
- Shared or dense experts remain on the exact current hidden state path.

## Requirements

The script expects:

- `torch`
- `transformers`
- `requests`

The default model path is already configured to the local Qwen3-VL MoE checkpoint:

```text
/home/shufan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c
```

Supported model types:

- `qwen3_vl_moe`
- `qwen3_moe`
- `qwen2_moe`

## Recommended Usage

### 1. Verify The ShareGPT Trace Locality

Run this first to confirm the routing metric you care about.

```bash
python tools/speculative_dispatch_ppl.py \
  --router_trace_path artifacts/case_study/router_lora_case_v1/router_trace/router_trace.jsonl \
  --trace_only
```

To print per-layer trace statistics:

```bash
python tools/speculative_dispatch_ppl.py \
  --router_trace_path artifacts/case_study/router_lora_case_v1/router_trace/router_trace.jsonl \
  --trace_only \
  --debug_print_trace_layer_stats
```

### 2. Run Teacher-Forced PPL On A Local Text File

If you have a plain-text evaluation corpus:

```bash
python tools/speculative_dispatch_ppl.py \
  --model_path /home/shufan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c \
  --text_path /path/to/eval.txt \
  --seq_len 512 \
  --stride 512 \
  --device_map auto \
  --dtype auto \
  --local_files_only
```

This will run:

- one clean pass
- one speculative pass with the monkey patch enabled

### 3. Run PPL And Print Online Layer Debug Stats

```bash
python tools/speculative_dispatch_ppl.py \
  --model_path /home/shufan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c \
  --text_path /path/to/eval.txt \
  --seq_len 512 \
  --stride 512 \
  --device_map auto \
  --dtype auto \
  --debug_print_layer_stats \
  --debug_sample_limit 32 \
  --debug_match_report_path /tmp/spec_dispatch_debug.json
```

This adds:

- per-layer same-layer previous-token routing stats from the live model run
- mismatch samples in `/tmp/spec_dispatch_debug.json`

### 4. Run PPL And Trace Reconciliation Together

If you want the online PPL run and the offline trace stats in the same output:

```bash
python tools/speculative_dispatch_ppl.py \
  --model_path /home/shufan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c \
  --text_path /path/to/eval.txt \
  --router_trace_path artifacts/case_study/router_lora_case_v1/router_trace/router_trace.jsonl \
  --seq_len 512 \
  --stride 512 \
  --device_map auto \
  --dtype auto \
  --debug_print_layer_stats
```

## Important Interpretation Notes

The routing trace and the teacher-forced PPL run only align if they use the same locality definition and comparable input distribution.

Correct comparisons:

- Compare the online `Matched-Token Rate` against the offline `Same-Layer Previous-Token / Same-Set Rate`.
- Compare the online `Swapped-Expert Rate` against the offline `Same-Layer Previous-Token / Mean Overlap Hit Rate`.

Incorrect comparisons:

- Do not compare the online `Matched-Token Rate` against the offline `Adjacent-Layer Same-Token / Same-Set Rate`.
- Do not compare the online `Swapped-Expert Rate` against the offline `Adjacent-Layer Same-Token / Mean Overlap Hit Rate`.

Also note:

- If your trace came from ShareGPT decode traffic but your PPL run uses WikiText, the match rate does not need to agree.
- If you want the online matched rate to approach the ShareGPT trace number, run the PPL probe on text that resembles the traced workload.

## Verified Results So Far

### Default Full Evaluation

Using the exact command below:

```bash
python tools/speculative_dispatch_ppl.py
```

This run used:

- the local `Qwen3-VL-30B-A3B-Instruct` checkpoint
- the script defaults `seq_len=512`, `stride=512`, `max_eval_tokens=8192`
- the fallback WikiText-2 validation text fetched from the script's default URL
- `device_map=auto`
- `dtype=auto`

Measured output:

- `Model Type: qwen3_vl_moe`
- `Evaluated Tokens: 8192`
- `Predicted Tokens: 8176`
- `Clean PPL: 6.677874`
- `Speculative PPL: 9.835621`
- `Delta PPL: +3.157747`
- `Matched-Token Rate: 0.53% (2050/384977)`
- `Partial-Swap Token Rate: 94.97% (365622/384977)`
- `Swapped-Expert Rate: 44.76% (1378610/3079816)`
- `Average Expert Overlap: 3.5810`

Interpretation:

- On this default WikiText-style run, the partial-overlap approximation substantially degrades perplexity.
- The live `Matched-Token Rate` is only `0.53%`, far below the ShareGPT trace `Same-Set Rate` of `11.88%`.
- The live `Swapped-Expert Rate` is `44.76%`, below the ShareGPT trace mean overlap hit rate of `72.23%`.
- Those gaps are expected because the trace came from ShareGPT decode traffic, while this run used the default WikiText fallback corpus.

### ShareGPT Router Trace

Using:

```bash
python tools/speculative_dispatch_ppl.py \
  --router_trace_path artifacts/case_study/router_lora_case_v1/router_trace/router_trace.jsonl \
  --trace_only
```

We verified:

- `Trace Phase: decode`
- `Trace Events: 5297088`
- `Trace Top-K: 8`

Same-layer previous-token locality:

- `Same-Set Rate: 11.88%`
- `Ordered Rate: 3.27%`
- `Top-1 Rate: 69.00%`
- `Any-Overlap Rate: 99.51%`
- `Average Expert Overlap: 5.7786`
- `Mean Overlap Hit Rate: 72.23%`

Adjacent-layer same-token locality:

- `Same-Set Rate: 0.00%`
- `Ordered Rate: 0.00%`
- `Top-1 Rate: 1.40%`
- `Any-Overlap Rate: 38.80%`
- `Average Expert Overlap: 0.4837`
- `Mean Overlap Hit Rate: 6.05%`

These numbers explain both the earlier debugging result and the revised overlap policy:

- the original cross-layer patch logic naturally produced `Matched-Token Rate = 0%`
- the ShareGPT `72%` number comes from same-layer previous-token overlap, not adjacent-layer equality
- under the revised overlap policy, the relevant trace-side reuse metric is `Mean Overlap Hit Rate: 72.23%`

### Current Expectation For The Updated PPL Patch

With the revised partial-overlap patch:

- the online exact-match statistic uses the same same-layer previous-token notion as the ShareGPT trace
- the `Matched-Token Rate` should be compared against the trace `Same-Set Rate: 11.88%`
- the `Swapped-Expert Rate` should be compared against the trace `Mean Overlap Hit Rate: 72.23%`
- the actual PPL delta depends strongly on the evaluation corpus
- on the default WikiText fallback run, the measured `Delta PPL` was `+3.157747`

## Useful Flags

- `--trace_only`
  Only analyze `router_trace.jsonl`. Skip model loading.
- `--router_trace_path`
  Path to canonical `router_trace.jsonl`.
- `--debug_print_trace_layer_stats`
  Print per-layer trace statistics.
- `--debug_print_layer_stats`
  Print per-layer live-model routing stats during the speculative PPL run.
- `--debug_sample_limit`
  Record a bounded number of mismatched routing samples.
- `--debug_match_report_path`
  Dump the live run debug report as JSON.
- `--max_eval_tokens`
  Limit evaluation tokens for quick experiments.
- `--seq_len` and `--stride`
  Control teacher-forced scoring windows.

## Practical Workflow

Use this sequence:

1. Run `--trace_only` to confirm the target locality metric.
2. Run the PPL probe on your chosen text corpus.
3. Check whether `Matched-Token Rate` is in the same regime as the trace `Same-Set Rate`, and whether `Swapped-Expert Rate` is in the same regime as the trace `Mean Overlap Hit Rate`.
4. If it is not, inspect the corpus mismatch first before changing the patch logic again.
5. Only then interpret `Delta PPL` as evidence about the speculative approximation.

## Known Limitation

This probe only simulates the routed expert branch. It does not yet model the full serving-engine pipeline, PCIe timing, or LoRA-adapter transport path. It is intended as a fast offline accuracy study, not a latency simulator.
