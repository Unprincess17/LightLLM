# CoLoRA HTML Opening Slides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the CoLoRA opening slide deck from markdown to a self-contained HTML deck and make the HTML file the only maintained source of truth.

**Architecture:** Replace `docs/colora-project-opening-slides.md` with `docs/colora-project-opening-slides.html`. The HTML is a static, dependency-free presentation with a compact status dashboard before Slide 1, one `<section class="slide">` per slide, and inline CSS for rendering. Roadmap and checklist content are updated during conversion; stale or distracting narrative is preserved only as HTML comments where useful.

**Tech Stack:** Static HTML5, inline CSS, native browser rendering, no JavaScript, no build system.

---

## File Structure

- Create: `docs/colora-project-opening-slides.html`
  - Self-contained HTML deck.
  - Contains a top dashboard card and Slides 1-12.
  - Uses inline CSS and semantic sections.
- Delete: `docs/colora-project-opening-slides.md`
  - Removed because HTML becomes the only maintained source.
- Existing reference: `docs/superpowers/specs/2026-06-10-colora-html-opening-slides-design.md`
  - Source spec for expected checklist content and cleanup rules.

---

### Task 1: Add HTML Deck Shell and Dashboard

**Files:**
- Create: `docs/colora-project-opening-slides.html`
- Reference: `docs/superpowers/specs/2026-06-10-colora-html-opening-slides-design.md`

- [ ] **Step 1: Create the HTML file with static deck shell and dashboard**

Write this full initial file to `docs/colora-project-opening-slides.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CoLoRA: Network-Aware LoRA Miss Recovery for Distributed MoE Serving</title>
  <style>
    :root {
      --bg: #0b1020;
      --slide: #f8fafc;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --accent: #2563eb;
      --good: #15803d;
      --warn: #b45309;
      --bad: #b91c1c;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .deck {
      height: 100vh;
      overflow-y: auto;
      scroll-snap-type: y mandatory;
    }

    .dashboard,
    .slide {
      min-height: 100vh;
      scroll-snap-align: start;
      background: var(--slide);
      padding: 52px 72px;
      border-bottom: 6px solid var(--bg);
    }

    .slide {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    h1 {
      margin: 0 0 18px;
      font-size: 48px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }

    h2 {
      margin: 0 0 12px;
      font-size: 36px;
      line-height: 1.15;
      letter-spacing: -0.02em;
    }

    h3 {
      margin: 14px 0 8px;
      font-size: 23px;
    }

    p,
    li {
      font-size: 21px;
      line-height: 1.45;
    }

    ul,
    ol {
      margin: 6px 0 0 28px;
      padding: 0;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border-radius: 12px;
      overflow: hidden;
      font-size: 17px;
      box-shadow: 0 8px 20px rgb(15 23 42 / 6%);
    }

    th,
    td {
      border: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }

    th {
      background: #e2e8f0;
      font-weight: 700;
    }

    pre {
      margin: 0;
      padding: 18px;
      overflow: auto;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 14px;
      font-size: 16px;
      line-height: 1.25;
    }

    blockquote {
      margin: 8px 0;
      padding: 10px 16px;
      border-left: 5px solid var(--accent);
      background: #eff6ff;
      color: #1e3a8a;
    }

    blockquote p {
      margin: 0;
    }

    .tag {
      display: inline-block;
      margin-right: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #dbeafe;
      color: #1d4ed8;
      font-size: 16px;
      font-weight: 700;
    }

    .muted { color: var(--muted); }
    .lead { font-size: 25px; }
    .ok { color: var(--good); font-weight: 800; }
    .warn { color: var(--warn); font-weight: 800; }
    .bad { color: var(--bad); font-weight: 800; }
    .win { color: var(--good); font-weight: 800; }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }

    .grid3 {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: 0 8px 20px rgb(15 23 42 / 6%);
    }

    .card h3:first-child {
      margin-top: 0;
    }

    .dashboard-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }

    .dashboard-title p {
      margin: 8px 0 0;
      color: var(--muted);
    }

    .summary-counts {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }

    .count-card {
      padding: 14px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
    }

    .count-card strong {
      display: block;
      font-size: 30px;
      line-height: 1;
    }

    .count-card span {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    details {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
    }

    summary {
      cursor: pointer;
      font-weight: 700;
      font-size: 18px;
    }

    details li {
      font-size: 17px;
    }

    .checklist-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }

    .checklist-grid li {
      font-size: 16px;
    }

    .footer {
      margin-top: auto;
      color: #64748b;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <main class="deck">
    <section class="dashboard" aria-labelledby="project-status-title">
      <div class="dashboard-header">
        <div class="dashboard-title">
          <span class="tag">Project Status</span>
          <h1 id="project-status-title">CoLoRA Opening Deck TODOs</h1>
          <p>Claude-readable and human-readable status summary. This card is the canonical checklist for the opening-deck artifact.</p>
        </div>
      </div>

      <div class="summary-counts" aria-label="Checklist counts">
        <div class="count-card"><strong>6</strong><span>Done / Keep</span></div>
        <div class="count-card"><strong>3</strong><span>Stale / Retire</span></div>
        <div class="count-card"><strong>3</strong><span>Active Next</span></div>
        <div class="count-card"><strong>2</strong><span>Decisions</span></div>
      </div>

      <div class="grid">
        <details open>
          <summary>Done / Keep</summary>
          <ul>
            <li>Local CPU/GPU crossover is measured and optimized; CPU wins for typical decode batches up to N_dec ≤ 8.</li>
            <li>Cross-node S1/S2/S3 benchmark is collected, including the 60/60 configurations and focused EP run.</li>
            <li>Calibration anchor file exists.</li>
            <li>System TPOT simulation and synthetic sweeps exist.</li>
            <li>Small live validation pipeline exists.</li>
            <li>Figure assembly pipeline exists.</li>
          </ul>
        </details>
        <details open>
          <summary>Stale / Retire</summary>
          <ul>
            <li>“Case studies need anchor file” is stale because the calibration anchor exists.</li>
            <li>“Minimal E2E validation is only planned” is stale because the B11 live-validation pipeline exists.</li>
            <li>Any monotonic EP-degradation claim should be retired unless supported by a stronger generator or trace-backed study.</li>
          </ul>
        </details>
        <details open>
          <summary>Active Next</summary>
          <ul>
            <li>Run an RDMA/QP-level all-to-all or real MoE trace pressure study.</li>
            <li>Produce per-load IB counter summaries.</li>
            <li>Run a paper claim-to-artifact audit before final writing.</li>
          </ul>
        </details>
        <details open>
          <summary>Decisions Needed</summary>
          <ul>
            <li>Decide whether to invest in a lower-level RDMA generator or frame the current EP result as qualitative contention evidence.</li>
            <li>Decide which claims belong in the main paper versus appendix.</li>
          </ul>
        </details>
      </div>
    </section>
  </main>
</body>
</html>
```

- [ ] **Step 2: Open the file in a browser or preview server**

Run:

```bash
python -m http.server 8000 --directory docs
```

Expected: server starts and prints `Serving HTTP on ... port 8000`.

Open `http://127.0.0.1:8000/colora-project-opening-slides.html`.
Expected: the dashboard card renders with four count cards and four details sections.

Stop the server with `Ctrl+C` after checking.

- [ ] **Step 3: Commit the shell and dashboard**

Run:

```bash
git add docs/colora-project-opening-slides.html
git commit -m "docs(slides): add HTML opening deck dashboard"
```

Expected: commit succeeds with one new file.

---

### Task 2: Convert Slides 1-9 Into HTML Sections

**Files:**
- Modify: `docs/colora-project-opening-slides.html`
- Reference: `docs/colora-project-opening-slides.md`

- [ ] **Step 1: Insert Slides 1-9 before `</main>`**

In `docs/colora-project-opening-slides.html`, insert the following HTML after the closing `</section>` of the dashboard and before `</main>`:

```html
    <section class="slide" aria-labelledby="slide-1-title">
      <div><span class="tag">Slide 1</span><span class="muted">Title</span></div>
      <h1 id="slide-1-title">CoLoRA: Network-Aware LoRA Miss Recovery for Distributed MoE Serving</h1>
      <p class="lead">Gong Shufan</p>
      <div class="grid3">
        <div class="card">
          <h3>Problem</h3>
          <p>MoE + multi-LoRA serving hits cache misses — current systems stall the GPU.</p>
        </div>
        <div class="card">
          <h3>Insight</h3>
          <p>Local decode misses are CPU-first for typical batches (N≤8); cross-machine misses need pre-cache + relay.</p>
        </div>
        <div class="card">
          <h3>Contribution</h3>
          <p>Hybrid local/cross-machine recovery + network-aware scheduling.</p>
        </div>
      </div>
    </section>

    <section class="slide" aria-labelledby="slide-2-title">
      <div><span class="tag">Slide 2</span></div>
      <h2 id="slide-2-title">The Miss Problem in MoE + LoRA Serving</h2>
      <p><strong>Setup:</strong> MoE model such as Qwen3-30B-A3B serving thousands of LoRA adapters.</p>
      <div class="grid">
        <div class="card">
          <h3>Cache miss is common</h3>
          <ul>
            <li>GPU memory is finite and cannot hold all LoRA weights.</li>
            <li>Zipf-distributed adapter popularity creates a long tail of cold adapters.</li>
            <li>Miss rate at budget=2048, out of roughly 4000 experts, is about <strong>15–30%</strong> depending on workload.</li>
            <li>Each miss stalls the decode step until the LoRA is recovered.</li>
          </ul>
        </div>
        <div class="card">
          <h3>Current approach</h3>
          <ul>
            <li>S-LoRA / Punica path: miss → H2D transfer LoRA weights to GPU → GPU compute merge.</li>
            <li>Locally, CPU-first recovery is faster for typical decode batches.</li>
            <li>Cross-machine misses add network delay and contention.</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="slide" aria-labelledby="slide-3-title">
      <div><span class="tag">Slide 3</span></div>
      <h2 id="slide-3-title">Local Decode Miss Recovery: CPU Wins at Typical Decode Batches</h2>
      <table>
        <tr><th>Parameter</th><th>Prefill</th><th>Decode</th></tr>
        <tr><td>Batch size (N)</td><td>64–256</td><td><strong>1–16</strong></td></tr>
        <tr><td>Sequence length (S)</td><td>512–4096</td><td><strong>1</strong></td></tr>
        <tr><td>Activation size</td><td>S × 2048 bf16</td><td><strong>4KB</strong> (1 × 2048 bf16)</td></tr>
        <tr><td>LoRA weight size</td><td>Fixed</td><td>R=64: 361KB; R=128: 722KB</td></tr>
        <tr><td>GPU compute time</td><td>Amortized</td><td><strong>Dominated by H2D transfer</strong></td></tr>
      </table>
      <div class="grid">
        <div class="card">
          <h3>GPU path, R=64 N=4</h3>
          <ul>
            <li>H2D weight transfer: 28.3μs.</li>
            <li>GPU matmul: 45.6μs.</li>
            <li><strong>Total: 72.4μs.</strong></li>
          </ul>
        </div>
        <div class="card">
          <h3>CPU path, R=64 N=4</h3>
          <ul>
            <li>D2H activation: 14.1μs.</li>
            <li>CPU AVX compute: 32.4μs.</li>
            <li>H2D result: 16.6μs.</li>
            <li><strong>Total: 63.2μs.</strong></li>
          </ul>
        </div>
      </div>
      <blockquote><p>CPU is 1.15× faster at N=4, R=64. CPU wins up to N=8 for R≤64.</p></blockquote>
      <!-- Original detailed implementation note intentionally hidden from rendered deck: both paths use no-expand optimization with AVX-512 BF16 dot-product instructions; CPU path benefits from pre-allocated output buffers, OpenMP token parallelism, and 32-wide Stage 2 vectorization. -->
    </section>

    <section class="slide" aria-labelledby="slide-4-title">
      <div><span class="tag">Slide 4</span></div>
      <h2 id="slide-4-title">Crossover Curve: CPU Wins at Typical Decode Batches</h2>
      <p><strong>Benchmark source:</strong> motivation_microbench.py, single MoE expert, A100-SXM4-80GB. Model: Qwen3-VL-30B-A3B, H=2048, I=768, top_k=2.</p>
      <table>
        <tr><th>Rank</th><th>N_dec</th><th>GPU total (μs)</th><th>CPU total (μs)</th><th>Winner</th></tr>
        <tr><td>8</td><td>1</td><td>64.5</td><td class="win">36.8</td><td>CPU (1.75×)</td></tr>
        <tr><td>8</td><td>4</td><td>65.8</td><td class="win">46.5</td><td>CPU (1.42×)</td></tr>
        <tr><td>8</td><td>16</td><td>65.8</td><td class="win">63.8</td><td>CPU (1.03×)</td></tr>
        <tr><td>16</td><td>8</td><td>65.7</td><td class="win">51.6</td><td>CPU (1.27×)</td></tr>
        <tr><td>32</td><td>8</td><td>66.4</td><td class="win">57.2</td><td>CPU (1.16×)</td></tr>
        <tr><td>64</td><td>4</td><td>72.4</td><td class="win">63.2</td><td>CPU (1.15×)</td></tr>
        <tr><td>64</td><td>16</td><td class="win">72.4</td><td>141.6</td><td>GPU (1.95×)</td></tr>
        <tr><td>128</td><td>8</td><td>85.6</td><td class="win">84.4</td><td>CPU (1.01×)</td></tr>
        <tr><td>128</td><td>16</td><td class="win">86.3</td><td>221.9</td><td>GPU (2.57×)</td></tr>
      </table>
      <ul>
        <li><strong>CPU wins at N_dec ≤ 8 for all ranks R ≤ 64</strong>, with speedups of 1.01–1.75×.</li>
        <li>CPU wins at N_dec ≤ 8 even for R=128, though margins are thin.</li>
        <li>GPU wins at N_dec ≥ 16 for all ranks, where AVX compute cost dominates.</li>
      </ul>
      <!-- Hidden from rendered deck: the crossover shifted after eliminating allocation overhead, adding OpenMP token parallelism, and removing unnecessary CUDA synchronize after CPU compute. -->
    </section>

    <section class="slide" aria-labelledby="slide-5-title">
      <div><span class="tag">Slide 5</span></div>
      <h2 id="slide-5-title">The Cross-Machine Challenge</h2>
      <p><strong>In distributed MoE with Expert Parallelism, misses span machines.</strong></p>
      <pre>UM253 A100 GPU  ◄──── RDMA 200Gbps / InfiniBand ────►  UM251 CPU LoRA store</pre>
      <table>
        <tr><th>Strategy</th><th>Flow</th><th>Network payload</th></tr>
        <tr><td>S1: Weight transfer</td><td>Remote CPU → RDMA → Local GPU → GPU compute</td><td><strong>R × (H+I) × 2B</strong> (large)</td></tr>
        <tr><td>S2: Activation transfer</td><td>Local GPU → CPU → RDMA → Remote CPU compute → RDMA → Local GPU</td><td><strong>(H + I) × 2B</strong> (small, but 2 RDMA round-trips)</td></tr>
        <tr><td>S3: Pre-cached + relay</td><td>Local GPU → RDMA → Remote relay → RDMA → Local GPU compute</td><td><strong>H × 2B</strong> (smallest, but needs prior weight transfer)</td></tr>
      </table>
      <blockquote><p>Key question: Which strategy wins, and how does EP background traffic affect the choice?</p></blockquote>
    </section>

    <section class="slide" aria-labelledby="slide-6-title">
      <div><span class="tag">Slide 6</span></div>
      <h2 id="slide-6-title">Cross-Node Results: Strategy Comparison</h2>
      <p><strong>Measured:</strong> UM253 A100 SXM4 ↔ UM251 CPU, HDR IB 200Gbps, GLOO backend.</p>
      <table>
        <tr><th>Rank</th><th>N_miss</th><th>S1 total (ms)</th><th>S2 total (ms)</th><th>S3 total (ms)</th></tr>
        <tr><td>16</td><td>1</td><td>1.37</td><td>0.89</td><td class="win">0.58</td></tr>
        <tr><td>16</td><td>4</td><td>1.82</td><td>0.69</td><td class="win">0.86</td></tr>
        <tr><td>64</td><td>1</td><td>1.81</td><td>1.81</td><td class="win">0.75</td></tr>
        <tr><td>64</td><td>2</td><td>3.50</td><td>3.09</td><td class="win">0.84</td></tr>
        <tr><td>64</td><td>4</td><td>6.81</td><td>4.68</td><td class="win">0.94</td></tr>
        <tr><td>128</td><td>4</td><td>6.65</td><td>1.58</td><td class="win">1.02</td></tr>
      </table>
      <ul>
        <li><strong>S3 is consistently fastest</strong>, 1.5–7× faster than S1.</li>
        <li>S1 scales poorly with rank.</li>
        <li>S2 suffers from multiple RDMA round-trips at high rank.</li>
        <li>The winning strategy depends on whether weights are already cached locally.</li>
      </ul>
    </section>

    <section class="slide" aria-labelledby="slide-7-title">
      <div><span class="tag">Slide 7</span></div>
      <h2 id="slide-7-title">EP Contention: Network Awareness Matters</h2>
      <p>Focused counter-validated run with Python IPoIB all-to-all traffic, R=64, N_miss=2, warmup=10, iters=50.</p>
      <table>
        <tr><th>EP load</th><th>S1 total (ms)</th><th>S2 total (ms)</th><th>S3 total (ms)</th></tr>
        <tr><td>0%</td><td>3.44</td><td>3.33</td><td>1.59</td></tr>
        <tr><td>25%</td><td>3.92</td><td>3.68</td><td class="win">0.81</td></tr>
        <tr><td>50%</td><td>3.85</td><td class="bad">6.84</td><td>1.64</td></tr>
        <tr><td>75%</td><td>4.57</td><td>5.68</td><td>1.55</td></tr>
        <tr><td>90%</td><td>3.20</td><td>4.94</td><td>1.22</td></tr>
      </table>
      <ul>
        <li>Counter validation confirms the background generator injects traffic on the IB port.</li>
        <li>S2 can degrade materially under contention: 3.33ms → 6.84ms at 50% EP load.</li>
        <li><strong>Do not claim a monotonic measured EP-degradation curve yet.</strong></li>
        <li>Next evidence should use a lower-level RDMA/QP all-to-all generator or real MoE traces.</li>
      </ul>
      <!-- Hidden from rendered deck: the old ib_write_bw table should not be used as evidence because it had invalid rate flags, -z misuse, one-shot server lifecycle bugs, and missing liveness checks. -->
    </section>

    <section class="slide" aria-labelledby="slide-7b-title">
      <div><span class="tag">Slide 7b</span></div>
      <h2 id="slide-7b-title">Insight: EP Load ≠ Effective Contention</h2>
      <p>Counter-validated traffic is present at every EP level, yet S2 latency jumps at 50% requested load and then improves at 75% and 90%.</p>
      <div class="grid">
        <div class="card">
          <h3>Why nominal load fails</h3>
          <ul>
            <li><strong>Direction:</strong> IB is full-duplex; traffic may not share the same path.</li>
            <li><strong>Burstiness and pacing:</strong> higher requested rate can reorganize bursts and gaps.</li>
            <li><strong>Bottleneck identity:</strong> bottlenecks can shift between IB bandwidth, CPU scheduling, DMA descriptors, and remote compute.</li>
            <li><strong>Sustained vs. instantaneous rate:</strong> requested load is a target, not a guaranteed rate during each benchmark window.</li>
          </ul>
        </div>
        <div class="card">
          <h3>Actionable pressure signals</h3>
          <ul>
            <li>IB port counter deltas over the last few milliseconds.</li>
            <li>RDMA CQ depth or posted-but-uncompleted work requests.</li>
            <li>Recent activation RTT over the same QP.</li>
            <li>Optional PCIe H2D/D2H busy fraction.</li>
          </ul>
        </div>
      </div>
      <blockquote><p>CoLoRA should react to measured, path-specific, time-local fabric pressure — not static EP load percentage.</p></blockquote>
    </section>

    <section class="slide" aria-labelledby="slide-8-title">
      <div><span class="tag">Slide 8</span></div>
      <h2 id="slide-8-title">CoLoRA Design: Hybrid Recovery Policy</h2>
      <pre>Miss event
  ├─ local GPU cache hit → warm GPU compute
  ├─ local cold miss
  │    ├─ N_dec ≤ 8  → CPU AVX compute
  │    └─ N_dec ≥ 16 → GPU H2D + compute
  └─ cross-machine miss
       ├─ pre-cached weights → S3 activation relay
       └─ fallback → S2 remote CPU activation compute</pre>
      <ul>
        <li><strong>Local warm hit:</strong> GPU matmul only, with weights already on GPU.</li>
        <li><strong>Local cold miss:</strong> CPU-first for N_dec ≤ 8; GPU path for N_dec ≥ 16.</li>
        <li><strong>Cross-machine:</strong> pre-cache weights during idle, use S3 relay at miss time, fallback to S2 if not cached.</li>
      </ul>
    </section>

    <section class="slide" aria-labelledby="slide-9-title">
      <div><span class="tag">Slide 9</span></div>
      <h2 id="slide-9-title">CoLoRA Design: Network-Aware Scheduling</h2>
      <p><strong>Key design principle:</strong> schedule on measured fabric pressure, not nominal EP load.</p>
      <pre>EP traffic pattern:     ████████░░░░░░████████░░░░░░████████░░░░░░
CoLoRA pre-cache:       ░░░░░░░░██████░░░░░░░░██████░░░░░░░░██████
At miss time:           activation relay only, no weight transfer needed</pre>
      <ol>
        <li><p><strong>Sense fabric pressure:</strong> IB counter deltas, RDMA CQ depth, activation RTT, and PCIe busy fraction.</p></li>
        <li><p><strong>Pre-cache in low-pressure windows:</strong> push popular cold LoRA weights to the local GPU in the background.</p></li>
        <li><p><strong>Pick the cheapest path at miss time:</strong> warm GPU, local CPU/GPU, S3 relay, S2 fallback, or S1 reusable transfer.</p></li>
      </ol>
      <blockquote><p>EP workload analysis and LoRA recovery are one shared-fabric scheduling problem.</p></blockquote>
    </section>
```

- [ ] **Step 2: Preview Slides 1-9**

Run:

```bash
python -m http.server 8000 --directory docs
```

Open `http://127.0.0.1:8000/colora-project-opening-slides.html`.
Expected: dashboard plus Slides 1-9 render. Scroll-snap should move between sections; tables and preformatted diagrams should be legible.

Stop the server with `Ctrl+C` after checking.

- [ ] **Step 3: Commit Slides 1-9 conversion**

Run:

```bash
git add docs/colora-project-opening-slides.html
git commit -m "docs(slides): convert CoLoRA opening narrative to HTML"
```

Expected: commit succeeds with only `docs/colora-project-opening-slides.html` modified.

---

### Task 3: Add Updated Roadmap, Takeaways, and Checklist Slides

**Files:**
- Modify: `docs/colora-project-opening-slides.html`
- Reference: `docs/superpowers/specs/2026-06-10-colora-html-opening-slides-design.md`

- [ ] **Step 1: Insert Slides 10-12 after Slide 9 and before `</main>`**

In `docs/colora-project-opening-slides.html`, insert this HTML after the Slide 9 `</section>` and before `</main>`:

```html
    <section class="slide" aria-labelledby="slide-10-title">
      <div><span class="tag">Slide 10</span></div>
      <h2 id="slide-10-title">Updated Experimental Roadmap</h2>
      <table>
        <tr><th>Study</th><th>Status</th><th>What it proves / remaining gap</th></tr>
        <tr>
          <td>Single-layer crossover</td>
          <td class="ok">Done</td>
          <td>CPU wins at N_dec ≤ 8 for typical decode batches; GPU wins at N_dec ≥ 16.</td>
        </tr>
        <tr>
          <td>Cross-node benchmark</td>
          <td class="ok">Done</td>
          <td>S3 is fastest when weights are pre-cached; S2 is sensitive to contention.</td>
        </tr>
        <tr>
          <td>Analytical / trace-driven case studies</td>
          <td class="ok">Mostly done</td>
          <td>Calibration anchor, system TPOT simulation, synthetic sweeps, small live validation, and figure assembly pipeline exist.</td>
        </tr>
        <tr>
          <td>EP contention evidence</td>
          <td class="warn">Active gap</td>
          <td>Current generator is counter-validated, but not a faithful RDMA/QP or real MoE all-to-all workload.</td>
        </tr>
        <tr>
          <td>Paper-facing claim audit</td>
          <td class="warn">Next</td>
          <td>Align claims, figures, and artifact paths before final writing.</td>
        </tr>
      </table>
      <blockquote><p>Current evidence is sufficient for motivation, challenge, and design. The main remaining risk is over-claiming EP contention without a stronger pressure study.</p></blockquote>
    </section>

    <section class="slide" aria-labelledby="slide-11-title">
      <div><span class="tag">Slide 11</span></div>
      <h2 id="slide-11-title">Key Takeaways</h2>
      <ol>
        <li><p><strong>LoRA miss recovery is the bottleneck</strong> in MoE + LoRA serving at practical cache budgets.</p></li>
        <li><p><strong>Local decode miss recovery is CPU-first</strong> at typical decode batches, especially N_dec ≤ 8.</p></li>
        <li><p><strong>Cross-machine recovery requires pre-caching</strong>; S3 is 1.5–7× faster than naive weight transfer.</p></li>
        <li><p><strong>EP load is a poor proxy for contention</strong>; CoLoRA should schedule on measured fabric pressure.</p></li>
        <li><p><strong>CoLoRA adapts to the regime</strong>: local CPU-first, local GPU for large batches, cross-machine pre-cache plus relay, and pressure-aware scheduling.</p></li>
      </ol>
    </section>

    <section class="slide" aria-labelledby="slide-12-title">
      <div><span class="tag">Slide 12</span></div>
      <h2 id="slide-12-title">Structured Next-step Checklist</h2>
      <div class="checklist-grid">
        <div class="card">
          <h3 class="ok">Done / Keep</h3>
          <ul>
            <li>Local CPU/GPU crossover evidence.</li>
            <li>Cross-node S1/S2/S3 comparison.</li>
            <li>Calibration anchor file.</li>
            <li>System TPOT and synthetic sweeps.</li>
            <li>Small live validation pipeline.</li>
            <li>Figure assembly pipeline.</li>
          </ul>
        </div>
        <div class="card">
          <h3 class="warn">Stale / Retire</h3>
          <ul>
            <li>“Case studies need anchor file.”</li>
            <li>“Minimal E2E validation is only planned.”</li>
            <li>Any monotonic EP-degradation claim.</li>
          </ul>
        </div>
        <div class="card">
          <h3>Active Next</h3>
          <ul>
            <li>RDMA/QP-level all-to-all or real MoE trace pressure study.</li>
            <li>Per-load IB counter summary.</li>
            <li>Paper claim-to-artifact audit.</li>
          </ul>
        </div>
        <div class="card">
          <h3>Decisions Needed</h3>
          <ul>
            <li>Invest in lower-level RDMA generator?</li>
            <li>Frame current EP result as qualitative contention evidence?</li>
            <li>Which claims are main-paper versus appendix?</li>
          </ul>
        </div>
      </div>
      <blockquote><p>Source-of-truth rule: this HTML deck is the maintained opening-slide artifact.</p></blockquote>
    </section>
```

- [ ] **Step 2: Preview updated Slides 10-12**

Run:

```bash
python -m http.server 8000 --directory docs
```

Open `http://127.0.0.1:8000/colora-project-opening-slides.html`.
Expected: Slides 10-12 render after Slide 9. Slide 10 does not say the anchor file is missing. Slide 12 has four structured checklist columns.

Stop the server with `Ctrl+C` after checking.

- [ ] **Step 3: Commit updated roadmap and checklist slides**

Run:

```bash
git add docs/colora-project-opening-slides.html
git commit -m "docs(slides): update roadmap and checklist in HTML deck"
```

Expected: commit succeeds with only `docs/colora-project-opening-slides.html` modified.

---

### Task 4: Remove Markdown Source and Verify Single Source of Truth

**Files:**
- Delete: `docs/colora-project-opening-slides.md`
- Modify: no other files expected

- [ ] **Step 1: Delete the markdown deck**

Run:

```bash
rm docs/colora-project-opening-slides.md
```

Expected: file is removed from the working tree.

- [ ] **Step 2: Confirm the HTML file is the only opening-slide artifact**

Run:

```bash
python - <<'PY'
from pathlib import Path
html = Path('docs/colora-project-opening-slides.html')
md = Path('docs/colora-project-opening-slides.md')
assert html.exists(), 'HTML deck is missing'
assert not md.exists(), 'Markdown deck still exists'
text = html.read_text(encoding='utf-8')
required = [
    'CoLoRA Opening Deck TODOs',
    'Slide 1',
    'Updated Experimental Roadmap',
    'Structured Next-step Checklist',
    'RDMA/QP-level all-to-all or real MoE trace pressure study',
    'Any monotonic EP-degradation claim should be retired',
]
missing = [item for item in required if item not in text]
assert not missing, f'Missing required HTML content: {missing}'
print('HTML deck single-source check passed')
PY
```

Expected output:

```text
HTML deck single-source check passed
```

- [ ] **Step 3: Check repository status**

Run:

```bash
git status --short
```

Expected output includes:

```text
 D docs/colora-project-opening-slides.md
```

and includes `docs/colora-project-opening-slides.html` as modified or added.

- [ ] **Step 4: Commit markdown removal**

Run:

```bash
git add docs/colora-project-opening-slides.md docs/colora-project-opening-slides.html
git commit -m "docs(slides): make HTML deck the opening source of truth"
```

Expected: commit succeeds and includes deletion of the markdown file.

---

### Task 5: Final Manual Review and Preview

**Files:**
- Read/verify: `docs/colora-project-opening-slides.html`

- [ ] **Step 1: Run final content check**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('docs/colora-project-opening-slides.html').read_text(encoding='utf-8')
checks = {
    'dashboard before slides': text.index('CoLoRA Opening Deck TODOs') < text.index('Slide 1'),
    'no markdown source': not Path('docs/colora-project-opening-slides.md').exists(),
    'has all 12 slides': all(f'Slide {i}' in text for i in range(1, 13)),
    'has visible checklist': 'Structured Next-step Checklist' in text,
    'hidden old callout preserved': 'Hidden from rendered deck' in text,
}
failed = [name for name, ok in checks.items() if not ok]
assert not failed, f'Final checks failed: {failed}'
print('Final HTML deck checks passed')
PY
```

Expected output:

```text
Final HTML deck checks passed
```

- [ ] **Step 2: Serve final deck locally**

Run:

```bash
python -m http.server 8000 --directory docs
```

Open `http://127.0.0.1:8000/colora-project-opening-slides.html`.
Expected:

- The dashboard appears first.
- Slide 1 is the CoLoRA title slide.
- Slides 2-9 preserve the technical story.
- Slide 10 uses updated roadmap status.
- Slide 12 shows the structured checklist.
- Hidden implementation notes do not appear in the browser.

Stop the server with `Ctrl+C` after checking.

- [ ] **Step 3: Verify clean working tree**

Run:

```bash
git status --short
```

Expected: no output.

If there is output, inspect it with:

```bash
git diff --stat
```

Only commit if the remaining output is part of this HTML deck conversion.

- [ ] **Step 4: Report final result**

Tell the user:

```text
Implemented the HTML source-of-truth opening deck. The maintained file is docs/colora-project-opening-slides.html; the markdown source was removed. Final checks passed and the working tree is clean.
```

---

## Self-Review

Spec coverage:

- Create `docs/colora-project-opening-slides.html`: Task 1.
- Remove `docs/colora-project-opening-slides.md`: Task 4.
- Preserve slide narrative and evidence: Task 2 and Task 3.
- Visible dashboard card before Slide 1: Task 1.
- Update Slides 10 and 12: Task 3.
- Comment out unnecessary call-outs instead of deleting: Task 2 uses HTML comments for old detailed notes.
- No build system or external dependencies: all tasks use static HTML and inline CSS.
- Manual review and repository checks: Task 5.

Placeholder scan:

- No `TBD`, `TODO`, `implement later`, or unresolved placeholder instructions remain.
- Every code-changing step includes exact content or exact commands.

Type/path consistency:

- The plan consistently uses `docs/colora-project-opening-slides.html` as the HTML artifact.
- The plan consistently removes `docs/colora-project-opening-slides.md`.
- All verification snippets use the same paths and expected strings.
