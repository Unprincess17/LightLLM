# CoLoRA HTML Opening Slides Design

## Goal

Convert `docs/colora-project-opening-slides.md` into a presentation-first HTML deck and make the HTML file the only maintained source of truth for the CoLoRA opening slides, roadmap, and next-step checklist.

## Scope

In scope:

- Create `docs/colora-project-opening-slides.html` as a self-contained HTML presentation.
- Remove `docs/colora-project-opening-slides.md` after conversion.
- Preserve the current slide narrative and evidence content.
- Add a visible project-status dashboard card before Slide 1 so Claude and human readers can quickly see current TODOs without re-reading the full background.
- Update stale roadmap/TODO language in Slides 10 and 12.
- Remove unnecessary comments, meta call-outs, and implementation notes that distract from the opening-deck narrative.

Out of scope:

- Re-running experiments.
- Changing benchmark scripts.
- Creating a separate checklist document.
- Maintaining parallel markdown and HTML slide sources.

## Artifact Design

`docs/colora-project-opening-slides.html` is the single maintained artifact. It should be directly openable in a browser with no build step and should be publishable through the visual companion server for quick review.

The deck remains presentation-first:

1. A compact project-status dashboard appears at the very top.
2. Slide 1 remains the title/opening slide.
3. Slides 2-9 preserve the current technical motivation, measurements, and design narrative.
4. Slide 10 becomes the updated experimental roadmap.
5. Slide 11 remains key takeaways.
6. Slide 12 becomes the structured next-step checklist.

The markdown file is removed after the HTML conversion so there is no stale duplicate source.

## Dashboard Card

The HTML file starts with a visible dashboard card before Slide 1. It should be compact enough not to feel like a separate slide, but explicit enough that Claude can inspect the top of the file and understand current project status.

The dashboard contains four groups:

- Done
- Stale / Retire
- Active Next
- Decisions Needed

The card can use a simple visible layout; it does not need JavaScript. If expand/collapse behavior is used, prefer native HTML such as `<details>` / `<summary>`.

## Checklist Content

### Done / Keep

- Local CPU/GPU crossover is measured and optimized; CPU wins for typical decode batches up to `N_dec <= 8`.
- Cross-node S1/S2/S3 benchmark is collected, including the 60/60 configurations and focused EP run.
- Calibration anchor file exists.
- System TPOT simulation and synthetic sweeps exist.
- Small live validation pipeline exists.
- Figure assembly pipeline exists.

### Stale / Retire

- “Case studies need anchor file” is stale because the calibration anchor exists.
- “Minimal E2E validation is only planned” is stale because the B11 live-validation pipeline exists.
- Any monotonic EP-degradation claim should be retired unless supported by a stronger generator or trace-backed study.

### Active Next

- Run an RDMA/QP-level all-to-all or real MoE trace pressure study.
- Produce per-load IB counter summaries.
- Run a paper claim-to-artifact audit before final writing.

### Decisions Needed

- Decide whether to invest in a lower-level RDMA generator or frame the current EP result as qualitative contention evidence.
- Decide which claims belong in the main paper versus appendix.

## Roadmap Updates

Slide 10 should no longer say the analytical case studies need an anchor file. It should state that the case-study and TPOT pipeline is mostly done, with remaining work focused on paper-facing validation and claim alignment.

Slide 12 should no longer list generic stale next steps. It should become the structured checklist above.

## Cleanup Rules

Comment out unnecessary words and call-outs from the presentation content using HTML comments (`<!-- -->`) rather than deleting them. This preserves the original reasoning for future reference while keeping the rendered slides clean. Keep visible only remarks that affect interpretation of results or prevent an incorrect claim, such as the warning not to claim a monotonic EP-degradation curve.

Do not add hidden TODO comments that duplicate the visible dashboard. The visible dashboard is the canonical checklist.

## Testing and Review

Manual review is sufficient for this conversion:

- Open the HTML deck in a browser.
- Confirm each slide renders legibly.
- Confirm the dashboard appears before Slide 1.
- Confirm Slide 10 and Slide 12 reflect current project status.
- Confirm the markdown file is removed.
- Confirm there is no duplicate markdown source left to maintain.

A lightweight repository check should also be run:

- `git status --short`
- `python -m py_compile` is not needed because this task does not modify Python code.

## Implementation Constraints

- Do not add a build system.
- Do not add external CSS or JavaScript dependencies.
- Do not create a separate checklist file.
- Do not preserve the markdown deck as a second maintained source.
