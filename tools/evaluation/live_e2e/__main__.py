import argparse
from pathlib import Path
import sys

from tools.evaluation.live_e2e.runner import run_manifest
from tools.evaluation.live_e2e.summarize import collect_and_summarize_manifest
from tools.evaluation.live_e2e.runner import get_summaries_dir
from tools.evaluation.live_e2e.manifest import load_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Live E2E evaluation runner for COLoRA paper evidence."
    )
    parser.add_argument(
        "--manifest", "-m",
        required=True,
        type=Path,
        help="Path to YAML manifest file describing the evaluation runs",
    )
    parser.add_argument(
        "--summarize-only", "-s",
        action="store_true",
        help="Only run summarization from existing results, do not re-execute runs",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.summarize_only:
        # Load manifest to get run_id and output location
        manifest = load_manifest(args.manifest)
        summaries_dir = get_summaries_dir(manifest)
        results_json_path = summaries_dir / "manifest_results.json"
        if not results_json_path.exists():
            print(f"Error: manifest_results.json not found at {results_json_path}. Run the full evaluation first.", file=sys.stderr)
            sys.exit(1)
        summary = collect_and_summarize_manifest(
            manifest_run_id=manifest.run_id,
            results_json_path=results_json_path,
            output_root=summaries_dir,
        )
        print(f"\nSummary complete:")
        print(f"  Total runs: {summary['total_runs']}")
        print(f"  Valid runs: {summary['valid_runs']}")
        print(f"  Output: {summaries_dir / 'live_e2e_summary.json'}")
        print(f"  Comparison CSV: {summaries_dir / 'live_e2e_comparison.csv'}")
        sys.exit(0)
    else:
        # Run all the evaluation
        results = run_manifest(args.manifest)
        manifest = load_manifest(args.manifest)
        summaries_dir = get_summaries_dir(manifest)
        # After running, automatically do the summarization
        results_json_path = summaries_dir / "manifest_results.json"
        summary = collect_and_summarize_manifest(
            manifest_run_id=manifest.run_id,
            results_json_path=results_json_path,
            output_root=summaries_dir,
        )
        print(f"\nEvaluation and summarization complete:")
        print(f"  Total runs executed: {len(results)}")
        print(f"  Valid runs executed: {sum(1 for r in results if r['valid'])}")
        print(f"  Summary: {summaries_dir / 'live_e2e_summary.json'}")
        print(f"  Paper comparison CSV: {summaries_dir / 'live_e2e_comparison.csv'}")
        # Exit with non-zero if no valid paper runs
        valid_paper = sum(
            1 for s in summary["summaries"]
            if s.get("valid") and s.get("suite_kind") == "paper"
        )
        expected_paper = sum(
            1 for r in manifest.runs
            if r.suite_kind == "paper"
        )
        if valid_paper < expected_paper:
            print(f"\nWarning: Only {valid_paper}/{expected_paper} paper runs completed successfully.", file=sys.stderr)
            if valid_paper == 0:
                sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
