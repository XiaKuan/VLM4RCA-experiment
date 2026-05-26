from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vlm4rca.candidates.metric_builder import build_metric_candidates_for_case
from vlm4rca.candidates.models import MetricCandidateBuildResult, RcaCandidate, VariantRecallSummary
from vlm4rca.evaluation.recall import (
    candidates_by_case_to_targets,
    evaluate_component_recall_at_k,
    summarize_variant_recall,
)
from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.manifest import load_case_manifest
from vlm4rca.openrca.models import Phase1CaseSidecar
from vlm4rca.openrca.phase1 import build_phase1_sidecars


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _candidate_payload(result: MetricCandidateBuildResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def build_metric_baseline_report_markdown(report_row: VariantRecallSummary) -> str:
    return "\n".join(
        [
            "# Metric-only Baseline Report",
            "",
            "| Variant | Recall@3 | Recall@5 | Recall@8 | Soft Recall@8 | Avg Candidates |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| {report_row.variant} | "
                f"{report_row.recall_at_3:.3f} | "
                f"{report_row.recall_at_5:.3f} | "
                f"{report_row.recall_at_8:.3f} | "
                f"{report_row.soft_recall_at_8:.3f} | "
                f"{report_row.avg_candidates:.2f} |"
            ),
            "",
        ]
    )


def _build_metric_results(
    adapter: OpenRCABankAdapter,
    sidecars: Sequence[Phase1CaseSidecar],
    max_candidates: int,
) -> dict[str, MetricCandidateBuildResult]:
    results: dict[str, MetricCandidateBuildResult] = {}
    for sidecar in sidecars:
        metrics_path = adapter.telemetry_path(sidecar.case_id, "metrics")
        results[sidecar.case_id] = build_metric_candidates_for_case(
            case_id=sidecar.case_id,
            metrics_path=metrics_path,
            windows=sidecar.incident_window,
            max_candidates=max_candidates,
        )
    return results


def run_metric_baseline(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    max_candidates: int = 8,
) -> dict[str, Any]:
    manifest = load_case_manifest(manifest_path)
    adapter = OpenRCABankAdapter(data_root)
    sidecars = build_phase1_sidecars(adapter, manifest)
    metric_results = _build_metric_results(adapter, sidecars, max_candidates=max_candidates)
    candidates_by_case: dict[str, list[RcaCandidate]] = {
        case_id: result.candidates for case_id, result in metric_results.items()
    }

    targets_by_case = candidates_by_case_to_targets(candidates_by_case)
    recall_summary = evaluate_component_recall_at_k(sidecars, targets_by_case, ks=[3, 5, 8])
    report_row = summarize_variant_recall(
        "M", sidecars, candidates_by_case, targets_by_case=targets_by_case
    )
    markdown = build_metric_baseline_report_markdown(report_row)

    for case_id, result in metric_results.items():
        _write_json(output_dir / "candidates" / "M" / f"{case_id}.json", _candidate_payload(result))

    _write_json(output_dir / "recall_M.json", recall_summary.model_dump(mode="json"))
    _write_json(output_dir / "report_M.json", report_row.model_dump(mode="json"))
    _write_json(
        output_dir / "candidate_rankings_M.json",
        {case_id: targets for case_id, targets in sorted(targets_by_case.items())},
    )
    (output_dir / "baseline_report.md").write_text(markdown, encoding="utf-8")

    return {
        "metric_results": {
            case_id: result.model_dump(mode="json")
            for case_id, result in sorted(metric_results.items())
        },
        "recall_summary": recall_summary.model_dump(mode="json"),
        "report_row": report_row.model_dump(mode="json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenRCA Phase 2 Metric-only baseline")
    parser.add_argument("--manifest", type=Path, default=Path("configs/openrca_pilot15.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/OpenRCA/Bank"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/openrca_phase2_metric_pilot15")
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        run_metric_baseline(
            manifest_path=args.manifest,
            data_root=args.data_root,
            output_dir=args.output_dir,
            max_candidates=args.max_candidates,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
