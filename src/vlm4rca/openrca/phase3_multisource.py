from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from vlm4rca.candidates.log_builder import build_log_candidates_for_case
from vlm4rca.candidates.merge_rank import (
    build_variant_result,
    update_present_in_variants,
)
from vlm4rca.candidates.metric_builder import build_metric_candidates_for_case
from vlm4rca.candidates.models import RcaCandidate
from vlm4rca.candidates.multisource_models import (
    MultiSourceVariantResult,
    ShadowEdgeCandidate,
)
from vlm4rca.candidates.topology_builder import (
    expand_topology_candidates,
    load_static_topology_edges,
    topology_edges_from_shadow_edges,
)
from vlm4rca.candidates.trace_builder import build_trace_candidates_for_case
from vlm4rca.evaluation.ablation import (
    VARIANT_CHAIN,
    build_candidate_recall_ablation,
    build_checkpoint_decision,
    build_first_hit_rows,
    summarize_shadow_edges,
)
from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.manifest import load_case_manifest
from vlm4rca.openrca.models import CaseManifest, Phase1CaseSidecar
from vlm4rca.openrca.phase1 import build_phase1_sidecars


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_case_candidates(
    sidecar: Phase1CaseSidecar,
    adapter: OpenRCABankAdapter,
    *,
    trace_edges: list[tuple[str, str]] | None = None,
) -> tuple[
    list[RcaCandidate],
    list[ShadowEdgeCandidate],
    list,
    list,
    list[tuple[str, str]],
]:
    """Build metric, trace, log candidates and topology edges for one case.

    Returns (metric_candidates, shadow_edges, trace_source_candidates,
    log_source_candidates, topology_edges).
    """
    windows = sidecar.incident_window

    # --- metric ---
    metrics_result = build_metric_candidates_for_case(
        sidecar.case_id,
        adapter.telemetry_path(sidecar.case_id, "metrics"),
        windows,
    )
    metric_candidates = list(metrics_result.candidates)

    # --- trace ---
    trace_source_candidates: list = []
    shadow_edges: list[ShadowEdgeCandidate] = []
    if sidecar.modality_availability.traces_available:
        trace_result = build_trace_candidates_for_case(
            sidecar.case_id,
            adapter.telemetry_path(sidecar.case_id, "traces"),
            windows,
        )
        trace_source_candidates = list(trace_result.source_candidates)
        shadow_edges = list(trace_result.shadow_edges)

    # --- log ---
    log_source_candidates: list = []
    if sidecar.modality_availability.logs_available:
        log_result = build_log_candidates_for_case(
            sidecar.case_id,
            adapter.telemetry_path(sidecar.case_id, "logs"),
            windows,
        )
        log_source_candidates = list(log_result.source_candidates)

    # --- topology edges ---
    topo_edges: list[tuple[str, str]] = []
    if sidecar.modality_availability.topology_available:
        if sidecar.modality_availability.topology_source == "static_topology":
            static_path = adapter.telemetry_path(sidecar.case_id, "topology")
            if static_path.exists() and static_path.stat().st_size > 0:
                topo_edges = load_static_topology_edges(static_path)
        elif shadow_edges:
            topo_edges = topology_edges_from_shadow_edges(shadow_edges)

    return (
        metric_candidates,
        shadow_edges,
        trace_source_candidates,
        log_source_candidates,
        topo_edges,
    )


def _build_topology_source_candidates(
    case_id: str,
    metric_candidates: list[RcaCandidate],
    topo_edges: list[tuple[str, str]],
) -> list:
    """Build topology-expanded source candidates from metric seed candidates."""
    if not topo_edges:
        return []
    return expand_topology_candidates(case_id, list(metric_candidates), topo_edges)


def _build_variants_for_case(
    sidecar: Phase1CaseSidecar,
    metric_candidates: list[RcaCandidate],
    shadow_edges: list[ShadowEdgeCandidate],
    trace_source_candidates: list,
    log_source_candidates: list,
    topo_source_candidates: list,
) -> dict[str, MultiSourceVariantResult]:
    """Build all four variant results for a single case."""
    all_non_metric = [*trace_source_candidates, *log_source_candidates, *topo_source_candidates]
    results: dict[str, MultiSourceVariantResult] = {}

    for variant in VARIANT_CHAIN:
        variant_non_metric: list = []
        if variant in ("M+T", "M+T+L", "M+T+L+Topo"):
            variant_non_metric.extend(trace_source_candidates)
        if variant in ("M+T+L", "M+T+L+Topo"):
            variant_non_metric.extend(log_source_candidates)
        if variant == "M+T+L+Topo":
            variant_non_metric.extend(topo_source_candidates)

        result = build_variant_result(
            case_id=sidecar.case_id,
            variant=variant,
            metric_candidates=metric_candidates,
            source_candidates=variant_non_metric,
            shadow_edges=shadow_edges,
        )
        results[variant] = result

    return results


def _build_ablation_markdown_row(variant: str, row: dict) -> str:
    label = {
        "M": "Metric",
        "M+T": "Metric+Trace",
        "M+T+L": "Metric+Trace+Log",
        "M+T+L+Topo": "Full",
    }.get(variant, variant)
    return (
        f"| {label} | {row['recall_at_3']:.3f} | {row['recall_at_5']:.3f} "
        f"| {row['recall_at_8']:.3f} | {row['soft_recall_at_8']:.3f} "
        f"| {row['avg_candidates']:.2f} |"
    )


def build_candidate_recall_markdown(ablation: dict) -> str:
    """Build markdown report from ablation dict."""
    lines: list[str] = []

    for section_key, title in (("primary", "Primary Analysis"), ("secondary", "Secondary Analysis")):
        section = ablation.get(section_key, {})
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Variant | Recall@3 | Recall@5 | Recall@8 | Soft@8 | Avg Cands |")
        lines.append("|---------|----------|----------|----------|--------|-----------|")
        for variant in VARIANT_CHAIN:
            row = section.get(variant)
            if row is not None:
                lines.append(_build_ablation_markdown_row(variant, row))
        lines.append("")

    return "\n".join(lines)


def run_multisource_candidate_recall(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
) -> dict:
    """Run the full Phase 3 multi-source candidate recall pipeline.

    Returns a dict containing ablation, first_hit_rows, checkpoint_decision,
    and shadow_edge_metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_case_manifest(manifest_path)
    effective_root = Path(manifest.data_root) if data_root is None else data_root
    adapter = OpenRCABankAdapter(effective_root)
    sidecars = build_phase1_sidecars(adapter, manifest)

    # --- Build per-case candidates ---
    per_case_metric: dict[str, list[RcaCandidate]] = defaultdict(list)
    per_case_shadow_edges: dict[str, list[ShadowEdgeCandidate]] = defaultdict(list)
    per_case_trace_src: dict[str, list] = defaultdict(list)
    per_case_log_src: dict[str, list] = defaultdict(list)
    per_case_topo_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for sidecar in sidecars:
        (
            metric_candidates,
            shadow_edges,
            trace_src,
            log_src,
            topo_edges,
        ) = _build_case_candidates(sidecar, adapter)
        per_case_metric[sidecar.case_id] = metric_candidates
        per_case_shadow_edges[sidecar.case_id] = shadow_edges
        per_case_trace_src[sidecar.case_id] = trace_src
        per_case_log_src[sidecar.case_id] = log_src
        per_case_topo_edges[sidecar.case_id] = topo_edges

    # --- Build variants for each case ---
    all_variants: dict[str, dict[str, MultiSourceVariantResult]] = defaultdict(dict)
    for sidecar in sidecars:
        case_id = sidecar.case_id
        topo_src = _build_topology_source_candidates(
            case_id,
            per_case_metric[case_id],
            per_case_topo_edges[case_id],
        )
        case_variants = _build_variants_for_case(
            sidecar,
            per_case_metric[case_id],
            per_case_shadow_edges[case_id],
            per_case_trace_src[case_id],
            per_case_log_src[case_id],
            topo_src,
        )
        for variant, result in case_variants.items():
            all_variants[variant][case_id] = result

    # --- Update present_in_variants across all cases ---
    for variant in VARIANT_CHAIN:
        all_variants[variant] = update_present_in_variants(all_variants[variant])

    # --- Write per-variant candidate JSON files ---
    candidates_dir = output_dir / "candidates"
    for variant in VARIANT_CHAIN:
        variant_dir = candidates_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        for case_id, result in all_variants[variant].items():
            _write_json(
                variant_dir / f"{case_id}.json",
                {
                    "case_id": case_id,
                    "variant": variant,
                    "candidates": [c.model_dump(mode="json") for c in result.candidates],
                },
            )

    # --- Build ablation evaluation ---
    candidates_by_variant: dict[str, dict[str, list[RcaCandidate]]] = {}
    for variant in VARIANT_CHAIN:
        candidates_by_variant[variant] = {
            case_id: list(result.candidates)
            for case_id, result in all_variants[variant].items()
        }

    ablation = build_candidate_recall_ablation(sidecars, candidates_by_variant)

    # --- Write ablation JSON ---
    _write_json(output_dir / "candidate_recall_ablation.json", ablation)

    # --- Write ablation Markdown ---
    markdown = build_candidate_recall_markdown(ablation)
    (output_dir / "candidate_recall_ablation.md").write_text(markdown, encoding="utf-8")

    # --- Write first-hit sources ---
    first_hit_rows = build_first_hit_rows(sidecars, candidates_by_variant)
    _write_json(output_dir / "first_hit_sources.json", first_hit_rows)

    # --- Write checkpoint decision ---
    checkpoint = build_checkpoint_decision(sidecars, candidates_by_variant)
    _write_json(output_dir / "checkpoint_decision.json", checkpoint)

    # --- Write shadow edge metrics ---
    relevant_edges_by_case: dict[str, list[str]] = {}
    for sidecar in sidecars:
        gt_components = {mapping.mapped_component for mapping in sidecar.gt_mapping}
        edges = per_case_shadow_edges.get(sidecar.case_id, [])
        relevant = [
            edge.edge_key
            for edge in edges
            if edge.caller in gt_components or edge.callee in gt_components
        ]
        relevant_edges_by_case[sidecar.case_id] = relevant

    shadow_summary = summarize_shadow_edges(relevant_edges_by_case, per_case_shadow_edges)
    _write_json(output_dir / "shadow_edge_metrics.json", shadow_summary)

    return {
        "ablation": ablation,
        "first_hit_rows": first_hit_rows,
        "checkpoint_decision": checkpoint,
        "shadow_edge_metrics": shadow_summary,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenRCA Phase 3 multi-source candidate recall ablation."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_multisource_candidate_recall(args.manifest, args.data_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
