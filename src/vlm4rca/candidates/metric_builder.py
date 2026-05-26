from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from vlm4rca.candidates.metric_features import score_metrics_dataframe
from vlm4rca.candidates.models import (
    METRIC_CANDIDATE_BUDGET,
    MetricCandidateBuildResult,
    MetricCategory,
    MetricFeatureEvidence,
    RcaCandidate,
)
from vlm4rca.openrca.models import IncidentWindows

MAX_EVIDENCE_CATEGORIES = 3


def _evidence_sort_key(evidence: MetricFeatureEvidence) -> tuple[float, str]:
    return (-evidence.score, evidence.metric_column)


def _select_top_evidence_by_category(
    evidence_items: list[MetricFeatureEvidence],
) -> list[MetricFeatureEvidence]:
    best_by_category: dict[MetricCategory, MetricFeatureEvidence] = {}
    for evidence in sorted(evidence_items, key=_evidence_sort_key):
        current = best_by_category.get(evidence.metric_category)
        if current is None or evidence.score > current.score:
            best_by_category[evidence.metric_category] = evidence
    return sorted(best_by_category.values(), key=_evidence_sort_key)[:MAX_EVIDENCE_CATEGORIES]


def _candidate_score(evidence_items: list[MetricFeatureEvidence]) -> float:
    top_evidence = _select_top_evidence_by_category(evidence_items)
    return round(sum(item.score for item in top_evidence), 6)


def _evidence_summary(evidence_items: list[MetricFeatureEvidence]) -> list[str]:
    summaries: list[str] = []
    for evidence in _select_top_evidence_by_category(evidence_items):
        summaries.append(
            f"{evidence.metric_category}: {evidence.metric_column} "
            f"robust_z={evidence.robust_z_score:.2f} "
            f"relative_change={evidence.relative_change:.2f} "
            f"p95_shift={evidence.p95_shift:.2f}"
        )
    return summaries


def build_metric_candidates_from_dataframe(
    case_id: str,
    metrics: pd.DataFrame,
    windows: IncidentWindows,
    max_candidates: int = METRIC_CANDIDATE_BUDGET,
) -> MetricCandidateBuildResult:
    scored_metrics, warnings = score_metrics_dataframe(metrics, windows)
    grouped: dict[str, list[MetricFeatureEvidence]] = defaultdict(list)
    raw_targets: dict[str, str] = {}

    for evidence in scored_metrics:
        grouped[evidence.canonical_target].append(evidence)
        raw_targets.setdefault(evidence.canonical_target, evidence.raw_target)

    ranked_components = sorted(
        grouped.items(),
        key=lambda item: (
            -_candidate_score(item[1]),
            -len(item[1]),
            item[0],
        ),
    )

    candidates: list[RcaCandidate] = []
    for rank, (canonical_target, evidence_items) in enumerate(
        ranked_components[:max_candidates],
        start=1,
    ):
        candidate_key = f"service:{canonical_target}"
        candidates.append(
            RcaCandidate(
                case_id=case_id,
                variant="M",
                candidate_key=candidate_key,
                variant_candidate_id=f"cand:{case_id}:M:{rank}:service:{canonical_target}",
                target_type="service",
                canonical_target=canonical_target,
                raw_target=raw_targets[canonical_target],
                introduced_by="metric",
                metric_only_present=True,
                present_in_variants=["M"],
                sources=["metric"],
                source_scores={"metric": _candidate_score(evidence_items)},
                evidence_summary=_evidence_summary(evidence_items),
                rank=rank,
                selected_for_rendering=True,
            )
        )

    return MetricCandidateBuildResult(
        case_id=case_id,
        variant="M",
        candidates=candidates,
        warnings=warnings,
    )


def build_metric_candidates_for_case(
    case_id: str,
    metrics_path: Path,
    windows: IncidentWindows,
    max_candidates: int = METRIC_CANDIDATE_BUDGET,
) -> MetricCandidateBuildResult:
    metrics = pd.read_csv(metrics_path)
    return build_metric_candidates_from_dataframe(
        case_id=case_id,
        metrics=metrics,
        windows=windows,
        max_candidates=max_candidates,
    )
