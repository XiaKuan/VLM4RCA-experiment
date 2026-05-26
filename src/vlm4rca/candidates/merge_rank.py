from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from vlm4rca.candidates.models import CandidateSource, RcaCandidate, VariantName
from vlm4rca.candidates.multisource_models import (
    MultiSourceVariantResult,
    ShadowEdgeCandidate,
    SourceBucket,
    SourceCandidate,
    SourceQuotaConfig,
)

VARIANT_BUCKETS: dict[VariantName, tuple[SourceBucket, ...]] = {
    "M": ("metric",),
    "M+T": ("metric", "trace_service", "trace_edge_projected_service"),
    "M+T+L": ("metric", "trace_service", "trace_edge_projected_service", "log"),
    "M+T+L+Topo": ("metric", "trace_service", "trace_edge_projected_service", "log", "topology"),
}

SOURCE_PRIORITY: dict[CandidateSource, int] = {
    "metric": 0,
    "trace": 1,
    "log": 2,
    "topology": 3,
}

SOURCE_ORDER: tuple[CandidateSource, ...] = ("metric", "trace", "log", "topology")


def convert_metric_candidates_to_source_candidates(
    metric_candidates: Iterable[RcaCandidate],
) -> list[SourceCandidate]:
    converted: list[SourceCandidate] = []
    for candidate in metric_candidates:
        converted.append(
            SourceCandidate(
                case_id=candidate.case_id,
                candidate_key=candidate.candidate_key,
                target_type=candidate.target_type,
                canonical_target=candidate.canonical_target,
                raw_target=candidate.raw_target,
                source="metric",
                source_bucket="metric",
                score=float(candidate.source_scores.get("metric", 0.0)),
                evidence_summary=list(candidate.evidence_summary),
                source_refs=[f"metric:{candidate.canonical_target}"],
            )
        )
    return converted


def _normalize_and_protect(
    candidates: list[SourceCandidate],
    quotas: SourceQuotaConfig,
) -> list[SourceCandidate]:
    by_bucket: dict[SourceBucket, list[SourceCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_bucket[candidate.source_bucket].append(candidate)

    normalized: list[SourceCandidate] = []
    for bucket, bucket_candidates in by_bucket.items():
        ordered = sorted(
            bucket_candidates, key=lambda candidate: (-candidate.score, candidate.canonical_target)
        )
        max_score = max((candidate.score for candidate in ordered), default=0.0)
        denominator = max_score if max_score > 0.0 else 1.0
        quota = quotas.quota_for(bucket)
        for index, candidate in enumerate(ordered):
            normalized.append(
                candidate.model_copy(
                    update={
                        "normalized_score": round(candidate.score / denominator, 6),
                        "protected": index < quota,
                    }
                )
            )
    return normalized


def _introduced_by(sources: set[CandidateSource]) -> CandidateSource:
    for source in SOURCE_ORDER:
        if source in sources:
            return source
    return "metric"


def _source_sort(sources: set[CandidateSource]) -> tuple[CandidateSource, ...]:
    return tuple(source for source in SOURCE_ORDER if source in sources)


def _merge_source_candidates(
    case_id: str,
    variant: VariantName,
    source_candidates: list[SourceCandidate],
    metric_only_keys: set[str],
    max_final_candidates: int,
) -> list[RcaCandidate]:
    grouped: dict[str, list[SourceCandidate]] = defaultdict(list)
    for candidate in source_candidates:
        if candidate.target_type != "edge":
            grouped[candidate.candidate_key].append(candidate)

    merged_rows: list[dict[str, object]] = []
    for candidate_key, items in grouped.items():
        sources = {item.source for item in items}
        source_scores: dict[CandidateSource, float] = {}
        for source in sources:
            source_scores[source] = round(
                max(item.normalized_score for item in items if item.source == source),
                6,
            )
        representative = sorted(
            items, key=lambda item: (-item.normalized_score, item.canonical_target)
        )[0]
        evidence: list[str] = []
        for item in sorted(
            items,
            key=lambda item: (
                SOURCE_PRIORITY[item.source],
                -item.normalized_score,
                item.canonical_target,
            ),
        ):
            evidence.extend(item.evidence_summary)
        merged_rows.append(
            {
                "candidate_key": candidate_key,
                "target_type": representative.target_type,
                "canonical_target": representative.canonical_target,
                "raw_target": representative.raw_target,
                "introduced_by": _introduced_by(sources),
                "metric_only_present": candidate_key in metric_only_keys,
                "sources": _source_sort(sources),
                "source_scores": source_scores,
                "evidence_summary": evidence[:6],
                "protected": any(item.protected for item in items),
                "aggregate_score": round(sum(source_scores.values()), 6),
                "source_priority": min(SOURCE_PRIORITY[source] for source in sources),
            }
        )

    def row_sort_key(row: dict[str, object]) -> tuple[int, float, int, str]:
        return (
            -len(row["sources"]),  # type: ignore[arg-type]
            -float(row["aggregate_score"]),
            int(row["source_priority"]),
            str(row["canonical_target"]),
        )

    protected = sorted([row for row in merged_rows if bool(row["protected"])], key=row_sort_key)
    non_protected = sorted(
        [row for row in merged_rows if not bool(row["protected"])], key=row_sort_key
    )
    ordered = (protected + non_protected)[:max_final_candidates]

    candidates: list[RcaCandidate] = []
    for rank, row in enumerate(ordered, start=1):
        canonical_target = str(row["canonical_target"])
        target_type = str(row["target_type"])
        candidates.append(
            RcaCandidate(
                case_id=case_id,
                variant=variant,
                candidate_key=str(row["candidate_key"]),
                variant_candidate_id=f"cand:{case_id}:{variant}:{rank}:{target_type}:{canonical_target}",
                target_type=target_type,  # type: ignore[arg-type]
                canonical_target=canonical_target,
                raw_target=str(row["raw_target"]),
                introduced_by=row["introduced_by"],  # type: ignore[arg-type]
                metric_only_present=bool(row["metric_only_present"]),
                present_in_variants=(variant,),
                sources=row["sources"],  # type: ignore[arg-type]
                source_scores=row["source_scores"],  # type: ignore[arg-type]
                evidence_summary=row["evidence_summary"],  # type: ignore[arg-type]
                rank=rank,
                selected_for_rendering=True,
            )
        )
    return candidates


def build_variant_result(
    case_id: str,
    variant: VariantName,
    metric_candidates: list[RcaCandidate],
    source_candidates: list[SourceCandidate],
    shadow_edges: list[ShadowEdgeCandidate],
    quotas: SourceQuotaConfig = SourceQuotaConfig(),
) -> MultiSourceVariantResult:
    enabled = set(VARIANT_BUCKETS[variant])
    metric_sources = convert_metric_candidates_to_source_candidates(metric_candidates)
    all_sources = [
        candidate
        for candidate in [*metric_sources, *source_candidates]
        if candidate.source_bucket in enabled
    ]
    normalized = _normalize_and_protect(all_sources, quotas)
    metric_only_keys = {candidate.candidate_key for candidate in metric_candidates}
    final_candidates = _merge_source_candidates(
        case_id=case_id,
        variant=variant,
        source_candidates=normalized,
        metric_only_keys=metric_only_keys,
        max_final_candidates=quotas.max_final_candidates,
    )
    return MultiSourceVariantResult(
        case_id=case_id,
        variant=variant,
        candidates=final_candidates,
        shadow_edges=shadow_edges if "trace_service" in enabled else [],
        warnings=[],
    )


def update_present_in_variants(
    results_by_variant: dict[VariantName, MultiSourceVariantResult],
) -> dict[VariantName, MultiSourceVariantResult]:
    presence: dict[str, list[VariantName]] = defaultdict(list)
    for variant in ("M", "M+T", "M+T+L", "M+T+L+Topo"):
        result = results_by_variant.get(variant)
        if result is None:
            continue
        for candidate in result.candidates:
            if variant not in presence[candidate.candidate_key]:
                presence[candidate.candidate_key].append(variant)

    updated: dict[VariantName, MultiSourceVariantResult] = {}
    for variant, result in results_by_variant.items():
        candidates = [
            candidate.model_copy(
                update={"present_in_variants": tuple(presence[candidate.candidate_key])}
            )
            for candidate in result.candidates
        ]
        updated[variant] = result.model_copy(update={"candidates": candidates})
    return updated


def update_present_in_variants_per_case(
    results_by_variant_by_case: dict[VariantName, dict[str, MultiSourceVariantResult]],
) -> dict[VariantName, dict[str, MultiSourceVariantResult]]:
    """Update present_in_variants for per-case variant results.

    Unlike update_present_in_variants which expects one result per variant,
    this function handles the structure variant -> case_id -> result.
    """
    presence: dict[str, list[VariantName]] = defaultdict(list)
    for variant in ("M", "M+T", "M+T+L", "M+T+L+Topo"):
        case_results = results_by_variant_by_case.get(variant, {})
        for result in case_results.values():
            for candidate in result.candidates:
                if variant not in presence[candidate.candidate_key]:
                    presence[candidate.candidate_key].append(variant)

    updated: dict[VariantName, dict[str, MultiSourceVariantResult]] = {}
    for variant, case_results in results_by_variant_by_case.items():
        updated[variant] = {}
        for case_id, result in case_results.items():
            candidates = [
                candidate.model_copy(
                    update={"present_in_variants": tuple(presence[candidate.candidate_key])}
                )
                for candidate in result.candidates
            ]
            updated[variant][case_id] = result.model_copy(update={"candidates": candidates})
    return updated
