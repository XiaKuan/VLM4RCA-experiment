from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vlm4rca.candidates.models import MetricCategory, MetricFeatureEvidence
from vlm4rca.openrca.canonicalization import canonicalize_component
from vlm4rca.openrca.models import IncidentWindows

EPSILON = 1e-9
MIN_POINTS_PER_WINDOW = 2


@dataclass(frozen=True)
class MetricColumn:
    metric_column: str
    raw_target: str
    canonical_target: str
    metric_name: str
    metric_category: MetricCategory


def categorize_metric(metric_name: str) -> MetricCategory:
    lowered = metric_name.lower()
    if any(token in lowered for token in ("latency", "duration", "response_time", "responsetime")):
        return "latency"
    if any(token in lowered for token in ("error", "err", "failed", "failure", "exception")):
        return "error"
    if any(token in lowered for token in ("request", "throughput", "qps", "tps", "count")):
        return "traffic"
    if "cpu" in lowered:
        return "cpu"
    if any(token in lowered for token in ("memory", "mem", "heap", "swap")):
        return "memory"
    if any(token in lowered for token in ("disk", "dsk", "filesystem", "fsavailable", "fsused")):
        return "disk"
    if any(token in lowered for token in ("network", "net", "tcp", "packet", "bandwidth")):
        return "network"
    return "unknown"


def parse_metric_column(column_name: str) -> MetricColumn | None:
    if column_name == "timestamp":
        return None

    parts = column_name.split("__")
    if len(parts) >= 2:
        raw_target = parts[0]
        metric_name = "__".join(parts[1:])
    else:
        raw_target = "unknown"
        metric_name = column_name

    canonical_target = canonicalize_component(raw_target)
    if not canonical_target:
        return None

    return MetricColumn(
        metric_column=column_name,
        raw_target=raw_target,
        canonical_target=canonical_target,
        metric_name=metric_name,
        metric_category=categorize_metric(metric_name),
    )


def _finite_values(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _scale_for_baseline(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > EPSILON:
        # 1.4826 is the consistency constant for MAD as a normal distribution estimator
        return 1.4826 * mad
    std = float(np.std(values))
    if std > EPSILON:
        return std
    return 1.0


def _relative_delta(incident_value: float, baseline_value: float) -> float:
    denominator = abs(baseline_value)
    if denominator <= EPSILON:
        denominator = 1.0
    return (incident_value - baseline_value) / denominator


def score_metric_series(
    metric: MetricColumn,
    baseline: pd.Series,
    incident: pd.Series,
) -> MetricFeatureEvidence:
    baseline_values = _finite_values(baseline)
    incident_values = _finite_values(incident)

    if baseline_values.size == 0 or incident_values.size == 0:
        robust_z_score = 0.0
        relative_change = 0.0
        p95_shift = 0.0
    else:
        baseline_median = float(np.median(baseline_values))
        incident_median = float(np.median(incident_values))
        robust_z_score = (incident_median - baseline_median) / _scale_for_baseline(baseline_values)

        baseline_mean = float(np.mean(baseline_values))
        incident_mean = float(np.mean(incident_values))
        relative_change = _relative_delta(incident_mean, baseline_mean)

        baseline_p95 = float(np.percentile(baseline_values, 95))
        incident_p95 = float(np.percentile(incident_values, 95))
        p95_shift = _relative_delta(incident_p95, baseline_p95)

    # Baseline scoring: simple unweighted sum; z-score dominates when large.
    # Future work: normalize features to comparable scales.
    score = abs(robust_z_score) + abs(relative_change) + abs(p95_shift)
    return MetricFeatureEvidence(
        metric_column=metric.metric_column,
        raw_target=metric.raw_target,
        canonical_target=metric.canonical_target,
        metric_category=metric.metric_category,
        robust_z_score=round(float(robust_z_score), 6),
        relative_change=round(float(relative_change), 6),
        p95_shift=round(float(p95_shift), 6),
        score=round(float(score), 6),
        baseline_points=int(baseline_values.size),
        incident_points=int(incident_values.size),
    )


def score_metrics_dataframe(
    metrics: pd.DataFrame,
    windows: IncidentWindows,
) -> tuple[list[MetricFeatureEvidence], list[str]]:
    if "timestamp" not in metrics.columns:
        raise ValueError("metrics.csv must contain a timestamp column")

    frame = metrics.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    baseline_mask = (frame["timestamp"] >= windows.baseline_start) & (
        frame["timestamp"] < windows.baseline_end
    )
    incident_mask = (frame["timestamp"] >= windows.incident_start) & (
        frame["timestamp"] < windows.incident_end
    )

    evidence: list[MetricFeatureEvidence] = []
    warnings: list[str] = []
    for column_name in frame.columns:
        parsed = parse_metric_column(column_name)
        if parsed is None:
            continue

        baseline = frame.loc[baseline_mask, column_name]
        incident = frame.loc[incident_mask, column_name]
        scored = score_metric_series(parsed, baseline, incident)
        if (
            scored.baseline_points < MIN_POINTS_PER_WINDOW
            or scored.incident_points < MIN_POINTS_PER_WINDOW
        ):
            warnings.append(
                f"{column_name} skipped: baseline_points={scored.baseline_points} incident_points={scored.incident_points}"
            )
            continue
        evidence.append(scored)

    return evidence, warnings
