from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vlm4rca.openrca.models import IncidentWindows


@dataclass(frozen=True)
class WindowConfig:
    pre_window_seconds: int = 10 * 60
    post_window_seconds: int = 20 * 60
    baseline_window_seconds: int = 30 * 60
    timezone: str = "UTC"


def extract_incident_windows(
    case_meta: Mapping[str, Any],
    config: WindowConfig = WindowConfig(),
) -> IncidentWindows:
    if "inject_time" not in case_meta:
        raise KeyError("case_meta is missing inject_time")

    inject_time = int(case_meta["inject_time"])
    baseline_start = inject_time - config.pre_window_seconds - config.baseline_window_seconds
    baseline_end = inject_time - config.pre_window_seconds
    incident_start = baseline_end
    incident_end = inject_time + config.post_window_seconds

    warnings: list[str] = []
    if "context_start" in case_meta and int(case_meta["context_start"]) > baseline_start:
        warnings.append(
            f"context_start {int(case_meta['context_start'])} is after baseline_start {baseline_start}"
        )
    if "context_end" in case_meta and int(case_meta["context_end"]) < incident_end:
        warnings.append(f"context_end {int(case_meta['context_end'])} is before incident_end {incident_end}")

    return IncidentWindows(
        inject_time=inject_time,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        incident_start=incident_start,
        incident_end=incident_end,
        pre_window_seconds=config.pre_window_seconds,
        post_window_seconds=config.post_window_seconds,
        baseline_window_seconds=config.baseline_window_seconds,
        timezone=config.timezone,
        warnings=warnings,
    )
