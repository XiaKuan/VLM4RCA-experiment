from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

TelemetryKind = Literal["metrics", "logs", "traces", "topology"]

_TELEMETRY_FILES: dict[TelemetryKind, str] = {
    "metrics": "metrics.csv",
    "logs": "logs.csv",
    "traces": "traces.csv",
    "topology": "topology.csv",
}


class OpenRCAAdapter(Protocol):
    def list_case_ids(self) -> list[str]:
        raise NotImplementedError

    def case_path(self, case_id: str) -> Path:
        raise NotImplementedError

    def load_case_meta(self, case_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def telemetry_path(self, case_id: str, kind: TelemetryKind) -> Path:
        raise NotImplementedError

    def telemetry_exists(self, case_id: str, kind: TelemetryKind) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class OpenRCABankAdapter:
    data_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))

    @property
    def cases_dir(self) -> Path:
        return self.data_root / "cases"

    def list_case_ids(self) -> list[str]:
        if not self.cases_dir.exists():
            raise FileNotFoundError(f"OpenRCA cases directory does not exist: {self.cases_dir}")
        return sorted(
            path.name
            for path in self.cases_dir.iterdir()
            if path.is_dir() and (path / "case_meta.json").exists()
        )

    def case_path(self, case_id: str) -> Path:
        path = self.cases_dir / case_id
        if not path.exists():
            raise FileNotFoundError(f"OpenRCA case directory does not exist: {path}")
        return path

    def load_case_meta(self, case_id: str) -> dict[str, Any]:
        meta_path = self.case_path(case_id) / "case_meta.json"
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def telemetry_path(self, case_id: str, kind: TelemetryKind) -> Path:
        return self.case_path(case_id) / _TELEMETRY_FILES[kind]

    def telemetry_exists(self, case_id: str, kind: TelemetryKind) -> bool:
        path = self.telemetry_path(case_id, kind)
        return path.exists() and path.stat().st_size > 0
