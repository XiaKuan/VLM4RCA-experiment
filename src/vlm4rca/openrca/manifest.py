from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vlm4rca.openrca.models import CaseManifest


def load_case_manifest(path: Path) -> CaseManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a YAML mapping: {path}")

    manifest = CaseManifest.model_validate(payload)
    case_ids = [case.case_id for case in manifest.cases]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate case_id values in manifest: {joined}")
    return manifest


def manifest_to_jsonable(manifest: CaseManifest) -> dict[str, Any]:
    return manifest.model_dump(mode="json")
