from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vlm4rca.evaluation.recall import evaluate_component_recall_at_k
from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.ground_truth import build_ground_truth_mappings, extract_raw_ground_truth
from vlm4rca.openrca.manifest import load_case_manifest, manifest_to_jsonable
from vlm4rca.openrca.modality import detect_modality_availability
from vlm4rca.openrca.models import CaseManifest, Phase1CaseSidecar
from vlm4rca.openrca.windows import extract_incident_windows


def build_phase1_sidecars(
    adapter: OpenRCABankAdapter,
    manifest: CaseManifest,
) -> list[Phase1CaseSidecar]:
    available_case_ids = set(adapter.list_case_ids())
    sidecars: list[Phase1CaseSidecar] = []

    for manifest_case in manifest.cases:
        if manifest_case.case_id not in available_case_ids:
            raise FileNotFoundError(f"Manifest case_id is not present in OpenRCA-Bank cases: {manifest_case.case_id}")

        meta = adapter.load_case_meta(manifest_case.case_id)
        sidecars.append(
            Phase1CaseSidecar(
                case_id=manifest_case.case_id,
                case_group=manifest_case.case_group,
                task_index=meta.get("task_index"),
                inject_time=int(meta["inject_time"]),
                case_path=str(adapter.case_path(manifest_case.case_id)),
                raw_ground_truth=extract_raw_ground_truth(meta),
                incident_window=extract_incident_windows(meta),
                modality_availability=detect_modality_availability(adapter, manifest_case.case_id),
                gt_mapping=build_ground_truth_mappings(meta),
            )
        )

    return sidecars


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_phase1_outputs(
    sidecars: Sequence[Phase1CaseSidecar],
    manifest: CaseManifest,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest_cases.json", manifest_to_jsonable(manifest))

    for sidecar in sidecars:
        _write_json(
            output_dir / "case_sidecars" / f"{sidecar.case_id}.json",
            sidecar.model_dump(mode="json"),
        )

    _write_json(
        output_dir / "modality_availability.json",
        [
            {
                "case_id": sidecar.case_id,
                "case_group": sidecar.case_group,
                "modality_availability": sidecar.modality_availability.model_dump(mode="json"),
            }
            for sidecar in sidecars
        ],
    )
    _write_json(
        output_dir / "ground_truth_mapping.json",
        [
            {
                "case_id": sidecar.case_id,
                "case_group": sidecar.case_group,
                "gt_mapping": [mapping.model_dump(mode="json") for mapping in sidecar.gt_mapping],
            }
            for sidecar in sidecars
        ],
    )
    empty_recall = evaluate_component_recall_at_k(sidecars, candidates_by_case={}, ks=[3, 5, 8])
    _write_json(output_dir / "recall_empty_run.json", empty_recall.model_dump(mode="json"))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenRCA Phase 1 data and evaluation sidecars.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/openrca_pilot15.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/openrca_phase1_pilot15"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = load_case_manifest(args.manifest)
    data_root = args.data_root if args.data_root is not None else Path(manifest.data_root)
    adapter = OpenRCABankAdapter(data_root)
    sidecars = build_phase1_sidecars(adapter, manifest)
    write_phase1_outputs(sidecars, manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
