import json
from pathlib import Path

from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.manifest import load_case_manifest
from vlm4rca.openrca.phase1 import build_phase1_sidecars, main, write_phase1_outputs


def _write_case(root: Path, case_id: str, component: str) -> None:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "task_index": "task_3",
                "inject_time": 1000,
                "context_start": 0,
                "context_end": 3000,
                "matched_faults": [{"component": component, "reason": "network latency", "timestamp": 1000}],
                "system_components": [component],
                "evidence_components": [component],
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "metrics.csv").write_text("timestamp,cpu\n1,2\n", encoding="utf-8")
    (case_dir / "logs.csv").write_text("time,timestamp,service,log_name,message\n1,1,a,app,ok\n", encoding="utf-8")
    (case_dir / "traces.csv").write_text(
        "time,timestamp,service,trace_id,span_id,parent_span_id,duration\n1,1,a,t,s,p,2\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, data_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "dataset: openrca_bank",
                f"data_root: {data_root.as_posix()}",
                "cases:",
                "  - case_id: case_1",
                "    case_group: metric_obvious",
                "    selection_reason: fixture one",
                "  - case_id: case_2",
                "    case_group: soft_latency",
                "    selection_reason: fixture two",
            ]
        ),
        encoding="utf-8",
    )


def test_build_phase1_sidecars_from_adapter_and_manifest(tmp_path: Path) -> None:
    _write_case(tmp_path, "case_1", "Tomcat01")
    _write_case(tmp_path, "case_2", "Mysql02")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, tmp_path)

    manifest = load_case_manifest(manifest_path)
    sidecars = build_phase1_sidecars(OpenRCABankAdapter(tmp_path), manifest)

    assert [sidecar.case_id for sidecar in sidecars] == ["case_1", "case_2"]
    assert sidecars[0].modality_availability.topology_source == "trace_derived"
    assert sidecars[0].gt_mapping[0].raw_ground_truth == "Tomcat01"
    assert sidecars[0].gt_mapping[0].mapped_component == "tomcat01"


def test_write_phase1_outputs_without_candidate_or_image_files(tmp_path: Path) -> None:
    _write_case(tmp_path / "data", "case_1", "Tomcat01")
    _write_case(tmp_path / "data", "case_2", "Mysql02")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, tmp_path / "data")
    sidecars = build_phase1_sidecars(OpenRCABankAdapter(tmp_path / "data"), load_case_manifest(manifest_path))

    output_dir = tmp_path / "outputs"
    write_phase1_outputs(sidecars, load_case_manifest(manifest_path), output_dir)

    assert (output_dir / "case_sidecars" / "case_1.json").exists()
    assert (output_dir / "modality_availability.json").exists()
    assert (output_dir / "ground_truth_mapping.json").exists()
    assert (output_dir / "recall_empty_run.json").exists()
    written_paths = [path.name for path in output_dir.rglob("*") if path.is_file()]
    assert not any("candidate" in name for name in written_paths)
    assert not any(name.endswith((".png", ".jpg", ".jpeg", ".svg", ".html")) for name in written_paths)


def test_cli_main_writes_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_case(data_root, "case_1", "Tomcat01")
    _write_case(data_root, "case_2", "Mysql02")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, data_root)
    output_dir = tmp_path / "phase1"

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "recall_empty_run.json").read_text(encoding="utf-8"))
    assert summary["n_cases"] == 2
    assert summary["component_recall_at_k"] == {"3": 0.0, "5": 0.0, "8": 0.0}
