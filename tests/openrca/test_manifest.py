from pathlib import Path

import pytest

from vlm4rca.openrca.manifest import load_case_manifest


def test_loads_pilot_manifest_with_fixed_15_cases() -> None:
    manifest = load_case_manifest(Path("configs/openrca_pilot15.yaml"))

    assert manifest.dataset == "openrca_bank"
    assert manifest.data_root == "data/OpenRCA/Bank"
    assert len(manifest.cases) == 15
    assert [case.case_id for case in manifest.cases[:3]] == [
        "task3_008",
        "task3_009",
        "task3_010",
    ]
    assert {case.case_group for case in manifest.cases} == {
        "metric_obvious",
        "soft_latency",
        "mixed_ambiguous",
    }


def test_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "dataset: openrca_bank",
                "data_root: data/OpenRCA/Bank",
                "cases:",
                "  - case_id: task3_001",
                "    case_group: soft_latency",
                "    selection_reason: first",
                "  - case_id: task3_001",
                "    case_group: soft_latency",
                "    selection_reason: duplicate",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate case_id values"):
        load_case_manifest(manifest_path)
