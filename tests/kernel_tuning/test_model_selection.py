"""Explicit discovery scopes and immutable evidence, without Torch or GPU execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from scripts.kernel_tuning import batch
from scripts.kernel_tuning.__main__ import main
from scripts.kernel_tuning.artifacts import atomic_json
from scripts.kernel_tuning.contracts import ContractError, TaskPackage, digest, read_json, tree_hashes
from scripts.kernel_tuning.discovery.contracts import SCHEMA, ModelCapture
from scripts.kernel_tuning.discovery.numerics import PROFILES, SEEDS, VERSION, implementation_digest
from scripts.kernel_tuning.discovery.tasks import render_model


def capture_fixture(root: Path, *, calibrated: bool = False, blocked: bool = False) -> ModelCapture:
    """Write synthetic capture metadata; fixtures are never executed or called real calibration."""
    root.mkdir()
    operators = []
    for name in (("sum" if calibrated else "relu"), "neg", "sigmoid"):
        identity = {
            "kind": "aten",
            "name": f"aten.{name}.default",
            "arguments": {"tuple": [{"tuple": [{"tensor": 0}]}, {"dict": {}}]},
            "returns": {"tensor": 0},
            "input_dtypes": ["float32"],
            "output_dtypes": ["float32"],
            "mutates": [],
        }
        cases = []
        for size in (2, 4):
            fixture = f"fixtures/{name}-{size}.bin"
            path = root / fixture
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"synthetic, non-executable fixture")
            tensor = {"shape": [size], "stride": [1], "offset": 0, "dtype": "float32", "device": "cpu"}
            case = {
                "inputs": [{**tensor, "storage": 0, "fixture": fixture}],
                "outputs": [{**tensor, **({"shape": [], "stride": []} if name == "sum" else {})}],
                "output_aliases": [[]],
                "output_groups": [[0]],
            }
            cases.append({**case, "id": digest(case)[:20], "count": size})
        operators.append(
            {
                "id": digest(identity)[:20],
                "name": identity["name"],
                "identity": identity,
                "status": "blocked" if blocked and name == "sigmoid" else "ready",
                "workloads": cases,
            }
        )
    precision = {"matmul_precision": "highest", "allow_tf32": False}
    if calibrated:
        op = operators[0]
        numerical = {
            "version": VERSION,
            "implementation_digest": implementation_digest(),
            "reference": "aten",
            "operator_identity": digest(op["identity"]),
            "precision": precision,
            "flags": {},
            "profiles": [list(pair) for pair in PROFILES],
            "seeds": list(SEEDS),
            "environment": {"synthetic": True},
            "workloads": {case["id"]: digest(case) for case in op["workloads"]},
            "cases": {},
        }
        for index, case in enumerate(op["workloads"]):
            residual = (index + 1) / 100
            bound = {
                "dtype": "float32",
                "reference_dtype": "float64",
                "atol": 2 * residual,
                "rtol": 0.01,
                "baseline_max_abs": 1.0,
                "residual_max": residual,
                "rms": 1.0,
                "epsilon": 0.01,
                "tiny": 0.0,
            }
            numerical["cases"][case["id"]] = {f"{p}:{s}": [bound] for p, s in PROFILES}
        atomic_json(root / "calibration.json", {op["id"]: numerical})
    atomic_json(root / "operators.json", operators)
    atomic_json(
        root / "coverage.json", {"operators": {"ready": 2 if blocked else 3, "blocked": int(blocked)}}
    )
    atomic_json(
        root / "manifest.json",
        {
            "schema": SCHEMA,
            "target": {"model": "test", "type": "synthetic"},
            "complete": True,
            "errors": [],
            "precision": precision,
            "flags": {},
            "device_type": "cpu",
            "torch_version": "synthetic",
            "files": tree_hashes(root),
        },
    )
    return ModelCapture.load(root, require_ready=False)


def create(capture: ModelCapture, tmp_path: Path, selection: dict | None = None) -> Path:
    return batch.create_model(
        capture,
        tmp_path / "batches",
        agent="fake/model",
        overrides={},
        hardware_notes=None,
        selection=selection,
    )


def rebind_task(root: Path, task_root: Path) -> None:
    """Simulate rewritten task hashes; capture-derived evidence must still reject tampering."""
    identity = TaskPackage.load(task_root).identity
    index = read_json(root / "tasks/model.json")
    index["tasks"][task_root.name] = identity
    atomic_json(root / "tasks/model.json", index)
    record = read_json(root / "batch.json")
    record["items"][task_root.name]["task_digest"] = identity
    atomic_json(root / "batch.json", record)


def test_selection_excludes_operators_and_workloads_but_preserves_capture(tmp_path: Path) -> None:
    capture = capture_fixture(tmp_path / "capture", blocked=True)
    op, _, _ = capture.operators
    workload = op["workloads"][1]["id"]
    root = create(capture, tmp_path, {op["id"]: [workload]})
    assert ModelCapture.load(root / "capture", require_ready=False).identity == capture.identity
    record = read_json(root / "batch.json")
    assert record["selection"] == {op["id"]: [workload]}
    assert set(record["items"]) == {op["id"]}
    assert read_json(root / "tasks/model.json")["selection"] == record["selection"]
    task = TaskPackage.load(root / "tasks" / op["id"])
    assert task.workload_ids == (workload,)
    assert read_json(task.root / "replay.json")["cases"] == {workload: op["workloads"][1]}
    assert op["workloads"][0]["inputs"][0]["fixture"] not in task.hashes
    batch._verify_model(root, record)
    batch.write_summary(root)
    assert "not full-model coverage" in (root / "summary.md").read_text(encoding="utf-8")
    with pytest.raises(ContractError, match="incomplete"):
        render_model(capture, tmp_path / "full")


def test_default_mode_keeps_every_ready_operator_and_workload(tmp_path: Path) -> None:
    capture = capture_fixture(tmp_path / "capture")
    root = create(capture, tmp_path)
    record = read_json(root / "batch.json")
    assert record["selection"] is None
    assert len(record["items"]) == 3
    assert all(len(TaskPackage.load(root / item["task"]).workloads) == 2 for item in record["items"].values())
    batch._verify_model(root, record)
    record["items"].pop(next(iter(record["items"])))
    with pytest.raises(ContractError, match="coverage gaps"):
        batch._verify_model(root, record)


@pytest.mark.parametrize(
    "kind", ["empty", "unknown_operator", "unknown_workload", "empty_workloads", "duplicate", "blocked"]
)
def test_invalid_selection_writes_no_tasks(tmp_path: Path, kind: str) -> None:
    capture = capture_fixture(tmp_path / "capture", blocked=True)
    op = capture.operators[0]
    work = op["workloads"][0]["id"]
    selection = {
        "empty": {},
        "unknown_operator": {"missing": None},
        "unknown_workload": {op["id"]: ["missing"]},
        "empty_workloads": {op["id"]: []},
        "duplicate": {op["id"]: [work, work]},
        "blocked": {capture.operators[-1]["id"]: None},
    }[kind]
    with pytest.raises(ContractError):
        render_model(capture, tmp_path / "tasks", selection=selection)
    assert not (tmp_path / "tasks").exists()


def test_numerical_subset_retains_all_frozen_profiles_and_original_bounds(tmp_path: Path) -> None:
    capture = capture_fixture(tmp_path / "capture", calibrated=True)
    op = capture.operators[0]
    key = op["workloads"][0]["id"]
    original = read_json(capture.root / "calibration.json")[op["id"]]
    root = create(capture, tmp_path, {op["id"]: [key]})
    task = TaskPackage.load(root / "tasks" / op["id"])
    selected = read_json(task.root / "numerics.json")
    assert selected == {
        **original,
        "cases": {key: original["cases"][key]},
        "workloads": {key: original["workloads"][key]},
    }
    assert task.settings.precision.atol == original["cases"][key]["recorded:0"][0]["atol"]
    assert read_json(root / "capture/calibration.json")[op["id"]] == original
    batch._verify_model(root, read_json(root / "batch.json"))
    selected["environment"] = {"synthetic": "changed"}
    atomic_json(task.root / "numerics.json", selected)
    rebind_task(root, task.root)
    with pytest.raises(ContractError, match="numerical evidence changed"):
        batch._verify_model(root, read_json(root / "batch.json"))


@pytest.mark.parametrize("edit", ["scope", "workload", "fixture", "drop_scope"])
def test_batch_rejects_changed_scope_and_capture_evidence(tmp_path: Path, edit: str) -> None:
    capture = capture_fixture(tmp_path / "capture")
    op = capture.operators[0]
    root = create(capture, tmp_path, {op["id"]: None})
    record = read_json(root / "batch.json")
    task_root = root / "tasks" / op["id"]
    if edit == "scope":
        record["selection"][op["id"]].pop()
        atomic_json(root / "batch.json", record)
    elif edit == "drop_scope":
        record.pop("selection")
        atomic_json(root / "batch.json", record)
    elif edit == "workload":
        replay = read_json(task_root / "replay.json")
        next(iter(replay["cases"].values()))["count"] += 1
        atomic_json(task_root / "replay.json", replay)
        rebind_task(root, task_root)
    else:
        (task_root / op["workloads"][0]["inputs"][0]["fixture"]).write_bytes(b"changed")
        rebind_task(root, task_root)
    with pytest.raises(ContractError, match="scope|coverage|index|fixtures"):
        batch._verify_model(root, read_json(root / "batch.json"))


def test_every_selected_task_still_requires_successful_preflight(tmp_path: Path, monkeypatch) -> None:
    from test_batch import measurement

    capture = capture_fixture(tmp_path / "capture")
    first, second, _ = capture.operators
    root = create(capture, tmp_path, {first["id"]: None, second["id"]: None})
    commands = []

    async def evaluate(store, output, **kwargs) -> dict[str, Any]:
        if read_json(store.task.root / "generated.json")["operator_id"] == second["id"]:
            raise ContractError("synthetic failure")
        return {"measurement": measurement(store.task, 10), "environment": {"identity": "fake"}}

    monkeypatch.setattr(batch, "evaluate", evaluate)
    with pytest.raises(ContractError, match="every task"):
        batch.execute(root, launcher=lambda *args: commands.append(args))
    with pytest.raises(ContractError, match="recorded scope"):
        batch.preflight(root, [first["id"]])
    result = batch.preflight(root)
    assert set(result) == {first["id"], second["id"]}
    assert result[second["id"]] == "synthetic failure"
    with pytest.raises(ContractError, match="every task"):
        batch.execute(root, launcher=lambda *args: commands.append(args))
    assert commands == []


def test_rewriting_both_indexes_cannot_silently_drop_a_selected_operator(tmp_path: Path) -> None:
    capture = capture_fixture(tmp_path / "capture")
    first, second, _ = capture.operators
    root = create(capture, tmp_path, {first["id"]: None, second["id"]: None})
    record = read_json(root / "batch.json")
    index = read_json(root / "tasks/model.json")
    del record["selection"][second["id"]]
    del record["items"][second["id"]]
    del index["selection"][second["id"]]
    del index["tasks"][second["id"]]
    atomic_json(root / "tasks/model.json", index)
    with pytest.raises(ContractError, match="selected scope changed"):
        batch._verify_model(root, record)


def test_cli_selection_generates_scoped_batches_and_cannot_change_on_resume(tmp_path: Path, capsys) -> None:
    capture = capture_fixture(tmp_path / "capture")
    op = capture.operators[0]
    selection = tmp_path / "selection.json"
    atomic_json(selection, {op["id"]: [op["workloads"][0]["id"]]})
    args = ["--model", "test", "--captured", str(capture.root), "--selection", str(selection)]
    assert main(["generate", *args, "--output", str(tmp_path / "tasks")]) == 0
    assert main(["tune-all", *args, "--dry-run", "--output", str(tmp_path / "batches")]) == 0
    assert "not full-model coverage" in capsys.readouterr().out
    (root,) = (tmp_path / "batches").iterdir()
    assert main(["tune-all", "--resume", str(root), "--selection", str(selection)]) == 2
    assert "continues the recorded batch" in capsys.readouterr().err
    assert main(["generate", "--catalog", "--selection", str(selection)]) == 2
    atomic_json(selection, None)
    assert main(["generate", *args, "--output", str(tmp_path / "invalid")]) == 2
    assert not (tmp_path / "invalid").exists()
