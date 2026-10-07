"""Developer commands for checking, running, resuming, and exporting tuning jobs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from .artifacts import REPOSITORY, RunStore, atomic_json, run_lock, runtime_identity
from .contracts import HMZ_REVISION, ContractError, TaskPackage, read_json
from .discovery.contracts import ModelCapture
from .runner import evaluate


def _execute(store: RunStore, *, resume: bool) -> dict:
    if "replay.json" in store.task.hashes:
        from .batch import verify_model_run

        verify_model_run(store)
    if os.name != "posix":
        raise ContractError(
            "This pinned Humanize2 release requires POSIX; run tuning on the Linux target machine"
        )
    if sys.version_info < (3, 12):
        raise ContractError("Humanize2 requires a separate Python 3.12+ tooling environment")
    try:
        from hmz.flows import BudgetExceeded
        from hmz.sdk import Hmz
    except ImportError as exc:
        raise ContractError("Install the optional scripts/kernel_tuning tooling environment first") from exc
    with run_lock(store.root):
        store.verify()
        if runtime_identity() != store.manifest["tool_runtime"]:
            raise ContractError("Humanize2/Python tooling environment changed; start a new run")
        remaining = store.task.settings.search.max_seconds - store.manifest["elapsed_seconds"]
        if remaining <= 0 or store.manifest["status"] == "completed":
            raise ContractError("Run budget already exhausted; start a new run with a new task contract")
        epic = store.root / "humanize.json"
        running = Hmz(workspace=store.root / "workspace").run(
            flow=Path(__file__).with_name("flow.py"),
            task=f"Optimize {store.task.definition['name']} under its frozen task contract",
            agents={"coder": store.manifest["agent"]},
            params={"run_dir": str(store.root)},
            budget={"duration": remaining, "graceful": False},
            # A killed invocation may be newer than humanize.json. The dedicated
            # workspace lets hmz select its newest journal instead of a stale path.
            resume=resume,
        )
        try:
            return running.run()
        except BudgetExceeded:
            stopped = RunStore(store.root)
            stopped.recover()
            stopped.manifest.update(status="completed", stop_reason="max_seconds")
            stopped.save()
            return stopped.manifest
        finally:
            archived = None
            if running.epic:
                archived = store.root / "humanize" / running.epic.name
                archived.mkdir(parents=True, exist_ok=True)
                # Preserve flow/resume journals, without copying agent account
                # directories. CLI-managed sessions stay at the recorded epic.
                for path in running.epic.glob("*.jsonl"):
                    if path.is_file() and not path.is_symlink():
                        shutil.copyfile(path, archived / path.name)
            atomic_json(
                epic,
                {
                    "epic": str(running.epic) if running.epic else None,
                    "archived": str(archived) if archived else None,
                    "expected_hmz_revision": HMZ_REVISION,
                },
            )


def _selection(parser: argparse.ArgumentParser, *, inference: bool = False) -> None:
    parser.add_argument("--model", action="append", help="One canonical target policy name, e.g. pi05")
    parser.add_argument(
        "--selection",
        type=Path,
        help="JSON mapping of captured operator IDs to workload ID lists (null means all)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override tuning.yaml, e.g. evaluator_python=/abs/python or search.max_candidates=10",
    )
    parser.add_argument("--hardware-notes", type=Path, help="Target-hardware notes copied into each task")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--captured", type=Path, help="Completed model capture directory")
    if inference:
        sources.add_argument(
            "--inference-config",
            type=Path,
            help="Run one end-to-end inference from JSON model/engine/input settings, then calibrate and tune",
        )
        parser.add_argument(
            "--fixture-bytes",
            type=int,
            help="Input fixture budget for --inference-config (default: 64 MiB)",
        )


def _model_capture(args: argparse.Namespace) -> ModelCapture:
    if not args.model or len(args.model) != 1:
        raise ContractError("Model tuning requires exactly one --model")
    if args.captured is None:
        raise ContractError(
            "Model tuning requires --captured MODEL_CAPTURE; run capture --inference-config first"
        )
    return ModelCapture.load(
        args.captured, args.model[0], require_ready=not (getattr(args, "list", False) or args.selection)
    )


def _read_selection(path: Path | None) -> dict[str, list[str] | None] | None:
    if path is None:
        return None
    selection = read_json(path)
    if not isinstance(selection, dict) or not selection:
        raise ContractError("--selection requires a nonempty JSON object of operator IDs")
    return selection


def _generate(args: argparse.Namespace) -> None:
    from .discovery.tasks import render_model
    from .generate import parse_overrides

    capture = _model_capture(args)
    if args.list:
        if args.selection:
            raise ContractError("--list shows the complete capture; omit --selection")
        for op in capture.operators:
            print(
                f"{op['id']} {op['status']:20} {op['name']} "
                f"({len(op['workloads'])} cases) {op.get('reason') or ''}"
            )
        return
    tasks = render_model(
        capture,
        args.output,
        overrides=parse_overrides(args.set),
        hardware_notes=args.hardware_notes,
        selection=_read_selection(args.selection),
    )
    for task in tasks:
        print(f"{task.definition['description']}: {task.root} ({len(task.workloads)} workloads)")


def _tune_all(args: argparse.Namespace) -> int:
    from . import batch
    from .generate import parse_overrides

    if args.resume:
        if (
            args.model
            or args.set
            or args.hardware_notes
            or args.agent
            or args.captured
            or args.selection
            or args.inference_config
            or args.fixture_bytes is not None
        ):
            raise ContractError("--resume continues the recorded batch; omit selection, --set, and --agent")
        root = args.resume.resolve(strict=True)
    else:
        if args.fixture_bytes is not None and not args.inference_config:
            raise ContractError("--fixture-bytes requires --inference-config")
        agent = args.agent or ""
        if not (args.dry_run or args.preflight_only) and "/" not in agent:
            raise ContractError("Choose an explicit Humanize2 agent: --agent harness/model:effort")
        options = {
            "agent": agent,
            "overrides": parse_overrides(args.set),
            "hardware_notes": args.hardware_notes,
        }
        if args.inference_config:
            from .discovery.inference import prepare_inference

            if not args.model or len(args.model) != 1 or args.selection:
                raise ContractError(
                    "End-to-end tuning requires one --model and every observed operator; omit --selection"
                )
            captured = prepare_inference(
                args.model[0],
                args.inference_config,
                args.output,
                options["overrides"].get("evaluator_python", ""),
                fixture_bytes=args.fixture_bytes if args.fixture_bytes is not None else 64 * 1024 * 1024,
            )
        else:
            captured = _model_capture(args)
        root = batch.create_model(
            captured,
            args.output,
            selection=_read_selection(args.selection),
            **options,
        )
    print(f"Batch: {root}", flush=True)
    if args.dry_run:
        batch.write_summary(root)
        print((root / "summary.md").read_text(encoding="utf-8"))
        return 0
    results = batch.preflight(root)
    items = read_json(root / "batch.json")["items"]
    for name, error in results.items():
        timings = ", ".join(f"{work} {us:.1f} us" for work, us in items[name].get("baseline_us", {}).items())
        print(f"preflight {name}: {f'ok ({timings})' if error is None else error}", flush=True)
    if args.preflight_only:
        batch.write_summary(root)
        return (
            0
            if all(
                i["status"] != "preflight_failed" for i in read_json(root / "batch.json")["items"].values()
            )
            else 1
        )
    record = batch.execute(root)
    print((root / "summary.md").read_text(encoding="utf-8"))
    return 0 if all(item["status"] == "completed" for item in record["items"].values()) else 1


def main(argv: list[str] | None = None) -> int:
    """Run only the requested command; checking/status/export never invoke an agent."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Validate a task without executing its code")
    check.add_argument("task", type=Path)
    check.add_argument(
        "--environment", action="store_true", help="Also probe the selected evaluator environment (no tuning)"
    )
    run = commands.add_parser("run", help="Start real agent-driven tuning on this machine")
    run.add_argument("task", type=Path)
    run.add_argument("--agent", help="Explicit Humanize2 harness/model:effort spec; overrides tuning.yaml")
    run.add_argument("--output", type=Path, default=REPOSITORY / "results/kernel_tuning")
    for name in ("resume", "status"):
        commands.add_parser(name).add_argument("run", type=Path)
    export = commands.add_parser("export", help="Export best verified source and evidence; no execution")
    export.add_argument("run", type=Path)
    export.add_argument("destination", type=Path)
    generate = commands.add_parser("generate", help="Write every discovered model task; no execution")
    _selection(generate)
    generate.add_argument("--output", type=Path, default=REPOSITORY / "results/kernel_tuning/tasks")
    generate.add_argument("--list", action="store_true", help="List discovered operators and coverage gaps")
    tune_all = commands.add_parser("tune-all", help="Generate, preflight, and tune every selected operator")
    _selection(tune_all, inference=True)
    tune_all.add_argument("--agent", help="Explicit Humanize2 harness/model:effort spec for every operator")
    tune_all.add_argument("--output", type=Path, default=REPOSITORY / "results/kernel_tuning/batches")
    tune_all.add_argument("--resume", type=Path, metavar="BATCH", help="Continue an existing batch directory")
    mode = tune_all.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate tasks without preflight or agents; inference inputs still run capture and calibration",
    )
    mode.add_argument("--preflight-only", action="store_true", help="Stop after measuring the baselines")
    capture = commands.add_parser(
        "capture", help="Run one complete synchronous model inference and record its operators"
    )
    capture.add_argument("--model", required=True, help="Canonical policy name")
    capture.add_argument(
        "--inference-config",
        type=Path,
        required=True,
        help="JSON model/engine/input settings for one end-to-end inference",
    )
    capture.add_argument(
        "--fixture-bytes", type=int, default=64 * 1024 * 1024, help="Maximum stored input fixture bytes"
    )
    capture.add_argument("--output", type=Path, required=True, help="New model capture directory")
    calibration = commands.add_parser(
        "calibrate", help="Freeze baseline/FP64 numerical bounds in the model runtime before generation"
    )
    calibration.add_argument("--captured", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True, help="New calibrated capture directory")
    skeleton = commands.add_parser(
        "skeleton", help="Random-weight copy of a Hugging Face checkpoint for shape capture (model runtime)"
    )
    skeleton.add_argument("repo", help="Model repository, e.g. org/name; HF_ENDPOINT selects a mirror")
    skeleton.add_argument("--revision", required=True, help="Immutable commit to copy")
    skeleton.add_argument("--output", type=Path, required=True)
    raw = sys.argv[1:] if argv is None else argv
    try:
        args = parser.parse_args(raw)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        if args.command == "capture":
            from .discovery.inference import capture_inference

            captured = capture_inference(
                args.model, args.inference_config, args.output, fixture_bytes=args.fixture_bytes
            )
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "identity": captured.identity,
                        "operators": len(captured.operators),
                        "execution": captured.manifest["execution"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif args.command == "calibrate":
            from benchmarks.kernel_tuning.calibration import calibrate

            result = calibrate(ModelCapture.load(args.captured), args.output)
            print(json.dumps({"capture": str(result.root), "identity": result.identity}, indent=2))
        elif args.command == "skeleton":
            from .skeleton import build

            summary = build(args.repo, args.revision, args.output)
            tensors = sum(summary["synthesized"].values())
            print(
                f"{args.output}: {len(summary['copied'])} files copied, "
                f"{tensors} random tensors in {len(summary['synthesized'])} safetensors files"
            )
        elif args.command == "generate":
            _generate(args)
        elif args.command == "tune-all":
            return _tune_all(args)
        elif args.command == "check":
            task = TaskPackage.load(args.task)
            result = {
                "task": task.definition["name"],
                "digest": task.identity,
                "workloads": len(task.workloads),
                "validation": "structural; no task code executed",
            }
            if args.environment:
                import tempfile

                with tempfile.TemporaryDirectory(prefix="kernel-preflight-") as directory:
                    store = RunStore.create(task, Path(directory), agent="")
                    result["environment"] = asyncio.run(
                        evaluate(store, store.root / "probe", operation="probe")
                    )["environment"]
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "run":
            task = TaskPackage.load(args.task)
            agent = args.agent or task.settings.agent
            if not agent or "/" not in agent:
                raise ContractError("Choose an explicit Humanize2 agent: --agent harness/model:effort")
            model_batch = None
            if "replay.json" in task.hashes:
                from .batch import verify_model_task

                model_batch = verify_model_task(task)
            store = RunStore.create(task, args.output, agent)
            if model_batch is not None:
                store.manifest["model_batch"] = str(model_batch)
                item = read_json(model_batch / "batch.json")["items"][task.root.name]
                store.manifest["environment"] = item["preflight_environment"]
                store.save()
            print(f"Run archive: {store.root}", flush=True)
            print(json.dumps(_execute(store, resume=False), indent=2, ensure_ascii=False))
        else:
            store = RunStore(args.run)
            if args.command == "resume":
                print(json.dumps(_execute(store, resume=True), indent=2, ensure_ascii=False))
            elif args.command == "status":
                print(json.dumps(store.manifest, indent=2, ensure_ascii=False))
            else:
                with run_lock(store.root):
                    print(store.export(args.destination))
        return 0
    except (ContractError, OSError, ValueError, ImportError) as exc:
        print(f"kernel_tuning: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; the run archive can be resumed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
