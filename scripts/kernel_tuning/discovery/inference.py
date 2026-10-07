"""Run the public synchronous inference path once and capture its executed operators."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..artifacts import REPOSITORY, atomic_json
from ..contracts import ContractError, read_json
from .contracts import ModelCapture


def load_config(path: Path) -> dict[str, Any]:
    """Read model kwargs, engine settings and an Observation tensor archive.

    Only ``inputs`` is resolved relative to this JSON file. Policy kwargs pass
    unchanged to the factory; checkpoint paths should be absolute.
    """
    config = read_json(path)
    allowed = {"policy", "engine", "inputs", "num_steps", "seed"}
    if not isinstance(config, dict) or config.keys() - allowed:
        raise ContractError(f"Inference configuration must contain only {sorted(allowed)}")
    if not isinstance(config.get("inputs"), str) or not config["inputs"]:
        raise ContractError("Inference configuration requires an inputs tensor archive")
    for key in ("policy", "engine"):
        if not isinstance(config.get(key, {}), dict):
            raise ContractError(f"Inference {key} must be a JSON object")
    if config.get("policy", {}).get("compile_backend", "none") != "none":
        raise ContractError("End-to-end discovery requires compile_backend='none'")
    if config.get("num_steps") is not None and (
        type(config["num_steps"]) is not int or config["num_steps"] < 1
    ):
        raise ContractError("Inference num_steps must be a positive integer")
    seed = config.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ContractError("Inference seed must be an integer in [0, 2**63)")
    inputs = (path.resolve().parent / config["inputs"]).resolve(strict=True)
    if not inputs.is_file():
        raise ContractError("Inference inputs must be a file")
    return {**config, "inputs": str(inputs), "seed": seed}


def capture_inference(
    model: str, config_path: Path, output: Path, *, fixture_bytes: int = 64 * 1024 * 1024
) -> ModelCapture:
    """Load one model and observe one complete ``GenerationBackend.generate`` call.

    The input archive is a plain Observation field mapping, or a nonempty list
    of them, saved with ``torch.save`` and loaded with ``weights_only=True``.
    Recurrent observations each start an explicit fresh session. Construction,
    input deserialization and artifact writes are outside the observed call.
    """
    import torch

    from embodiinfer import EngineConfig, EngineCore, GenerationBackend
    from embodiinfer.policies import factory
    from embodiinfer.types import Observation, SessionKey, validate_observation

    from .recorder import CaptureSession

    config = load_config(config_path)
    inputs = Path(config["inputs"])
    with inputs.open("rb") as handle:
        hasher = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
        input_digest = hasher.hexdigest()
        handle.seek(0)
        records = torch.load(handle, map_location="cpu", weights_only=True)
    records = [records] if isinstance(records, dict) else records
    if not isinstance(records, list) or not records or any(not isinstance(row, dict) for row in records):
        raise ContractError("Inference inputs must be an Observation mapping or a nonempty list of mappings")
    try:
        observations = [Observation(**row) for row in records]
    except TypeError as exc:
        raise ContractError(f"Invalid Observation fields: {exc}") from exc
    for observation in observations:
        validate_observation(observation)

    engine_options = dict(config.get("engine", {}))
    device = torch.device(engine_options.get("device", "cuda"))
    if device.type not in {"cpu", "cuda"}:
        raise ContractError("Operator capture supports CPU or CUDA execution")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ContractError("CUDA inference requested but CUDA is unavailable; refusing a CPU fallback")
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    session = CaptureSession(model, output, fixture_bytes=fixture_bytes, device_type=device.type)
    session.execution = {"kind": "end_to_end_inference", "entry": "GenerationBackend.generate", "calls": 0}
    complete = False
    try:
        session.install()
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(config["seed"])
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(config["seed"])
            kwargs = config.get("policy", {})
            policy = factory.make_policy(model, **kwargs)
            engine_options.setdefault("max_batch_size", 1 if policy.is_recurrent else len(observations))
            engine_options.setdefault("use_cuda_graph", not policy.is_recurrent or policy.manages_cuda_graph)
            engine_config = EngineConfig(**engine_options)
            core = EngineCore(policy, engine_config)
            backend = GenerationBackend(core)
            sessions = (
                [SessionKey(index, "kernel-tuning-capture") for index in range(len(observations))]
                if policy.is_recurrent
                else None
            )
            atomic_json(
                session.root / "inference.json",
                {
                    "model": model,
                    "policy": kwargs,
                    "engine": asdict(engine_config),
                    "num_steps": config.get("num_steps"),
                    "seed": config["seed"],
                    "input_sha256": input_digest,
                    "batch_size": len(observations),
                    "sessions": [asdict(key) for key in sessions] if sessions is not None else None,
                },
            )
            # No policy method whitelist: this covers collation, preprocessing,
            # all nested model/decoder calls, and the final CPU action conversion.
            with session.observe(policy, **kwargs):
                outputs = backend.generate(observations, config.get("num_steps"), session_ids=sessions)
            if len(outputs) != len(observations):
                raise ContractError("End-to-end inference did not return one action chunk per observation")
            torch.save([chunk.actions for chunk in outputs], session.root / "actions.pt")
            session.execution.update(calls=1, outputs=len(outputs))
            complete = True
    except Exception as exc:
        session.errors.append(f"End-to-end inference failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        session.close(complete=complete)
    return ModelCapture.load(session.root, model)


def prepare_inference(
    model: str,
    config_path: Path,
    output: Path,
    evaluator_python: str,
    *,
    fixture_bytes: int = 64 * 1024 * 1024,
) -> ModelCapture:
    """Capture and calibrate in the selected model runtime before task generation.

    This controller imports no Torch or model dependencies. The original and
    calibrated captures remain available under ``output/preparations`` even if
    a later phase fails; a successful batch freezes its own capture snapshot.
    """
    interpreter = Path(evaluator_python)
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise ContractError("Inference requires --set evaluator_python=/absolute/model/runtime/python")
    load_config(config_path)
    root = output.resolve() / "preparations" / uuid4().hex
    root.mkdir(parents=True)
    print(f"Inference capture and calibration: {root}", flush=True)
    commands = [
        [
            "capture",
            "--model",
            model,
            "--inference-config",
            str(config_path.resolve()),
            "--fixture-bytes",
            str(fixture_bytes),
            "--output",
            str(root / "captured"),
        ],
        ["calibrate", "--captured", str(root / "captured"), "--output", str(root / "calibrated")],
    ]
    for command in commands:
        result = subprocess.run(
            [str(interpreter), "-m", "scripts.kernel_tuning", *command], cwd=REPOSITORY, check=False
        )
        if result.returncode:
            raise ContractError(f"Inference {command[0]} failed ({result.returncode}); artifacts: {root}")
    return ModelCapture.load(root / "calibrated", model)
