"""End-to-end discovery observes the public inference call, without model benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scripts.kernel_tuning.__main__ import main  # noqa: E402
from scripts.kernel_tuning.artifacts import atomic_json  # noqa: E402
from scripts.kernel_tuning.contracts import ContractError, read_json  # noqa: E402
from scripts.kernel_tuning.discovery.contracts import ModelCapture  # noqa: E402
from scripts.kernel_tuning.discovery.inference import capture_inference, prepare_inference  # noqa: E402
from scripts.kernel_tuning.discovery.tasks import render_model  # noqa: E402

from embodiinfer import EngineConfig, EngineCore, GenerationBackend  # noqa: E402
from embodiinfer.policies import factory  # noqa: E402
from embodiinfer.policies.base import VLAPolicy  # noqa: E402
from embodiinfer.policies.config import VLAPolicyConfig  # noqa: E402
from embodiinfer.policies.decoder import ActionDecoder, DecodeResult  # noqa: E402
from embodiinfer.types import Observation  # noqa: E402


class _Decoder(ActionDecoder):
    def __init__(self, policy):
        self.policy = policy

    def init_state(self, batch_size, generator=None):
        return torch.randn(batch_size, 1, 1, generator=generator)

    def produce_chunk(self, state, prefix, num_steps, bucket, graphs):
        self.policy.calls.append("decode")
        return self.policy.convert_actions(self.policy.unlisted_decode(prefix) + state[:, 0]).unsqueeze(1)

    def decode(self, state, prefix, num_steps, bucket, graphs, **kwargs):
        return DecodeResult(
            self.produce_chunk(state, prefix, num_steps, bucket, graphs),
            next_memory=SimpleNamespace(seq_len=1),
        )


class _Policy(VLAPolicy):
    def __init__(self, *, recurrent=False):
        super().__init__(VLAPolicyConfig(action_dim=1, action_horizon=1, default_num_steps=1))
        self.weight = torch.nn.Parameter(torch.ones(1).exp())
        self.calls = []
        self.recurrent = recurrent
        self._decoder = _Decoder(self)

    @property
    def is_recurrent(self):
        return self.recurrent

    @property
    def decoder(self):
        return self._decoder

    def collate(self, observations, request_ids):
        self.calls.append("collate")
        result = super().collate(observations, request_ids)
        result.state = result.state.sin()
        return result

    def encode_prefix(self, batch, memory=None):
        self.calls.append("encode")
        assert memory is None
        return batch.state.sigmoid()

    def unlisted_decode(self, prefix):
        return prefix.cos()

    def convert_actions(self, actions):
        return actions.tanh()


def _config(tmp_path: Path, **changes):
    row = {
        "images": torch.ones(1, 3, 2, 2),
        "state": torch.ones(1),
        "instruction_tokens": torch.ones(1).long(),
    }
    torch.save(row, tmp_path / "observation.pt")
    config = {"inputs": "observation.pt", "engine": {"device": "cpu", "use_cuda_graph": False}, "seed": 7}
    config.update(changes)
    path = tmp_path / "inference.json"
    atomic_json(path, config)
    return path, row


@pytest.mark.parametrize("model", factory.available_policies())
def test_every_registered_id_uses_complete_inference_scope(tmp_path: Path, monkeypatch, model: str):
    instances = []

    def build(**kwargs):
        policy = _Policy()
        instances.append(policy)
        return policy

    monkeypatch.setitem(factory._REGISTRY, model, build)
    path, row = _config(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        reference = GenerationBackend(EngineCore(build(), EngineConfig(device="cpu", use_cuda_graph=False)))
        expected = reference.generate([Observation(**row)])[0].actions
    before = torch.get_rng_state().clone()
    capture = capture_inference(model, path, tmp_path / "capture")
    assert torch.equal(torch.get_rng_state(), before)
    assert instances[-1].calls == ["collate", "encode", "decode"]
    assert torch.equal(torch.load(capture.root / "actions.pt", weights_only=True)[0], expected)
    names = {op["name"] for op in capture.operators}
    assert {"aten.sin.default", "aten.sigmoid.default", "aten.cos.default", "aten.tanh.default"} <= names
    assert "aten.exp.default" not in names  # Loading/initialization is outside inference.
    assert capture.manifest["execution"]["calls"] == 1
    assert capture.manifest["execution"]["outputs"] == 1
    tasks = render_model(capture, tmp_path / "tasks")
    assert len(tasks) == sum(op["status"] == "ready" for op in capture.operators)


def test_recurrent_inference_uses_a_fresh_explicit_session(tmp_path: Path, monkeypatch):
    policy = _Policy(recurrent=True)
    monkeypatch.setitem(factory._REGISTRY, "mock_flow_vla", lambda **kw: policy)
    path, _ = _config(tmp_path)
    capture = capture_inference("mock_flow_vla", path, tmp_path / "capture")
    metadata = read_json(capture.root / "inference.json")
    assert metadata["sessions"] == [{"env_id": 0, "episode_id": "kernel-tuning-capture", "rollout_id": 0}]
    assert policy.calls == ["collate", "encode", "decode"]


def test_failed_inference_cannot_publish_a_complete_capture(tmp_path: Path, monkeypatch):
    policy = _Policy()

    def fail(prefix):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(policy, "unlisted_decode", fail)
    monkeypatch.setitem(factory._REGISTRY, "mock_flow_vla", lambda **kw: policy)
    path, _ = _config(tmp_path)
    original_factory = factory.make_policy
    with pytest.raises(RuntimeError, match="decoder failed"):
        capture_inference("mock_flow_vla", path, tmp_path / "capture")
    assert factory.make_policy is original_factory
    capture = ModelCapture.load(tmp_path / "capture", require_ready=False)
    assert not capture.manifest["complete"]
    assert capture.manifest["execution"]["calls"] == 0
    with pytest.raises(ContractError, match="did not complete"):
        capture.require_ready()


def test_inference_rejects_compile_and_unavailable_cuda(tmp_path: Path, monkeypatch):
    path, _ = _config(tmp_path, policy={"compile_backend": "inductor"})
    with pytest.raises(ContractError, match="compile_backend"):
        capture_inference("mock_flow_vla", path, tmp_path / "compiled")
    path, _ = _config(tmp_path, engine={"device": "cuda"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ContractError, match="refusing a CPU fallback"):
        capture_inference("mock_flow_vla", path, tmp_path / "cuda")


def test_cli_runs_inference_without_a_preparation_script(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setitem(factory._REGISTRY, "mock_flow_vla", lambda **kw: _Policy())
    path, _ = _config(tmp_path)
    args = [
        "capture",
        "--model",
        "mock_flow_vla",
        "--inference-config",
        str(path),
        "--output",
        str(tmp_path / "capture"),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["execution"]["calls"] == 1
    assert main([*args, "--", "unused.py"]) == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_runtime_failure_stops_before_calibration(tmp_path: Path, monkeypatch):
    import sys

    from scripts.kernel_tuning.discovery import inference

    path, _ = _config(tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(inference.subprocess, "run", run)
    with pytest.raises(ContractError, match="capture failed"):
        prepare_inference("mock_flow_vla", path, tmp_path / "batches", sys.executable)
    assert len(commands) == 1
    assert commands[0][0] == sys.executable
    assert "--inference-config" in commands[0]


@pytest.mark.parametrize("extra", [["--selection", "scope.json"], ["--skip-preflight"], ["--catalog"]])
def test_end_to_end_tuning_cannot_skip_recorded_operators(tmp_path: Path, extra):
    path, _ = _config(tmp_path)
    assert (
        main(
            [
                "tune-all",
                "--model",
                "mock_flow_vla",
                "--inference-config",
                str(path),
                "--agent",
                "fake/model",
                *extra,
            ]
        )
        == 2
    )


@pytest.mark.parametrize("command", ["capture", "generate", "tune-all"])
@pytest.mark.parametrize("obsolete", ["--catalog", "--estimated", "--force", "--skip-preflight", "rms_norm"])
def test_removed_entrypoints_fail_before_execution(tmp_path: Path, monkeypatch, command: str, obsolete: str):
    calls = []
    monkeypatch.setitem(factory._REGISTRY, "mock_flow_vla", lambda **kw: calls.append(kw))
    path, _ = _config(tmp_path)
    output = tmp_path / "removed"
    args = [command, "--model", "mock_flow_vla", "--output", str(output)]
    if command == "capture":
        args += ["--inference-config", str(path)]
    assert main([*args, obsolete]) == 2
    assert not calls
    assert not output.exists()


def test_cli_requires_inference_config_instead_of_an_external_script(tmp_path: Path):
    script = tmp_path / "old.py"
    marker = tmp_path / "executed"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n", encoding="utf-8")
    assert (
        main(
            ["capture", "--model", "mock_flow_vla", "--output", str(tmp_path / "capture"), "--", str(script)]
        )
        == 2
    )
    assert not marker.exists()
    assert not (tmp_path / "capture").exists()
