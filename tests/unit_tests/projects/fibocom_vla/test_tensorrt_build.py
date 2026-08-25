# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests for fail-closed TensorRT engine construction and loading."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rlinf.projects.fibocom_vla.errors import (
    ConfigurationError,
    OptionalDependencyError,
)
from rlinf.projects.fibocom_vla.inference.tensorrt_build import (
    ArtifactDigest,
    BuildEnvironment,
    EngineManifest,
    EnginePrecision,
    InputShapeRange,
    OptimizationProfile,
    SourceCheckpointIdentity,
    TensorRTBuildSpec,
    TensorRTV3Runtime,
    build_tensorrt_engine,
)


def _write(root: Path, relative: str, payload: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _environment(
    *, gpu_name: str = "Fake GPU", gpu_device_index: int = 0
) -> BuildEnvironment:
    return BuildEnvironment(
        tensorrt_version="10.8.0",
        cuda_version="12.6",
        gpu_name=gpu_name,
        gpu_compute_capability="8.9",
        gpu_device_index=gpu_device_index,
        gpu_total_memory_bytes=24 * 1024**3,
    )


def _profile(*, maximum_batch: int = 2) -> OptimizationProfile:
    return OptimizationProfile(
        name="robot_batch",
        inputs=(
            InputShapeRange(
                name="input",
                min_shape=(1, 3),
                opt_shape=(1, 3),
                max_shape=(maximum_batch, 3),
            ),
        ),
    )


def _source(root: Path) -> SourceCheckpointIdentity:
    return SourceCheckpointIdentity(
        revision="a" * 40,
        files=(ArtifactDigest.from_path(root, "checkpoint/model.safetensors"),),
        norm_stats=ArtifactDigest.from_path(root, "checkpoint/norm_stats.json"),
        transforms=(ArtifactDigest.from_path(root, "source/transforms.py"),),
    )


def _prepare_root(tmp_path: Path) -> Path:
    _write(tmp_path, "checkpoint/model.safetensors", b"checkpoint-bytes")
    _write(tmp_path, "checkpoint/norm_stats.json", b'{"norm":"bound"}\n')
    _write(tmp_path, "source/transforms.py", b"TRANSFORM_VERSION = 1\n")
    _write(tmp_path, "export/model.onnx", b"exported-network")
    _write(tmp_path, "engines/pi05.plan", b"serialized-engine")
    return tmp_path


def _manifest(
    root: Path, *, environment: BuildEnvironment | None = None
) -> EngineManifest:
    return EngineManifest(
        engine=ArtifactDigest.from_path(root, "engines/pi05.plan"),
        source_checkpoint=_source(root),
        build_inputs=(ArtifactDigest.from_path(root, "export/model.onnx"),),
        build_source_kind="onnx",
        environment=environment or _environment(),
        precision=EnginePrecision.BF16,
        optimization_profiles=(_profile(),),
        workspace_size_bytes=1024**3,
    )


def test_manifest_round_trip_verifies_all_bound_artifacts(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    manifest = _manifest(root)
    manifest_path = root / "engines/pi05.manifest.json"
    manifest.write(manifest_path)

    parsed = EngineManifest.from_path(manifest_path)
    verified = parsed.verify(
        root,
        expected_source_checkpoint=manifest.source_checkpoint,
        expected_environment=_environment(),
        expected_precision=EnginePrecision.BF16,
        expected_profiles=(_profile(),),
    )

    assert parsed == manifest
    assert verified.engine_path == (root / "engines/pi05.plan").resolve()
    assert parsed.to_json() == manifest.to_json()


def test_manifest_precision_flag_is_explicit_and_schema_v1_remains_readable(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    manifest = _manifest(root)

    payload = manifest.to_dict()

    assert payload["schema_version"] == 2
    assert payload["enabled_builder_precision_flag"] == "bf16"
    assert "precision" not in payload
    assert manifest.enabled_builder_precision_flag is EnginePrecision.BF16
    assert (
        EngineManifest.from_dict(payload).enabled_builder_precision_flag
        is EnginePrecision.BF16
    )

    legacy_payload = dict(payload)
    legacy_payload["schema_version"] = 1
    legacy_payload["precision"] = legacy_payload.pop("enabled_builder_precision_flag")
    legacy = EngineManifest.from_dict(legacy_payload)
    assert legacy.schema_version == 1
    assert legacy.enabled_builder_precision_flag is EnginePrecision.BF16
    assert legacy.to_dict() == legacy_payload


def test_manifest_rejects_same_size_engine_tampering(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    manifest = _manifest(root)
    (root / "engines/pi05.plan").write_bytes(b"tampered-engine!!")

    with pytest.raises(ConfigurationError, match="SHA-256 mismatch"):
        manifest.verify(root)


def test_manifest_rejects_environment_and_profile_mismatches(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    manifest = _manifest(root)

    with pytest.raises(ConfigurationError, match="environment mismatch"):
        manifest.verify(root, expected_environment=_environment(gpu_name="Other GPU"))
    with pytest.raises(ConfigurationError, match="profile set mismatch"):
        manifest.verify(root, expected_profiles=(_profile(maximum_batch=4),))
    with pytest.raises(ConfigurationError, match="rejected shapes"):
        manifest.require_profile("robot_batch", {"input": (3, 3)})
    with pytest.raises(ConfigurationError, match="input mismatch"):
        manifest.require_profile("robot_batch", {"wrong": (1, 3)})


def test_manifest_json_and_paths_are_strict(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    data = _manifest(root).to_dict()
    data["unreviewed"] = True
    with pytest.raises(ConfigurationError, match="unknown=unreviewed"):
        EngineManifest.from_json(json.dumps(data))

    with pytest.raises(ConfigurationError, match="duplicate JSON key"):
        EngineManifest.from_json('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ConfigurationError, match="canonical relative path"):
        ArtifactDigest(path="../escape.plan", sha256="0" * 64, size_bytes=1)


class _FakeProfile:
    def __init__(self) -> None:
        self.shapes: list[tuple[Any, ...]] = []

    def set_shape(self, *shape: Any) -> bool:
        self.shapes.append(shape)
        return True


class _FakeConfig:
    def __init__(self) -> None:
        self.profiles: list[_FakeProfile] = []
        self.flags: list[Any] = []

    def set_memory_pool_limit(self, pool: Any, size: int) -> None:
        assert pool == "workspace"
        assert size > 0

    def set_flag(self, flag: Any) -> None:
        self.flags.append(flag)

    def add_optimization_profile(self, profile: _FakeProfile) -> int:
        self.profiles.append(profile)
        return len(self.profiles) - 1


class _FakeBuilder:
    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def create_network(self, flags: int) -> dict[str, int]:
        return {"flags": flags}

    def create_builder_config(self) -> _FakeConfig:
        return _FakeConfig()

    def create_optimization_profile(self) -> _FakeProfile:
        return _FakeProfile()

    def build_serialized_network(self, network: Any, config: Any) -> bytes:
        assert network["populated"] is True
        assert len(config.profiles) == 1
        return b"fake-trt10-engine"


class _FakeLogger:
    WARNING = "warning"

    def __init__(self, severity: Any) -> None:
        self.severity = severity


def _fake_builder_trt() -> Any:
    return SimpleNamespace(
        __version__="10.8.0",
        Logger=_FakeLogger,
        Builder=_FakeBuilder,
        NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0),
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
        BuilderFlag=SimpleNamespace(BF16="bf16", FP16="fp16", INT8="int8"),
    )


def test_callback_builder_writes_engine_and_complete_manifest(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    (root / "engines/pi05.plan").unlink()
    spec = TensorRTBuildSpec(
        source_checkpoint=_source(root),
        build_inputs=(ArtifactDigest.from_path(root, "export/model.onnx"),),
        precision=EnginePrecision.BF16,
        optimization_profiles=(_profile(),),
        workspace_size_bytes=1024**2,
    )

    def populate(builder: Any, network: dict[str, Any], logger: Any) -> object:
        assert builder is not None
        assert logger is not None
        network["populated"] = True
        return object()

    verified = build_tensorrt_engine(
        root,
        engine_path="engines/pi05.plan",
        manifest_path="engines/pi05.manifest.json",
        spec=spec,
        network_populator=populate,
        observed_environment=_environment(),
        trt_module=_fake_builder_trt(),
    )

    assert verified.manifest.build_source_kind == "network_callback"
    assert verified.engine_path.read_bytes() == b"fake-trt10-engine"
    reloaded = EngineManifest.from_path(root / "engines/pi05.manifest.json")
    assert reloaded == verified.manifest


def test_builder_and_runtime_report_missing_optional_tensorrt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rlinf.projects.fibocom_vla.inference import tensorrt_build

    root = _prepare_root(tmp_path)
    manifest_path = root / "engines/pi05.manifest.json"
    _manifest(root).write(manifest_path)
    spec = TensorRTBuildSpec(
        source_checkpoint=_source(root),
        build_inputs=(ArtifactDigest.from_path(root, "export/model.onnx"),),
        precision=EnginePrecision.BF16,
        optimization_profiles=(_profile(),),
    )
    original_import = tensorrt_build.importlib.import_module

    def missing_tensorrt(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tensorrt":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(tensorrt_build.importlib, "import_module", missing_tensorrt)
    with pytest.raises(OptionalDependencyError, match="requires the tensorrt package"):
        build_tensorrt_engine(
            root,
            engine_path="engines/new.plan",
            manifest_path="engines/new.manifest.json",
            spec=spec,
            onnx_path="export/model.onnx",
            observed_environment=_environment(),
        )
    with pytest.raises(OptionalDependencyError, match="requires the tensorrt package"):
        TensorRTV3Runtime.from_manifest(
            manifest_path,
            root,
            profile_name="robot_batch",
            observed_environment=_environment(),
        )


class _FakeTensor:
    _next_pointer = 100

    def __init__(
        self, shape: tuple[int, ...], dtype: Any, device: str = "cuda:0"
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.is_cuda = True
        self.pointer = _FakeTensor._next_pointer
        _FakeTensor._next_pointer += 100

    def is_contiguous(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return self.pointer


class _FakeStream:
    def __init__(self, cuda_stream: int = 77) -> None:
        self.cuda_stream = cuda_stream
        self.synchronized = False

    def synchronize(self) -> None:
        self.synchronized = True


class _FakeDeviceScope:
    def __init__(self, cuda: _TrackingCuda, index: int) -> None:
        self.cuda = cuda
        self.index = index
        self.previous: int | None = None

    def __enter__(self) -> None:
        self.previous = getattr(self.cuda.local, "active_device", None)
        self.cuda.local.active_device = self.index
        self.cuda.entered_devices.append(self.index)

    def __exit__(self, *args: object) -> None:
        self.cuda.local.active_device = self.previous


class _TrackingCuda:
    def __init__(self, stream: _FakeStream) -> None:
        self.stream = stream
        self.local = threading.local()
        self.entered_devices: list[int] = []

    @property
    def active_device(self) -> int | None:
        return getattr(self.local, "active_device", None)

    def device(self, index: int) -> _FakeDeviceScope:
        return _FakeDeviceScope(self, index)

    def current_stream(self, *, device: str) -> _FakeStream:
        expected = int(device.split(":", maxsplit=1)[1])
        assert self.active_device == expected
        return self.stream


class _FakeContext:
    def __init__(self) -> None:
        self.addresses: dict[str, int] = {}
        self.executed_stream: int | None = None

    def set_optimization_profile_async(self, index: int, stream: int) -> bool:
        return index == 0 and stream > 0

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        return name == "input" and shape == (1, 3)

    def set_tensor_address(self, name: str, pointer: int) -> bool:
        self.addresses[name] = pointer
        return True

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        assert name == "output"
        return (1, 2)

    def execute_async_v3(self, stream: int) -> bool:
        self.executed_stream = stream
        return True


class _FakeEngine:
    num_io_tensors = 2
    num_optimization_profiles = 1

    def __init__(self) -> None:
        self.contexts: list[_FakeContext] = []
        self.context: _FakeContext | None = None

    def get_tensor_name(self, index: int) -> str:
        return ("input", "output")[index]

    def get_tensor_mode(self, name: str) -> str:
        return "input" if name == "input" else "output"

    def get_tensor_dtype(self, name: str) -> str:
        return "float32"

    def get_tensor_profile_shape(
        self, name: str, profile_index: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert name == "input"
        assert profile_index == 0
        return ((1, 3), (1, 3), (2, 3))

    def create_execution_context(self) -> _FakeContext:
        self.context = _FakeContext()
        self.contexts.append(self.context)
        return self.context


class _FakeRuntime:
    def __init__(self, logger: Any) -> None:
        self.logger = logger
        self.engine = _FakeEngine()

    def deserialize_cuda_engine(self, payload: bytes) -> _FakeEngine:
        assert payload == b"serialized-engine"
        return self.engine


class _FakeBadProfileEngine(_FakeEngine):
    def get_tensor_profile_shape(
        self, name: str, profile_index: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert name == "input"
        assert profile_index == 0
        return ((1, 3), (1, 3), (4, 3))


class _FakeBadProfileRuntime(_FakeRuntime):
    def __init__(self, logger: Any) -> None:
        self.logger = logger
        self.engine = _FakeBadProfileEngine()


def test_v3_runtime_rejects_profile_not_encoded_by_engine(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    manifest_path = root / "engines/pi05.manifest.json"
    _manifest(root).write(manifest_path)
    fake_trt = SimpleNamespace(
        Logger=_FakeLogger,
        Runtime=_FakeBadProfileRuntime,
        TensorIOMode=SimpleNamespace(INPUT="input"),
    )

    with pytest.raises(ConfigurationError, match="profile mismatch"):
        TensorRTV3Runtime.from_manifest(
            manifest_path,
            root,
            profile_name="robot_batch",
            observed_environment=_environment(),
            trt_module=fake_trt,
            torch_module=SimpleNamespace(),
        )


def test_v3_runtime_binds_torch_cuda_pointers_and_executes(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    manifest_path = root / "engines/pi05.manifest.json"
    _manifest(root).write(manifest_path)
    stream = _FakeStream()
    fake_torch = SimpleNamespace(
        float32="float32",
        cuda=SimpleNamespace(current_stream=lambda *, device: stream),
        empty=lambda shape, *, dtype, device: _FakeTensor(shape, dtype, device),
    )
    fake_trt = SimpleNamespace(
        Logger=_FakeLogger,
        Runtime=_FakeRuntime,
        TensorIOMode=SimpleNamespace(INPUT="input"),
    )
    runtime = TensorRTV3Runtime.from_manifest(
        manifest_path,
        root,
        profile_name="robot_batch",
        expected_source_checkpoint=_source(root),
        expected_precision=EnginePrecision.BF16,
        expected_profiles=(_profile(),),
        observed_environment=_environment(),
        trt_module=fake_trt,
        torch_module=fake_torch,
    )

    outputs = runtime.execute(
        {"input": _FakeTensor((1, 3), "float32")}, synchronize=True
    )

    assert tuple(outputs) == ("output",)
    assert outputs["output"].shape == (1, 2)
    assert runtime.context.addresses.keys() == {"input", "output"}
    assert runtime.context.executed_stream == 77
    assert stream.synchronized is True


def test_v3_runtime_rejects_input_on_gpu_other_than_manifest(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    environment = _environment(gpu_device_index=1)
    manifest_path = root / "engines/pi05.manifest.json"
    _manifest(root, environment=environment).write(manifest_path)
    stream = _FakeStream()
    fake_torch = SimpleNamespace(
        float32="float32",
        cuda=SimpleNamespace(current_stream=lambda *, device: stream),
        empty=lambda shape, *, dtype, device: _FakeTensor(shape, dtype, device),
    )
    fake_trt = SimpleNamespace(
        Logger=_FakeLogger,
        Runtime=_FakeRuntime,
        TensorIOMode=SimpleNamespace(INPUT="input"),
    )
    runtime = TensorRTV3Runtime.from_manifest(
        manifest_path,
        root,
        profile_name="robot_batch",
        observed_environment=environment,
        trt_module=fake_trt,
        torch_module=fake_torch,
    )

    with pytest.raises(ConfigurationError, match="manifest requires cuda:1"):
        runtime.execute({"input": _FakeTensor((1, 3), "float32", "cuda:0")})


def test_v3_runtime_creates_and_executes_under_manifest_device_scope(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    environment = _environment(gpu_device_index=1)
    manifest_path = root / "engines/pi05.manifest.json"
    _manifest(root, environment=environment).write(manifest_path)
    tracking_cuda = _TrackingCuda(_FakeStream())

    class GuardedContext(_FakeContext):
        def execute_async_v3(self, stream: int) -> bool:
            assert tracking_cuda.active_device == 1
            return super().execute_async_v3(stream)

    class GuardedEngine(_FakeEngine):
        def create_execution_context(self) -> _FakeContext:
            assert tracking_cuda.active_device == 1
            self.context = GuardedContext()
            self.contexts.append(self.context)
            return self.context

    class GuardedRuntime(_FakeRuntime):
        def __init__(self, logger: Any) -> None:
            assert tracking_cuda.active_device == 1
            self.logger = logger
            self.engine = GuardedEngine()

        def deserialize_cuda_engine(self, payload: bytes) -> _FakeEngine:
            assert tracking_cuda.active_device == 1
            return super().deserialize_cuda_engine(payload)

    fake_torch = SimpleNamespace(
        float32="float32",
        cuda=tracking_cuda,
        empty=lambda shape, *, dtype, device: _FakeTensor(shape, dtype, device),
    )
    fake_trt = SimpleNamespace(
        Logger=_FakeLogger,
        Runtime=GuardedRuntime,
        TensorIOMode=SimpleNamespace(INPUT="input"),
    )

    runtime = TensorRTV3Runtime.from_manifest(
        manifest_path,
        root,
        profile_name="robot_batch",
        observed_environment=environment,
        trt_module=fake_trt,
        torch_module=fake_torch,
    )
    runtime.execute({"input": _FakeTensor((1, 3), "float32", "cuda:1")})

    assert tracking_cuda.entered_devices == [1, 1]
    assert tracking_cuda.active_device is None


def test_v3_runtime_uses_independent_contexts_for_concurrent_threads(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    manifest_path = root / "engines/pi05.manifest.json"
    _manifest(root).write(manifest_path)
    fake_torch = SimpleNamespace(
        float32="float32",
        cuda=SimpleNamespace(current_stream=lambda *, device: _FakeStream()),
        empty=lambda shape, *, dtype, device: _FakeTensor(shape, dtype, device),
    )
    fake_trt = SimpleNamespace(
        Logger=_FakeLogger,
        Runtime=_FakeRuntime,
        TensorIOMode=SimpleNamespace(INPUT="input"),
    )
    runtime = TensorRTV3Runtime.from_manifest(
        manifest_path,
        root,
        profile_name="robot_batch",
        observed_environment=_environment(),
        trt_module=fake_trt,
        torch_module=fake_torch,
    )
    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def execute() -> None:
        try:
            start.wait()
            runtime.execute({"input": _FakeTensor((1, 3), "float32")})
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=execute) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(runtime.engine.contexts) == 2
    assert len({id(context) for context in runtime.engine.contexts}) == 2
    assert all(context.executed_stream == 77 for context in runtime.engine.contexts)


def test_v3_runtime_uses_independent_contexts_for_distinct_streams(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    manifest_path = root / "engines/pi05.manifest.json"
    _manifest(root).write(manifest_path)
    selected_stream = [_FakeStream(77)]
    fake_torch = SimpleNamespace(
        float32="float32",
        cuda=SimpleNamespace(current_stream=lambda *, device: selected_stream[0]),
        empty=lambda shape, *, dtype, device: _FakeTensor(shape, dtype, device),
    )
    fake_trt = SimpleNamespace(
        Logger=_FakeLogger,
        Runtime=_FakeRuntime,
        TensorIOMode=SimpleNamespace(INPUT="input"),
    )
    runtime = TensorRTV3Runtime.from_manifest(
        manifest_path,
        root,
        profile_name="robot_batch",
        observed_environment=_environment(),
        trt_module=fake_trt,
        torch_module=fake_torch,
    )

    runtime.execute({"input": _FakeTensor((1, 3), "float32")})
    selected_stream[0] = _FakeStream(88)
    runtime.execute({"input": _FakeTensor((1, 3), "float32")})

    assert len(runtime.engine.contexts) == 2
    assert [context.executed_stream for context in runtime.engine.contexts] == [77, 88]
