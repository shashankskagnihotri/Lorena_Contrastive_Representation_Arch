import numpy as np
import pytest
import torch
import json
import os
from pathlib import Path
from types import SimpleNamespace

from sparse_contrast.benchmark import (SCOPES, _profile, _serial_measure, _sustained_measure,
                                      summarize_times, timing_batches, validate_configuration_groups,
                                      canonical_gpu_uuid, gpu_isolation, hardware_class_identity,
                                      save_measurement_session, validate_measurement_session, measure_cell,
                                      _warmup_scope)

LEGACY_SCOPE = {"schema_version": 1, "active_representations": {
    "mnist": ["single_color", "grayscale", "color_opponency"],
    "cifar10": ["single_color", "grayscale", "color_opponency"]}}
SCOPED_SCOPE = {"schema_version": 1, "active_representations": {
    "mnist": ["grayscale"], "cifar10": ["single_color", "grayscale", "color_opponency"]}}


def test_paired_timing_membership_deterministic_and_private_rng():
    torch.manual_seed(8)
    state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    warm, measured = timing_batches(list(range(1600)), 8)
    assert len(warm) == 50 and len(measured) == 200
    assert len({i for batch in measured for i in batch}) == 1600
    assert all(len(batch) == 8 for batch in warm + measured)
    assert (warm, measured) == timing_batches(list(range(1600)), 8)
    assert torch.equal(state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    _, batch_one = timing_batches(list(range(1600)), 1)
    assert {i for batch in batch_one for i in batch} == set(range(200))


def test_latency_population_summary_and_separate_observed_throughput():
    rows = [{"elapsed_ms": value} for value in (1., 2., 3., 4.)]
    result = summarize_times(rows, 8)
    assert result["mean_ms"] == result["median_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["serial_images_per_second"] == 3200
    assert result["amortized_mean_ms_per_image"] == 2.5 / 8
    calls = []
    throughput = _sustained_measure(lambda i: calls.append(i), 10, 8, torch.device("cpu"))
    assert calls == list(range(10))
    assert throughput["images"] == 80 and throughput["elapsed_seconds"] > 0


def test_complete_nineteen_configuration_groups_detect_missing_control():
    shared = {"dataset": "mnist", "architecture": "vit_small", "seed": 0}
    configs = [{**shared, "representation": "raw", "execution": "dense", "sparsity_percent": None}]
    for representation in ("single_color", "grayscale", "color_opponency"):
        configs.append({**shared, "representation": representation, "execution": "dense", "sparsity_percent": 0})
        for percent in (0, 20, 40, 60, 80):
            configs.append({**shared, "representation": representation, "execution": "compact", "sparsity_percent": percent})
    assert len(configs) == 19
    validate_configuration_groups(configs, LEGACY_SCOPE)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_configuration_groups(configs[1:], LEGACY_SCOPE)
    with pytest.raises(ValueError, match="duplicated"):
        validate_configuration_groups(configs + configs[:1], LEGACY_SCOPE)


class InstrumentedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.profiling = False
        self.calls = 0

    def set_profiling(self, enabled):
        self.profiling = enabled

    def forward(self, x, execution=None):
        self.calls += 1
        if self.profiling:
            with torch.profiler.record_function("frontend/test_native_conversion"):
                x = x.float() / 255
            with torch.profiler.record_function("normalization/test_frozen_mean_std"):
                return (x - 0.5) / 0.25
        return (x.float() / 255 - 0.5) / 0.25


def test_primary_timer_runs_online_call_and_profiler_is_separate(tmp_path):
    model = InstrumentedModel()
    raw = torch.ones(2, 1, 4, 4, dtype=torch.uint8)
    rows = _serial_measure(lambda i: model(raw), 4, torch.device("cpu"))
    assert model.calls == 4 and not model.profiling
    assert len(rows) == 4 and all(row["cuda_stream_elapsed_ms"] is None for row in rows)
    profile = _profile(model, raw, "dense", torch.device("cpu"), tmp_path, rows[0]["elapsed_ms"])
    assert model.calls == 5 and not model.profiling
    assert profile["stage_intervals_additive"] is False
    assert profile["cuda_profiler_available"] is False
    assert (tmp_path / "trace.json").exists() and (tmp_path / "profile.json").exists()
    names = {row["name"] for row in profile["events"]}
    assert "frontend/test_native_conversion" in names
    assert "normalization/test_frozen_mean_std" in names
    assert SCOPES[:3] == ("gpu_raw_to_logits", "host_raw_to_logits", "loader_to_cpu_prediction")


BARE_UUID = "ad155c1d-e69a-2415-2753-0199cbf30f1e"
OTHER_UUID = "aa155c1d-e69a-2415-2753-0199cbf30f1e"


@pytest.mark.parametrize("value,expected", [
    (BARE_UUID, "GPU-" + BARE_UUID),
    (BARE_UUID.upper(), "GPU-" + BARE_UUID),
    ("GPU-" + BARE_UUID, "GPU-" + BARE_UUID),
    ("gpu-" + BARE_UUID.upper(), "GPU-" + BARE_UUID),
    ("MIG-" + BARE_UUID, "MIG-" + BARE_UUID),
    ("mig-gpu-" + BARE_UUID.upper() + "/1/0", "MIG-GPU-" + BARE_UUID + "/1/0"),
])
def test_nvidia_uuid_canonicalization(value, expected):
    assert canonical_gpu_uuid(value) == expected


@pytest.mark.parametrize("value", ["", "N/A", "GPU-not-a-uuid", "MIG-GPU-" + BARE_UUID + "/oops"])
def test_invalid_uuid_fails_explicitly(value):
    with pytest.raises(ValueError):
        canonical_gpu_uuid(value)


def mock_nvidia(monkeypatch, selected_uuid, compute_rows):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid=BARE_UUID))
    def query(command, **kwargs):
        if "--query-gpu=uuid" in command:
            assert "--id=GPU-" + BARE_UUID in command
            return SimpleNamespace(stdout=selected_uuid + "\n")
        assert "--query-compute-apps=gpu_uuid,pid" in command
        return SimpleNamespace(stdout=compute_rows)
    monkeypatch.setattr("sparse_contrast.benchmark.subprocess.run", query)


def test_isolation_matches_bare_torch_and_prefixed_nvidia_uuid(monkeypatch):
    mock_nvidia(monkeypatch, "gpu-" + BARE_UUID.upper(), f"GPU-{BARE_UUID}, {os.getpid()}\n")
    result = gpu_isolation(torch.device("cuda:0"))
    assert result["verified"] is True
    assert result["device_uuid"] == "GPU-" + BARE_UUID
    assert result["observed_compute_pids"] == [os.getpid()]


def test_isolation_catches_foreign_pid_despite_uuid_prefix_difference(monkeypatch):
    foreign_pid = os.getpid() + 100000
    mock_nvidia(monkeypatch, "GPU-" + BARE_UUID, f"gpu-{BARE_UUID.upper()}, {foreign_pid}\n")
    with pytest.raises(RuntimeError, match="competing compute PIDs"):
        gpu_isolation(torch.device("cuda:0"))


@pytest.mark.parametrize("selected", ["", "GPU-" + OTHER_UUID])
def test_isolation_rejects_unidentified_or_wrong_selected_gpu(monkeypatch, selected):
    mock_nvidia(monkeypatch, selected, "")
    with pytest.raises(RuntimeError, match="not uniquely identified"):
        gpu_isolation(torch.device("cuda:0"))


def session_metadata(session, uuid, gpu="NVIDIA H100 NVL"):
    metadata = dict(gpu=gpu, compute_capability=[9, 0], driver_version="580.65.06",
                    torch="2.9.1+cu126", cuda="12.6", cudnn=91002,
                    precision="float32", backend="PyTorch_explicit_attention",
                    device_uuid=uuid, measurement_session_id=session, host="allocated-node", job_id="12345")
    metadata["hardware_identity"] = hardware_class_identity(metadata)
    return metadata


def test_same_gpu_class_resumes_across_physical_sessions_without_overwrite(tmp_path):
    first = session_metadata("session_first", "GPU-" + BARE_UUID)
    second = session_metadata("session_second", "GPU-" + OTHER_UUID)
    assert first["hardware_identity"] == second["hardware_identity"]
    assert hardware_class_identity(session_metadata("other", "GPU-" + OTHER_UUID, gpu="NVIDIA RTX 2080")) != first["hardware_identity"]
    reference_first = save_measurement_session(tmp_path, first)
    original = (tmp_path / "hardware.json").read_bytes()
    reference_second = save_measurement_session(tmp_path, second)
    assert (tmp_path / "hardware.json").read_bytes() == original
    assert validate_measurement_session(reference_first) == first
    assert validate_measurement_session(reference_second) == second
    # A completed cell belongs to its original session, but its scientific
    # identity is reusable after a scheduler continuation on another same-class GPU.
    directory = tmp_path / "cells" / "clean__test"
    directory.mkdir(parents=True)
    identity = {"config_hash": "frozen_config", "hardware_identity": first["hardware_identity"]}
    saved = {"state": "complete", "identity": identity, "measurement": reference_first, "file_hashes": {}}
    (directory / "summary.json").write_text(json.dumps(saved))
    reused = measure_cell(None, {}, "compact", None, "clean", "test", 1,
                          torch.device("cpu"), directory, identity, measurement=reference_second)
    assert reused == saved
    assert reused["measurement"]["measurement_session_id"] == "session_first"
    Path(reference_first["hardware_metadata_file"]).write_text("{}");
    with pytest.raises(ValueError, match="provenance changed"):
        validate_measurement_session(reference_first)


def test_scope_allocator_cleanup_precedes_all_warmups_and_peak_reset(monkeypatch):
    """Mock CUDA API only: test ordering without a GPU context or allocation."""
    order = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: order.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: order.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: order.append("reset_peak"))
    monkeypatch.setattr("sparse_contrast.benchmark.gc.collect", lambda: order.append("gc_collect"))
    _warmup_scope(lambda index: order.append(("warmup", index)), 50, torch.device("cuda:0"))
    assert order == ["synchronize", "gc_collect", "empty_cache"] + [("warmup", index) for index in range(50)] + ["synchronize", "reset_peak"]


def full_registry(scope=LEGACY_SCOPE):
    """Hand-countable 2 datasets x 2 architectures x 3 seeds x 19 methods."""
    from sparse_contrast.common import digest
    configs = []
    for dataset in ("mnist", "cifar10"):
        for architecture in ("vit_small", "swin_tiny"):
            for seed in (0, 1, 2):
                methods = [("raw", "dense", None)]
                for representation in scope["active_representations"][dataset]:
                    methods.append((representation, "dense", 0))
                    methods.extend((representation, "compact", p) for p in (0, 20, 40, 60, 80))
                for representation, execution, percent in methods:
                    config = dict(dataset=dataset, architecture=architecture, seed=seed,
                                  representation=representation, execution=execution, sparsity_percent=percent,
                                  source_hash="frozen_source", protocol="heldout_val", normalization_hash="frozen_norm",
                                  registry_id=f"{dataset}_{architecture}_{seed}_{representation}_{execution}_{percent}")
                    config["config_hash"] = digest(config)
                    configs.append(config)
    return configs


def test_matched_blocks_exact_24_times_34_no_overlap_and_stable_order():
    from sparse_contrast.benchmark import make_matched_blocks
    configs = full_registry()
    paths = [f"/frozen/{config['config_hash']}.json" for config in configs]
    blocks = make_matched_blocks(configs, paths, LEGACY_SCOPE)
    assert len(configs) == 228 and len(blocks) == 24
    assert blocks == make_matched_blocks(configs[::-1], paths[::-1], LEGACY_SCOPE)
    identities = []
    by_hash = {config["config_hash"]: config for config in configs}
    for block in blocks:
        work = block["work"]
        assert len(work) == 34
        assert {execution: sum(item["execution"] == execution for item in work)
                for execution in ("dense", "compact", "dense_masked")} == {"dense": 4, "compact": 15, "dense_masked": 15}
        assert {item["config_hash"] for item in work if item["execution"] == "compact"} == {
            item["config_hash"] for item in work if item["execution"] == "dense_masked"}
        for item in work:
            config = by_hash[item["config_hash"]]
            assert (config["dataset"], config["architecture"], config["seed"], item["batch_size"]) == (
                block["dataset"], block["architecture"], block["seed"], block["batch_size"])
            identities.append((item["config_hash"], item["execution"], item["batch_size"]))
    assert len(identities) == len(set(identities)) == 816
    with pytest.raises(ValueError, match="Incomplete|228"):
        make_matched_blocks(configs[:-1], paths[:-1], LEGACY_SCOPE)
    with pytest.raises(ValueError, match="duplicated|228"):
        make_matched_blocks(configs + configs[:1], paths + paths[:1], LEGACY_SCOPE)


@pytest.mark.parametrize("node,gpu,expected", [
    ("NodeName=n1 Gres=gpu:nvidia_h100_nvl:4(S:0-1) State=MIXED", "NVIDIA H100 NVL", "nvidia_h100_nvl"),
    ("NodeName=n2 Gres=gpu:nvidia_geforce_rtx_2080_ti:4(S:0-1)", "NVIDIA GeForce RTX 2080 Ti", "nvidia_geforce_rtx_2080_ti"),
    ("NodeName=n3 Gres=gpu:a100:2(S:0),gpu:h100:2(S:1)", "NVIDIA H100 PCIe", "h100"),
])
def test_gres_selector_matches_cuda_model(node, gpu, expected):
    from sparse_contrast.benchmark import parse_gpu_gres_type
    assert parse_gpu_gres_type(node, gpu) == expected


@pytest.mark.parametrize("node,gpu", [
    ("NodeName=n Gres=gpu:4", "NVIDIA H100 NVL"),
    ("NodeName=n Gres=gpu:a100:4", "NVIDIA H100 NVL"),
    ("NodeName=n Gres=(null)", "NVIDIA H100 NVL"),
    ("NodeName=n", "NVIDIA H100 NVL"),
])
def test_gres_selector_rejects_untyped_missing_and_wrong_model(node, gpu):
    from sparse_contrast.benchmark import parse_gpu_gres_type
    with pytest.raises(ValueError):
        parse_gpu_gres_type(node, gpu)


def test_scheduler_gpu_selection_records_allocated_node(monkeypatch):
    from sparse_contrast.benchmark import scheduler_gpu_selection
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu-vram-94gb")
    monkeypatch.setenv("SLURMD_NODENAME", "allocated-node")
    def run(command, **kwargs):
        assert command == ["scontrol", "-o", "show", "node", "allocated-node"]
        return SimpleNamespace(stdout="NodeName=allocated-node Gres=gpu:nvidia_h100_nvl:4(S:0-1)\n")
    monkeypatch.setattr("sparse_contrast.benchmark.subprocess.run", run)
    selected = scheduler_gpu_selection({"gpu": "NVIDIA H100 NVL", "host": "allocated-node.local"})
    assert selected["gpu_partition"] == selected["slurm_partition"] == "gpu-vram-94gb"
    assert selected["gpu_gres_type"] == "nvidia_h100_nvl"
    assert selected["selector_node"] == "allocated-node"


@pytest.fixture
def frozen_plan(tmp_path, monkeypatch, request):
    import sparse_contrast.benchmark as benchmark
    from sparse_contrast.common import atomic_json
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    scope = getattr(request, "param", LEGACY_SCOPE)
    cohort = ""
    if isinstance(scope, tuple):
        scope, cohort = scope
    monkeypatch.setenv("SPARSE_CONTRAST_TIMING_COHORT", cohort)
    atomic_json(tmp_path / "experiment_scope.json", scope)
    configs = full_registry(scope)
    entries = []
    for config in configs:
        path = tmp_path / "configs" / f"{config['config_hash']}.json"
        atomic_json(path, config)
        entries.append({"config": config, "config_path": str(path)})
    campaign = tmp_path / "campaign.json"
    atomic_json(campaign, {"main": entries, "heartbeat": 1, "source_hash": "frozen_source",
                           "orchestration_source_hash": "measurement_source",
                           **({"latency_cohort": {"id": cohort}} if cohort else {})})
    plan, returned = benchmark.freeze_campaign_plan(campaign)
    assert returned == configs
    assert benchmark.load_frozen_blocks(root=tmp_path) == plan
    return tmp_path, campaign, plan


def test_frozen_plan_resumes_despite_mutable_campaign_heartbeat(frozen_plan):
    from sparse_contrast.benchmark import freeze_campaign_plan, load_frozen_blocks
    from sparse_contrast.common import atomic_json
    root, path, plan = frozen_plan
    campaign = json.loads(path.read_text())
    campaign["heartbeat"] = 999999
    campaign["main"].reverse()
    atomic_json(path, campaign)
    assert freeze_campaign_plan(path)[0] == plan
    assert load_frozen_blocks(root=root) == plan
    order_path = root / "outputs/benchmarks/order.json"
    order = json.loads(order_path.read_text())
    order["work"].reverse()
    atomic_json(order_path, order)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_frozen_blocks(root=root)


@pytest.fixture
def completed_block(frozen_plan, monkeypatch):
    """Real saved evidence graph, small corruption enumeration for the fixture."""
    import sparse_contrast.benchmark as benchmark
    from sparse_contrast.common import atomic_json, file_hash
    root, campaign, plan = frozen_plan
    monkeypatch.setattr(benchmark, "corruption_cells", lambda dataset: [("fixture_corruption", 1)])
    meta = session_metadata("fixture_session", "GPU-" + BARE_UUID)
    contract = {key: meta[key] for key in benchmark.HARDWARE_CLASS_FIELDS}
    contract["hardware_identity"] = meta["hardware_identity"]
    atomic_json(benchmark.benchmark_root(root) / "hardware_contract.json", {"contract": contract})
    measurement = save_measurement_session(root / "measurements", meta)
    block_id = next(block["id"] for block in plan["blocks"] if block["dataset"] == "mnist")
    workers = plan["blocks"][block_id]["work"]
    receipts = []
    for item in workers:
        config = json.loads(Path(item["config_path"]).read_text())
        rd = root / "outputs/runs" / config["protocol"] / config["registry_id"] / config["config_hash"]
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "final.pt").write_bytes(b"fixture_checkpoint")
        checkpoint_hash = file_hash(rd / "final.pt")
        atomic_json(rd / "completed.json", {"checkpoint_sha256": checkpoint_hash})
        out = rd / "benchmark" / meta["hardware_identity"] / item["execution"] / f"batch{item['batch_size']}"
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "zero_input_diagnostic.json", {"measurement": measurement})
        identity = dict(config_hash=config["config_hash"], execution=item["execution"],
                        training_source_hash=config["source_hash"], measurement_source_hash=plan["measurement_source_hash"],
                        hardware_identity=meta["hardware_identity"], checkpoint_sha256=checkpoint_hash,
                        normalization_hash=config["normalization_hash"])
        if benchmark.timing_cohort():
            identity["timing_cohort"] = benchmark.timing_cohort()
        cells = []
        for corruption, severity in [("clean", "test"), ("fixture_corruption", 1)]:
            directory = out / "cells" / f"{corruption}__{severity}"
            hashes = {}
            for name in ("timings.jsonl", "tokens.jsonl", "batch_membership.json", "profile.json", "trace.json"):
                atomic_json(directory / name, {"fixture_file": name})
                hashes[name] = file_hash(directory / name)
            atomic_json(directory / "summary.json", dict(state="complete", identity=identity,
                        corruption=corruption, severity=severity, measurement=measurement, file_hashes=hashes))
            cells.append(dict(corruption=corruption, severity=severity, path=str(directory / "summary.json"), measurement=measurement))
        atomic_json(out / "summary.json", dict(state="complete", identity=identity, cells=cells,
                    expected_cells=2, batch_size=item["batch_size"], metadata=meta,
                    zero_input_diagnostic_sha256=file_hash(out / "zero_input_diagnostic.json"),
                    measurement_sessions=[measurement], physical_device_uuids=[meta["device_uuid"]]))
        receipts.append(benchmark.worker_receipt(item, meta["hardware_identity"], root=root))
    directory = benchmark.benchmark_root(root) / "blocks" / f"block_{block_id:03d}"
    progress = dict(state="completed", block_id=block_id, workers=len(workers), source_hash=plan["source_hash"],
                    measurement_source_hash=plan["measurement_source_hash"], scope_sha256=plan["scope_sha256"],
                    blocks_sha256=plan["blocks_sha256"], order_sha256=plan["order_sha256"])
    atomic_json(directory / "progress.json", progress)
    receipt = dict(state="completed", block_id=block_id, expected_workers=len(workers), source_hash=plan["source_hash"],
                   measurement_source_hash=plan["measurement_source_hash"], scope_sha256=plan["scope_sha256"],
                   blocks_sha256=plan["blocks_sha256"], order_sha256=plan["order_sha256"], workers=receipts,
                   hardware_identity=meta["hardware_identity"])
    if benchmark.timing_cohort():
        progress["cohort"] = receipt["cohort"] = benchmark.timing_cohort()
        atomic_json(directory / "progress.json", progress)
    atomic_json(directory / "completed.json", receipt)
    return root, campaign, plan, receipt, directory


@pytest.mark.parametrize("frozen_plan", [LEGACY_SCOPE, SCOPED_SCOPE], indirect=True)
def test_completed_block_revalidates_all_evidence_and_preserves_resume(completed_block, monkeypatch):
    import sparse_contrast.benchmark as benchmark
    root, campaign, plan, receipt, directory = completed_block
    assert benchmark.validate_block_completion(receipt["block_id"], root=root) == receipt
    # An already completed block returns its verified receipt without making a
    # CUDA context, invoking a worker, or rewriting progress/evidence.
    monkeypatch.setattr(benchmark, "load_frozen_blocks", lambda root=None: plan)
    original_validate = benchmark.validate_block_completion
    monkeypatch.setattr(benchmark, "validate_block_completion", lambda block_id: original_validate(block_id, root=root))
    monkeypatch.setattr(benchmark, "validate_config_artifacts", lambda config: None)
    monkeypatch.setattr(benchmark, "executing_source_hash", lambda config: plan["measurement_source_hash"])
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: pytest.fail("completed block relaunched work"))
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("block parent initialized CUDA"))
    original_progress = (directory / "progress.json").read_bytes()
    assert benchmark.run_block(campaign, receipt["block_id"]) == receipt
    assert (directory / "progress.json").read_bytes() == original_progress


@pytest.mark.parametrize("frozen_plan", [LEGACY_SCOPE, SCOPED_SCOPE], indirect=True)
def test_interrupted_block_reuses_completed_workers_and_launches_only_missing(completed_block, monkeypatch):
    import sparse_contrast.benchmark as benchmark
    from sparse_contrast.common import atomic_json
    root, campaign, plan, receipt, directory = completed_block
    (directory / "completed.json").unlink()
    atomic_json(directory / "progress.json", {"state": "running", "completed_workers": len(receipt["workers"]) - 1})
    missing = receipt["workers"][11]
    summary_path = Path(missing["worker_summary_path"])
    original_summary = summary_path.read_bytes()
    summary_path.unlink()
    original_receipt = benchmark.worker_receipt
    monkeypatch.setattr(benchmark, "load_frozen_blocks", lambda root=None: plan)
    monkeypatch.setattr(benchmark, "worker_receipt", lambda item, hardware: original_receipt(item, hardware, root=root))
    monkeypatch.setattr(benchmark, "validate_config_artifacts", lambda config: None)
    monkeypatch.setattr(benchmark, "executing_source_hash", lambda config: plan["measurement_source_hash"])
    monkeypatch.setattr(benchmark, "run_dir", lambda config: root / "outputs/runs" / config["protocol"] / config["registry_id"] / config["config_hash"])
    monkeypatch.setattr(benchmark, "stop_requested", lambda: False)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("block parent initialized CUDA"))
    launches = []
    def launch(command, **kwargs):
        launches.append(command)
        assert command[command.index("--execution") + 1] == missing["execution"]
        assert command[command.index("--batch-size") + 1] == str(missing["batch_size"])
        assert json.loads(Path(command[command.index("--config") + 1]).read_text())["config_hash"] == missing["config_hash"]
        summary_path.write_bytes(original_summary)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(benchmark.subprocess, "run", launch)
    resumed = benchmark.run_block(campaign, receipt["block_id"])
    assert len(launches) == 1 and resumed["workers"] == receipt["workers"]
    progress = json.loads((directory / "progress.json").read_text())
    assert progress["state"] == "completed" and progress["workers"] == len(receipt["workers"])
    assert json.loads((directory / "completed.json").read_text()) == resumed


@pytest.mark.parametrize("frozen_plan", [LEGACY_SCOPE, SCOPED_SCOPE], indirect=True)
@pytest.mark.parametrize("target", ["raw_timing", "cell_summary", "worker_summary", "session", "checkpoint", "progress", "receipt_order", "contract", "measurement_source", "scope"])
def test_completed_block_rejects_changed_saved_evidence(completed_block, target):
    from sparse_contrast.benchmark import validate_block_completion
    from sparse_contrast.common import atomic_json
    root, _, _, receipt, directory = completed_block
    worker = receipt["workers"][0]
    if target == "raw_timing":
        Path(worker["cell_summary_hashes"][0]["path"]).with_name("timings.jsonl").write_text("changed\n")
    elif target == "cell_summary":
        path = Path(worker["cell_summary_hashes"][0]["path"])
        value = json.loads(path.read_text()); value["severity"] = 999
        atomic_json(path, value)
    elif target == "worker_summary":
        path = Path(worker["worker_summary_path"])
        value = json.loads(path.read_text()); value["identity"]["hardware_identity"] = "other_gpu_class"
        atomic_json(path, value)
    elif target == "session":
        Path(worker["measurement_sessions"][0]["hardware_metadata_file"]).write_text("{}")
    elif target == "checkpoint":
        Path(worker["worker_summary_path"]).parents[4].joinpath("final.pt").write_bytes(b"changed_checkpoint")
    elif target == "progress":
        path = directory / "progress.json"
        value = json.loads(path.read_text()); value["workers"] = len(receipt["workers"]) - 1
        atomic_json(path, value)
    elif target == "receipt_order":
        receipt["workers"].reverse(); atomic_json(directory / "completed.json", receipt)
    elif target == "measurement_source":
        path = Path(worker["worker_summary_path"])
        value = json.loads(path.read_text()); value["identity"]["measurement_source_hash"] = "wrong_source"
        atomic_json(path, value)
    elif target == "scope":
        receipt["scope_sha256"] = "wrong_scope"; atomic_json(directory / "completed.json", receipt)
    else:
        path = root / "outputs/benchmarks/hardware_contract.json"
        value = json.loads(path.read_text()); value["contract"]["hardware_identity"] = "other_class"
        atomic_json(path, value)
    with pytest.raises(ValueError):
        validate_block_completion(receipt["block_id"], root=root)


def test_scoped_matrix_has_156_configs_and_552_unique_workers():
    from sparse_contrast.benchmark import make_matched_blocks
    configs = full_registry(SCOPED_SCOPE)
    paths = [f"/frozen/{config['config_hash']}.json" for config in configs]
    blocks = make_matched_blocks(configs, paths, SCOPED_SCOPE)
    assert len(configs) == 156 and len(blocks) == 24
    assert sum(config['dataset'] == 'mnist' for config in configs) == 42
    assert sum(config['dataset'] == 'cifar10' for config in configs) == 114
    assert blocks == make_matched_blocks(configs[::-1], paths[::-1], SCOPED_SCOPE)
    identities = []
    by_hash = {config['config_hash']:config for config in configs}
    for block in blocks:
        work = block['work']
        expected = {'dense':2, 'compact':5, 'dense_masked':5} if block['dataset']=='mnist' else {
            'dense':4, 'compact':15, 'dense_masked':15}
        assert len(work) == (12 if block['dataset']=='mnist' else 34)
        assert {execution:sum(item['execution']==execution for item in work) for execution in expected} == expected
        assert {item['config_hash'] for item in work if item['execution']=='compact'} == {
            item['config_hash'] for item in work if item['execution']=='dense_masked'}
        if block['dataset']=='mnist':
            assert {by_hash[item['config_hash']]['representation'] for item in work} == {'raw','grayscale'}
        identities.extend((item['config_hash'],item['execution'],item['batch_size']) for item in work)
    assert len(identities) == len(set(identities)) == 552
    with pytest.raises(ValueError, match='scope|Incomplete'):
        legacy = full_registry()
        make_matched_blocks(legacy, [f"/frozen/{config['config_hash']}.json" for config in legacy], SCOPED_SCOPE)
    dropped = [config for config in configs if not (config['dataset']=='cifar10' and config['representation']=='single_color')]
    with pytest.raises(ValueError, match='Incomplete|scope'):
        make_matched_blocks(dropped, [f"/frozen/{config['config_hash']}.json" for config in dropped], SCOPED_SCOPE)


@pytest.mark.parametrize('frozen_plan', [SCOPED_SCOPE], indirect=True)
def test_revised_plan_freezes_scope_and_distinct_measurement_lineage(frozen_plan):
    from sparse_contrast.benchmark import load_frozen_blocks
    from sparse_contrast.common import atomic_json, digest
    root, campaign, plan = frozen_plan
    assert plan['source_hash'] == 'frozen_source'
    assert plan['measurement_source_hash'] == 'measurement_source'
    assert plan['scope'] == SCOPED_SCOPE and plan['scope_sha256'] == digest(SCOPED_SCOPE)
    assert sum(len(block['work']) for block in plan['blocks']) == 552
    order = json.loads((root/'outputs/benchmarks/order.json').read_text())
    assert order['scope_sha256'] == plan['scope_sha256']
    assert order['measurement_source_hash'] == plan['measurement_source_hash']
    assert {entry['config']['source_hash'] for entry in json.loads(campaign.read_text())['main']} == {'frozen_source'}
    atomic_json(root/'experiment_scope.json', LEGACY_SCOPE)
    with pytest.raises(ValueError, match='scope'):
        load_frozen_blocks(root=root)


@pytest.mark.parametrize('frozen_plan', [SCOPED_SCOPE], indirect=True)
def test_scoped_mnist_receipt_cannot_claim_legacy_34_workers(completed_block):
    from sparse_contrast.benchmark import validate_block_completion
    from sparse_contrast.common import atomic_json
    root, _, plan, receipt, directory = completed_block
    assert plan['blocks'][receipt['block_id']]['dataset'] == 'mnist'
    assert receipt['expected_workers'] == len(receipt['workers']) == 12
    receipt['expected_workers'] = 34
    atomic_json(directory/'completed.json',receipt)
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_block_completion(receipt['block_id'],root=root)


@pytest.mark.parametrize('cohort', ['../legacy', '/absolute', 'bad/name', '.', 'a.b', 'a b'])
def test_timing_cohort_rejects_path_escape(tmp_path, cohort):
    from sparse_contrast.benchmark import benchmark_root
    with pytest.raises(ValueError, match='cohort'):
        benchmark_root(tmp_path, cohort)


def test_cohort_root_is_explicit_and_restores_environment(tmp_path, monkeypatch):
    from sparse_contrast.benchmark import benchmark_root, selected_cohort
    monkeypatch.delenv('SPARSE_CONTRAST_TIMING_COHORT', raising=False)
    assert benchmark_root(tmp_path) == tmp_path/'outputs/benchmarks'
    with selected_cohort('a6000_20261004'):
        assert benchmark_root(tmp_path) == tmp_path/'outputs/benchmarks/cohorts/a6000_20261004'
        assert benchmark_root(tmp_path, '') == tmp_path/'outputs/benchmarks'
    assert 'SPARSE_CONTRAST_TIMING_COHORT' not in os.environ


@pytest.mark.parametrize('frozen_plan', [(SCOPED_SCOPE, 'a6000_20261004')], indirect=True)
def test_new_cohort_plan_keeps_legacy_artifacts_and_exact_552_workers(frozen_plan):
    import sparse_contrast.benchmark as benchmark
    root, campaign, plan = frozen_plan
    legacy = root/'outputs/benchmarks/blocks.json'
    legacy.write_text('preserved legacy evidence')
    assert plan['cohort'] == 'a6000_20261004'
    assert len(benchmark.indexed_workers(plan)) == 552
    ids = [(item['config_hash'],item['execution'],item['batch_size']) for _,item in benchmark.indexed_workers(plan)]
    assert len(set(ids)) == 552
    assert benchmark.freeze_campaign_plan(campaign)[0] == plan
    assert legacy.read_text() == 'preserved legacy evidence'
    assert benchmark.load_frozen_blocks(root, cohort='a6000_20261004') == plan
    with pytest.raises(FileNotFoundError):
        benchmark.load_frozen_blocks(root, cohort='other')


def _mock_completed_campaign_worker(monkeypatch, root, plan):
    import sparse_contrast.benchmark as benchmark
    monkeypatch.setattr(benchmark, 'validate_config_artifacts', lambda config: None)
    monkeypatch.setattr(benchmark, 'executing_source_hash', lambda config: plan['measurement_source_hash'])
    monkeypatch.setattr(benchmark, 'run_dir', lambda config: root/'outputs/runs'/config['protocol']/config['registry_id']/config['config_hash'])
    monkeypatch.setattr(benchmark, 'stop_requested', lambda: False)
    monkeypatch.setattr(torch.cuda, 'init', lambda: pytest.fail('receipt reuse initialized CUDA'))
    monkeypatch.setattr(benchmark, 'run_worker', lambda *a, **k: pytest.fail('completed worker was relaunched'))


@pytest.mark.parametrize('frozen_plan', [(SCOPED_SCOPE, 'a6000_20261004')], indirect=True)
def test_independent_workers_resume_without_relaunch_and_finalize_matched_block(completed_block, monkeypatch):
    import sparse_contrast.benchmark as benchmark
    root, campaign, plan, old_receipt, blockdir = completed_block
    cohort = plan['cohort']
    block_id = old_receipt['block_id']
    (blockdir/'completed.json').unlink()
    (blockdir/'progress.json').unlink()
    _mock_completed_campaign_worker(monkeypatch, root, plan)
    worker_ids = [index for index,(block,_) in enumerate(benchmark.indexed_workers(plan)) if block == block_id]
    with pytest.raises(FileNotFoundError):
        benchmark.finalize_block(block_id, root=root, cohort=cohort)
    # Finish out of order as independent GPUs do; the final block retains frozen order.
    for index in reversed(worker_ids):
        receipt = benchmark.run_campaign_worker(campaign, index, cohort=cohort)
        assert receipt['worker_id'] == index and receipt['block_id'] == block_id
        before = (benchmark.benchmark_root(root, cohort)/'workers'/f'worker_{index:06d}'/'completed.json').read_bytes()
        assert benchmark.run_campaign_worker(campaign, index, cohort=cohort) == receipt
        assert benchmark.validate_worker_completion(index, root=root, cohort=cohort) == receipt
        assert (benchmark.benchmark_root(root, cohort)/'workers'/f'worker_{index:06d}'/'completed.json').read_bytes() == before
    final = benchmark.finalize_block(block_id, root=root, cohort=cohort)
    assert final['workers'] == old_receipt['workers'] and final['expected_workers'] == 12
    assert benchmark.validate_block_completion(block_id, root=root, cohort=cohort) == final
    assert benchmark.finalize_block(block_id, root=root, cohort=cohort) == final


@pytest.mark.parametrize('frozen_plan', [(SCOPED_SCOPE, 'a6000_20261004')], indirect=True)
@pytest.mark.parametrize('corruption', ['worker_id', 'work', 'cohort', 'raw'])
def test_independent_worker_receipt_rejects_wrong_identity_and_changed_raw(completed_block, monkeypatch, corruption):
    import sparse_contrast.benchmark as benchmark
    from sparse_contrast.common import atomic_json
    root,campaign,plan,block_receipt,_ = completed_block
    _mock_completed_campaign_worker(monkeypatch, root, plan)
    index = next(i for i,(block,_) in enumerate(benchmark.indexed_workers(plan)) if block == block_receipt['block_id'])
    receipt = benchmark.run_campaign_worker(campaign,index,cohort=plan['cohort'])
    path = benchmark.benchmark_root(root,plan['cohort'])/'workers'/f'worker_{index:06d}'/'completed.json'
    if corruption == 'raw':
        cell = Path(receipt['worker']['cell_summary_hashes'][0]['path']).parent
        (cell/'timings.jsonl').write_text('changed raw evidence')
    else:
        receipt[corruption] = {'config_hash':'wrong'} if corruption == 'work' else 'wrong'
        atomic_json(path,receipt)
    with pytest.raises(ValueError):
        benchmark.validate_worker_completion(index,root=root,cohort=plan['cohort'])


@pytest.mark.parametrize('frozen_plan', [(SCOPED_SCOPE, 'a6000_20261004')], indirect=True)
def test_independent_worker_missing_summary_launches_exact_item_and_stop_prevents_launch(completed_block, monkeypatch):
    import sparse_contrast.benchmark as benchmark
    root,campaign,plan,block_receipt,_ = completed_block
    _mock_completed_campaign_worker(monkeypatch,root,plan)
    index = next(i for i,(block,_) in enumerate(benchmark.indexed_workers(plan)) if block == block_receipt['block_id'])
    item = benchmark.indexed_workers(plan)[index][1]
    summary_path = Path(block_receipt['workers'][0]['worker_summary_path'])
    original = summary_path.read_bytes()
    summary_path.unlink()
    calls=[]
    def execute(config,execution,batch_size):
        calls.append((config['config_hash'],execution,batch_size))
        summary_path.write_bytes(original)
    monkeypatch.setattr(benchmark,'run_worker',execute)
    monkeypatch.setattr(benchmark,'stop_requested',lambda:True)
    with pytest.raises(InterruptedError,match='STOP'):
        benchmark.run_campaign_worker(campaign,index,cohort=plan['cohort'])
    assert calls == []
    monkeypatch.setattr(benchmark,'stop_requested',lambda:False)
    result=benchmark.run_campaign_worker(campaign,index,cohort=plan['cohort'])
    assert calls == [(item['config_hash'],item['execution'],item['batch_size'])]
    assert benchmark.validate_worker_completion(index,root=root,cohort=plan['cohort']) == result


def test_cpu_model_parser_rejects_missing_and_heterogeneous_hosts(tmp_path):
    from sparse_contrast.benchmark import cpu_model_name
    path=tmp_path/'cpuinfo'
    path.write_text('processor: 0\nmodel name : Fixture CPU 24-Core\nprocessor: 1\nmodel name: Fixture CPU 24-Core\n')
    assert cpu_model_name(path)=='Fixture CPU 24-Core'
    path.write_text('model name: CPU A\nmodel name: CPU B\n')
    with pytest.raises(RuntimeError,match='homogeneous CPU'):cpu_model_name(path)
    path.write_text('processor: 0\n')
    with pytest.raises(RuntimeError,match='CPU model'):cpu_model_name(path)


def test_new_hardware_contract_includes_cpu_and_threads_and_rejects_drift(tmp_path,monkeypatch):
    import sparse_contrast.benchmark as benchmark
    monkeypatch.setattr(benchmark,'ROOT',tmp_path)
    monkeypatch.setenv('SPARSE_CONTRAST_TIMING_COHORT','a6000_cpu_matched')
    meta=session_metadata('cpu_session','GPU-'+BARE_UUID,gpu='NVIDIA RTX A6000')
    meta.update(cpu_model='AMD EPYC Fixture 24-Core',torch_num_threads=4,torch_num_interop_threads=48)
    meta['hardware_identity']=hardware_class_identity(meta)
    benchmark.hardware_contract(meta)
    saved=json.loads((benchmark.benchmark_root()/'hardware_contract.json').read_text())
    assert all(saved['contract'][key]==meta[key] for key in benchmark.CPU_CLASS_FIELDS)
    for field,value in [('cpu_model','Other CPU'),('torch_num_threads',8),('torch_num_interop_threads',24)]:
        changed={**meta,field:value}
        changed['hardware_identity']=hardware_class_identity(changed)
        assert changed['hardware_identity']!=meta['hardware_identity']
        with pytest.raises(RuntimeError,match='hardware/backend drift'):benchmark.hardware_contract(changed)
