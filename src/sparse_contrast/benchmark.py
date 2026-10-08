"""Paired online latency, separate sustained throughput, and bounded profiling.

One worker owns one checkpoint/execution/batch-size and a fresh CUDA process.
The campaign freezes 24 randomized matched blocks. Each worker may run in an
independent reserved GPU process; block receipts preserve the matched grouping.
Independent workers share one frozen hardware class within an explicit cohort. No model/sample-derived cache is
used by any online scope. Profiler data are separate, nonadditive evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .common import (ROOT, atomic_json, digest, directory_lock, file_hash, hardware,
                     run_dir, seed_all, stop_requested, tb_dir, validate_config_artifacts, executing_source_hash)
from .data import CorruptionDataset, corruption_cells, fixed_panel_indices, load_clean
from .pipeline import Classifier
from .train import load_stats
from .scope import (configurations, expected_main_count, expected_worker_count,
                    expected_workers_per_block, read_scope, validate_registry)

BENCHMARK_SEED = 20260930
SCOPES = ("gpu_raw_to_logits", "host_raw_to_logits", "loader_to_cpu_prediction", "cached_input_model_only_diagnostic")
HARDWARE_CLASS_FIELDS = ("gpu", "compute_capability", "driver_version", "torch", "cuda", "cudnn", "precision", "backend")
CPU_CLASS_FIELDS = ("cpu_model", "torch_num_threads", "torch_num_interop_threads")



def timing_cohort(cohort=None):
    """Explicit experiment namespace; absence preserves the legacy cohort."""
    value = os.getenv("SPARSE_CONTRAST_TIMING_COHORT", "") if cohort is None else cohort
    if not isinstance(value, str) or (value and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value) is None):
        raise ValueError("Invalid timing cohort identifier")
    return value


def benchmark_root(root=None, cohort=None):
    out = Path(ROOT if root is None else root) / "outputs" / "benchmarks"
    cohort = timing_cohort(cohort)
    return out / "cohorts" / cohort if cohort else out


@contextmanager
def selected_cohort(cohort=None):
    previous = os.environ.get("SPARSE_CONTRAST_TIMING_COHORT")
    os.environ["SPARSE_CONTRAST_TIMING_COHORT"] = timing_cohort(cohort)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SPARSE_CONTRAST_TIMING_COHORT", None)
        else:
            os.environ["SPARSE_CONTRAST_TIMING_COHORT"] = previous


def canonical_gpu_uuid(value):
    """PyTorch may return bare hex UUIDs; NVIDIA tools require GPU-/MIG-."""
    value = str(value).strip()
    is_mig = value.lower().startswith("mig-")
    if is_mig:
        value = value[4:]
        if value.lower().startswith("gpu-"):
            parent, *instances = value.split("/")
            if len(instances) != 2 or not all(part.isdigit() for part in instances):
                raise ValueError("Invalid legacy MIG GPU/instance UUID")
            return "MIG-" + canonical_gpu_uuid(parent) + "/" + "/".join(instances)
    elif value.lower().startswith("gpu-"):
        value = value[4:]
    if re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value) is None:
        raise ValueError(f"Unrecognized CUDA device UUID: {value!r}")
    return ("MIG-" if is_mig else "GPU-") + value.lower()


def hardware_class_identity(metadata):
    """Physical UUID/session are provenance, never a new same-class workload."""
    fields = HARDWARE_CLASS_FIELDS + (CPU_CLASS_FIELDS if "cpu_model" in metadata else ())
    return digest({key: metadata.get(key) for key in fields})


def cpu_model_name(path=Path("/proc/cpuinfo")):
    names = {" ".join(line.split(":", 1)[1].split()) for line in Path(path).read_text().splitlines()
             if line.split(":", 1)[0].strip() == "model name"}
    if len(names) != 1 or not next(iter(names), ""):
        raise RuntimeError("Cannot identify one homogeneous CPU model for timing")
    return next(iter(names))


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def timing_batches(indices, batch_size, warmup=50, repeats=200, seed=BENCHMARK_SEED):
    """Private RNG, no prediction-dependent selection, identical across methods.

    Dataset panels interleave classes. The first repeats*B panel entries define
    the measured balanced set; a private permutation changes their order only.
    Warmup cycles the same predetermined panel using a different permutation.
    """
    if batch_size < 1 or len(indices) == 0 or repeats < 1 or warmup < 0:
        raise ValueError("Invalid timing panel dimensions")
    panel = np.asarray(indices, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = np.resize(panel, repeats * batch_size)
    measured = selected[rng.permutation(len(selected))].reshape(repeats, batch_size)
    warm = np.resize(panel[rng.permutation(len(panel))], warmup * batch_size).reshape(warmup, batch_size)
    return warm.tolist(), measured.tolist()


def summarize_times(rows, batch_size):
    if not rows:
        raise ValueError("No timing measurements")
    values = np.asarray([row["elapsed_ms"] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Nonpositive/nonfinite latency")
    return {
        "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)), "count": len(values),
        "serial_images_per_second": float(len(values) * batch_size * 1000 / values.sum()),
        "amortized_mean_ms_per_image": float(values.mean() / batch_size),
        "batch_size": batch_size, "latency_unit": "milliseconds_per_batch",
    }


def gpu_isolation(device):
    if device.type != "cuda":
        return {"verified": False, "reason": "CPU diagnostic only"}
    props = torch.cuda.get_device_properties(device)
    uuid = canonical_gpu_uuid(getattr(props, "uuid", ""))
    identify = subprocess.run(["nvidia-smi", f"--id={uuid}", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
                              check=True, capture_output=True, text=True)
    identified = [canonical_gpu_uuid(line) for line in identify.stdout.splitlines() if line.strip()]
    if identified != [uuid]:
        raise RuntimeError(f"Selected CUDA device {uuid} was not uniquely identified by NVIDIA: {identified}")
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    observed = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        gpu_uuid, pid = [x.strip() for x in line.split(",", 1)]
        if canonical_gpu_uuid(gpu_uuid) == uuid:
            observed.append(int(pid))
    foreign = sorted(set(observed) - {os.getpid()})
    if foreign:
        raise RuntimeError(f"Benchmark GPU {uuid} has competing compute PIDs {foreign}; no timings accepted")
    return {"verified": True, "device_uuid": uuid, "observed_compute_pids": sorted(set(observed)),
            "self_pid": os.getpid(), "scheduler_job_id": os.getenv("SLURM_JOB_ID"),
            "policy": "check each cell before/after; scheduler allocation must reserve GPU"}


def device_metadata(device):
    info = hardware()
    info.update(device=str(device), precision="float32", compilation="eager_no_compile",
                backend="PyTorch_explicit_attention", input_layout="contiguous_BCHW_uint8",
                benchmark_seed=BENCHMARK_SEED, process_id=os.getpid(),
                cpu_model=cpu_model_name(),
                slurm_cpus_per_task=os.getenv("SLURM_CPUS_PER_TASK"),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                torch_num_threads=torch.get_num_threads(), torch_num_interop_threads=torch.get_num_interop_threads(),
                omp_num_threads=os.getenv("OMP_NUM_THREADS"), mkl_num_threads=os.getenv("MKL_NUM_THREADS"))
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        uuid = canonical_gpu_uuid(props.uuid)
        info.update(device_uuid=uuid, torch_reported_device_uuid=str(props.uuid), memory_bytes=props.total_memory,
                    compute_capability=[props.major, props.minor])
        command = ["nvidia-smi", f"--id={uuid}",
                   "--query-gpu=uuid,name,driver_version,pstate,power.draw,power.limit,clocks.current.sm,clocks.current.memory,temperature.gpu",
                   "--format=csv,noheader,nounits"]
        info["nvidia_smi_state_fields"] = ["uuid", "name", "driver_version", "pstate", "power_watts", "power_limit_watts", "sm_MHz", "memory_MHz", "temperature_C"]
        info["nvidia_smi_state_values"] = [x.strip() for x in subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().split(",")]
        if len(info["nvidia_smi_state_values"]) != len(info["nvidia_smi_state_fields"]) or canonical_gpu_uuid(info["nvidia_smi_state_values"][0]) != uuid:
            raise RuntimeError("NVIDIA metadata did not uniquely identify the selected CUDA device")
        info["driver_version"] = info["nvidia_smi_state_values"][2]
    info["hardware_identity"] = hardware_class_identity(info)
    info["session_started_unix_ns"] = time.time_ns()
    info["measurement_session_id"] = digest({key: info.get(key) for key in
        ("host", "job_id", "process_id", "device_uuid", "session_started_unix_ns")})[:24]
    return info


def save_measurement_session(directory, metadata):
    """Append immutable per-worker metadata; preserve the original hardware file."""
    path = directory / "sessions" / metadata["measurement_session_id"] / "hardware.json"
    if path.exists():
        if json.loads(path.read_text()) != metadata:
            raise ValueError("Measurement session ID reused with different hardware metadata")
    else:
        atomic_json(path, metadata)
    first = directory / "hardware.json"
    if not first.exists():
        atomic_json(first, metadata)
    return {"measurement_session_id": metadata["measurement_session_id"],
            "measurement_device_uuid": metadata.get("device_uuid"),
            "hardware_identity": metadata["hardware_identity"],
            "hardware_metadata_file": str(path), "hardware_metadata_sha256": file_hash(path),
            "host": metadata.get("host"), "scheduler_job_id": metadata.get("job_id")}


def validate_measurement_session(measurement):
    path = Path(measurement["hardware_metadata_file"])
    if file_hash(path) != measurement["hardware_metadata_sha256"]:
        raise ValueError("Saved measurement session hardware provenance changed")
    metadata = json.loads(path.read_text())
    if (metadata["measurement_session_id"] != measurement["measurement_session_id"] or
            metadata.get("device_uuid") != measurement["measurement_device_uuid"] or
            metadata["hardware_identity"] != measurement["hardware_identity"]):
        raise ValueError("Saved measurement session identity disagrees with hardware metadata")
    return metadata


def hardware_contract(metadata, scheduler_selection=None):
    """Freeze device class/backend across primary workers and continuations."""
    path = benchmark_root() / "hardware_contract.json"
    contract = {"hardware_identity": metadata["hardware_identity"],
                "gpu": metadata["gpu"], "compute_capability": metadata["compute_capability"],
                "driver_version": metadata["driver_version"], "torch": metadata["torch"],
                "cuda": metadata["cuda"], "cudnn": metadata["cudnn"],
                "precision": metadata["precision"], "backend": metadata["backend"]}
    if timing_cohort() and (not metadata.get("cpu_model") or any(
            not isinstance(metadata.get(key), int) or metadata[key] < 1 for key in CPU_CLASS_FIELDS[1:])):
        raise ValueError("New timing cohorts require CPU model and positive PyTorch thread counts")
    if "cpu_model" in metadata:
        contract.update({key: metadata[key] for key in CPU_CLASS_FIELDS})
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["contract"] != contract or previous.get("cohort", "") != timing_cohort():
            raise RuntimeError("Benchmark hardware/backend drift: resume on the frozen GPU class, or create a fully separate complete timing campaign")
        if scheduler_selection:
            for key, value in scheduler_selection.items():
                if key in previous and previous[key] != value and key in ("gpu_partition", "gpu_gres_type"):
                    raise RuntimeError(f"Frozen benchmark scheduler selector changed: {key}")
            if not all(key in previous for key in scheduler_selection):
                atomic_json(path, {**previous, **scheduler_selection})
    else:
        atomic_json(path, {"contract": contract, "cohort": timing_cohort(), "first_device_uuid": metadata["device_uuid"],
                    "first_host": metadata["host"], "first_scheduler_job": os.getenv("SLURM_JOB_ID"),
                    "first_partition": os.getenv("SLURM_JOB_PARTITION"),
                    "slurm_partition": os.getenv("SLURM_JOB_PARTITION"),
                    "policy": "same_device_class_and_backend; physical UUID retained for every cell; never pool GPU types",
                    **(scheduler_selection or {})})


def parse_gpu_gres_type(node_description, gpu_name):
    """Resolve the site's actual typed GRES, without inventing a GPU selector."""
    match = re.search(r"(?:^|\s)Gres=(\S+)", node_description)
    if match is None:
        raise ValueError("Allocated node has no readable Gres field")
    value = re.sub(r"\([^)]*\)", "", match.group(1))
    types = sorted({entry.split(":")[1] for entry in value.split(",")
                    if entry.startswith("gpu:") and len(entry.split(":")) == 3 and entry.split(":")[2].isdigit()})
    normalized_name = re.sub(r"[^a-z0-9]", "", gpu_name.lower())
    matched = [kind for kind in types if re.sub(r"[^a-z0-9]", "", kind.lower()) in normalized_name]
    if len(matched) == 1:
        return matched[0]
    raise ValueError(f"Cannot identify one typed Slurm GPU GRES for {gpu_name!r}; observed types={types}")


def scheduler_gpu_selection(metadata):
    if not os.getenv("SLURM_JOB_ID") or not os.getenv("SLURM_JOB_PARTITION"):
        raise RuntimeError("Latency preparation requires an allocated Slurm GPU job")
    node = os.getenv("SLURMD_NODENAME") or str(metadata["host"]).split(".")[0]
    observed = subprocess.run(["scontrol", "-o", "show", "node", node], check=True, capture_output=True, text=True).stdout
    return {"gpu_partition": os.environ["SLURM_JOB_PARTITION"],
            "gpu_gres_type": parse_gpu_gres_type(observed, metadata["gpu"]),
            "slurm_partition": os.environ["SLURM_JOB_PARTITION"], "selector_node": node,
            "selector_node_description": observed.strip()}


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _serial_measure(call, count, device):
    """Only top-level boundaries; includes CPU dispatch/dynamic-length handling."""
    rows = []
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if device.type == "cuda" else None
    if events:
        events[0].record(); events[1].record(); events[1].synchronize()  # lazy event init outside measurements
    for repeat in range(count):
        _synchronize(device)
        started = time.perf_counter_ns()
        if events:
            events[0].record()
        output = call(repeat)
        if events:
            events[1].record()
        _synchronize(device)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        cuda_elapsed = events[0].elapsed_time(events[1]) if events else None
        # Output checks and recording happen AFTER the complete measured interval.
        if repeat == 0 and torch.is_tensor(output) and not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite benchmark output")
        rows.append({"repeat_id": repeat, "elapsed_ms": elapsed, "cuda_stream_elapsed_ms": cuda_elapsed,
                     "measurement_flags": ["valid", "synchronized_wall", "all_observations_retained"]})
        del output
    return rows


def _sustained_measure(call, count, batch_size, device):
    """One synchronization at each end, independently observed throughput."""
    _synchronize(device)
    started = time.perf_counter_ns()
    for repeat in range(count):
        output = call(repeat)
    _synchronize(device)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    del output
    return {"elapsed_seconds": elapsed, "images": count * batch_size, "batches": count,
            "images_per_second": count * batch_size / elapsed,
            "policy": "serial_Python_dispatch_no_per_batch_explicit_synchronization_except_required_model_or_D2H"}


def _warmup_scope(call, count, device):
    """Discard unreferenced prior-scope reserve, warm this mode, then reset peaks.

    Live weights/inputs remain allocated. Cache management and warmup are setup,
    outside every serial/throughput timing interval. This avoids charging a
    prior dense parity check or profiler's allocator reserve to compact mode.
    """
    _synchronize(device)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    for index in range(count):
        call(index)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _make_batches(dataset, batches, device):
    host, ids = [], []
    for indices in batches:
        samples = [dataset[int(i)] for i in indices]
        tensor = torch.stack([sample[0] for sample in samples]).contiguous()
        host.append(tensor.pin_memory() if device.type == "cuda" else tensor)
        ids.append([sample[2] for sample in samples])
    return host, ids


def _loader(dataset, batches, device):
    # No prefetch: observed wait contains fetch/layout/collation/pinning, with
    # warm mapped pages. Dataset construction is explicitly setup, not timed.
    return DataLoader(dataset, batch_sampler=batches, num_workers=0,
                      pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(BENCHMARK_SEED))


def _profile(model, raw, execution, device, destination, uninstrumented_ms, measurement=None):
    destination.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda_supported = device.type == "cuda" and torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
    if cuda_supported:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    model.set_profiling(True)
    try:
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            _synchronize(device)
            start = time.perf_counter_ns()
            with torch.profiler.record_function("complete_gpu_raw_to_logits"):
                model(raw, execution=execution)
                _synchronize(device)
            profiled_ms = (time.perf_counter_ns() - start) / 1e6
        table = []
        for event in profiler.key_averages(group_by_input_shape=True):
            table.append({"name": event.key, "count": event.count, "input_shapes": event.input_shapes,
                          "cpu_total_ms_inclusive": event.cpu_time_total / 1000,
                          "cpu_self_ms": event.self_cpu_time_total / 1000,
                          "device_total_ms_inclusive": event.device_time_total / 1000,
                          "device_self_ms": event.self_device_time_total / 1000,
                          "cpu_memory_bytes": event.cpu_memory_usage,
                          "device_memory_bytes": event.device_memory_usage})
        temporary = destination / "trace.json.tmp"
        profiler.export_chrome_trace(str(temporary))
        os.replace(temporary, destination / "trace.json")
        summary = {"profiled_call_wall_ms": profiled_ms, "matched_uninstrumented_wall_ms": uninstrumented_ms,
                   "measurement": measurement,
                   "profiling_overhead_ratio": profiled_ms / uninstrumented_ms,
                   "cuda_profiler_available": cuda_supported,
                   "torch_tb_profiler_plugin": "not_required_standard_Chrome_trace_exported",
                   "stage_intervals_additive": False,
                   "attribution": "inclusive scopes overlap/nest; self CPU and device times are separate; never sum as end-to-end latency",
                   "events": table, "trace_sha256": file_hash(destination / "trace.json")}
        atomic_json(destination / "profile.json", summary)
        return summary
    finally:
        model.set_profiling(False)


def _write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


@torch.inference_mode()
def measure_cell(model, config, execution, dataset, corruption, severity, batch_size,
                 device, directory, identity, warmup=50, repeats=200, measurement=None):
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Benchmark cell identity changed")
        for file_name, expected_hash in previous["file_hashes"].items():
            if file_hash(directory / file_name) != expected_hash:
                raise ValueError(f"Corrupt benchmark artifact: {directory / file_name}")
        validate_measurement_session(previous["measurement"])
        return previous
    if measurement is None:
        raise ValueError("A new timing cell requires immutable measurement-session provenance")
    validate_measurement_session(measurement)
    isolation_before = gpu_isolation(device)
    if isolation_before.get("device_uuid") != measurement["measurement_device_uuid"]:
        raise ValueError("Timing cell device differs from measurement-session device")
    warm_batches, measured_batches = timing_batches(fixed_panel_indices(dataset, "timing"), batch_size, warmup, repeats)
    host_warm, warm_ids = _make_batches(dataset, warm_batches, device)
    host, measured_ids = _make_batches(dataset, measured_batches, device)
    gpu_warm = [x.to(device, non_blocking=True) for x in host_warm]
    gpu = [x.to(device, non_blocking=True) for x in host]
    _synchronize(device)
    atomic_json(directory / "batch_membership.json", {"warmup_indices": warm_batches, "measured_indices": measured_batches,
                "warmup_ids": warm_ids, "measured_ids": measured_ids, "seed": BENCHMARK_SEED,
                "dataset_split_hash": dataset.split_hash, "independent_of_predictions_sparsity_latency": True})
    model.set_profiling(False)
    model.collect_diagnostics = False
    if config["execution"] == "compact":
        compact = model(gpu[0], execution="compact")
        dense = model(gpu[0], execution="dense_masked")
        torch.testing.assert_close(compact, dense, rtol=1e-4, atol=2e-5)
        del compact, dense
    raw_rows, summaries, memory = [], {}, {}
    cached = None
    for scope in SCOPES:
        if stop_requested():
            raise InterruptedError("STOP marker before timing scope")
        if scope == "gpu_raw_to_logits":
            warm_call = lambda i: model(gpu_warm[i], execution=execution)
            call = lambda i: model(gpu[i], execution=execution)
        elif scope == "host_raw_to_logits":
            warm_call = lambda i: model(host_warm[i].to(device, non_blocking=True), execution=execution)
            call = lambda i: model(host[i].to(device, non_blocking=True), execution=execution)
        elif scope == "loader_to_cpu_prediction":
            warm_iterator = iter(_loader(dataset, warm_batches, device))
            measured_iterator = iter(_loader(dataset, measured_batches, device))
            def loader_call(iterator):
                raw, _label, _ids = next(iterator)
                return model(raw.to(device, non_blocking=True), execution=execution).argmax(1).cpu()
            warm_call = lambda i: loader_call(warm_iterator)
            call = lambda i: loader_call(measured_iterator)
        else:
            cached_warm = [model.prepare(x, execution=execution) for x in gpu_warm]
            cached = [model.prepare(x, execution=execution) for x in gpu]
            warm_call = lambda i: model.backbone.forward_patches(*cached_warm[i], execution=execution)
            call = lambda i: model.backbone.forward_patches(*cached[i], execution=execution)
        _warmup_scope(warm_call, warmup, device)
        rows = _serial_measure(call, repeats, device)
        if device.type == "cuda":
            memory[scope] = {key: getattr(torch.cuda, key)(device) for key in ("max_memory_allocated", "max_memory_reserved")}
        if scope == "loader_to_cpu_prediction":
            measured_iterator = iter(_loader(dataset, measured_batches, device))
        sustained = _sustained_measure(call, repeats, batch_size, device)
        for row in rows:
            row.update(scope=scope if device.type == "cuda" else "cpu_diagnostic/" + scope,
                       sample_ids=measured_ids[row["repeat_id"]], token_record_id=row["repeat_id"],
                       measurement_device_uuid=isolation_before.get("device_uuid"),
                       measurement_session_id=measurement["measurement_session_id"],
                       corruption=corruption, severity=severity, **identity)
        raw_rows.extend(rows)
        summaries[scope] = {**summarize_times(rows, batch_size), "sustained_throughput": sustained,
                            "memory": memory.get(scope), "cpu_diagnostic_only": device.type != "cuda",
                            "memory_policy": {
                                "fresh_process_per_checkpoint_execution_batch_size": True,
                                "scope_setup": "synchronize_gc_empty_unreferenced_CUDA_cache_then_50_mode_specific_warmups_then_synchronize_reset_peak_counters",
                                "measurement": "peak_allocated_and_reserved_over_serial_measured_calls_before_throughput_and_profiler",
                                "live_allocations_included": "weights_raw_input_panel_and_required_model_state; cached_inputs_only_in_explicit_cached_diagnostic",
                                "cache_or_peak_reset_inside_timing": False,
                            }}
    del cached, cached_warm, call, warm_call
    # Diagnostics rerun raw input outside all headline/throughput windows. Their
    # support is recomputed from each corrupted image, never its clean partner.
    tokens = []
    model.collect_diagnostics = True
    for index, raw in enumerate(gpu):
        model(raw, execution=execution)
        tokens.append({"token_record_id": index, "sample_ids": measured_ids[index],
                       "measurement_session_id": measurement["measurement_session_id"],
                       "measurement_device_uuid": measurement["measurement_device_uuid"],
                       "native": jsonable(model.latest_diagnostics),
                       "execution": jsonable(model.backbone.latest_telemetry)})
    model.collect_diagnostics = False
    profile = _profile(model, gpu[0], execution, device, directory,
                       next(row["elapsed_ms"] for row in raw_rows if row["repeat_id"] == 0), measurement)
    isolation_after = gpu_isolation(device)
    _write_jsonl(directory / "timings.jsonl", raw_rows)
    _write_jsonl(directory / "tokens.jsonl", tokens)
    result = {
        "state": "complete", "identity": identity, "corruption": corruption, "severity": severity,
        "measurement": measurement,
        "comparison_policy": "matched_hardware_class_and_backend; physical_GPU_UUIDs_and_sessions_may_differ_and_are_retained_per_measurement; never_pool_different_GPU_types",
        "batch_size": batch_size, "warmup_batches_per_scope": warmup, "measured_batches_per_scope": repeats,
        "scopes": summaries, "isolation_before": isolation_before, "isolation_after": isolation_after,
        "loader_policy": {"num_workers": 0, "prefetch": False, "cache": "warm_OS_and_dataset_cache",
                          "dataset_construction_and_archive_decode": "setup_excluded",
                          "measured": "next_loader_batch_collation_pinning_H2D_online_model_argmax_D2H",
                          "storage": str(getattr(dataset, "directory", "validated_clean_arrays_in_RAM"))},
        "online_boundaries": {"gpu_raw_to_logits": "decoded_uint8_raw_GPU_to_GPU_logits_includes_online_conversion_and_all_model_work",
                              "host_raw_to_logits": "decoded_uint8_raw_pinned_host_to_GPU_logits_includes_H2D_and_all_online_work",
                              "excluded_from_online": "disk_network_decode_model_load_context_setup_warmup_statistics_fitting_CPU_prediction_return",
                              "timer": "synchronized_perf_counter_ns_complete_call; CUDA_event_stream_interval_saved_separately"},
        "raw_GPU_panel_bytes": sum(x.numel() * x.element_size() for x in gpu + gpu_warm),
        "stage_applicability": {
            "frontend": "NA_raw_input" if config["representation"] == "raw" else "source_matched_blur_then_fused_color_mapping_and_subtraction",
            "coefficient_ranking": "NA_raw_input" if config["representation"] == "raw" else (
                "NA_0percent_bypass_no_sort" if config["sparsity_percent"] == 0 else "stable_native_abs_sort_and_zero"),
            "support_and_gather": "NA_dense_control" if execution == "dense" else "native_support_then_saved_mask_gather_or_dense_masked_reference",
            "normalization": "executed_fused_with_patch_layout",
            "swin_merging": "executed_feature_space_only" if config["architecture"] == "swin_tiny" else "NA_ViT",
        },
        "stage_intervals_additive": False, "profile_overhead_ratio": profile["profiling_overhead_ratio"],
        "file_hashes": {name: file_hash(directory / name) for name in
                        ("timings.jsonl", "tokens.jsonl", "batch_membership.json", "profile.json", "trace.json")},
    }
    atomic_json(summary_path, result)
    return result


def _log_cell(writer, row, index):
    prefix = f"{row['corruption']}/severity_{row['severity']}"
    for scope, summary in row["scopes"].items():
        for key in ("mean_ms", "median_ms", "p95_ms", "serial_images_per_second", "amortized_mean_ms_per_image"):
            writer.add_scalar(f"{prefix}/{scope}/{key}", summary[key], index)
        writer.add_scalar(f"{prefix}/{scope}/sustained_images_per_second", summary["sustained_throughput"]["images_per_second"], index)
        for key, value in (summary.get("memory") or {}).items():
            writer.add_scalar(f"{prefix}/{scope}/{key}_bytes", value, index)
    writer.add_scalar(f"{prefix}/profile/overhead_ratio", row["profile_overhead_ratio"], index)


@torch.inference_mode()
def measure_zero_input(model, config, execution, batch_size, device, destination, measurement):
    """Explicit controlled synthetic input diagnostic; never official accuracy."""
    if destination.exists():
        previous = json.loads(destination.read_text())
        validate_measurement_session(previous["measurement"])
        return previous
    size = 28 if config["dataset"] == "mnist" else 32
    channels = 1 if config["dataset"] == "mnist" else 3
    host = torch.zeros(batch_size, channels, size, size, dtype=torch.uint8)
    if device.type == "cuda":
        host = host.pin_memory()
    raw = host.to(device)
    _patches, support = model.prepare(raw, execution=execution)
    if execution in ("compact", "dense_masked") and bool(support.any()):
        raise AssertionError("Native all-zero contrast input did not follow empty-image path")
    scopes = {}
    for scope in ("gpu_raw_to_logits", "host_raw_to_logits"):
        call = (lambda i: model(raw, execution=execution)) if scope == "gpu_raw_to_logits" else (
            lambda i: model(host.to(device, non_blocking=True), execution=execution))
        for index in range(50):
            call(index)
        rows = _serial_measure(call, 200, device)
        for row in rows:
            row.update(measurement_session_id=measurement["measurement_session_id"],
                       measurement_device_uuid=measurement["measurement_device_uuid"], scope=scope)
        scopes[scope] = {"summary": summarize_times(rows, batch_size), "measurements": rows}
    result = {"origin": "controlled_synthetic_all_zero_raw_uint8", "role": "all_empty_path_diagnostic_not_accuracy_or_real_input_aggregate",
              "native_support_all_empty": support is not None and not bool(support.any()),
              "execution": execution, "batch_size": batch_size, "scopes": scopes,
              "config_hash": config["config_hash"], "normalization_hash": config.get("normalization_hash"),
              "measurement": measurement,
              "measurement_device_uuid": measurement["measurement_device_uuid"],
              "measurement_session_id": measurement["measurement_session_id"],
              "cpu_diagnostic_only": device.type != "cuda"}
    atomic_json(destination, result)
    return result


def run_worker(config, execution, batch_size, device_name="cuda", smoke=False):
    validate_config_artifacts(config)
    if config["precision"] != "fp32":
        raise ValueError("Frozen benchmark protocol requires fp32 for all compared hardware/methods")
    device = torch.device(device_name)
    if device.type != "cuda" and not smoke:
        raise RuntimeError("Primary online benchmark requires the reserved CUDA device")
    seed_all(config["seed"])
    rd = run_dir(config)
    checkpoint = rd / "final.pt"
    completed = json.loads((rd / "completed.json").read_text())
    checkpoint_hash = file_hash(checkpoint)
    if checkpoint_hash != completed["checkpoint_sha256"]:
        raise ValueError("Benchmark final checkpoint hash mismatch")
    if execution == "dense_masked" and config["execution"] != "compact":
        raise ValueError("Dense-masked is a checkpoint-equivalent diagnostic for compact models")
    setup_start = time.perf_counter()
    model = Classifier(config, load_stats(config))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config_hash"] != config["config_hash"] or state.get("normalization_hash") != config.get("normalization_hash"):
        raise ValueError("Checkpoint config/normalization mismatch")
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    del state
    _synchronize(device)
    metadata = device_metadata(device)
    gpu_isolation(device)
    if not smoke:
        hardware_contract(metadata)
    identity = {"config_hash": config["config_hash"], "checkpoint_sha256": checkpoint_hash,
                "training_source_hash": config["source_hash"],
                "measurement_source_hash": executing_source_hash(config),
                "normalization_hash": config.get("normalization_hash"), "dataset": config["dataset"],
                "architecture": config["architecture"], "representation": config["representation"],
                "sparsity_percent": config["sparsity_percent"], "seed": config["seed"],
                "trained_execution": config["execution"], "execution": execution,
                "hardware_identity": metadata["hardware_identity"], "benchmark_rng": BENCHMARK_SEED,
                "latency_protocol_version": 2, "warmup_batches": 50, "measured_batches": 200,
                "batch_size": batch_size,
                "study_role": "smoke_diagnostic" if smoke else "primary"}
    if timing_cohort():
        identity["timing_cohort"] = timing_cohort()
    out = rd / ("benchmark_smoke" if smoke else "benchmark") / metadata["hardware_identity"] / execution / f"batch{batch_size}"
    out.mkdir(parents=True, exist_ok=True)
    from .campaign import recover_dead_lock
    recover_dead_lock(out / "writer.lock")
    with directory_lock(out / "writer.lock"):
        measurement = save_measurement_session(out, metadata)
        setup_seconds = time.perf_counter() - setup_start
        empty = measure_zero_input(model, config, execution, batch_size, device, out / "zero_input_diagnostic.json", measurement)
        cells = [("clean", "val" if smoke else "test")] + ([] if smoke else corruption_cells(config["dataset"]))
        rows = []
        for index, (corruption, severity) in enumerate(cells):
            if stop_requested():
                raise InterruptedError("STOP marker before benchmark cell")
            dataset = load_clean(config["dataset"], "val" if smoke else "test") if corruption == "clean" else CorruptionDataset(config["dataset"], corruption, severity)
            destination = out / "cells" / f"{corruption}__{severity}"
            row = measure_cell(model, config, execution, dataset, corruption, severity, batch_size,
                               device, destination, identity, measurement=measurement)
            rows.append({"corruption": corruption, "severity": severity, "path": str(destination / "summary.json"),
                         "measurement": row["measurement"]})
            # No live asynchronous writer exists during ANY timing/profiler pass.
            writer = SummaryWriter(str(tb_dir(config, f"latency/{metadata['hardware_identity']}/{execution}/batch{batch_size}" + ("/smoke" if smoke else ""))))
            try:
                if index == 0:
                    writer.add_scalar("setup/model_context_load_seconds", setup_seconds, 0)
                    writer.add_text("provenance/hardware", json.dumps(metadata), 0)
                    for scope, values in empty["scopes"].items():
                        writer.add_scalar(f"synthetic_zero_input/{scope}/median_ms", values["summary"]["median_ms"], 0)
                _log_cell(writer, row, index)
                timing_rows = [json.loads(line) for line in (destination / "timings.jsonl").read_text().splitlines()]
                for scope in SCOPES:
                    values = [x["elapsed_ms"] for x in timing_rows if x["scope"].endswith(scope)]
                    writer.add_histogram(f"{corruption}/severity_{severity}/{scope}/latency_ms", np.asarray(values), index)
                profile = json.loads((destination / "profile.json").read_text())
                for event in profile["events"]:
                    if "/" in event["name"] or event["name"] in ("input_conversion", "native_sparsification", "native_support", "frozen_normalization_and_patch_layout"):
                        writer.add_scalar(f"{corruption}/severity_{severity}/profile/{event['name']}/cpu_self_ms", event["cpu_self_ms"], index)
                        writer.add_scalar(f"{corruption}/severity_{severity}/profile/{event['name']}/device_inclusive_ms", event["device_total_ms_inclusive"], index)
            finally:
                writer.flush(); writer.close()
            print(json.dumps({"benchmark": str(destination), "state": "complete", "gpu_median_ms": row["scopes"]["gpu_raw_to_logits"]["median_ms"]}), flush=True)
            del dataset
        sessions = {row["measurement"]["measurement_session_id"]: row["measurement"] for row in rows}
        sessions[empty["measurement"]["measurement_session_id"]] = empty["measurement"]
        summary = {"state": "complete", "identity": identity, "cells": rows, "expected_cells": len(cells),
                   "batch_size": batch_size, "metadata": metadata,
                   "zero_input_diagnostic_sha256": file_hash(out / "zero_input_diagnostic.json"),
                   "measurement_sessions": list(sessions.values()),
                   "physical_device_uuids": sorted({value["measurement_device_uuid"] for value in sessions.values() if value["measurement_device_uuid"] is not None}),
                   "comparison_policy": "same_hardware_class_and_backend; session_and_physical_device_provenance_retained; different_GPU_types_not_pooled"}
        atomic_json(out / "summary.json", summary)
        return summary


def validate_configuration_groups(configs, scope=None):
    scope = read_scope(ROOT) if scope is None else scope
    groups = {}
    for config in configs:
        groups.setdefault((config["dataset"], config["architecture"], config["seed"]), []).append(config)
    for key, group in groups.items():
        expected = set(configurations(key[0], scope))
        actual = [(c["representation"], c["execution"], c["sparsity_percent"]) for c in group]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError(f"Incomplete/duplicated or out-of-scope benchmark group: {key}; expected {len(expected)} configurations")


def make_matched_blocks(configs, config_paths, scope=None):
    """Independent matched comparisons with dataset-specific scoped workers."""
    scope = read_scope(ROOT) if scope is None else scope
    validate_configuration_groups(configs, scope)
    validate_registry(configs, scope)
    expected_count = expected_main_count(scope)
    if len(configs) != expected_count or len(config_paths) != expected_count:
        raise ValueError(f"Latency preparation requires the complete {expected_count}-configuration registry")
    if len({config["config_hash"] for config in configs}) != expected_count:
        raise ValueError("Duplicate resolved configuration identities")
    grouped = {}
    for config, path in zip(configs, config_paths):
        for batch in (1, 8):
            key = (config["dataset"], config["architecture"], config["seed"], batch)
            for execution in (["compact", "dense_masked"] if config["execution"] == "compact" else ["dense"]):
                grouped.setdefault(key, []).append({"config_path": str(path), "config_hash": config["config_hash"],
                                                    "execution": execution, "batch_size": batch})
    rng = np.random.Generator(np.random.PCG64(BENCHMARK_SEED))
    keys = sorted(grouped)
    keys = [keys[index] for index in rng.permutation(len(keys))]
    blocks = []
    for block_id, (dataset, architecture, seed, batch) in enumerate(keys):
        work = sorted(grouped[(dataset, architecture, seed, batch)],
                      key=lambda item: (item["config_hash"], item["execution"]))
        work = [work[index] for index in rng.permutation(len(work))]
        if len(work) != expected_workers_per_block(dataset, scope):
            raise ValueError(f"Incorrect scoped worker count in matched {dataset} latency block")
        blocks.append({"id": block_id, "dataset": dataset, "architecture": architecture,
                       "seed": seed, "batch_size": batch, "work": work})
    return blocks


def freeze_campaign_plan(path):
    """Pure CPU planning; no CUDA initialization in any block parent process."""
    campaign = json.loads(Path(path).read_text())
    if campaign.get("latency_cohort", {}).get("id", "") != timing_cohort():
        raise ValueError("Campaign and selected timing cohort disagree")
    scope = read_scope(ROOT)
    entries = campaign["main"]
    configs = [entry.get("config") or json.loads(Path(entry["config_path"]).read_text()) for entry in entries]
    for entry, config in zip(entries, configs):
        if json.loads(Path(entry["config_path"]).read_text()) != config:
            raise ValueError("Campaign and saved worker configuration disagree")
        if digest({key: value for key, value in config.items() if key != "config_hash"}) != config["config_hash"]:
            raise ValueError("Campaign configuration digest mismatch")
    source_hashes = {config["source_hash"] for config in configs}
    if len(source_hashes) != 1:
        raise ValueError("Matched latency campaign must use one immutable training source snapshot")
    source_hash = next(iter(source_hashes))
    measurement_source_hash = campaign.get("orchestration_source_hash", campaign.get("source_hash", source_hash))
    blocks = make_matched_blocks(configs, [entry["config_path"] for entry in entries], scope)
    work = [item for block in blocks for item in block["work"]]
    out = benchmark_root()
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"source_hash": source_hash, "measurement_source_hash": measurement_source_hash,
                  "scope": scope, "scope_sha256": digest(scope)}
    if timing_cohort():
        provenance["cohort"] = timing_cohort()
    order = {"schema_version": 3, **provenance, "frozen_work_sha256": digest(work),
             "benchmark_rng": BENCHMARK_SEED, "work": work,
             "ordering": "randomized_dataset_architecture_seed_batch_blocks;dataset_scoped_randomized_sequential_matched_workers;real_input_batches_paired"}
    order_path = out / "order.json"
    if order_path.exists() and json.loads(order_path.read_text()) != order:
        raise ValueError("Frozen benchmark campaign/order changed")
    if not order_path.exists():
        atomic_json(order_path, order)
    plan = {"schema_version": 2, **provenance, "frozen_work_sha256": digest(work),
            "order_sha256": file_hash(order_path), "blocks": blocks}
    plan["blocks_sha256"] = digest(plan)
    plan_path = out / "blocks.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen matched latency block plan changed")
    if not plan_path.exists():
        atomic_json(plan_path, plan)
    return plan, configs


def load_frozen_blocks(root=ROOT, cohort=None):
    out = benchmark_root(root, cohort)
    plan = json.loads((out / "blocks.json").read_text())
    order = json.loads((out / "order.json").read_text())
    scope = read_scope(root)
    if plan.get("cohort", "") != timing_cohort(cohort) or order.get("cohort", "") != timing_cohort(cohort):
        raise ValueError("Frozen timing cohort identity mismatch")
    if plan["blocks_sha256"] != digest({key: value for key, value in plan.items() if key != "blocks_sha256"}):
        raise ValueError("Frozen block plan hash mismatch")
    if plan["order_sha256"] != file_hash(out / "order.json") or order["source_hash"] != plan["source_hash"]:
        raise ValueError("Frozen block/order source identity mismatch")
    if (plan.get("scope") != scope or plan.get("scope_sha256") != digest(scope) or
            order.get("scope") != scope or order.get("scope_sha256") != digest(scope) or
            not plan.get("measurement_source_hash") or order.get("measurement_source_hash") != plan["measurement_source_hash"]):
        raise ValueError("Frozen scope or measurement source identity mismatch")
    work = [item for block in plan["blocks"] for item in block["work"]]
    if order["work"] != work or digest(work) != plan["frozen_work_sha256"] or digest(work) != order["frozen_work_sha256"]:
        raise ValueError("Frozen worker order identity mismatch")
    expected = {(dataset, architecture, seed, batch) for dataset in ("mnist", "cifar10")
                for architecture in ("vit_small", "swin_tiny") for seed in (0, 1, 2) for batch in (1, 8)}
    actual = {(block["dataset"], block["architecture"], block["seed"], block["batch_size"]) for block in plan["blocks"]}
    if actual != expected or [block["id"] for block in plan["blocks"]] != list(range(len(expected))):
        raise ValueError("Frozen plan does not contain all matched dataset/architecture/seed/batch blocks")
    identities = [(item["config_hash"], item["execution"], item["batch_size"]) for item in work]
    expected_workers = expected_worker_count(scope)
    if len(identities) != expected_workers or len(set(identities)) != expected_workers:
        raise ValueError(f"Frozen plan must cover exactly {expected_workers} distinct scoped workers")
    configs = {}
    for block in plan["blocks"]:
        if len(block["work"]) != expected_workers_per_block(block["dataset"], scope):
            raise ValueError("Invalid scoped matched worker block dimensions")
        for item in block["work"]:
            config = json.loads(Path(item["config_path"]).read_text())
            if (config["config_hash"] != item["config_hash"] or
                    digest({key: value for key, value in config.items() if key != "config_hash"}) != config["config_hash"] or
                    config["source_hash"] != plan["source_hash"]):
                raise ValueError("Frozen worker configuration or training source identity mismatch")
            if (config["dataset"], config["architecture"], config["seed"], item["batch_size"]) != (
                    block["dataset"], block["architecture"], block["seed"], block["batch_size"]):
                raise ValueError("Worker is outside its matched comparison block")
            allowed = ("compact", "dense_masked") if config["execution"] == "compact" else ("dense",)
            if item["execution"] not in allowed:
                raise ValueError("Worker execution is incompatible with its trained configuration")
            configs[config["config_hash"]] = config
    validate_registry(list(configs.values()), scope)
    expected_identities = {(config["config_hash"], execution, batch) for config in configs.values()
                           for batch in (1, 8) for execution in
                           (("compact", "dense_masked") if config["execution"] == "compact" else ("dense",))}
    if set(identities) != expected_identities:
        raise ValueError("Frozen workers do not exactly cover the active registry")
    return plan


def prepare_campaign(path):
    from .campaign import recover_dead_lock
    out = benchmark_root()
    out.mkdir(parents=True, exist_ok=True)
    recover_dead_lock(out / "prepare.lock")
    with directory_lock(out / "prepare.lock"):
        plan, configs = freeze_campaign_plan(path)
        validate_config_artifacts(configs[0])
        if executing_source_hash(configs[0]) != plan["measurement_source_hash"]:
            raise ValueError("Latency preparation is not executing the frozen measurement source")
        if not os.getenv("SLURM_JOB_ID"):
            raise RuntimeError("Latency preparation requires an allocated Slurm GPU job")
        seed_all(BENCHMARK_SEED)
        metadata = device_metadata(torch.device("cuda"))
        isolation = gpu_isolation(torch.device("cuda"))
        selection = scheduler_gpu_selection(metadata)
        hardware_contract(metadata, scheduler_selection=selection)
        session = save_measurement_session(out / "preparation", metadata)
        atomic_json(out / "prepared.json", {"state": "completed", "source_hash": plan["source_hash"],
                    "cohort": timing_cohort(),
                    "blocks_sha256": plan["blocks_sha256"], "order_sha256": plan["order_sha256"],
                    "expected_blocks": len(plan["blocks"]), "expected_workers": sum(len(block["work"]) for block in plan["blocks"]),
                    "measurement_source_hash": plan["measurement_source_hash"], "scope_sha256": plan["scope_sha256"], "measurement": session,
                    "isolation": isolation, "hardware_contract_sha256": file_hash(out / "hardware_contract.json")})
    return plan


def worker_receipt(item, hardware_identity, root=ROOT, cohort=None):
    """Validate one existing worker's completion and return its durable receipt."""
    config = json.loads(Path(item["config_path"]).read_text())
    if (config["config_hash"] != item["config_hash"] or
            digest({key: value for key, value in config.items() if key != "config_hash"}) != config["config_hash"]):
        raise ValueError("Worker configuration identity changed")
    rd = Path(root) / "outputs" / "runs" / config["protocol"] / config["registry_id"] / config["config_hash"]
    path = rd / "benchmark" / hardware_identity / item["execution"] / f"batch{item['batch_size']}" / "summary.json"
    summary = json.loads(path.read_text())
    identity = summary["identity"]
    plan_provenance = json.loads((benchmark_root(root, cohort) / "blocks.json").read_text())
    expected_cells = {("clean", "test"), *corruption_cells(config["dataset"])}
    actual_cells = [(cell["corruption"], cell["severity"]) for cell in summary["cells"]]
    completed = json.loads((rd / "completed.json").read_text())
    contract = json.loads((benchmark_root(root, cohort) / "hardware_contract.json").read_text())["contract"]
    if (summary["state"] != "complete" or identity["config_hash"] != item["config_hash"] or
            identity.get("training_source_hash") != config["source_hash"] or
            identity.get("measurement_source_hash") != plan_provenance["measurement_source_hash"] or
            identity.get("timing_cohort", "") != timing_cohort(cohort) or
            identity["execution"] != item["execution"] or summary["batch_size"] != item["batch_size"] or
            identity["hardware_identity"] != hardware_identity or identity["checkpoint_sha256"] != completed["checkpoint_sha256"] or
            file_hash(rd / "final.pt") != identity["checkpoint_sha256"] or
            identity["normalization_hash"] != config["normalization_hash"] or set(actual_cells) != expected_cells or
            len(actual_cells) != len(expected_cells) or summary["expected_cells"] != len(expected_cells) or
            any(summary["metadata"].get(key) != value for key, value in contract.items())):
        raise ValueError("Incomplete or mismatched benchmark worker completion")
    if file_hash(path.parent / "zero_input_diagnostic.json") != summary["zero_input_diagnostic_sha256"]:
        raise ValueError("Worker contains corrupt zero-input timing evidence")
    cells = []
    for cell in summary["cells"]:
        cell_path = Path(cell["path"])
        saved = json.loads(cell_path.read_text())
        if (saved["state"] != "complete" or saved["identity"] != identity or
                saved["corruption"] != cell["corruption"] or saved["severity"] != cell["severity"]):
            raise ValueError("Worker contains mismatched cell completion")
        if set(saved["file_hashes"]) != {"timings.jsonl", "tokens.jsonl", "batch_membership.json", "profile.json", "trace.json"}:
            raise ValueError("Worker cell omits required timing/profile evidence")
        for name, expected_hash in saved["file_hashes"].items():
            if file_hash(cell_path.parent / name) != expected_hash:
                raise ValueError("Worker contains corrupt timing/profile evidence")
        if saved["measurement"] != cell["measurement"]:
            raise ValueError("Worker cell measurement session changed")
        metadata = validate_measurement_session(saved["measurement"])
        if any(metadata.get(key) != value for key, value in contract.items()):
            raise ValueError("Worker cell contains a different hardware class")
        cells.append({"path": str(cell_path), "sha256": file_hash(cell_path)})
    for session in summary["measurement_sessions"]:
        metadata = validate_measurement_session(session)
        if any(metadata.get(key) != value for key, value in contract.items()):
            raise ValueError("Worker session contains a different hardware class")
    return {"config_hash": item["config_hash"], "execution": item["execution"], "batch_size": item["batch_size"],
            "worker_summary_path": str(path), "worker_summary_sha256": file_hash(path),
            "cell_summary_hashes": cells, "measurement_sessions": summary["measurement_sessions"],
            "physical_device_uuids": summary["physical_device_uuids"]}


def validate_block_completion(block_id, root=ROOT, cohort=None):
    plan = load_frozen_blocks(root) if cohort is None else load_frozen_blocks(root, cohort=cohort)
    if not 0 <= block_id < len(plan["blocks"]):
        raise ValueError("Block id is outside the frozen scoped plan")
    path = benchmark_root(root, cohort) / "blocks" / f"block_{block_id:03d}" / "completed.json"
    receipt = json.loads(path.read_text())
    expected_workers = len(plan["blocks"][block_id]["work"])
    progress = json.loads(path.with_name("progress.json").read_text())
    contract = json.loads((benchmark_root(root, cohort) / "hardware_contract.json").read_text())["contract"]
    if (receipt.get("cohort", "") != timing_cohort(cohort) or progress.get("cohort", "") != timing_cohort(cohort) or
            receipt["state"] != "completed" or receipt["block_id"] != block_id or receipt["expected_workers"] != expected_workers or
            receipt.get("measurement_source_hash") != plan["measurement_source_hash"] or
            receipt.get("scope_sha256") != plan["scope_sha256"] or
            receipt["source_hash"] != plan["source_hash"] or receipt["blocks_sha256"] != plan["blocks_sha256"] or
            receipt["order_sha256"] != plan["order_sha256"] or receipt["hardware_identity"] != contract["hardware_identity"]):
        raise ValueError("Matched block completion identity mismatch")
    if (progress["state"] != "completed" or progress["block_id"] != block_id or progress["workers"] != expected_workers or
            progress.get("measurement_source_hash") != plan["measurement_source_hash"] or
            progress.get("scope_sha256") != plan["scope_sha256"] or
            progress["source_hash"] != plan["source_hash"] or progress["blocks_sha256"] != plan["blocks_sha256"] or
            progress["order_sha256"] != plan["order_sha256"]):
        raise ValueError("Matched block progress is not a matching completed receipt")
    expected = [(item["config_hash"], item["execution"], item["batch_size"]) for item in plan["blocks"][block_id]["work"]]
    actual = [(item["config_hash"], item["execution"], item["batch_size"]) for item in receipt["workers"]]
    if actual != expected:
        raise ValueError("Matched block worker coverage/order mismatch")
    for expected_item, saved_item in zip(plan["blocks"][block_id]["work"], receipt["workers"]):
        # Re-read all raw timing/profile evidence as well as the summaries.
        # Existence or hashes of summary files alone cannot certify completion.
        current = worker_receipt(expected_item, contract["hardware_identity"], root=root, cohort=cohort)
        if current != saved_item:
            raise ValueError("Completed worker evidence changed after block receipt")
    return receipt


def indexed_workers(plan):
    """Stable IDs retain the frozen matched order regardless of launch order."""
    return [(block["id"], item) for block in plan["blocks"] for item in block["work"]]


def _worker_coordinates(plan, worker_id):
    work = indexed_workers(plan)
    if not isinstance(worker_id, int) or not 0 <= worker_id < len(work):
        raise ValueError("Worker id is outside the frozen scoped plan")
    return work[worker_id]


def _worker_envelope(plan, worker_id, hardware_identity, root, cohort):
    block_id, item = _worker_coordinates(plan, worker_id)
    return {"worker_id": worker_id, "block_id": block_id, "cohort": timing_cohort(cohort),
            "source_hash": plan["source_hash"], "measurement_source_hash": plan["measurement_source_hash"],
            "scope_sha256": plan["scope_sha256"], "blocks_sha256": plan["blocks_sha256"],
            "order_sha256": plan["order_sha256"], "hardware_identity": hardware_identity,
            "work": item}


def validate_worker_completion(worker_id, root=ROOT, cohort=None):
    """Revalidate the full evidence graph, including raw files and checkpoint."""
    plan = load_frozen_blocks(root, cohort=cohort)
    _block_id, item = _worker_coordinates(plan, worker_id)
    out = benchmark_root(root, cohort)
    hardware_identity = json.loads((out / "hardware_contract.json").read_text())["contract"]["hardware_identity"]
    directory = out / "workers" / f"worker_{worker_id:06d}"
    receipt = json.loads((directory / "completed.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    expected = _worker_envelope(plan, worker_id, hardware_identity, root, cohort)
    for saved in (receipt, progress):
        if saved.get("state") != "completed" or any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Independent latency worker completion identity mismatch")
    current = worker_receipt(item, hardware_identity, root=root, cohort=cohort)
    if receipt.get("worker") != current:
        raise ValueError("Independent latency worker evidence changed after completion")
    return receipt


def run_campaign_worker(path, worker_id, cohort=None):
    """One scheduler job, one fresh CUDA process, one immutable work identity."""
    with selected_cohort(cohort):
        plan = load_frozen_blocks(ROOT)
        _block_id, item = _worker_coordinates(plan, worker_id)
        campaign = json.loads(Path(path).read_text())
        if (campaign.get("latency_cohort", {}).get("id", "") != timing_cohort() or
                campaign.get("orchestration_source_hash", campaign["source_hash"]) != plan["measurement_source_hash"]):
            raise ValueError("Campaign timing cohort or measurement source changed")
        config = json.loads(Path(item["config_path"]).read_text())
        validate_config_artifacts(config)
        if executing_source_hash(config) != plan["measurement_source_hash"]:
            raise ValueError("Latency worker is not executing the frozen measurement source")
        out = benchmark_root()
        hardware_identity = json.loads((out / "hardware_contract.json").read_text())["contract"]["hardware_identity"]
        directory = out / "workers" / f"worker_{worker_id:06d}"
        from .campaign import recover_dead_lock
        recover_dead_lock(directory / "controller.lock")
        with directory_lock(directory / "controller.lock"):
            if (directory / "completed.json").exists():
                return validate_worker_completion(worker_id, root=ROOT)
            if stop_requested():
                raise InterruptedError("STOP marker before benchmark worker launch")
            envelope = _worker_envelope(plan, worker_id, hardware_identity, ROOT, None)
            atomic_json(directory / "progress.json", {**envelope, "state": "running", "time": time.time(),
                                                      "job_id": os.getenv("SLURM_JOB_ID")})
            summary_path = run_dir(config) / "benchmark" / hardware_identity / item["execution"] / f"batch{item['batch_size']}" / "summary.json"
            if not summary_path.exists():
                run_worker(config, item["execution"], item["batch_size"])
            evidence = worker_receipt(item, hardware_identity, root=ROOT)
            result = {**envelope, "state": "completed", "worker": evidence,
                      "job_id": os.getenv("SLURM_JOB_ID"), "time": time.time()}
            # A crash between these atomic writes resumes from existing worker evidence.
            atomic_json(directory / "progress.json", {**envelope, "state": "completed", "time": time.time()})
            atomic_json(directory / "completed.json", result)
            return result


def finalize_block(block_id, root=ROOT, cohort=None):
    """Assemble a matched block only after every independent worker validates."""
    plan = load_frozen_blocks(root, cohort=cohort)
    if not 0 <= block_id < len(plan["blocks"]):
        raise ValueError("Block id is outside the frozen scoped plan")
    out = benchmark_root(root, cohort)
    directory = out / "blocks" / f"block_{block_id:03d}"
    from .campaign import recover_dead_lock
    recover_dead_lock(directory / "controller.lock")
    with directory_lock(directory / "controller.lock"):
        if (directory / "completed.json").exists():
            return validate_block_completion(block_id, root=root, cohort=cohort)
        receipts = [validate_worker_completion(index, root=root, cohort=cohort)["worker"]
                    for index, (owner_block, _item) in enumerate(indexed_workers(plan)) if owner_block == block_id]
        hardware_identity = json.loads((out / "hardware_contract.json").read_text())["contract"]["hardware_identity"]
        envelope = {"block_id": block_id, "cohort": timing_cohort(cohort), "source_hash": plan["source_hash"],
                    "measurement_source_hash": plan["measurement_source_hash"], "scope_sha256": plan["scope_sha256"],
                    "blocks_sha256": plan["blocks_sha256"], "order_sha256": plan["order_sha256"]}
        result = {**envelope, "state": "completed", "expected_workers": len(plan["blocks"][block_id]["work"]),
                  "workers": receipts, "hardware_identity": hardware_identity,
                  "physical_device_uuids": sorted({uuid for item in receipts for uuid in item["physical_device_uuids"]}),
                  "device_policy": "independent_reserved_GPU_processes;same_frozen_hardware_class;per_session_UUIDs_preserved"}
        atomic_json(directory / "progress.json", {**envelope, "state": "completed", "workers": len(receipts), "time": time.time()})
        atomic_json(directory / "completed.json", result)
        return result


def run_block(path, block_id):
    """Parent performs CPU orchestration only; each child gets a fresh context."""
    plan = load_frozen_blocks()
    if not 0 <= block_id < len(plan["blocks"]):
        raise ValueError("Block id is outside the frozen scoped plan")
    campaign = json.loads(Path(path).read_text())
    if {(entry.get("config") or json.loads(Path(entry["config_path"]).read_text()))["source_hash"]
            for entry in campaign["main"]} != {plan["source_hash"]}:
        raise ValueError("Campaign source changed after latency preparation")
    measurement_source_hash = campaign.get("orchestration_source_hash", campaign.get("source_hash", plan["source_hash"]))
    if measurement_source_hash != plan["measurement_source_hash"]:
        raise ValueError("Campaign measurement source changed after latency preparation")
    block = plan["blocks"][block_id]
    expected_workers = len(block["work"])
    first_config = json.loads(Path(block["work"][0]["config_path"]).read_text())
    validate_config_artifacts(first_config)
    if executing_source_hash(first_config) != plan["measurement_source_hash"]:
        raise ValueError("Latency block is not executing the frozen measurement source")
    out = benchmark_root() / "blocks" / f"block_{block_id:03d}"
    out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((benchmark_root() / "hardware_contract.json").read_text())
    hardware_identity = contract["contract"]["hardware_identity"]
    from .campaign import recover_dead_lock
    recover_dead_lock(out / "controller.lock")
    with directory_lock(out / "controller.lock"):
        if (out / "completed.json").exists():
            return validate_block_completion(block_id)
        receipts = []
        for index, item in enumerate(block["work"]):
            if stop_requested():
                raise InterruptedError("STOP marker before benchmark worker launch")
            config = json.loads(Path(item["config_path"]).read_text())
            summary_path = run_dir(config) / "benchmark" / hardware_identity / item["execution"] / f"batch{item['batch_size']}" / "summary.json"
            atomic_json(out / "progress.json", {"state": "running", "block_id": block_id, "index": index, "total": expected_workers, "current": item,
                        "cohort": timing_cohort(),
                        "completed_workers": len(receipts), "source_hash": plan["source_hash"], "blocks_sha256": plan["blocks_sha256"],
                        "measurement_source_hash": plan["measurement_source_hash"], "scope_sha256": plan["scope_sha256"],
                        "order_sha256": plan["order_sha256"], "job_id": os.getenv("SLURM_JOB_ID"), "time": time.time()})
            if summary_path.exists():
                receipts.append(worker_receipt(item, hardware_identity))
                continue
            code_root = Path(__file__).resolve().parents[1]
            command = [sys.executable, str(code_root / "benchmark.py"), "--config", item["config_path"],
                       "--execution", item["execution"], "--batch-size", str(item["batch_size"]), "--device", "cuda"]
            subprocess.run(command, check=True, cwd=code_root)
            receipts.append(worker_receipt(item, hardware_identity))
        result = {"state": "completed", "block_id": block_id, "source_hash": plan["source_hash"],
                  "cohort": timing_cohort(),
                  "blocks_sha256": plan["blocks_sha256"], "order_sha256": plan["order_sha256"], "expected_workers": expected_workers,
                  "measurement_source_hash": plan["measurement_source_hash"], "scope_sha256": plan["scope_sha256"],
                  "workers": receipts, "hardware_identity": hardware_identity, "job_id": os.getenv("SLURM_JOB_ID"),
                  "physical_device_uuids": sorted({uuid for item in receipts for uuid in item["physical_device_uuids"]}),
                  "device_policy": "sequential_fresh_workers_share_reserved_GPU_within_uninterrupted_block;walltime_continuations_may_change_physical_UUID_with_same_hardware_class;per_session_provenance_preserved"}
        atomic_json(out / "progress.json", {"state": "completed", "block_id": block_id, "workers": expected_workers, "time": time.time(),
                    "cohort": timing_cohort(),
                    "measurement_source_hash": plan["measurement_source_hash"], "scope_sha256": plan["scope_sha256"],
                    "source_hash": plan["source_hash"], "blocks_sha256": plan["blocks_sha256"], "order_sha256": plan["order_sha256"]})
        atomic_json(out / "completed.json", result)
    return result


def run_campaign(path):
    """Compatibility driver: run the already prepared 24 blocks sequentially."""
    plan = load_frozen_blocks()
    for block in plan["blocks"]:
        run_block(path, block["id"])
    atomic_json(benchmark_root() / "progress.json",
                {"state": "completed", "workers": sum(len(block["work"]) for block in plan["blocks"]), "blocks": len(plan["blocks"]),
                 "cohort": timing_cohort(),
                 "measurement_source_hash": plan["measurement_source_hash"], "scope_sha256": plan["scope_sha256"],
                 "source_hash": plan["source_hash"], "blocks_sha256": plan["blocks_sha256"], "order_sha256": plan["order_sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--campaign")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-campaign", action="store_true")
    mode.add_argument("--block-id", type=int)
    mode.add_argument("--worker-id", type=int)
    parser.add_argument("--cohort", help="Independent homogeneous timing campaign namespace")
    parser.add_argument("--execution", choices=("dense", "compact", "dense_masked"))
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Clean-only diagnostic, never primary evidence")
    args = parser.parse_args()
    if args.cohort is not None:
        os.environ["SPARSE_CONTRAST_TIMING_COHORT"] = timing_cohort(args.cohort)
    if (args.prepare_campaign or args.block_id is not None or args.worker_id is not None) and not args.campaign:
        parser.error("--prepare-campaign, --block-id and --worker-id require --campaign")
    if args.campaign:
        if args.prepare_campaign:
            prepare_campaign(args.campaign)
        elif args.block_id is not None:
            run_block(args.campaign, args.block_id)
        elif args.worker_id is not None:
            run_campaign_worker(args.campaign, args.worker_id)
        else:
            run_campaign(args.campaign)
    else:
        config = json.loads(Path(args.config).read_text())
        run_worker(config, args.execution or config["execution"], args.batch_size, args.device, args.smoke)


if __name__ == "__main__":
    main()
