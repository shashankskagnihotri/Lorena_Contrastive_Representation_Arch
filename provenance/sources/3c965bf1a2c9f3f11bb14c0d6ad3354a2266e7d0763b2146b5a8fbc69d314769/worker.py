"""Explicit recovery worker: original computation, separately recorded validation.

The two copied orchestration functions are checked against their frozen originals
at startup. The measured scopes and called timing/model kernels are unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

PROJECT = Path('/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch')
STUDY_ROOT = PROJECT / 'training_dense_testing_sparse'
SCIENTIFIC_ROOT = STUDY_ROOT / 'outputs/source/8cf4d0c1e0ce5443d5080ea397df9a8f085d1bf2c4292912676545d38533a0cb'
sys.path.insert(0, str(SCIENTIFIC_ROOT))

import torch
from torch.utils.tensorboard import SummaryWriter
from dense_sparse.common import (ROOT, atomic_json, case_dir, digest, effective_config,
    file_hash, load_case, load_model, now, read_json)
from dense_sparse.scheduler import recover_lock
from dense_sparse.worker import (_check_stop, _dataset, _freeze_hardware_contract,
    _identity, _save_summary, _timing_stage_metadata, _verify_timing_cell, expected_cells)
from sparse_contrast import benchmark
from sparse_contrast.common import directory_lock, seed_all
from sparse_contrast.data import fixed_panel_indices
from sparse_contrast.benchmark import (BENCHMARK_SEED, SCOPES, _make_batches,
    _synchronize, _loader, _warmup_scope, _serial_measure, _sustained_measure,
    _profile, _write_jsonl, summarize_times, gpu_isolation,
    validate_measurement_session, stop_requested, jsonable, timing_batches)
from validation import (POLICY, source_receipt, validate_parity, validation_reference,
    verify_validation_reference, verify_copied_orchestration, verify_global_bindings)


def measure_cell_with_validation_receipt(model, config, execution, dataset, corruption,
        severity, batch_size, device, directory, identity, **options):
    previous = (directory / 'summary.json').exists()
    row = measure_cell(model, config, execution, dataset, corruption, severity,
                       batch_size, device, directory, identity, **options)
    if 'validation_amendment' in row:
        verify_validation_reference(row['validation_amendment'], identity, row['measurement'])
    elif not previous and config['execution'] == 'compact':
        raise ValueError('New compact timing cell omitted its validation receipt')
    return row


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
        validate_parity(model, gpu[0], compact, dense, directory, identity, measurement, measured_ids[0])
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
    if config["execution"] == "compact":
        result["validation_amendment"] = validation_reference(directory)
    atomic_json(summary_path, result)
    return result


def benchmark_case(case_or_path, execution, batch_size, shard=0, num_shards=1, device='cuda'):
    case = load_case(case_or_path)
    allowed = ('compact', 'dense_masked') if case['inference_execution'] == 'compact' else ('dense',)
    if execution not in allowed or batch_size not in (1, 8):
        raise ValueError('Timing execution/batch differs from the registered inference intervention')
    if (isinstance(shard, bool) or isinstance(num_shards, bool) or
            not isinstance(shard, int) or not isinstance(num_shards, int) or
            num_shards < 1 or not 0 <= shard < num_shards):
        raise ValueError('Invalid timing shard coordinates')
    all_cells = expected_cells(case['dataset'])
    cells = [cell for index, cell in enumerate(all_cells) if index % num_shards == shard]
    if not cells:
        raise ValueError('Timing shard contains no official cells')
    _check_stop()
    device = torch.device(device)
    if device.type != 'cuda' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Primary timing requires an allocated CUDA GPU')
    seed_all(case['seed'])
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Timing workers require one scheduler-visible GPU')
    model = load_model(case, device)
    original = model.config
    if original['precision'] != 'fp32':
        raise ValueError('Matched timing requires the frozen FP32 protocol')
    control = effective_config(case, original)
    identity = _identity(case, original)
    metadata = benchmark.device_metadata(device)
    benchmark.gpu_isolation(device)
    contract_path, saved_contract = _freeze_hardware_contract(metadata, case['source_hash'])
    identity.update(execution=execution, batch_size=batch_size, shard=shard, num_shards=num_shards,
                    hardware_identity=metadata['hardware_identity'], benchmark_rng=benchmark.BENCHMARK_SEED,
                    latency_protocol_version=2, warmup_batches=50, measured_batches=200)
    out = case_dir(case) / 'latency' / execution / f'batch{batch_size}' / f'shard{shard}'
    out.mkdir(parents=True, exist_ok=True)
    recover_lock(out / 'writer.lock')
    with directory_lock(out / 'writer.lock'):
        measurement = benchmark.save_measurement_session(out, metadata)
        # config_hash in this diagnostic identifies the TEST, never a relabelled checkpoint.
        zero_config = {**control, 'config_hash': case['test_config_hash']}
        zero_path = out / 'zero_input_diagnostic.json'
        empty = benchmark.measure_zero_input(model, zero_config, execution, batch_size,
                                             device, zero_path, measurement)
        if (empty['config_hash'] != case['test_config_hash'] or empty['execution'] != execution or
                empty['batch_size'] != batch_size or empty['normalization_hash'] != case['normalization_hash']):
            raise ValueError('All-empty diagnostic belongs to a different intervention')
        empty_metadata = benchmark.validate_measurement_session(empty['measurement'])
        if any(empty_metadata.get(key) != value for key, value in saved_contract['contract'].items()):
            raise ValueError('All-empty diagnostic hardware class changed')
        references, sessions = [], {empty['measurement']['measurement_session_id']: empty['measurement']}
        for index, (corruption, severity) in enumerate(cells):
            _check_stop()
            dataset = _dataset(case, corruption, severity)
            destination = out / 'cells' / f'{corruption}__{severity}'
            row = measure_cell_with_validation_receipt(model, control, execution, dataset, corruption, severity,
                                         batch_size, device, destination, identity, measurement=measurement)
            before = digest(row)
            row = _timing_stage_metadata(row, case)
            if digest(row) != before:
                atomic_json(destination / 'summary.json', row)
            row = _verify_timing_cell(destination / 'summary.json', identity, corruption, severity,
                                      saved_contract['contract'], dataset)
            references.append(dict(corruption=corruption, severity=severity,
                                   path=str(destination / 'summary.json'),
                                   sha256=file_hash(destination / 'summary.json'), measurement=row['measurement']))
            sessions[row['measurement']['measurement_session_id']] = row['measurement']
            # No writer exists during primary timings, sustained throughput or profiler.
            logdir = (ROOT / 'outputs/tensorboard' / case['test_config_hash'] / 'latency' /
                      metadata['hardware_identity'] / execution / f'batch{batch_size}' / f'shard{shard}')
            with SummaryWriter(str(logdir)) as writer:
                writer.add_text('identity', json.dumps(identity, sort_keys=True), 0)
                benchmark._log_cell(writer, row, index)
                if index == 0:
                    for scope, values in empty['scopes'].items():
                        writer.add_scalar(f'synthetic_zero_input/{scope}/median_ms',
                                          values['summary']['median_ms'], 0)
                profile = read_json(destination / 'profile.json')
                for event in profile['events']:
                    if '/' in event['name'] or event['name'] in (
                            'input_conversion', 'native_sparsification', 'native_support',
                            'frozen_normalization_and_patch_layout'):
                        writer.add_scalar(f'{corruption}/severity_{severity}/profile/{event["name"]}/cpu_self_ms',
                                          event['cpu_self_ms'], index)
                        writer.add_scalar(f'{corruption}/severity_{severity}/profile/{event["name"]}/device_inclusive_ms',
                                          event['device_total_ms_inclusive'], index)
            print(json.dumps(dict(mode='latency', test_config_hash=case['test_config_hash'],
                                  corruption=corruption, severity=severity, execution=execution,
                                  batch_size=batch_size, shard=shard, state='complete')), flush=True)
            del dataset
        summary = dict(state='complete', identity=identity, case=case, execution=execution,
                       batch_size=batch_size, shard=shard, num_shards=num_shards,
                       expected_cells=cells, all_expected_cells=all_cells, cells=references,
                       hardware_contract_path=str(contract_path), hardware_contract_sha256=file_hash(contract_path),
                       metadata=metadata, measurement_sessions=list(sessions.values()),
                       physical_device_uuids=sorted({session['measurement_device_uuid'] for session in sessions.values()}),
                       zero_input_diagnostic_sha256=file_hash(zero_path), completed_at=now(),
                       comparison_policy='same_GPU_CPU_threads_backend; physical-device/session transitions retained')
        if (out / 'summary.json').exists():
            previous = read_json(out / 'summary.json')
            if any(previous['metadata'].get(key) != value for key, value in saved_contract['contract'].items()):
                raise ValueError('Completed worker summary hardware class changed')
        summary = _save_summary(out / 'summary.json', summary)
    return summary


def verify_entrypoint():
    provenance=source_receipt()
    provenance['copied_function_verification']=verify_copied_orchestration(
        Path(benchmark.__file__), SCIENTIFIC_ROOT/'dense_sparse/worker.py', Path(__file__))
    from dense_sparse import worker as original_worker
    verify_global_bindings(benchmark.measure_cell, measure_cell,
                           {'validate_parity', 'validation_reference'})
    verify_global_bindings(original_worker.benchmark_case, benchmark_case,
                           {'measure_cell_with_validation_receipt'})
    provenance['all_copied_function_global_dependencies_match_frozen'] = True
    return provenance


@torch.inference_mode()
def validate_only(case, execution, batch_size, directory):
    """Real failure verification in an allocation; never a primary measurement."""
    _check_stop()
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
        raise RuntimeError('Allocated single GPU required')
    seed_all(case['seed']); device=torch.device('cuda'); model=load_model(case,device)
    model.set_profiling(False); model.collect_diagnostics=False
    metadata=benchmark.device_metadata(device); isolation=benchmark.gpu_isolation(device)
    if metadata['gpu']!='NVIDIA RTX A6000' or metadata['cpu_model']!='AMD EPYC 7413 24-Core Processor':
        raise RuntimeError('Use the original timing hardware class')
    directory.mkdir(parents=True,exist_ok=False)
    measurement=benchmark.save_measurement_session(directory,metadata)
    dataset=_dataset(case,'clean','test')
    _, batches=benchmark.timing_batches(fixed_panel_indices(dataset,'timing'),batch_size,50,200)
    ids=[dataset.sample_ids[i] for i in batches[0]]
    raw=torch.stack([dataset[i][0] for i in batches[0]]).to(device)
    identity=_identity(case,model.config)
    identity.update(execution=execution,batch_size=batch_size,diagnostic_only=True)
    def parameter_hash():
        h=hashlib.sha256()
        for key,value in model.state_dict().items():
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    original=parameter_hash()
    compact=model(raw,execution='compact');dense=model(raw,execution='dense_masked')
    validate_parity(model,raw,compact,dense,directory,identity,measurement,ids)
    ref=validation_reference(directory); saved=verify_validation_reference(ref,identity,measurement)
    if parameter_hash()!=original:raise ValueError('Production model state changed')
    atomic_json(directory/'validated.json',dict(state='passed',diagnostic_only=True,
        source=verify_entrypoint(),validation=ref,model_state_unchanged=True,
        fallback_used=saved['fallback_used'],isolation_before=isolation,
        isolation_after=benchmark.gpu_isolation(device),completed=now()))
    return {'state':'passed','cells':[]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--execution',choices=['compact','dense_masked','dense'],required=True)
    p.add_argument('--batch-size',type=int,choices=[1,8],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--validate-only',type=Path)
    args=p.parse_args(); source=verify_entrypoint();case=load_case(args.case)
    if args.validate_only:
        result=validate_only(case,args.execution,args.batch_size,args.validate_only)
        print(json.dumps(result),flush=True);return
    _check_stop()
    out=case_dir(case)/'latency'/args.execution/f'batch{args.batch_size}'/f'shard{args.shard}'
    attempt=out/'validation_attempts'/f"job{os.environ.get('SLURM_JOB_ID','local')}.json"
    record=dict(state='started',source=source,scientific_source_hash=case['source_hash'],
        test_config_hash=case['test_config_hash'],checkpoint_sha256=case['checkpoint_sha256'],
        execution=args.execution,batch_size=args.batch_size,shard=args.shard,num_shards=args.num_shards,
        job_id=os.environ.get('SLURM_JOB_ID'),command=sys.argv,started=now())
    atomic_json(attempt,record)
    try:
        result=benchmark_case(case,args.execution,args.batch_size,args.shard,args.num_shards)
        record.update(state='complete',completed=now(),summary_path=str(out/'summary.json'),
                      summary_sha256=file_hash(out/'summary.json'))
    except Exception as error:
        record.update(state='failed',completed=now(),error_type=type(error).__name__,
                      error=str(error),traceback=traceback.format_exc())
        atomic_json(attempt,record);raise
    atomic_json(attempt,record)
    print(json.dumps(dict(state=result['state'],test_config_hash=case['test_config_hash'],
                          cells=len(result['cells']),validation_source_hash=source['validation_source_hash'])),flush=True)


if __name__=='__main__':main()
