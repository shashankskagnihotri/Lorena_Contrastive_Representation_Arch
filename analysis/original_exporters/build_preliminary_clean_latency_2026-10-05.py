#!/usr/bin/env python3
"""Reproduce the full-condition clean A6000 latency audit from saved receipts.

No inference, training, checkpoint loads, or raw timing-file reads are performed.
Run from any directory with Python 3; defaults resolve relative to this script.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCOPES = ("gpu_raw_to_logits", "host_raw_to_logits", "loader_to_cpu_prediction",
          "cached_input_model_only_diagnostic")


def read_json(path, expected_hash=None):
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if expected_hash is not None:
        assert sha == expected_hash, (str(path), "receipt hash mismatch")
    return json.loads(data), {"path": str(path.resolve()), "sha256": sha}


def key(config):
    return (config["dataset"], config["architecture"], config["representation"],
            config["execution"], config["sparsity_percent"])


def csv_key(row):
    pct = None if row["sparsity_percent"] in ("", "None", "null") else int(row["sparsity_percent"])
    return (row["dataset"], row["architecture"], row["representation"], row["execution"], pct)


def average_sd(values):
    assert len(values) == 3 and all(math.isfinite(x) for x in values)
    return statistics.mean(values), statistics.stdev(values)


def write_csv(path, rows):
    assert rows
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--root", type=Path, default=here.parent / "sparse_contrast_benchmark_large_batch")
    parser.add_argument("--accuracy-csv", type=Path, default=here / "completed_main_conditions_large_batch_2026-10-04.csv")
    parser.add_argument("--output-prefix", type=Path, default=here / "a6000_all_conditions_clean_latency_2026-10-05")
    args = parser.parse_args()
    root = args.root.resolve()
    cohort = root / "outputs/benchmarks/cohorts/a6000-2026-10-04"
    campaign, campaign_binding = read_json(root / "outputs/campaign.json")
    _, manifest_binding = read_json(root / "experiment_manifest.json")
    contract_document, contract_binding = read_json(cohort / "hardware_contract.json")
    contract = contract_document["contract"]
    hardware_id = contract["hardware_identity"]
    assert contract["gpu"] == "NVIDIA RTX A6000"
    accuracy_bytes = args.accuracy_csv.read_bytes()
    with args.accuracy_csv.open() as stream:
        accuracy_rows = list(csv.DictReader(stream))
    accuracy = {csv_key(row): row for row in accuracy_rows}
    assert len(accuracy) == len(accuracy_rows) == 52
    workers, blocks = {}, []
    for path in sorted((cohort / "blocks").glob("block_*/completed.json")):
        block, binding = read_json(path)
        assert block["state"] == "completed" and block["hardware_identity"] == hardware_id
        assert len(block["workers"]) == block["expected_workers"]
        blocks.append(binding)
        for worker in block["workers"]:
            wkey = (worker["config_hash"], worker["execution"], worker["batch_size"])
            assert wkey not in workers
            workers[wkey] = (worker, binding, block)
    assert len(blocks) == 24 and len(workers) == 552
    items = campaign["main"]
    assert len(items) == 156 and len({item["config"]["config_hash"] for item in items}) == 156
    assert {key(item["config"]) for item in items} == set(accuracy)
    records, input_panels, hardware_sessions, configs = [], {}, {}, []
    for number, item in enumerate(items, 1):
        config = item["config"]
        assert config["role"] == "main" and config["protocol"] == "heldout_val"
        expected_train = 55000 if config["dataset"] == "mnist" else 45000
        expected_corruptions = 15 if config["dataset"] == "mnist" else 75
        run = root / "outputs/runs" / config["protocol"] / config["registry_id"] / config["config_hash"]
        done, done_binding = read_json(run / "completed.json")
        evaluation, eval_binding = read_json(run / "evaluation/summary.json")
        assert done["state"] == "completed" and done["config_hash"] == config["config_hash"]
        assert done["training_count"] == evaluation["training_count"] == expected_train
        assert evaluation["state"] == "complete" and evaluation["config"] == config
        assert evaluation["checkpoint_sha256"] == done["checkpoint_sha256"]
        assert evaluation["expected_cells"] == evaluation["completed_cells"] == expected_corruptions
        assert evaluation["total"] == 10000 * expected_corruptions
        assert evaluation["clean"]["total"] == 10000
        arow = accuracy[key(config)]
        assert int(arow["seeds"]) == 3 and int(arow["training_count"]) == expected_train
        assert config["config_hash"] in arow["config_hashes"].split(";")
        for metric, value in (("clean_accuracy", evaluation["clean"]["accuracy"]),
                              ("clean_loss", evaluation["clean"]["loss"]),
                              ("mCE_raw", evaluation["mCE_raw"])):
            assert math.isclose(float(arow[f"{metric}_seed{config['seed']}"]), value, rel_tol=1e-12, abs_tol=1e-12)
        configs.append({"config_hash": config["config_hash"], "registry_id": config["registry_id"],
                        "completed_receipt": done_binding, "evaluation_summary": eval_binding})
        executions = [config["execution"]]
        if config["execution"] == "compact":
            executions.append("dense_masked")
        for execution in executions:
            for batch in (1, 8):
                worker, block_binding, block = workers[(config["config_hash"], execution, batch)]
                summary, worker_binding = read_json(worker["worker_summary_path"], worker["worker_summary_sha256"])
                assert summary["state"] == "complete" and len(summary["cells"]) == expected_corruptions + 1
                assert block["source_hash"] == config["source_hash"]
                clean_refs = [ref for ref in worker["cell_summary_hashes"]
                              if Path(ref["path"]).parent.name == "clean__test"]
                assert len(clean_refs) == 1
                clean_ref = clean_refs[0]
                cell, cell_binding = read_json(clean_ref["path"], clean_ref["sha256"])
                assert cell["state"] == "complete" and cell["corruption"] == "clean" and cell["severity"] == "test"
                assert cell["batch_size"] == batch and cell["warmup_batches_per_scope"] == 50
                assert cell["measured_batches_per_scope"] == 200 and set(cell["scopes"]) == set(SCOPES)
                for identity in (summary["identity"], cell["identity"]):
                    assert identity["config_hash"] == config["config_hash"]
                    assert identity["checkpoint_sha256"] == done["checkpoint_sha256"]
                    assert identity["normalization_hash"] == config["normalization_hash"]
                    assert identity["execution"] == execution
                    assert identity["trained_execution"] == config["execution"]
                measurement = cell["measurement"]
                assert measurement["hardware_identity"] == hardware_id
                hw_hash = measurement["hardware_metadata_sha256"]
                if hw_hash not in hardware_sessions:
                    hardware_sessions[hw_hash] = read_json(measurement["hardware_metadata_file"], hw_hash)
                hw, hw_binding = hardware_sessions[hw_hash]
                assert all(hw.get(k) == value for k, value in contract.items())
                assert hw["device_uuid"] == measurement["measurement_device_uuid"]
                assert hw["measurement_session_id"] == measurement["measurement_session_id"]
                panel_key = (config["dataset"], config["architecture"], config["seed"], batch)
                panel_hash = cell["file_hashes"]["batch_membership.json"]
                assert panel_key not in input_panels or input_panels[panel_key] == panel_hash
                input_panels[panel_key] = panel_hash
                assert all(value["count"] == 200 and value["batch_size"] == batch for value in cell["scopes"].values())
                records.append({"dataset": config["dataset"], "architecture": config["architecture"],
                    "representation": config["representation"], "trained_execution": config["execution"],
                    "sparsity_percent": config["sparsity_percent"], "seed": config["seed"],
                    "execution": execution, "batch_size": batch,
                    "role": "primary" if execution == config["execution"] else "dense_masked_control",
                    "config_hash": config["config_hash"], "checkpoint_sha256": done["checkpoint_sha256"],
                    "normalization_hash": config["normalization_hash"], "training_source_hash": config["source_hash"],
                    "measurement_source_hash": block["measurement_source_hash"],
                    "scope_median_ms": {scope: cell["scopes"][scope]["median_ms"] for scope in SCOPES},
                    "scope_p95_ms": {scope: cell["scopes"][scope]["p95_ms"] for scope in SCOPES},
                    "scope_mean_ms": {scope: cell["scopes"][scope]["mean_ms"] for scope in SCOPES},
                    "block_receipt": block_binding, "worker_summary": worker_binding,
                    "cell_summary": cell_binding, "hardware_session": hw_binding,
                    "measurement": measurement, "panel_sha256": panel_hash})
        if number % 26 == 0:
            print(f"Verified {number}/156 models and {len(records)}/552 clean timing cells", flush=True)
    assert len(records) == 552 and len(input_panels) == 24
    groups, baseline = defaultdict(list), {}
    for record in records:
        gkey = tuple(record[k] for k in ("dataset", "architecture", "representation", "trained_execution",
                                         "sparsity_percent", "execution", "batch_size", "role"))
        groups[gkey].append(record)
        if record["representation"] == "raw":
            bkey = (record["dataset"], record["architecture"], record["seed"], record["batch_size"])
            assert bkey not in baseline
            baseline[bkey] = record
    assert len(baseline) == 24 and len(groups) == 184
    primary, controls = [], []
    for gkey, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        rows.sort(key=lambda row: row["seed"])
        assert [row["seed"] for row in rows] == [0, 1, 2]
        ds, arch, representation, trained_execution, pct, execution, batch, role = gkey
        arow = accuracy[(ds, arch, representation, trained_execution, pct)]
        row = dict(dataset=ds, architecture=arch, representation=representation,
                   trained_execution=trained_execution, execution=execution, sparsity_percent=pct,
                   role=role, batch_size=batch, seeds=3, protocol=arow["protocol"],
                   training_count=int(arow["training_count"]), validation_count=int(arow["validation_count"]),
                   training_batch=int(arow["training_batch"]), evaluation_batch=int(arow["evaluation_batch"]),
                   epochs=int(arow["epochs"]), base_lr=float(arow["base_lr"]),
                   hardware_identity=hardware_id, gpu=contract["gpu"], cpu_model=contract["cpu_model"],
                   torch_num_threads=contract["torch_num_threads"], clean_images_per_seed=10000)
        for metric, factor, output in (("clean_accuracy", 100, "clean_accuracy_percent"),
                                       ("clean_loss", 1, "clean_cross_entropy_loss"),
                                       ("mCE_raw", 100, "mCE_percent")):
            row[output + "_mean"] = factor * float(arow[metric + "_mean"])
            row[output + "_sample_sd"] = factor * float(arow[metric + "_sample_sd"])
        for metric in ("clean_accuracy", "mCE_raw"):
            name = metric + "_paired_method_minus_raw_pp"
            row[name + "_mean"] = float(arow[name + "_mean"])
            row[name + "_sample_sd"] = float(arow[name + "_sample_sd"])
        for scope in SCOPES:
            medians = [record["scope_median_ms"][scope] for record in rows]
            row[scope + "_mean_seed_median_ms"], row[scope + "_sample_sd_ms"] = average_sd(medians)
            row[scope + "_mean_seed_p95_ms"], row[scope + "_p95_sample_sd_ms"] = average_sd([record["scope_p95_ms"][scope] for record in rows])
            ratios = [baseline[(ds, arch, record["seed"], batch)]["scope_median_ms"][scope] / record["scope_median_ms"][scope] for record in rows]
            row[scope + "_paired_speedup_mean"], row[scope + "_paired_speedup_sample_sd"] = average_sd(ratios)
            row[scope + "_faster_seeds"] = sum(value > 1 for value in ratios)
            row[scope + "_paired_speedup_min"] = min(ratios)
            row[scope + "_paired_speedup_max"] = max(ratios)
        (primary if role == "primary" else controls).append(row)
    assert len(primary) == 104 and len(controls) == 80
    comparable = [row for row in primary if row["representation"] != "raw"]
    assert len(comparable) == 96
    signals = {}
    for scope in SCOPES:
        ordered = sorted(comparable, key=lambda row: row[scope + "_paired_speedup_mean"], reverse=True)
        def brief(row):
            fields = ("dataset", "architecture", "representation", "execution", "sparsity_percent", "batch_size",
                      "clean_accuracy_percent_mean", "mCE_percent_mean", "clean_accuracy_paired_method_minus_raw_pp_mean",
                      "mCE_raw_paired_method_minus_raw_pp_mean")
            result = {field: row[field] for field in fields}
            result.update({field: row[field] for field in row if field.startswith(scope + "_")})
            return result
        signals[scope] = {"non_raw_condition_batch_rows": len(ordered),
            "mean_speedup_above_one_rows": sum(row[scope + "_paired_speedup_mean"] > 1 for row in ordered),
            "all_three_seeds_faster_rows": sum(row[scope + "_faster_seeds"] == 3 for row in ordered),
            "strongest_mean_speedups": [brief(row) for row in ordered[:8]],
            "largest_slowdowns": [brief(row) for row in ordered[-4:]]}
    data = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "study": "training_sparse_testing_sparse", "status": "Preliminary complete-condition clean latency extraction; final report remains authoritative.",
        "definitions": {"latency": "Mean and sample SD of seed0/1/2 medians; each median uses200 measured repetitions after50 warmups. All latency units ms per batch.",
            "speedup": "For each seed, raw-baseline median divided by condition median, matched on dataset/architecture/batch/hardware/panel. Report mean and sample SD of these three ratios, not ratio of means.",
            "accuracy": "Saved full official clean accuracy and unnormalized mean corruption error; percentage units.52 conditions with all3 seeds. Dense-masked controls reuse checkpoint accuracy and are not independent training runs.",
            "hardware": "One frozen NVIDIA RTX A6000 GPU/CPU/thread class. Physical devices and sessions may differ; identities preserved. No Ada results pooled.",
            "interpretation": "Descriptive comparison of all52 conditions and both batches, without selecting a hidden subset.3/3 faster seeds is directional agreement, not a significance test or proof of generalization. Online scopes include frontend and backbone; cached input is separately measured diagnostic and cannot be subtracted as an additive stage.",
            "limits": "Clean latency only; corruption accuracy aggregates cover official15 MNIST-C or75 CIFAR-C cells. No raw timing/prediction/checkpoint files rehashed. Completed block/worker/clean-cell/hardware receipts checked; all-cell validation belongs to final report. No new runs launched."},
        "coverage": {"conditions": 52, "seeds_per_condition": 3, "models": 156,
                     "primary_worker_clean_cells": 312, "dense_masked_control_clean_cells": 240,
                     "primary_condition_batch_rows": 104, "control_condition_batch_rows": 80,
                     "hardware_sessions": len(hardware_sessions), "paired_panels": len(input_panels)},
        "sources": {"script": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "campaign_snapshot": campaign_binding, "manifest": manifest_binding, "hardware_contract": contract_binding,
            "accuracy_csv": {"path": str(args.accuracy_csv.resolve()), "sha256": hashlib.sha256(accuracy_bytes).hexdigest()},
            "completed_blocks": blocks, "model_receipts": configs},
        "hardware_contract": contract, "primary_rows": primary, "control_rows": controls,
        "seed_records": records, "signals": signals}
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    write_csv(prefix.with_suffix(".csv"), primary)
    write_csv(prefix.with_name(prefix.name + "_dense_masked_controls").with_suffix(".csv"), controls)
    print(json.dumps({"json": str(json_path), "sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
                      "coverage": data["coverage"], "signals": signals}, indent=2), flush=True)


if __name__ == "__main__":
    main()
