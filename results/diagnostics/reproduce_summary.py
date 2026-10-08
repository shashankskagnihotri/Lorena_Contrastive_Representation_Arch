"""Regenerate small diagnostic tables from the archived JSON evidence, on CPU.

No model execution, timing measurement, or external dependencies are used.
The arithmetic estimate counts multiply-add pairs as one MAC and excludes
normalization, nonlinearities, indexing, memory traffic, and launch overhead.
"""
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write_csv(name, rows):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = json.loads((ROOT / "source_sha256_manifest.json").read_text())
    checked = set()
    for record in manifest["records"]:
        destination = record["destination"]
        if destination in checked:
            continue
        payload = (ROOT / destination).read_bytes()
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Archived source hash mismatch: {destination}")
        checked.add(destination)
    selection = json.loads((ROOT / "selection.json").read_text())
    support_rows, profiles, arithmetic = [], [], []
    for item in selection["cells"]:
        path = ROOT / item["cell_path"]
        records = [json.loads(line) for line in (path / "tokens.jsonl").read_text().splitlines()]
        assert len(records) == 200
        batch = item["batch_size"]
        base = {key: item[key] for key in ("dataset", "architecture", "representation", "execution", "sparsity_percent", "seed", "batch_size")}
        row = dict(base)
        for key in ("natural_zero_fraction", "achieved_zero_fraction", "retained_patches"):
            row[key] = statistics.mean(float(x) for record in records for x in record["native"][key])
        executions = [record["execution"] for record in records]
        row["executed_sequence_length"] = statistics.mean(e.get("executed_sequence_length", 0) for e in executions)
        row["padding_tokens_per_image"] = statistics.mean(e.get("padding_tokens", 0) / batch for e in executions)
        row["window_length_groups_sum"] = statistics.mean(sum(b.get("window_length_groups", 0) for s in e.get("stages", []) for b in s["blocks"]) for e in executions)
        for stage in range(4):
            row[f"stage{stage}_active_per_image"] = statistics.mean(x for e in executions for x in e["stages"][stage]["blocks"][0]["active_per_image"]) if "stages" in executions[0] else ""
        support_rows.append(row)

        macs = []
        for e in executions:
            if item["architecture"] == "vit_small":
                length, dim = e["executed_sequence_length"], 384
                block_macs = 12 * (12 * length * dim**2 + 2 * length**2 * dim)
                merge_macs = 0
            else:
                block_macs = sum(12 * block["executed_rows"] * (96 * 2**stage)**2 + 2 * (96 * 2**stage) * block["executed_attention_pairs"] for stage, s in enumerate(e["stages"]) for block in s["blocks"]) / batch
                merge_macs = sum(8 * (96 * 2**stage)**2 * sum(e["stages"][stage + 1]["blocks"][0]["active_per_image"]) / batch for stage in range(3))
            macs.append((block_macs, merge_macs))
        arithmetic.append({**base, "transformer_block_MACs_per_image": statistics.mean(x[0] for x in macs), "merge_MACs_per_image": statistics.mean(x[1] for x in macs), "scope": "QKV, attention matmuls, output projection, MLP; merges separately; excludes stem/head and non-matmul work"})

        profile = json.loads((path / "profile.json").read_text())
        events = profile["events"]
        def count(name):
            return sum(e["count"] for e in events if e["name"] == name)
        def cpu(name):
            return sum(e["cpu_total_ms_inclusive"] for e in events if e["name"] == name)
        profiles.append({**base,
            "matched_uninstrumented_wall_ms": profile["matched_uninstrumented_wall_ms"],
            "profiled_call_wall_ms": profile["profiled_call_wall_ms"],
            "profiling_overhead_ratio": profile["profiling_overhead_ratio"],
            "cudaLaunchKernel_count": count("cudaLaunchKernel"),
            "aten_nonzero_count": count("aten::nonzero"),
            "cudaStreamSynchronize_count": count("cudaStreamSynchronize"),
            "transformer_blocks_CPU_inclusive_ms": sum(e["cpu_total_ms_inclusive"] for e in events if re.fullmatch(r"vit/block_\d+|swin/stage_\d+/block_\d+", e["name"])),
            "swin_attention_CPU_inclusive_ms": cpu("swin/attention"),
            "swin_window_group_CPU_inclusive_ms": cpu("swin/window_group_shift_position"),
            "source_profile": item["cell_path"] + "/profile.json"})
    write_csv("support_summary.csv", support_rows)
    write_csv("profile_summary.csv", profiles)
    write_csv("arithmetic_proxy.csv", arithmetic)


if __name__ == "__main__":
    main()
