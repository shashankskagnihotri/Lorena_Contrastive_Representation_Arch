#!/usr/bin/env python3
"""Rebuild the three published comparison PDFs from portable, saved tables.

This performs no model inference and never follows paths recorded inside CSVs.
The only non-standard dependencies are Matplotlib, Seaborn, and NumPy.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("vit_small", "swin_tiny")
ARCH_LABELS = {"vit_small": "ViT-Small", "swin_tiny": "Swin-Tiny"}
REPRESENTATIONS = {
    "mnist": ("grayscale",),
    "cifar10": ("color_opponency", "single_color", "grayscale"),
}
CORRUPTIONS = {
    "mnist": (
        "clean", "shot_noise", "impulse_noise", "glass_blur", "motion_blur",
        "shear", "scale", "rotate", "brightness", "translate", "stripe", "fog",
        "spatter", "dotted_line", "zigzag", "canny_edges",
    ),
    "cifar10": (
        "clean", "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
        "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    ),
}
COLORS = {"raw": "black", "color_opponency": "#7B3294", "single_color": "#E69F00", "grayscale": "#0072B2"}
LABELS = {"color_opponency": "Color opponency", "single_color": "Single color", "grayscale": "Grayscale contrast"}
SEEDS = (0, 1, 2)
MATCHED_LEVELS = (0, 20, 40, 60, 80)
FOLLOWUP_LEVELS = tuple(range(0, 91, 10))
ERROR_KEY = (
    "dataset", "architecture", "representation", "family", "trained_execution",
    "inference_execution", "sparsity_percent", "protocol", "training_count",
    "corruption", "severity_aggregation", "severity_count", "test_images_per_severity",
)
LATENCY_KEY = ("dataset", "architecture", "batch_size", "representation", "family", "sparsity_percent")
SCOPE = "gpu_raw_to_logits"
PDF_NAMES = ("MNIST_error_by_corruption.pdf", "CIFAR10_error_by_corruption.pdf", "latency_vs_sparsity.pdf")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), f"Empty CSV: {path}")
    require(all(None not in row for row in rows), f"Malformed CSV: {path}")
    return rows


def key(row, fields):
    return tuple(row[field] for field in fields)


def close(actual, expected, description):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10),
            f"Numerical mismatch for {description}: saved {actual}, recomputed {expected}")


def valid_hash(value, description):
    require(len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"Invalid SHA-256: {description}")


def bind_identity(identities, identity_key, value, description):
    require(identity_key not in identities or identities[identity_key] == value,
            f"Inconsistent {description}: {identity_key}")
    identities[identity_key] = value


def error_metadata(dataset, architecture, representation, family, sparsity, corruption):
    severity = "clean" if corruption == "clean" else ("fixed" if dataset == "mnist" else "equal_mean_of_1_2_3_4_5")
    return (
        dataset, architecture, representation, family,
        "compact" if family == "training_sparse_testing_sparse" else "dense",
        "dense" if family == "raw_dense_reference" else "compact", sparsity,
        "heldout_val", 55000 if dataset == "mnist" else 45000, corruption, severity,
        5 if dataset == "cifar10" and corruption != "clean" else 1, 10000,
    )


def load_errors(input_dir, inputs):
    """Check every expected panel/condition and recompute its three-seed statistics."""
    records, accuracy_identities = [], {}
    for dataset, stem in (("mnist", "MNIST"), ("cifar10", "CIFAR10")):
        aggregate_path = input_dir / f"{stem}_error_by_corruption.csv"
        seed_path = input_dir / "requested_comparisons" / f"{stem}_error_by_corruption_seeds.csv"
        inputs.extend((aggregate_path, seed_path))
        aggregates, seed_rows = read_csv(aggregate_path), read_csv(seed_path)
        expected = set()
        for architecture in ARCHITECTURES:
            for corruption in CORRUPTIONS[dataset]:
                expected.add(error_metadata(dataset, architecture, "raw", "raw_dense_reference", 0, corruption))
                for representation in REPRESENTATIONS[dataset]:
                    for family, levels in (("training_sparse_testing_sparse", MATCHED_LEVELS),
                                           ("training_dense_testing_sparse", FOLLOWUP_LEVELS)):
                        expected.update(error_metadata(dataset, architecture, representation, family, p, corruption)
                                        for p in levels)

        by_seed, checkpoint_families = defaultdict(dict), {}
        for row in aggregates + seed_rows:
            for field in ("sparsity_percent", "training_count", "severity_count", "test_images_per_severity"):
                row[field] = int(row[field])
        for row in seed_rows:
            group, seed = key(row, ERROR_KEY), int(row["seed"])
            require(group in expected and seed in SEEDS, f"Unexpected error seed row: {group}, seed {seed}")
            require(seed not in by_seed[group], f"Duplicate error seed: {group}, seed {seed}")
            value = float(row["error_percent"])
            require(math.isfinite(value) and 0 <= value <= 100, f"Invalid error percentage: {group}")
            by_seed[group][seed] = value
            for field in ("checkpoint_sha256", "training_config_hash"):
                valid_hash(row[field], field)
            identity = (row["checkpoint_sha256"], row["training_config_hash"])
            family_key = (dataset, row["architecture"], row["representation"], row["family"], seed)
            # A dense-trained sweep shares one checkpoint across all test sparsities.
            if row["family"] != "training_dense_testing_sparse":
                family_key += (row["sparsity_percent"],)
            bind_identity(checkpoint_families, family_key, identity, "error checkpoint family")
            point_key = (dataset, row["architecture"], row["representation"], row["family"], row["sparsity_percent"], seed)
            bind_identity(accuracy_identities, point_key, identity, "error checkpoint across corruptions")

        seen = set()
        for row in aggregates:
            group = key(row, ERROR_KEY)
            require(group in expected and group not in seen, f"Unexpected or duplicate error aggregate: {group}")
            seen.add(group)
            require(int(row["seeds"]) == 3 and set(by_seed[group]) == set(SEEDS),
                    f"Incomplete three-seed error evidence: {group}")
            values = [by_seed[group][seed] for seed in SEEDS]
            for seed, value in zip(SEEDS, values):
                close(float(row[f"error_percent_seed{seed}"]), value, f"{group}, seed {seed}")
            for field, value in (("error_percent_mean", statistics.mean(values)),
                                 ("error_percent_sample_sd", statistics.stdev(values))):
                row[field] = float(row[field])
                close(row[field], value, f"{group}, {field}")
            records.append(row)
        require(seen == expected and set(by_seed) == expected, f"Incomplete error matrix for {dataset}")
    return records, accuracy_identities


def load_latency(input_dir, inputs, accuracy_identities):
    """Validate complete and missing points, matched panels, and checkpoint lineage."""
    aggregate_path = input_dir / "latency_vs_sparsity.csv"
    seed_path = input_dir / "latency_vs_sparsity_seeds.csv"
    provenance_path = input_dir / "latency_vs_sparsity.provenance.json"
    inputs.extend((aggregate_path, seed_path, provenance_path))
    records, seed_rows = read_csv(aggregate_path), read_csv(seed_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    contract = provenance["hardware_contract"]
    hardware = contract["hardware_identity"]
    valid_hash(hardware, "hardware identity")
    # The published caption names this class. Reject other hardware rather than mislabel it.
    require(contract["gpu"] == "NVIDIA RTX A6000"
            and contract["cpu_model"] == "AMD EPYC 7413 24-Core Processor"
            and contract["precision"] == "float32" and contract["torch_num_threads"] == 4,
            "Latency hardware differs from the published A6000/EPYC 7413/FP32/four-thread class")
    # Historical absolute source paths are metadata only. Match output filenames;
    # never open a path recorded inside this JSON or either CSV.
    for path in (aggregate_path, seed_path):
        bindings = [item for item in provenance["outputs"] if Path(item["path"]).name == path.name]
        require(len(bindings) == 1 and bindings[0]["sha256"] == sha256(path),
                f"Published provenance hash mismatch: {path.name}")
    snapshot = datetime.fromisoformat(provenance["followup_snapshot"]["snapshot_updated"])
    require(snapshot.tzinfo is not None, "Latency snapshot lacks a timezone")

    expected = set()
    for dataset, representations in REPRESENTATIONS.items():
        for architecture in ARCHITECTURES:
            for batch in (1, 8):
                expected.add((dataset, architecture, batch, "raw", "raw_dense_reference", None))
                for representation in representations:
                    for family, levels in (("matched_sparse_training", MATCHED_LEVELS),
                                           ("dense_trained_sparse_inference", FOLLOWUP_LEVELS)):
                        expected.update((dataset, architecture, batch, representation, family, p) for p in levels)
    by_seed, panels = defaultdict(dict), {}
    families = {
        "raw_dense_reference": "raw_dense_reference",
        "matched_sparse_training": "training_sparse_testing_sparse",
        "dense_trained_sparse_inference": "training_dense_testing_sparse",
    }
    for row in records + seed_rows:
        row["batch_size"] = int(row["batch_size"])
        row["sparsity_percent"] = int(row["sparsity_percent"]) if row["sparsity_percent"] else None
        require(row["scope"] == SCOPE and row["hardware_identity"] == hardware,
                f"Mixed latency scope or hardware class: {key(row, LATENCY_KEY)}")
    for row in seed_rows:
        group, seed = key(row, LATENCY_KEY), int(row["seed"])
        require(group in expected and seed in SEEDS, f"Unexpected latency seed: {group}, {seed}")
        require(seed not in by_seed[group], f"Duplicate latency seed: {group}, {seed}")
        value = float(row["median_ms"])
        require(math.isfinite(value) and value > 0, f"Invalid latency: {group}, {seed}")
        by_seed[group][seed] = value
        for field in ("panel_sha256", "checkpoint_sha256", "training_config_hash", "summary_sha256"):
            valid_hash(row[field], field)
        family = families[row["family"]]
        require(row["source_study"] == ("training_dense_testing_sparse" if family == "training_dense_testing_sparse"
                                       else "training_sparse_testing_sparse"), "Incorrect latency source study")
        point_key = (row["dataset"], row["architecture"], row["representation"], family,
                     row["sparsity_percent"] if row["sparsity_percent"] is not None else 0, seed)
        require(accuracy_identities[point_key] == (row["checkpoint_sha256"], row["training_config_hash"]),
                f"Accuracy/latency checkpoint mismatch: {point_key}")
        panel_key = (row["dataset"], row["architecture"], row["batch_size"], seed)
        bind_identity(panels, panel_key, row["panel_sha256"], "latency sample panel")

    seen = set()
    for row in records:
        group = key(row, LATENCY_KEY)
        require(group in expected and group not in seen, f"Unexpected or duplicate latency aggregate: {group}")
        seen.add(group)
        require(row["latency_unit"] == "milliseconds_per_batch" and int(row["expected_seeds"]) == 3,
                f"Invalid latency units or expected seeds: {group}")
        actual_seeds = sorted(by_seed[group])
        require(int(row["available_seeds"]) == len(actual_seeds)
                and row["seed_ids"] == ";".join(map(str, actual_seeds)), f"Latency seed coverage mismatch: {group}")
        require(row["plotted"] in ("True", "False"), f"Invalid plotted flag: {group}")
        row["plotted"] = row["plotted"] == "True"
        complete = actual_seeds == list(SEEDS)
        require(row["plotted"] == complete, f"Partial latency point incorrectly plotted: {group}")
        if row["family"] != "dense_trained_sparse_inference":
            require(complete, f"Missing original-study reference or matched point: {group}")
        if complete:
            values = [by_seed[group][seed] for seed in SEEDS]
            for field, value in (("mean_seed_median_ms", statistics.mean(values)), ("sample_sd_ms", statistics.stdev(values))):
                row[field] = float(row[field])
                close(row[field], value, f"{group}, {field}")
        else:
            for field in ("mean_seed_median_ms", "sample_sd_ms"):
                require(row[field] == "" or math.isnan(float(row[field])), f"Partial seed mean supplied: {group}")
                row[field] = math.nan
    require(seen == expected, "Incomplete planned latency matrix")
    coverage = {}
    for family in families:
        points = [row for row in records if row["family"] == family]
        complete = sum(row["plotted"] for row in points)
        coverage[family] = {"planned": len(points), "plotted_three_seed": complete, "missing_three_seed": len(points) - complete}
    return records, {"coverage": coverage, "seed_rows": len(seed_rows), "hardware_contract": contract,
                     "snapshot_utc": snapshot.astimezone(timezone.utc).isoformat()}


def render_errors(rows, output_dir, preview):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    import seaborn as sns

    sns.set_theme(context="talk", style="whitegrid", font="serif", font_scale=1.0)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    for dataset, name in (("mnist", "MNIST"), ("cifar10", "CIFAR10")):
        stem = name + "_error_by_corruption"
        chosen = [row for row in rows if row["dataset"] == dataset]
        representations = REPRESENTATIONS[dataset]
        target = output_dir / (stem + ".pdf")
        temporary = output_dir / ("." + stem + ".tmp.pdf")
        with PdfPages(temporary, metadata={"Title": name + " error by corruption", "Subject": "Complete three-seed accuracy evidence; compact0-trained follow-up excluded"}) as pages:
            for architecture in ARCHITECTURES:
                fig, axes = plt.subplots(4, 4, figsize=(22, 18.6), sharex=True, sharey=True)
                for ax, corruption in zip(axes.flat, CORRUPTIONS[dataset]):
                    panel = [row for row in chosen if row["architecture"] == architecture and row["corruption"] == corruption]
                    baseline = next(row for row in panel if row["representation"] == "raw")
                    mean, sd = baseline["error_percent_mean"], baseline["error_percent_sample_sd"]
                    ax.axhspan(max(0, mean - sd), min(100, mean + sd), color=COLORS["raw"], alpha=.08, linewidth=0)
                    ax.axhline(mean, color=COLORS["raw"], linewidth=2, zorder=5)
                    for representation in representations:
                        for family, style, marker in (("training_sparse_testing_sparse", "-", "o"), ("training_dense_testing_sparse", "--", "s")):
                            points = sorted((row for row in panel if row["representation"] == representation and row["family"] == family), key=lambda row: row["sparsity_percent"])
                            x = [row["sparsity_percent"] for row in points]
                            y = [row["error_percent_mean"] for row in points]
                            sd = [row["error_percent_sample_sd"] for row in points]
                            ax.fill_between(x, [max(0, a-b) for a, b in zip(y, sd)], [min(100, a+b) for a, b in zip(y, sd)], color=COLORS[representation], alpha=.09, linewidth=0)
                            ax.plot(x, y, color=COLORS[representation], linestyle=style, marker=marker, markersize=4.5,
                                    linewidth=2.1, markerfacecolor=COLORS[representation] if style == "-" else "white", markeredgewidth=1.2)
                    ax.set_title("Clean (IID)" if corruption == "clean" else corruption.replace("_", " ").capitalize(), fontsize=17, pad=10)
                    ax.set_xlim(0, 90)
                    ax.set_ylim(0, 100)
                    ax.set_xticks([0, 20, 40, 60, 80, 90])
                    ax.set_yticks([0, 20, 40, 60, 80, 100])
                    ax.tick_params(labelsize=13)
                    ax.grid(alpha=.28)
                name_label = "CIFAR-10" if dataset == "cifar10" else "MNIST"
                fig.suptitle(f"{name_label} | {ARCH_LABELS[architecture]} | classification error by corruption", fontsize=25, y=.985)
                severity = "Each corruption: mean of severities 1–5 within each seed" if dataset == "cifar10" else "Each MNIST-C corruption: its official fixed severity"
                train_n = "45,000" if dataset == "cifar10" else "55,000"
                fig.text(.5, .950, f"{severity} | heldout_val | training N = {train_n}", ha="center", fontsize=16)
                handles = [Line2D([0], [0], color=COLORS["raw"], linewidth=2, label="Grayscale baseline" if dataset == "mnist" else "RGB baseline")]
                handles += [Line2D([0], [0], color=COLORS[rep], linewidth=3, label=LABELS[rep]) for rep in representations]
                handles += [Line2D([0], [0], color="#444444", linestyle="-", marker="o", label="Train sparse; test at same sparsity"),
                            Line2D([0], [0], color="#444444", linestyle="--", marker="s", markerfacecolor="white", label="Train dense; test sparse")]
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .932), ncol=3 if dataset == "cifar10" else 4,
                           fontsize=14, frameon=False, columnspacing=1.5, handlelength=3)
                fig.supxlabel("Imposed inference sparsity (%)", fontsize=20, y=.036)
                fig.supylabel("Classification error (%)", fontsize=20, x=.013)
                fig.text(.5, .010, "Lines: mean over seeds 0, 1, 2; bands: ±1 sample SD. Compact inference at 0% can remove naturally empty patches.", ha="center", fontsize=13)
                fig.subplots_adjust(left=.060, right=.985, bottom=.085, top=.852, hspace=.30, wspace=.12)
                pages.savefig(fig, dpi=300)
                if preview:
                    fig.savefig(output_dir / f"{stem}_{architecture}.png", dpi=150)
                plt.close(fig)
        temporary.replace(target)


def render_latency(records, metadata, output_dir, preview):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    import seaborn as sns

    sns.set_theme(context="talk", style="whitegrid", font="serif", rc={"figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "axes.titleweight": "regular"})
    target = output_dir / "latency_vs_sparsity.pdf"
    temporary = output_dir / ".latency_vs_sparsity.tmp.pdf"
    names = {"raw": "Raw baseline: RGB / grayscale", **LABELS}
    snapshot = datetime.fromisoformat(metadata["snapshot_utc"]).strftime("%Y-%m-%d %H:%M UTC")
    with PdfPages(temporary, metadata={"Title": "Clean online latency versus sparsity", "Subject": "Measured three-seed means and sample SD; matched A6000/EPYC7413 class"}) as pdf:
        for batch in (1, 8):
            fig, axes = plt.subplots(2, 2, figsize=(16, 13.0), sharex=True)
            fig.subplots_adjust(left=.09, right=.98, top=.83, bottom=.16, hspace=.31, wspace=.21)
            fig.suptitle(f"Clean online latency versus sparsity | Batch size {batch}", fontsize=23, y=.99)
            colors = [Line2D([0], [0], color=COLORS[rep], lw=2.5, label=names[rep]) for rep in ("raw", "color_opponency", "single_color", "grayscale")]
            fig.legend(handles=colors, loc="upper center", bbox_to_anchor=(.54, .955), ncol=4, frameon=False, fontsize=15)
            styles = [Line2D([0], [0], color=".3", lw=2.5, marker="o", label="Solid: trained and tested at matching sparsity"),
                      Line2D([0], [0], color=".3", lw=2.5, ls="--", marker="s", label="Dashed: trained dense; tested sparse")]
            fig.legend(handles=styles, loc="upper center", bbox_to_anchor=(.54, .914), ncol=2, frameon=False, fontsize=14)
            for row_index, (dataset, representations) in enumerate(REPRESENTATIONS.items()):
                for column, architecture in enumerate(ARCHITECTURES):
                    ax = axes[row_index, column]
                    subset = [row for row in records if (row["dataset"], row["architecture"], row["batch_size"]) == (dataset, architecture, batch)]
                    raw = next(row for row in subset if row["family"] == "raw_dense_reference")
                    mean, sd = raw["mean_seed_median_ms"], raw["sample_sd_ms"]
                    ax.axhline(mean, color="black", lw=2.0, zorder=2)
                    ax.axhspan(mean-sd, mean+sd, color="black", alpha=.08, lw=0)
                    for representation in representations:
                        for family, style, marker in (("matched_sparse_training", "-", "o"), ("dense_trained_sparse_inference", "--", "s")):
                            points = sorted((row for row in subset if row["family"] == family and row["representation"] == representation), key=lambda row: row["sparsity_percent"])
                            x = np.array([row["sparsity_percent"] for row in points])
                            y = np.array([row["mean_seed_median_ms"] if row["plotted"] else np.nan for row in points])
                            sd = np.array([row["sample_sd_ms"] if row["plotted"] else np.nan for row in points])
                            ax.plot(x, y, color=COLORS[representation], ls=style, marker=marker, ms=5.5, lw=2.1, zorder=3)
                            ax.fill_between(x, y-sd, y+sd, color=COLORS[representation], alpha=.12, lw=0)
                    dashed = [row for row in subset if row["family"] == "dense_trained_sparse_inference"]
                    complete = sum(row["plotted"] for row in dashed)
                    ax.text(.03, .05, f"Dashed coverage: {complete}/{len(dashed)} points", ha="left", va="bottom", transform=ax.transAxes, fontsize=11.5,
                            bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=2))
                    ax.set_title(("MNIST" if dataset == "mnist" else "CIFAR-10") + " | " + ARCH_LABELS[architecture], fontsize=19)
                    ax.set_xlim(0, 90)
                    ax.set_xticks(range(0, 91, 10))
                    ax.set_ylim(bottom=0)
                    ax.tick_params(labelsize=12, labelbottom=True)
                    ax.set_xlabel("Inference sparsity (%)", fontsize=16)
                    ax.set_ylabel("Latency (ms / batch)", fontsize=16)
                    ax.grid(axis="x", alpha=.18)
                    ax.grid(axis="y", alpha=.4)
            fig.text(.5, .103, "Mean of three seed medians; bands show sample SD. Each seed: 50 warm-up + 200 measured batches.", ha="center", fontsize=12)
            fig.text(.5, .078, "NVIDIA RTX A6000 | AMD EPYC 7413 | FP32 | 4 CPU threads. Missing three-seed points are not connected.", ha="center", fontsize=12)
            fig.text(.5, .053, "Black line: original dense-raw common reference. Exact checkpoints and timing panels match across studies; sessions differ.", ha="center", fontsize=11.2)
            fig.text(.5, .029, f"Follow-up coverage snapshot: {snapshot}. Raw-pixel sparse sweeps and compact-trained 0% follow-up family are excluded.", ha="center", fontsize=10.8)
            pdf.savefig(fig, dpi=300)
            if preview:
                fig.savefig(output_dir / f"latency_vs_sparsity_batch{batch}.png", dpi=150)
            plt.close(fig)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "plots", help="Directory containing the committed plot CSVs and provenance")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced_plots", help="Separate destination; defaults to reproduced_plots beside analysis")
    parser.add_argument("--preview", action="store_true", help="Also save six PNG page previews at 150 DPI")
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    require(output_dir != input_dir, "Choose a separate output directory to preserve the published plots")
    inputs = []
    errors, identities = load_errors(input_dir, inputs)
    latency, metadata = load_latency(input_dir, inputs, identities)
    input_bindings = [{"path": str(path.relative_to(input_dir)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in inputs]

    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = output_dir / "reproduction_provenance.json"
    # A rerun must not leave a previous success receipt if rendering fails.
    receipt.unlink(missing_ok=True)
    print(f"Validated {len(errors)} error aggregates, {len(errors) * len(SEEDS)} error seed rows, and {len(latency)} planned latency points.", flush=True)
    render_errors(errors, output_dir, args.preview)
    render_latency(latency, metadata, output_dir, args.preview)
    for item, path in zip(input_bindings, inputs):
        require(item["sha256"] == sha256(path), f"Input changed during rendering: {path.name}")
    outputs = []
    for name in PDF_NAMES:
        path = output_dir / name
        require(path.is_file() and path.stat().st_size > 0, f"Missing output: {name}")
        outputs.append({"path": name, "sha256": sha256(path), "bytes": path.stat().st_size, "pages": 2})
    result = {
        "state": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "inputs": input_bindings, "outputs": outputs,
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__, "seaborn": sns.__version__},
        "error_coverage": {"aggregate_rows": len(errors), "seed_rows": len(errors) * len(SEEDS), "by_dataset": dict(Counter(row["dataset"] for row in errors)), "seed_ids": list(SEEDS)},
        "latency": metadata,
        "validation": ["Complete protocol matrix and no duplicate rows", "Exact seed identities and mean/sample-SD agreement", "Checkpoint lineage shared between accuracy and latency", "Matched timing panels and a single hardware class", "Published latency CSV hashes", "Partial latency points preserved as NaN, never averaged or interpolated"],
        "limits": "Reproduces saved aggregate evidence, not inference or timing measurements. Historical paths are not opened; original checkpoints, predictions and raw timing rows are not revalidated. PDF bytes may vary with library versions, fonts and creation metadata.",
    }
    temporary = receipt.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    print(json.dumps({"state": "complete", "output_dir": str(output_dir), "pdfs": list(PDF_NAMES), "latency_coverage": metadata["coverage"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
