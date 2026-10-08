# Reproduce the three comparison PDFs offline

From a fresh repository clone, run:

```bash
python -m venv .venv-figures
source .venv-figures/bin/activate
python -m pip install matplotlib==3.10.7 seaborn==0.13.2 numpy==2.2.6
python analysis/reproduce_figures.py
```

The renderer needs Python 3.10 or newer. It requires no Torch, CUDA, datasets, checkpoints, cluster filesystem, scheduler, network access after dependency installation, or imports from the experiment implementation. It reads only the committed tables and one provenance JSON described below. Historical absolute paths inside those files remain opaque provenance text and are never opened.

The default destination is `reproduced_plots/`, separate from the published `plots/` directory. The script writes exactly three PDFs, each with two pages, plus `reproduction_provenance.json`:

| PDF | Pages |
| --- | --- |
| `MNIST_error_by_corruption.pdf` | ViT-Small, then Swin-Tiny; clean error and all 15 official fixed-severity MNIST-C corruptions |
| `CIFAR10_error_by_corruption.pdf` | ViT-Small, then Swin-Tiny; clean error and all 15 CIFAR-10-C corruptions, equally averaged over five severities within each seed |
| `latency_vs_sparsity.pdf` | Batch 1, then batch 8; each page compares both datasets and architectures |

Explicit locations and optional PNG previews are supported:

```bash
python analysis/reproduce_figures.py \
  --input-dir ./plots \
  --output-dir ./reproduced_plots \
  --preview
```

The input and output directories must differ. A rerun replaces only the renderer's named outputs in its chosen output directory. `--preview` additionally writes six PNG previews at 150 DPI; PDF export retains the publication's `savefig(dpi=300)` setting, vector lines, and embedded TrueType fonts.

## Evidence and validation

Required files under `--input-dir` are:

```text
MNIST_error_by_corruption.csv
CIFAR10_error_by_corruption.csv
requested_comparisons/MNIST_error_by_corruption_seeds.csv
requested_comparisons/CIFAR10_error_by_corruption_seeds.csv
latency_vs_sparsity.csv
latency_vs_sparsity_seeds.csv
latency_vs_sparsity.provenance.json
```

Before rendering, the script validates the complete specified comparison matrix: 512 MNIST and 1,472 CIFAR error aggregates, with all 5,952 corresponding seed rows. It recomputes every mean and sample standard deviation from seeds 0, 1, and 2 and checks each aggregate's explicit seed columns. It also checks training populations, official corruption and severity definitions, compact versus dense training identities, checkpoint consistency across corruption cells, and the absence of withdrawn MNIST color representations.

For latency, it validates all 248 planned points, per-seed medians, matched sample-panel hashes, shared checkpoint identities with the accuracy tables, one hardware class, and the published CSV hashes. It recomputes aggregate means and sample standard deviations only when all three seeds are present. Incomplete points must have missing aggregate values and remain `NaN` in plotting arrays, so lines do not bridge unmeasured gaps. The current committed snapshot contains 80 complete matched-training points, eight complete raw references, and 62 complete points out of 160 planned dense-trained sparse-inference points; 98 dashed points remain missing. These observed counts are calculated from the input rows, not encoded as benchmark answers. The hardware and snapshot caption are checked against the committed latency provenance.

These checks establish internal consistency of the published tables. They do not rerun inference, remeasure latency, or rehash the original checkpoint, prediction, and raw timing archives. See [the experiment protocol](../docs/experiment_protocol.md) and [results](../docs/results.md) for those scientific definitions and evidence limits.

## Plot definitions and reproducibility

Black horizontal lines are the original dense raw baseline: grayscale for MNIST and RGB for CIFAR-10. Blue is grayscale contrast, orange is single color contrast, and purple is color opponency. Solid curves train and test compact models at the same sparsity, including 0%, 20%, 40%, 60%, and 80%. Dashed curves use dense-trained checkpoints with compact inference at 0% through 90%; the separate compact-0-trained follow-up family is excluded. Bands show one sample standard deviation across the three seeds. Compact inference at 0% can still remove naturally empty patches.

The renderer preserves the publication's Seaborn `talk`/`whitegrid`/serif styling, figure dimensions, colors, axes, legends, panel order, and captions. Latency is clean GPU raw-image-to-logits time in milliseconds per batch, summarized as the mean of three per-seed medians, with 50 warm-up and 200 measured batches per seed. Its curves describe the frozen published partial-coverage snapshot, not subsequent live campaign progress.

`reproduction_provenance.json` records hashes of all seven inputs, the renderer, and its three outputs, together with actual Python/library versions and calculated coverage. Matching plotting-library and font versions minimizes visual differences; PDF creation metadata can differ even when pages render identically. No claim of byte-identical PDFs across platforms or dependency versions is made.

The complete renderer was executed on 2026-10-08 with Python 3.11.13 and the pinned plotting versions above. All three PDFs had exactly two pages. An independent PDF reader verified every page against the published originals: page dimensions and extracted text matched, and rendered pixels were identical at a comparison scale of 0.8. Representative exported pages were also inspected visually. This check uses the complete committed tables, not generated example data.

The committed verification records are the [input/output and coverage receipt](../provenance/figure_reproduction/reproduction_provenance.json), [six-page PDF comparison](../provenance/figure_reproduction/pdf_verification.json), and [original CPU rendering submission](../provenance/figure_reproduction/render_submission.json). A [copy manifest](../provenance/figure_reproduction/copy_manifest.json) verifies that these small records are byte-identical to the execution receipts. Generated PDF duplicates remain in the ignored `reproduced_plots/` directory. Historical execution paths in the receipts are provenance, not requirements for running the portable renderer.
