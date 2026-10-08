# Requested comparison figures

The requested set contains exactly three PDFs:

- [MNIST error by corruption](../MNIST_error_by_corruption.pdf): two pages, ViT-Small then Swin-Tiny. Each page compares clean error and all 15 official MNIST-C corruptions.
- [CIFAR-10 error by corruption](../CIFAR10_error_by_corruption.pdf): two pages, ViT-Small then Swin-Tiny. Each page compares clean error and all 15 official CIFAR-10-C corruptions.
- [Latency versus sparsity](../latency_vs_sparsity.pdf): two pages, batch size 1 then batch size 8. Each page contains MNIST/CIFAR-10 rows and ViT-Small/Swin-Tiny columns.

Every subplot uses the same colors: black for the unchanged dense input baseline (grayscale for MNIST, RGB for CIFAR-10), purple (`#7B3294`) for color opponency, orange (`#E69F00`) for single color, and blue (`#0072B2`) for grayscale contrast. MNIST contains only its retained baseline and grayscale-contrast conditions.

Solid curves show `training_sparse_testing_sparse`: compact models trained and tested at the same imposed sparsity, using every available level, 0%, 20%, 40%, 60%, and 80%. Dashed curves show `training_dense_testing_sparse`: models trained with dense execution and then evaluated with compact execution at 0% through 90% in steps of 10%. The follow-up models trained with compact execution at 0% are a separate family and are excluded from these requested dashed curves. Compact inference at 0% can already remove naturally empty patches.

The error plots report `100 × (1 − accuracy)` and share 0–100% vertical axes. CIFAR-10-C errors are averaged equally across its five official severities within each seed, then averaged across seeds 0, 1, and 2. MNIST-C uses each corruption's official fixed severity. Bands show one sample standard deviation across those three seed-level values. Every evaluation cell contains 10,000 test images. The protocol is `heldout_val`, with 55,000 MNIST training images or 45,000 CIFAR-10 training images.

The latency figure reports clean GPU-input-to-logits latency, including the representation frontend and backbone, on the matched RTX A6000/AMD EPYC hardware class. Its plotted values summarize the three seeds' measured medians; uncertainty is sample SD across seeds. Missing follow-up timing points remain missing while measurement continues. The CSV and provenance file record the exact available coverage.

The saved latency snapshot is from 2026-10-08 11:24:55 UTC: all 80 original matched-sparsity points and eight baseline points have three seeds, as do 62 of the 160 planned dashed contrast points. The other 98 dashed points remain blank. Available follow-up levels are 0/10/20% for MNIST ViT, 0/10/20/30% for MNIST Swin, and 0/10/20/30% for all CIFAR-10 contrast representations and architectures. Curves do not bridge missing levels. Follow-up raw-baseline remeasurements are preserved separately in [the raw-reference comparison CSV](../latency_vs_sparsity_raw_reference_comparison.csv), without pooling or rescaling the plotted original baseline.

Companion aggregate CSVs have the same stems as the three PDFs in the parent directory. Error seed-level CSVs, architecture-page PNG previews, and [error provenance](error_plots_provenance.json) are stored here. The latency companions are [aggregate CSV](../latency_vs_sparsity.csv), [seed CSV](../latency_vs_sparsity_seeds.csv), and [provenance](../latency_vs_sparsity.provenance.json).

Portable reconstruction: [offline renderer](../../analysis/reproduce_figures.py), which reproduces all three PDFs from the public CSVs. Original error-plot generation: [archived exporter](../../analysis/original_exporters/build_requested_error_plots_20261008.py). Its frozen source, exact command, scheduler job, complete input hashes, and exported artifact hashes are recorded in the launch receipt and provenance. These figures use saved experimental evidence; rendering performs no new model evaluation or timing measurement.

The final error render, job `361101`, completed successfully. Both PDFs contain two verified architecture pages, with 1,984 aggregate rows and 5,952 seed-level rows in their CSV companions. [Artifact verification](error_plots_verification.json) records exact hashes, all-page label/panel checks, and visual inspection of the actual exported PDFs. The final legend-only revision left all numerical CSV bytes unchanged.

The final latency render, job `361102`, also completed successfully. Its two pages contain 150 available aggregate points, with all 98 missing planned dashed points kept explicit. [Latency PDF verification](latency_pdf_audit.json) records the page and artifact checks; both exported pages were visually inspected. The complete requested set is three PDFs containing six pages.
