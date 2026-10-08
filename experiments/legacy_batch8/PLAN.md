# Sparse contrast benchmark execution plan

Original requested protocol: provenance/USER_TASK.txt. The user-revised active representation scope is experiment_scope.json: MNIST raw/grayscale and CIFAR-10 raw/all three contrast representations. Withdrawn MNIST single-color/opponent artifacts were removed as requested. Primary protocol: heldout_val. Preserve all existing data, environments, projects, and unrelated jobs.

1. Inspect authorized source/call sites and scheduler/environment; verify dataset originals and deterministic stratified splits. Pin source and environment provenance.
2. Port the three verified fixed frontends. Implement native coefficient sparsification, support-before-normalization, population statistics, and exact artifact identities. Block contrast runs if source parity is unavailable.
3. Implement small-image ViT-Small and Swin-Tiny, physically compact and dense-masked paths, native coordinates, merging semantics, and empty-image handling. Validate forward and parameter-gradient parity.
4. Run required tests, explicit tiny overfit/end-to-end checks, and full raw clean-validation pilots. Fit full permitted-training normalization first. Freeze one clean-validation recipe per dataset/architecture.
5. Train all 156 active registered runs (21 per MNIST architecture, 57 per CIFAR-10 architecture) from scratch with deterministic data randomness, full held-out validation every epoch, epoch-boundary resume, TensorBoard, and immutable attempt provenance.
6. Evaluate final checkpoints on complete official clean and corruption cells. Preserve per-sample predictions and counts.
7. On isolated scheduler-authorized hardware benchmark all trained paths and compact checkpoint dense-masked controls, batches 1/8, all required input cells/scopes, 50 warmups and 200 measurements. Save profiler diagnostics separately.
8. Reconcile coverage, aggregate across three seeds, render required scientific plots and TensorBoard figures, and generate REPORT.md from saved evidence.

Execution gates are fail-closed. Scheduler submission is not completion. STATUS.json is the single project-local execution record. No test/corruption-informed recipe tuning. No full_train_refit unless explicitly enabled; optional protocol remains separate.
