# Native-grid architectures and compact execution

Both models start from random weights and produce ten logits. No pretrained
weights or ImageNet resizing are used. Input patch vectors have channel-major
`(channel, local row, local column)` coefficient order and raster patch order.
The caller supplies a separate Boolean support measured before normalization.
`build_backbone(architecture, image_size, in_channels, drop_path=0.1,
execution='compact')` accepts `vit_small` or `swin_tiny`, native size 28 or 32,
and the actual frontend channel count. `forward_patches(patches, support,
execution=None)` returns logits. Dense execution accepts `support=None` and
retains every original image patch.

| Configuration | ViT-Small | Swin-Tiny |
|---|---|---|
| Patch size | 2 | 2 |
| Width | 384 | 96, 192, 384, 768 |
| Blocks | 12 | 2, 2, 6, 2 |
| Heads | 6 | 3, 6, 12, 24 |
| MLP ratio | 4 | 4 |
| LayerNorm epsilon | 1e-6 | 1e-5 |
| MNIST grids | 14 | 14, 7, 4, 2 |
| CIFAR-10 grids | 16 | 16, 8, 4, 2 |
| MNIST windows | global | 7, 7, 4, 2 |
| CIFAR-10 windows | global | 4, 4, 4, 2 |

Learned linear layers use biases except Swin merge reduction. Attention uses
scaled dot products with explicit softmax, QKV bias, and output projection.
GELU MLPs and pre-normalization residual blocks match the dense references.
Dropout is zero; the stochastic-depth probability increases linearly over the
twelve blocks from zero to the configured maximum. Each branch draws one mask
per original image, including empty images, then indexes that mask for packed
rows. It never treats window IDs as image IDs.

ViT has learned original-grid position embeddings and a class token. Compact
execution gathers active patch vectors before linear projection, gathers their
original positions, then packs them into tensors padded to the current batch's
maximum retained length. The class token is always present. Padding is masked
as keys and suppressed after residuals. The MLP executes padded sequence rows;
telemetry explicitly records useful tokens, class tokens, projected rows,
executed lengths and padding. An entirely empty image uses the class token only.
Batch padding can remain when other images contain retained patches.

Swin keeps flat original coordinates and stage extents. Registered geometry
maps them to cyclically shifted windows, local coordinates, and the official
shift-region mask. QKV, output projection, and MLP operate on active feature
rows. Nonempty windows are grouped by exact retained length, so attention has
no execution padding and empty windows are skipped. Relative bias indexes the
original within-window coordinate offsets. Shift restrictions use the
reference's additive -100 mask, while absent keys are excluded with negative
infinity. Shifts are disabled when a stage fits inside one window.

Patch merging creates a parent iff at least one original child exists. It
concatenates top-left, bottom-left, top-right, bottom-right children, inserting
zero feature vectors for absent children. It applies the standard 4C LayerNorm
and bias-free 4C-to-2C projection. Odd grids pad only right/bottom structural
children. No feature-based pruning follows merging. Final pooling averages
active normalized features; an empty image has a zero pooled vector and the
classifier-bias logits. Empty tensor paths retain zero gradients for parameters
without creating fake image tokens.

The `dense_masked` path executes the original dense grids but shares exactly
these sparse semantics: absent keys excluded, inactive residual states zeroed,
OR support propagated through merges, active-only pooling. The `dense` path
ignores sparse support and is the ordinary dense control. These controls are
not interchangeable.

`dense_reference(model)` instantiates the configurable timm VisionTransformer
or torchvision SwinTransformer and strictly transfers every parameter. ViT uses
the prescribed width/depth/heads and patch size. Swin uses a block factory to set
the dataset's per-stage windows and replaces only the stem input-channel count.
The tests compare all-active outputs at both image sizes, all-active parameter
gradients, and compact versus dense-masked outputs and every parameter gradient.
They also exercise empty/ragged batches, singleton windows, original positions,
odd merges, shifted wrapping, and original-image stochastic depth. FP32 output
tolerances are absolute 5e-6 / relative 4e-5 or tighter; full parameter-gradient
tolerance is absolute 3e-5 / relative 2e-4. Isolated FP64 structural tests use
exact equality or 1e-12. These are declared validation thresholds, not a claim
that the checks have passed; executed results are saved by the check job.

Architectural references are the installed versions pinned in the project
environment and provenance files. Source references inspected during design:
[timm VisionTransformer v1.0.22](https://github.com/huggingface/pytorch-image-models/blob/v1.0.22/timm/models/vision_transformer.py)
and [torchvision SwinTransformer v0.24.0](https://github.com/pytorch/vision/blob/v0.24.0/torchvision/models/swin_transformer.py).

Dynamic maximum extraction, nonzero extraction, sorting and retained-length
group decisions execute inside the forward call. Their synchronization and CPU
launch overhead are part of online latency. `latest_telemetry` stores detached
device tensors plus shape integers; nested Swin stage records include per-image
active counts, all window lengths, executed rows and attention-pair counts.
`set_profiling(True)` enables named embedding, grouping, attention, MLP/residual,
merge and head scopes. Profiling is disabled by default and must be disabled in
primary latency measurements. Parameter counts are computed from instantiated
models and logged per native channel count; no speedup is implied by sparsity.
