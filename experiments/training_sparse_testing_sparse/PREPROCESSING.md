# Verified fixed contrast frontend

The authoritative source is Shashank Agnihotri's
[`semseg/utils/blur_preprocessing.py`](https://github.com/shashankskagnihotri/semseg_using_VLMs_Lorena/blob/73deca8c15b8d91fee29bac71b6f73c013a47d22/semseg/utils/blur_preprocessing.py),
read from `/ceph/sagnihot/projects/semseg_using_VLMs_Lorena`.
Commit: `73deca8c15b8d91fee29bac71b6f73c013a47d22`.
File SHA-256: `f7c4420a3ab2bf4036323b9b4842e66d46225a88ce6f4440101c7a190861dff8`.
The source file is unmodified relative to that commit; other application files
in the checkout have local changes, which this study preserved. The exact
source is saved at `provenance/original_frontend.py`. No LICENSE, COPYING,
NOTICE, source copyright header, or repository AGENTS.md was found in the
source checkout. Attribution is retained; no upstream license is invented.

## Modes and native outputs

Let `I=[R,G,B]`, let `K` be a 3x3 box kernel with every entry `1/9`, and define
`U_0=I`, `U_d=K*U_(d-1)` independently per channel. Convolution has stride 1,
zero padding 1 on every application, no bias, and no boundary renormalization.
The signed center-surround input is conceptually
`D=I-(U_1+...+U_d)/d`. The actual implementation concatenates all levels and
uses one source-matched 1x1 convolution, preserving its floating-point order.

| Study representation | Source flags / current application name | Native outputs |
|---|---|---|
| `single_color` | `single_color=True`, `channels=3` / `single_color` | `D_R`, `D_G`, `D_B` |
| `grayscale` | `black_white=True`, other flags false, `channels=1` / `bw` | `(D_R+D_G+D_B)/3` |
| `color_opponency` | `color_opponency=True`, `channels=3` / `color_opponency` | `(D_R+D_G+D_B)/3`, `(D_R-D_G)/2`, `D_B/3-D_R/6-D_G/6` |

**Source discrepancy:** the README and newer module describe grayscale as
BT.601 luminance. The actual `calculate_weight` function uses BT.601 weights
`[0.299,0.587,0.114]` only when `blur_depth=0`. For any positive blur depth,
including this study, the grayscale branch uses equal RGB weights. We retain
that actual code behavior rather than substituting the prose description.
Depth zero does not perform center-surround contrast and is not a study mode.

The fixed study depth is **1**. This is the current source application's
`--preprocess-blur-depth` default and the newer module constructor default.
README examples use depth 3, so both depth 1 and depth 3 are covered by source
parity tests. All study training, statistics, evaluation, and timing use depth 1.

## Inspected application context and deliberate boundaries

Inspected call sites are `train.py` (argument parsing and construction),
`semseg/models/dinov3_m2f_cityscapes.py` (frontend before backbone),
`semseg/models/preprocessing.py` (local newer adapter), `README.md`, and data
transforms. The authoritative utility is tracked but not directly called by
the current application; the newer module uses the same contrast weights.
Current call-site files may include uncommitted work, recorded separately in
`provenance/source_context.json` rather than attributed to the source commit.

The old application transforms convert uint8 to `[0,1]`, then apply ImageNet
normalization before the frontend. The newer adapter's docstring likewise says
its input is ImageNet-normalized RGB. The utility itself imposes no range check.
This study's explicitly required input is **float32 raw `[0,1]`**, converted
from uint8 by division by 255, without any RGB normalization before contrast.
Parity means identical transform outputs for identical raw float32 inputs;
it does not claim identical end-to-end preprocessing to the segmentation app.

All source kernel/mixing weights are built in NumPy float64 and cast to
float32. The port preserves this construction and float32 convolution
arithmetic with autocast disabled. TF32 is disabled by the execution scripts.
Weights are registered buffers, with no trainable parameters and no optimizer
state; `.to(device)` moves every buffer correctly. There is no clipping,
magnitude replacement, reconstruction, RGB residual, or learned frontend.

The authoritative class has an additional randomly initialized, trainable
1-to-3-channel convolution (including bias) after native grayscale contrast.
The newer application replaces it with frozen channel replication. Neither
adapter belongs to native contrast. Both are omitted under the task's explicit
native-channel/fixed-frontend requirement. Source parity for grayscale is
tested at the output of `custom_layer`, before that adapter; replacing only
`change_channel_layer` with Identity exposes that boundary. No source contrast
math is reimplemented in the test oracle. Irrelevant source plotting and
mmseg imports are excluded by extracting the original class/helper AST nodes.

MNIST's raw baseline remains one channel. Its active grayscale contrast mode
receives three exact copies at the frontend input and emits its native one-channel
output. The user revised the study to exclude MNIST `single_color` and
`color_opponency` comparisons and delete their artifacts. CIFAR-10 retains all
three native contrast representations. `experiment_scope.json` is authoritative
for which fits, preprocessing rows, previews, and trained conditions belong in
results. Primary support uses exact epsilon 0; small floating-point cancellation
residue is not silently zeroed.

## Legacy threshold semantics versus this study

There is **no parameter named B** in the authoritative preprocessing file or
inspected preprocessing CLI. The unrelated symbol B in model shapes means
batch size. Thus a legacy label such as B=70 cannot be assigned a meaning from
this source, and is not interpreted as 70% zeros.

The source's `sparsity_threshold` has two meanings:

* With `sparsity_type='percentage'`, it is a **fraction** multiplied by
  `x.numel()` across the entire minibatch, including the legacy channel adapter
  output. It finds the k-th absolute magnitude and sets every magnitude `<=`
  that cutoff to zero, so ties can remove more than k entries.
* Otherwise it is an absolute value threshold in the incoming coefficient
  units and sets magnitudes strictly `< sparsity_threshold` to zero.

Legacy thresholding is disabled in this port. The explicit new study component
stable-sorts absolute **native unnormalized** coefficients independently per
image. It zeros exactly the first `floor(percent*M/100)` ranked positions,
including existing zeros; ties follow native channel, row, column order.
The 0% path performs no sorting. Actual zeros can exceed requested percent.
Legacy per-image standardization and remapping to ImageNet mean/std are also
disabled, replaced by frozen dataset/representation training statistics.

## Support, patches, and frozen statistics

Non-overlapping patch order is raster (row then column); each patch's vector
has order `(native_channel, local_row, local_column)`. Official images divide
exactly into 2x2 patches, so no artificial raw pixel padding is needed.
Support is `max(abs(native sparsified patch)) > 0.0`, saved before centering.
Positions stay in the original raster grid. A retained patch uses the ordinary
affine transform `(x-mu)/sigma_effective` for **every** coefficient, including
zeroed values. Empty patch normalization never determines token support.
Model-only sequence padding and absent feature-space children are not pixels.

`compute_normalization_stats.py` visits every ordered clean training sample once
for fitting: 55,000 MNIST or 45,000 CIFAR-10 images under `heldout_val`.
It uses unaugmented uint8 images, float32 frontend arithmetic, no thresholding,
and all native channels/pixels including zeros. CPU float64 `var_mean` chunks
with correction 0 are combined using ordered Chan population-moment updates.
Batch size, exact sample ID list/hash, chunk reduction order, CPU threads,
frontend configuration, source hashes, device, and versions are recorded.
Unequal last batches are included. Invalid finite/coverage/identity conditions
fail rather than drop data. A second complete pass verifies the actual
float32 dense standardized training representation has mean approximately 0
and population std approximately 1 on unguarded channels (tolerance 2e-5).

Measured std below `normalization_min_std=1e-6` gets effective std **1.0**;
measured values and guard flags are retained. Frozen buffers are shared across
all sparsities, seeds, models, dense controls, and evaluation inputs. Optional
`full_refit` artifacts use separate 60,000/50,000-image identities and paths.
Validation/test/corruption distributions never update these buffers.

Each artifact has a versioned JSON and NPZ, and a discoverable JSON alias
`normalization/{dataset}_{representation}_{protocol}.json`. Fitting uses a
shared filesystem advisory lock; writes use same-filesystem temporary files,
fsync, and atomic rename. The payload SHA-256 excludes its own hash field,
avoiding circular self-hashing; NPZ has an independent actual file SHA-256.
Loads verify both hashes, cross-check JSON/NPZ vectors, and check identity.
Changed source/frontend/split artifacts are rejected; select a new output
directory to fit a changed study identity without overwriting prior evidence.

## Verification evidence

`tests/test_frontend.py` checks the exact captured original on deterministic
synthetic inputs for all three modes at depths 1 and 3, and on fixed real MNIST
and CIFAR-10 images for all three modes. It also covers signs, native channels,
float32 under autocast, deterministic threshold ties, existing-zero semantics,
0% bypass, native denominators, and raster coordinates.
`tests/test_normalization.py` compares ordered unequal-batch moments with direct
float64 population moments, and checks training IDs, zeros, guards, cache
hashes/invalidation, nonzero mean offsets, all-empty support, and exact
gather/normalize equivalence. Runtime results are reported by the saved test
logs and report; this document describes the method rather than asserting an
unexecuted check passed.
