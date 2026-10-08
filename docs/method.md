# The implemented sparse-contrast method

This document describes the computation that produced this repository's experiments. It explains the fixed image transform, coefficient ranking, normalization, token support, compact ViT and Swin execution, training, and the later inference-only intervention. It distinguishes the mathematical method from implementation cost and floating-point behavior. The measured accuracy and latency results, and the evidence for possible causes of underperformance, belong in [performance_analysis.md](performance_analysis.md).

The central question is whether a classifier can learn from a sparse, signed local-contrast representation and save enough transformer computation to repay the cost of constructing and compacting that representation. The representation is the actual input to the classifier. RGB is not a hidden feature branch, and contrast is not merely a routing signal for selecting RGB embeddings. “Contrast” refers to an image transform. The learning objective is supervised cross-entropy, not contrastive learning.

## 1. Which implementation and experiments this describes

The public packages in `src/` are copies of the frozen scientific implementation. The original runtime directories were not moved or edited for publication.

| Role | Frozen source identity |
|---|---|
| Training of the completed large-batch study | `9359e4f10c0c2184eef4af8885edc7076bded30188775cc14222f79819f648c4` |
| Compatible original-study measurement/orchestration snapshot | `7ea1b6e8bd5a78b0ad411d0545ce060a5ae3597a2f29d788017f39cbeecf50d0` |
| Inference-only study and adapter | `8cf4d0c1e0ce5443d5080ea397df9a8f085d1bf2c4292912676545d38533a0cb` |
| Explicit numerical-validation amendment for a bounded group of inference timing workers | `3c965bf1a2c9f3f11bb14c0d6ad3354a2266e7d0763b2146b5a8fbc69d314769` |
| Controller extension preserving that amendment through retries and final verification | `fcfbdad09dd2aa67d5e62616a5ed77f19f205b324d0474ad15af15bef15466f7` |

The training and compatible measurement snapshots have byte-identical `frontend.py`, `normalization.py`, `pipeline.py`, `models.py`, `train.py`, `data.py`, and `compute_normalization_stats.py`. The public inference adapter matches its `8cf4d0c...` snapshot. The public packages preserve their exported scientific implementation; exactly three bootstrap files have documented path/interpreter adaptations: `sparse_contrast/common.py`, `dense_sparse/__init__.py`, and `dense_sparse/common.py`. These files must not be described as byte-identical to their originals. Their original and adapted hashes are recorded in [portability_adaptations.json](../provenance/portability_adaptations.json). Full unmodified snapshots and their manifests are retained under [provenance/sources](../provenance/sources/). A measurement-source hash is not substituted for the training-source hash of a checkpoint. See [reproducibility.md](reproducibility.md) for the distinction between importing the public package, reproducing figures, and rerunning a registered experiment.

The [original task](../experiments/training_sparse_testing_sparse/provenance/USER_TASK.txt) remains useful for the mathematical specification, but later explicit decisions changed the scope and training batch size. The operative original-study scope is:

- MNIST: native raw grayscale plus grayscale contrast.
- CIFAR-10: raw RGB plus single-color, grayscale, and color-opponency contrast.
- Two backbones: small-image ViT-Small and Swin-Tiny.
- Three training seeds: 0, 1, and 2.
- Contrast compact training at imposed coefficient sparsities 0%, 20%, 40%, 60%, and 80%.
- One additional ordinary dense 0%-contrast control for each retained dataset/representation/backbone/seed.
- One ordinary dense raw baseline per dataset/backbone/seed.

This gives 120 compact contrast models, 24 dense contrast controls, and 12 raw baselines: **156 primary trained models**, excluding pilots. The withdrawn MNIST single-color and color-opponency conditions are not part of these results. The later larger-batch campaign is a separate fresh training campaign; its recipes are documented below. It is not a continuation silently changing the batch size of the earlier batch-8 runs. See the [active scope](../experiments/training_sparse_testing_sparse/experiment_scope.json).

## 2. The entire forward computation

Let an input batch contain decoded `uint8` images. For one image, write its raw floating-point pixels as

$$
X = X_{\mathrm{uint8}}/255.
$$

The original contrast model computes

$$
X \xrightarrow{\text{fixed frontend}} C_0
\xrightarrow{\text{native coefficient ranking}} C_p
\xrightarrow{\text{native support test}} S
\quad\text{and}\quad
Z_{c,h,w}=\frac{(C_p)_{c,h,w}-\mu_c}{\sigma_{c,\mathrm{eff}}}.
$$

The backbone receives 2×2 patch vectors from $Z$, together with the separately saved support $S$. Compact execution gathers supported patches, projects those contrast values, and runs the sparse transformer. The logits have shape $B\times10$.

The implementation normalizes and patchifies the full small image before gathering. Because normalization is a fixed channelwise affine function and gathering uses the **original native support**, this gives the same retained patch values as gathering first and then normalizing. It does incur dense frontend/layout/normalization work. The code does not claim those operations become sparse merely because later tokens are absent. [Forward pipeline](../src/sparse_contrast/pipeline.py#L28).

For raw baselines, the frontend and coefficient-ranking steps are absent. Ordinary dense execution keeps every original patch. MNIST raw input is one-channel grayscale; CIFAR-10 raw input has three RGB channels. Both retain the native image resolution.

| Dataset and representation | Raw input | Native representation | Native coefficients per image $M$ | Patch grid | Patch vector width |
|---|---:|---:|---:|---:|---:|
| MNIST raw or grayscale contrast | $1\times28\times28$ | $1\times28\times28$ | 784 | $14\times14=196$ | $1\times2\times2=4$ |
| CIFAR-10 grayscale contrast | $3\times32\times32$ | $1\times32\times32$ | 1,024 | $16\times16=256$ | 4 |
| CIFAR-10 raw, single-color contrast, or opponency contrast | $3\times32\times32$ | $3\times32\times32$ | 3,072 | $16\times16=256$ | 12 |

There is no resizing to 224×224, no overlapping patch extraction, and no raw-image spatial padding to make these initial 2×2 patches fit. CIFAR training augmentation has its own padding, described later. Swin's later odd-grid feature padding is a separate operation.

## 3. The exact fixed frontend

### 3.1 Provenance and boundaries of the port

The frontend was derived from `semseg/utils/blur_preprocessing.py` at upstream commit `73deca8c15b8d91fee29bac71b6f73c013a47d22`, source-file SHA-256 `f7c4420a3ab2bf4036323b9b4842e66d46225a88ce6f4440101c7a190861dff8`. The captured upstream file, source context, and adaptations are retained in the experiment's [PREPROCESSING.md](../experiments/training_sparse_testing_sparse/PREPROCESSING.md) and [original source copy](../experiments/training_sparse_testing_sparse/provenance/original_frontend.py).

The port preserves the native fixed contrast transform. It deliberately excludes the legacy trainable grayscale-to-three-channel adapter, legacy thresholding, per-image normalization, and ImageNet range remapping. It receives raw `[0,1]` values as required by this study. That input contract differs from the old segmentation application's surrounding preprocessing. Numerical source parity means applying the same fixed transform to the same input, not reproducing the entire segmentation application.

No sRGB-to-linear-light conversion is performed. “Grayscale” below means the exact equal-weight channel combination in the code, not a claim of calibrated luminance. Kernel and mixing weights are constructed in NumPy float64, then stored and executed as float32 buffers. They are not optimizer parameters. The frontend runs outside autocast. [Weight construction and input contract](../src/sparse_contrast/frontend.py#L23).

### 3.2 Spatial kernel, blur cascade, and padding

The blur is a depthwise 3×3 box filter:

$$
K=\frac19
\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}.
$$

For three-channel input $X=[R,G,B]$, define

$$
U_0=X,\qquad U_i=K*U_{i-1}\quad(i=1,\ldots,d),
$$

where convolution is independent in each channel, stride 1, with one zero-padded pixel on every edge on **each** application. Padding is not reflected, replicated, circular, or renormalized at the boundary. The generic contrast operator is

$$
D=X-\frac1d\sum_{i=1}^{d}U_i.
$$

Every actual study configuration uses **$d=1$**, so conceptually $D=X-K*X$. This value follows the inspected application default. Depth-3 examples from the upstream documentation do not change the executed experiment. For $d>1$, the source would average all intermediate blurred images, not merely subtract the final blurred image.

The implementation does not first materialize $D$ and then multiply a color matrix. It concatenates $[U_0,U_1,\ldots,U_d]$ along channels and applies one fixed 1×1 convolution containing the center, surround, and color weights. That arithmetic order matters for exact float32 cancellation. Algebraically equivalent rewrites can produce different tiny residuals, which matter when token support uses exact zero. [Executed convolutions](../src/sparse_contrast/frontend.py#L103).

At depth 1, the conceptual single-channel spatial contrast kernel is

$$
\delta-K=\frac19
\begin{bmatrix}-1&-1&-1\\-1&8&-1\\-1&-1&-1\end{bmatrix}.
$$

A positive value says the center exceeds its box-filtered neighborhood; a negative value says it is below that neighborhood. Both signs are retained.

### 3.3 The three native channel mappings

Let $D_R,D_G,D_B$ denote the conceptual channelwise center-minus-surround responses.

**Single color** has three outputs:

$$
C_{\mathrm{single}}=[D_R,D_G,D_B].
$$

The name does not mean that one RGB channel is selected. It means separate contrast for each color channel, with no cross-channel mixing at this stage.

**Grayscale contrast** has one output:

$$
C_{\mathrm{gray}}=\frac{D_R+D_G+D_B}{3}.
$$

For positive blur depth, this is equal-weight RGB averaging. The legacy source uses `[0.299, 0.587, 0.114]` only in its depth-zero black-and-white branch. Depth zero does not perform the center-surround contrast used here. Substituting those luminance weights into this experiment would change the method.

**Color opponency** has three outputs, in this order:

$$
\begin{aligned}
C_I &= (D_R+D_G+D_B)/3,\\
C_{RG} &= (D_R-D_G)/2,\\
C_{BY} &= D_B/3-D_R/6-D_G/6.
\end{aligned}
$$

Equivalently,

$$
C_{\mathrm{opp}}=AD,\qquad
A=\begin{bmatrix}
1/3&1/3&1/3\\
1/2&-1/2&0\\
-1/6&-1/6&1/3
\end{bmatrix}.
$$

The yellow component is the average of red and green, with the exact scaling shown. The scales are consequential: coefficient pruning ranks **unnormalized values across all channels**, so changing any row scale changes which information survives. The later per-channel standardization does not undo a different pruning decision. [Exact native mixing weights](../src/sparse_contrast/frontend.py#L23).

For the actual depth-1 concatenation `[R,G,B,blur_R,blur_G,blur_B]`, the three opponency rows are exactly the float32 casts of

```text
I : [ 1/3,  1/3, 1/3, -1/3, -1/3, -1/3]
RG: [ 1/2, -1/2,   0, -1/2,  1/2,    0]
BY: [-1/6, -1/6, 1/3,  1/6,  1/6, -1/3]
```

The grayscale row is the first row above. Single-color rows select one unblurred channel with weight +1 and the corresponding blurred channel with weight −1.

### 3.4 MNIST and exact zeros

The grayscale MNIST image is replicated to three identical input channels **before** the source-matched transform. The grayscale frontend then emits one native channel. It is not copied back to three channels. The initial projection therefore consumes four values per MNIST patch.

In exact arithmetic, applying RGB opponency to three identical channels makes the opponent responses vanish. Float32 convolution order can leave small cancellation residuals. The active study excludes MNIST single-color/opponency conditions rather than treating their channels as useful independent color signals. The active grayscale pipeline still uses the exact source-matched float32 calculation; it does not replace it with a hand-simplified one-channel filter or round small results to zero.

There is no adjustable “B” controlling this transform. In tensor notation $B$ is batch size. A legacy label such as “B=70” is not interpreted as 70% coefficient sparsity.

### 3.5 What information this transform changes

In a constant-valued channel with value $a$, the depth-1 contrast is zero in the interior, but it is $a/3$ along a non-corner boundary pixel and $5a/9$ at a corner, in exact arithmetic. Zero padding therefore creates responses at an image boundary even without an interior edge. The transform need not have dataset-wide zero mean.

Away from boundaries, its spatial-frequency response is

$$
H(\omega_r,\omega_c)=1-
\frac{(1+2\cos\omega_r)(1+2\cos\omega_c)}9.
$$

This suppresses slowly varying content and emphasizes local variation. It does not imply that every corruption is suppressed: fine-scale noise can also create large responses.

There is an important distinction between suppression and irreversible information loss. For a finite image with this zero boundary convention, the box-filter matrix is a nonnegative substochastic matrix with boundary leakage and spectral radius below one. Consequently $I-K$ is invertible in exact arithmetic. The opponency matrix also has full rank, with determinant $-1/6$. Thus unpruned single-color/opponency contrast should not be described as necessarily deleting all global brightness or color information. Recovering attenuated components can nevertheless be poorly conditioned and difficult for the chosen learned model. Grayscale's 3-to-1 color projection and the subsequent hard coefficient pruning are genuinely many-to-one. These are mathematical properties of the stated operator, not evidence that a trained network learns an inverse.

## 4. Coefficient sparsification is not a token budget

For each image independently, flatten the native contrast tensor in **channel, row, column** order. If there are $M=CHW$ native coefficients and requested percentage $p$, compute

$$
n_{\mathrm{remove}}=\left\lfloor\frac{pM}{100}\right\rfloor.
$$

Stable-sort the absolute magnitudes of all $M$ entries. Set the first $n_{\mathrm{remove}}$ ranked positions to the exact floating-point value zero, retaining every other signed value. Equal magnitudes are resolved by the original channel/raster order. For $p=0$, return the coefficients directly without sorting. The implementation uses a full stable `torch.argsort`, clones the flat tensor, and scatters zeros into the selected positions. [Sparsifier](../src/sparse_contrast/frontend.py#L122).

This is:

- Per image, not per minibatch, class, or dataset.
- Across **all native channels together**, not a separate percentile in each channel.
- Applied before normalization, so the ranking is in native contrast units.
- A count of ranked positions, including positions already equal to zero.
- Not a cutoff that removes every value tied with the last selected magnitude.

For example, `[0, -0.05, +0.05, 0.20]` at 50% sparsity becomes `[0, 0, +0.05, 0.20]`: the two equal-magnitude entries are distinguished by position. At 25%, this example is unchanged because the sole selected entry was already zero.

The fraction of zeros after pruning can exceed $p$. If an image already has more zeros than the requested removal count, multiple requested percentages can yield exactly the same tensor. The percentage does **not** mean “remove $p$ percent of the currently nonzero coefficients.”

The legacy upstream percentage threshold was different: it ranked across the entire incoming minibatch, used a fractional parameter, operated after its legacy channel adapter, and removed all magnitudes at or below the selected cutoff. That code is not used for the new study's sparsifier. [Legacy threshold implementation](../experiments/training_sparse_testing_sparse/provenance/original_frontend.py#L188).

An especially relevant consequence for opponency is that intensity, red–green, and blue–yellow have different native scales. A global magnitude ranking can remove many small opponent-channel values before removing similarly informative but larger intensity values. This follows from the specified ranking rule; the study does not equalize channel variances before ranking.

## 5. Native support and frozen normalization

### 5.1 A patch is retained if any native coefficient is nonzero

For a non-overlapping 2×2 patch $j$, define

$$
S_j=\mathbf{1}\!\left[
\max_{c,\delta r,\delta c}|(C_p)_{c,2r_j+\delta r,2c_j+\delta c}|>0
\right].
$$

The maximum includes every native channel and all four spatial positions. The primary epsilon is exactly **0.0**, not a small numerical tolerance. One surviving coefficient retains the whole patch vector. A patch is removed only if every coefficient in it is exactly zero. [Support definition](../src/sparse_contrast/frontend.py#L156).

There is no second patch-ranking stage, target token count, learned router, coverage quota, minimum number of image tokens, or top-K patch selector. Empty images are allowed. A 60% coefficient target is therefore not a promise of 60% fewer tokens.

An illustrative independence calculation makes the distinction concrete: if each coefficient independently becomes zero with probability $q$, a $C$-channel 2×2 patch would be empty with probability $q^{4C}$. At $q=0.8$, this is about 41% for one channel and 6.9% for three channels. Actual image coefficients are correlated, so these figures are not estimates of this dataset's observed token counts; they explain why coefficient and patch sparsity are different quantities.

### 5.2 The statistics are fitted once on dense native contrast

For every retained dataset/contrast representation, the statistics fitter visits the entire permitted clean training partition at zero imposed sparsity and without random augmentation. It includes natural zeros, every image, every native pixel, and the unequal final batch.

For each channel, over $K=N_{\mathrm{train}}HW$ values,

$$
\mu_c=\frac1K\sum_i(C_0)_{i,c},\qquad
\sigma_c=\sqrt{\frac1K\sum_i((C_0)_{i,c}-\mu_c)^2}.
$$

This is population standard deviation, with correction 0. It is not the mean of per-image or per-batch standard deviations. The frontend runs in float32. Its outputs are converted to CPU float64 for within-chunk population moments and ordered Chan merging:

$$
\begin{aligned}
\delta&=\mu_B-\mu_A,\\
\mu_{A\cup B}&=\mu_A+\delta\frac{n_B}{n_A+n_B},\\
M_{2,A\cup B}&=M_{2,A}+M_{2,B}+\delta^2\frac{n_A n_B}{n_A+n_B}.
\end{aligned}
$$

The fitter records sample IDs, split identity, counts, reduction order, frontend/source identity, and JSON/NPZ hashes. A second full pass checks the actual float32 standardized dense training representation. [Moment accumulator](../src/sparse_contrast/normalization.py#L33); [fitting procedure](../experiments/training_sparse_testing_sparse/compute_normalization_stats.py#L33).

The fixed guard is

$$
\sigma_{c,\mathrm{eff}}=
\begin{cases}
1,&\sigma_c<10^{-6},\\
\sigma_c,&\text{otherwise}.
\end{cases}
$$

A guarded channel is retained, with its measured sigma recorded. It is not divided by a tiny number, deleted, or used to justify changing the support epsilon. No active normalization channel in the final retained scope triggered this guard.

The fitted vectors below are rounded for readability; the linked artifacts hold full precision and identities.

| Training representation | Mean vector | Population standard-deviation vector | Training pixels per channel |
|---|---|---|---:|
| [MNIST grayscale](../experiments/training_sparse_testing_sparse/normalization/mnist_grayscale_heldout_val.json) | `[0.00000850288]` | `[0.108462278]` | 43,120,000 |
| [CIFAR-10 grayscale](../experiments/training_sparse_testing_sparse/normalization/cifar10_grayscale_heldout_val.json) | `[0.020461346]` | `[0.083631396]` | 46,080,000 |
| [CIFAR-10 single color](../experiments/training_sparse_testing_sparse/normalization/cifar10_single_color_heldout_val.json) | `[0.020772852, 0.021002600, 0.019608619]` | `[0.085561938, 0.085474259, 0.083637526]` | 46,080,000 |
| [CIFAR-10 opponency](../experiments/training_sparse_testing_sparse/normalization/cifar10_color_opponency_heldout_val.json) | `[0.020461346, -0.000114874, -0.000426370]` | `[0.083631396, 0.010468590, 0.008367761]` | 46,080,000 |

These vectors are shared across architectures, seeds, training sparsities, dense controls, and all evaluation cells. They are not updated after pruning or on validation, clean test, corruptions, or the inference-only sparsity sweep. Retained patches and corrupted images need not have zero mean or unit variance after using these fixed statistics.

### 5.3 Native zeros, affine offsets, and absent features are different

The native coefficients are standardized as

$$
Z_c=\frac{(C_p)_c-\mu_c}{\sigma_{c,\mathrm{eff}}}.
$$

A coefficient set to zero therefore becomes $-\mu_c/\sigma_{c,\mathrm{eff}}$. That is expected. It does not change its native support and does not imply that new image evidence has appeared.

For a one-channel retained patch with $C_p=[0.10,0,0,-0.05]$, $\mu=0.02$, and $\sigma=0.08$, the actual patch vector is

$$
Z=[1,-0.25,-0.25,-0.875].
$$

The two zeroed coefficients are **not** reset to zero after normalization. An entirely zero native patch would standardize to four copies of −0.25, but its saved support is false and compact execution never projects it. The implementation never recomputes support from standardized values.

Three distinct kinds of zeros must therefore stay separate:

1. Native zeros in $C_p$, used to define coefficient sparsity and support.
2. Values inside retained standardized patch vectors, including affine offsets at zeroed coefficients.
3. Zero feature vectors used for masked sequence slots, absent Swin children, or structural padding.

Only the third category is repeatedly suppressed in the dense-masked reference to prevent learned biases and normalization from resurrecting absent locations. Re-zeroing category 2 would implement a different masked-normalization method. [Normalization API](../src/sparse_contrast/normalization.py#L98); [executed model-input path](../src/sparse_contrast/pipeline.py#L37).

The raw baseline uses the prescribed fixed published recipe constants instead of contrast statistics: MNIST mean/std `(0.1307)/(0.3081)` and CIFAR-10 means `(0.4914,0.4822,0.4465)` with stds `(0.2023,0.1994,0.2010)`. These are applied to `[0,1]` raw values and are not described as a population-statistic fit performed by this experiment. [Raw constants](../src/sparse_contrast/pipeline.py#L8).

## 6. Patch layout and the three execution modes

`patchify` maps `[B,C,H,W]` to `[B,N,4C]`. Patch positions are in raster order, first patch row then patch column. Within a vector the order is `(channel, local_row, local_column)`. For channel $c$, the four entries are top-left, top-right, bottom-left, bottom-right. The linear patch projection is equivalent to a stride-2, kernel-2 convolution with appropriately reshaped weights. [Layout](../src/sparse_contrast/frontend.py#L141).

| Execution | Spatial support | Physical transformer input | Purpose |
|---|---|---|---|
| `dense` | All original patches active | Full grid/sequence | Raw baseline and 0%-contrast dense training controls |
| `compact` | Saved native support | Retained rows, with ViT batch padding or Swin exact window-length groups | Actual sparse execution |
| `dense_masked` | Same saved support as compact | Full grid/sequence with inactive keys/states suppressed | Semantically matched correctness and timing control for the same checkpoint |

Passing a support mask to `dense` does not turn it into the sparse reference: the backbone deliberately treats all locations as active in that mode. The distinction is explicit in [input validation](../src/sparse_contrast/models.py#L99).

The frozen input standardization is distinct from learned feature LayerNorm. For a feature vector $z\in\mathbb R^D$, LayerNorm uses

$$
\operatorname{LN}(z)=\gamma\odot\frac{z-\bar z}{\sqrt{D^{-1}\sum_i(z_i-\bar z)^2+\epsilon}}+\beta.
$$

Its mean and variance are over that vector's feature dimension, not over images or the dataset. The affine parameters $\gamma,\beta$ are learned. Patch embeddings, QKV projections, attention output projections, both MLP linear layers, and classifier heads have learned biases; Swin's merge reduction is the stated bias-free exception. Thus an all-zero feature vector is not guaranteed to stay zero through a learned layer. Explicit support handling, rather than numerical zero alone, is what preserves absence. [Learned layers](../src/sparse_contrast/models.py#L24).

`dense_masked` is not a separately trained model and has no separate weights. It performs the same sparse computation using full-size arrays and masks. Ordinary dense attention over zero-valued or affine-standardized sparse patches is a different function: those locations can receive biases and positional embeddings, contribute keys/values, and affect normalization/pooling. It is therefore not the appropriate parity target.

## 7. ViT-Small, layer by layer

### 7.1 Dimensions and learned state

The implemented small-image ViT has:

- A linear $4C\rightarrow384$ patch embedding with bias.
- One learned class token of width 384.
- Learned absolute positional embeddings `[1,N+1,384]`, with the class position at index 0.
- Twelve pre-norm transformer blocks, six heads, head dimension 64.
- MLP width 1,536, GELU, and output width 384.
- LayerNorm epsilon $10^{-6}$.
- A final LayerNorm and a biased $384\rightarrow10$ head applied to the class token.
- No attention dropout, MLP dropout, token dropout, or pretrained weights.

The dense sequence lengths are 197 for MNIST and 257 for CIFAR-10, including the class token. Input channel count changes the first projection's input width; the transformer widths/depths remain fixed. [ViT construction](../src/sparse_contrast/models.py#L118).

### 7.2 Packing preserves original positions

Let image $b$ retain $n_b$ patches and $L=\max_b n_b$. Compact execution:

1. Computes the per-image lengths and extracts `(image_id, original_patch_position)` from the Boolean support.
2. Uses the cumulative support count to assign a compact slot inside each image.
3. Gathers only retained patch vectors and applies the learned patch projection to those rows.
4. Adds the positional embedding indexed by **original patch position plus one**.
5. Scatters the resulting features into `[B,L,384]`, leaving other slots zero and invalid.
6. Prepends the always-valid learned class token plus its own position embedding.

For original retained positions `[1,4,9]`, the model uses positional rows `[2,5,10]`, not `[1,2,3]`. Compaction changes storage, not spatial identity. [Packing and position indexing](../src/sparse_contrast/models.py#L134).

This implementation is padded by the maximum retained length in the current batch. The patch projection operates on exactly $\sum_b n_b$ retained rows, but attention, output projections, LayerNorms, and MLPs execute at sequence length $L+1$ for every image. Padding is explicitly counted. This is not fully ragged per-image ViT execution, and changing batch membership to obtain convenient lengths is not part of the method.

### 7.3 Attention, masks, and residual blocks

For an input sequence $X$ of width $D=384$,

$$
[Q,K,V]=\operatorname{Linear}_{D\rightarrow3D}(\operatorname{LN}_1(X)),
$$

and each head computes

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{64}}+M\right),
\qquad Y=\operatorname{Proj}(AV).
$$

$M$ is negative infinity for an invalid **key** and zero for a valid key. The class token is always a valid key, so softmax never sees a completely masked row merely because an image has no image tokens.

For each residual branch, stochastic depth is applied per original image during training. After each residual addition, invalid states are explicitly multiplied by the validity mask:

$$
\begin{aligned}
X'&=(X+\operatorname{DropPath}(Y))\odot S,\\
X''&=(X'+\operatorname{DropPath}(\operatorname{MLP}(\operatorname{LN}_2(X'))))\odot S.
\end{aligned}
$$

The validity mask includes the class token. Invalid queries may still incur arithmetic in padded/dense-masked execution, but their results are discarded. LayerNorm or projection biases at those slots cannot make them available as future keys. [Attention and blocks](../src/sparse_contrast/models.py#L54).

The classifier reads the final normalized class token. It does not average image tokens. An entirely empty image takes a **class-token-only semantic path**; an entirely empty batch has executed sequence length 1. In a mixed batch, an empty image can occupy padded physical slots, but all of its image slots remain masked. No arbitrary image patch is retained as a fallback.

## 8. Swin-Tiny, layer by layer

### 8.1 Native grids and stage settings

The stem is a biased $4C\rightarrow96$ linear projection followed by LayerNorm with epsilon $10^{-5}$. Unlike the ViT stem, it includes that patch normalization. There is no class token and no learned absolute positional embedding.

| Stage | Width | Blocks | Heads | Head width | MNIST grid / window / windows per image | CIFAR-10 grid / window / windows per image |
|---|---:|---:|---:|---:|---|---|
| 1 | 96 | 2 | 3 | 32 | $14\times14$ / $7\times7$ / 4 | $16\times16$ / $4\times4$ / 16 |
| 2 | 192 | 2 | 6 | 32 | $7\times7$ / $7\times7$ / 1 | $8\times8$ / $4\times4$ / 4 |
| 3 | 384 | 6 | 12 | 32 | $4\times4$ / $4\times4$ / 1 | $4\times4$ / $4\times4$ / 1 |
| 4 | 768 | 2 | 24 | 32 | $2\times2$ / $2\times2$ / 1 | $2\times2$ / $2\times2$ / 1 |

Each block uses an MLP expansion of four, GELU, two pre-norm residual branches, and LayerNorm epsilon $10^{-5}$. Patch merging doubles width between stages. The final head is biased $768\rightarrow10$. These are explicitly small-image adaptations of Swin-Tiny, not claims to reproduce the standard ImageNet input/window configuration. [Swin construction](../src/sparse_contrast/models.py#L345).

### 8.2 Sparse storage carries coordinates through the network

Compact Swin initially stores active features as `[T,96]`, where $T=\sum_b n_b$. Every row carries a flat integer identifier

$$
f=bG^2+rG+c,
$$

encoding original image ID and the current stage's original grid coordinates. The grid extent remains fixed even when many positions are absent. Tokens are not reshaped into a smaller rectangle, and window attention is not replaced by global attention.

There is no batch-max sequence padding for compact Swin's attention/MLP rows. QKV, output projection, and MLP operate on the stored active rows. There is still feature-space zero filling for missing children during merging and a tiny final scatter for pooling. “No attention padding” does not mean “no allocation, indexing, synchronization, or grouping overhead.”

### 8.3 Shifted windows and original offsets

A block alternates unshifted and shifted windows. The intended shift is $s=\lfloor W/2\rfloor$ in odd-indexed blocks, but it is disabled when $G\leq W$.

Therefore only stage 1 shifts for MNIST. CIFAR-10 shifts in stages 1 and 2. Both datasets' later stages fit into one window, so nominal alternation there does not create a shift.

For original stage coordinate $(r,c)$, the rolled coordinates are

$$
\tilde r=(r-s)\bmod G,\qquad \tilde c=(c-s)\bmod G.
$$

The window ID and within-window raster position are

$$
w=\left\lfloor\tilde r/W\right\rfloor(G/W)+\left\lfloor\tilde c/W\right\rfloor,
\qquad
\ell=(\tilde r\bmod W)W+(\tilde c\bmod W).
$$

Image ID is included in the global grouping key, so separate images cannot share attention. Original positions survive all packing and grouping. [Window geometry](../src/sparse_contrast/models.py#L178).

Each block has a learned relative-position bias table of shape `[(2W−1)^2, heads]`. For a query/key pair, its index is computed from the **original within-window coordinates**, not from indices 0, 1, 2 in the shortened list. The bias is added to scaled dot-product scores.

Cyclic shifting needs a separate boundary restriction to prevent wrapped edge regions from behaving like ordinary neighbors. The implementation partitions the shifted grid using the standard three slices along each dimension and adds **−100** to scores joining different shift regions. This is the finite penalty used by the chosen dense reference, not mathematical negative infinity. Missing keys in the dense-masked sparse reference receive negative infinity. The two masks have different purposes. [Relative-position and shift masks](../src/sparse_contrast/models.py#L196).

### 8.4 Exact retained-length grouping

For every block, compact execution computes active window IDs, stable-sorts rows by window, counts retained tokens in each window, and obtains the set of nonzero window lengths.

For each distinct length $k$, it batches together all windows containing exactly $k$ active tokens. Attention runs on tensors shaped approximately `[number_of_windows_with_length_k, heads, k, k]`. It gathers the corresponding original local positions and shift-region labels, applies attention, and scatters outputs back to the active-row order. Empty windows are skipped.

If $k_w$ is the number of active tokens in window $w$, the reported number of attention pairs is $\sum_w k_w^2$, before multiplying by heads. The dense implementation computes full-window pairs. Compact QKV and MLP row counts depend on active tokens, not padded full windows. [Compact block](../src/sparse_contrast/models.py#L243).

The implementation vectorizes token work, but it contains a Python loop over **distinct window lengths**. Obtaining those lengths through `.tolist()` requires making the dynamic group information available to the host. Stable sorting, counting, extracting indices, many small attention calls, and scatter operations are real online costs. This detail is central when interpreting latency: useful arithmetic can shrink while dispatch and indexing overhead rises. The measured evidence is discussed separately in [performance_analysis.md](performance_analysis.md).

### 8.5 Dense-masked Swin and empty windows

The dense-masked reference uses full stage arrays `[B,G^2,D]`, orders them into the same rolled windows, and applies the same relative bias and shift restrictions. It excludes inactive keys and multiplies states by support after both residual branches.

An entirely empty dense window would otherwise produce a softmax over all negative infinities. For such a window only, the attention calculation permits a dummy diagonal. Its resulting states are immediately suppressed by the false support. This is an internal numerical device in the dense reference, not a retained image token. Compact execution simply skips that empty window. [Dense block](../src/sparse_contrast/models.py#L281).

### 8.6 Patch merging: OR support and zero children

A stage parent exists if any of its four original children is active:

$$
S^{\mathrm{parent}}_{r,c}=
S_{2r,2c}\lor S_{2r+1,2c}\lor S_{2r,2c+1}\lor S_{2r+1,2c+1}.
$$

For each existing parent, concatenate child feature vectors in the exact reference order

```text
top-left, bottom-left, top-right, bottom-right
```

and insert a **zero feature vector** for a missing child. The merge is

$$
x_{\mathrm{parent}}=W_{\mathrm{merge}}\,
\operatorname{LN}_{4D}([x_{TL},x_{BL},x_{TR},x_{BR}]),
$$

with output width $2D$, LayerNorm epsilon $10^{-5}$, and a bias-free reduction. These feature zeros are not native pixels and do not receive the input mean/std transform. A present parent may legitimately receive nonzero values after its merge normalization; an absent parent must not reappear. Compact merging never creates it, and dense-masked merging masks it back to zero. [Merging](../src/sparse_contrast/models.py#L312).

MNIST's $7\times7$ stage requires a conceptual zero-padded eighth row/column before merging into $4\times4$. These positions are structurally absent. Compact merging handles them through missing child slots without inventing image evidence; dense merging explicitly pads both features and the Boolean mask. Thus the native stage grids are $14,7,4,2$.

There is no feature-magnitude pruning after merging. Parent support depends only on descendant input support. Consequently support can become effectively dense at later stages even when many initial patches were absent. Under an illustrative independent-child occupancy model, retained fraction $r$ becomes $1-(1-r)^4$ after one merge; real support is spatially correlated, so measured stage counts must be used for actual conclusions.

### 8.7 Final pooling and all-empty images

After stage 4, a final LayerNorm acts on active features. The model sums active spatial features for each image and divides by its active count. Compact execution uses an at-most-2×2 final scatter into zeros, avoiding duplicate-index atomic summation. Dense-masked execution multiplies by support before summing.

The denominator is clamped to at least one. An image with no active features therefore has a zero pooled vector and logits equal to the learned classifier bias. It does not receive a fake spatial token, a random surviving patch, or a dropped prediction. [Pooling](../src/sparse_contrast/models.py#L404).

## 9. Why compact and dense-masked should agree

In exact arithmetic and evaluation mode, the retained-state computation is the same in the two paths:

1. They receive identical standardized patch values and the identical pre-normalization support.
2. Patch projection, LayerNorm, and MLP act independently on each valid row.
3. Attention uses the same retained keys/values, original positional information, and permitted spatial relationships.
4. Inactive query states are discarded and never become future keys.
5. Swin's parent IDs, child ordering, inserted zeros, and support OR are identical.
6. Readout uses the same class token or active-feature pooling.

An induction over blocks and merges therefore gives matching valid hidden states and logits, up to ordering-dependent floating-point error. Learned biases do not invalidate that argument because absent states remain absent or are masked after every relevant operation.

The relevant pinned library versions are PyTorch `2.9.1+cu126`, torchvision `0.24.1+cu126`, and timm `1.0.22`. The all-active model is also compared against pinned third-party dense references. `dense_reference` constructs matching timm ViT or torchvision Swin configurations, maps every learned parameter, reshapes only the equivalent patch-projection weights, and loads with `strict=True`. There is no permissive loading that leaves unmatched random parameters. Swin's grayscale stem is explicitly replaced to match native channel count before strict loading. [Dense reference construction](../src/sparse_contrast/models.py#L433).

The verification suite covers all-active outputs and gradients, variable support lengths, singleton and empty inputs, nonzero learned biases, original-coordinate preservation, shifted boundaries, odd-grid merging, and compact-versus-dense-masked parameter gradients. Its scope is stronger than “an import succeeded,” but it is not a proof of identical floating-point outputs for every checkpoint/input/device.

### Floating-point comparison and the explicit recovery amendment

Matrix sizes, grouping, padding, and batch size can change float32 reduction order and selected kernels. This can make compact and dense-masked logits slightly different even when the real-valued operations agree. Deterministic execution means repeatability under the recorded stack and computation path; it does not imply bitwise equality across different computation paths or GPU types.

The original timing guard uses `rtol=1e-4, atol=2e-5`. A diagnosed MNIST raw-pixel inference case at batch 8 exceeded that guard for one of 80 logits: the absolute compact/dense-masked difference was approximately $3.55\times10^{-5}$. A copied-backbone float64 calculation agreed to approximately $4.62\times10^{-14}$, while the two original float32 paths differed from their respective float64 references by approximately $1.61\times10^{-5}$ and $4.22\times10^{-5}$. This is evidence for rounding in that diagnosed case, not permission to ignore arbitrary discrepancies.

The separately versioned amendment retains the original strict check first. If it fails, it requires identical float32 prepared patches and support, copies the stored float32 backbone weights into an independent float64 reference, and promotes those **already prepared float32 patches** to float64. It does not recompute the frontend, ranks, or support in higher precision. It then checks:

- Float64 compact versus dense-masked: `rtol=1e-10, atol=1e-11`.
- Each original float32 output against its corresponding float64 reference: `rtol=1e-4, atol=1e-4`.
- Finite outputs, unchanged production dtype/state, and cleanup of the temporary reference before primary warmup.

The scope is the explicitly diagnosed 16 MNIST raw-pixel, seed-1, batch-8 workers at inference sparsities 0–70%, with compact and dense-masked timing. Other workers retain their original path. Reference tensors, masks, IDs, logits, policy/source identities, and hashes are saved. The temporary float64 model is destroyed, RNG state restored, and unused CUDA cache cleared before the normal warmup/peak-reset sequence. The controller verifies these amendments at worker acceptance and again for all 16 workers before declaring final completion. No weights, support rules, measured precision, measurement count, or timed model operation is changed. [Validation policy and implementation](../provenance/sources/3c965bf1a2c9f3f11bb14c0d6ad3354a2266e7d0763b2146b5a8fbc69d314769/validation.py#L119).

## 10. Training recipe actually used

The completed larger-batch study uses one fixed recipe per dataset/backbone, shared across that group's raw, dense-contrast, and compact-contrast conditions. The later user instruction to rerun with larger useful batches superseded the original task's batch-8 starting recipe. There is no gradient accumulation: physical and nominal effective batch sizes are equal.

| Group | Train / evaluation batch | Epochs | Warmup epochs | Peak learning rate | Updates per epoch | Actual final batch |
|---|---:|---:|---:|---:|---:|---:|
| [MNIST / ViT-Small](../experiments/training_sparse_testing_sparse/configs/resolved/62004fa6b787a704ff5ee13663dbbc1aeaa90c7b0b292b27c4d9b1a1f63a9d13.json) | 256 / 256 | 50 | 5 | 0.0004242640687 | 215 | 216 |
| [MNIST / Swin-Tiny](../experiments/training_sparse_testing_sparse/configs/resolved/2862442744248fb27f135fca8176a4329e4389cf2b7a4b85035896938cfee3d1.json) | 1,024 / 1,024 | 50 | 5 | 0.0008485281374 | 54 | 728 |
| [CIFAR-10 / ViT-Small](../experiments/training_sparse_testing_sparse/configs/resolved/7c4187c643b7e210dc769d4c2b2adebfd5bcb5c678051889a482ac9b2ab223ad.json) | 128 / 128 | 200 | 10 | 0.0003 | 352 | 72 |
| [CIFAR-10 / Swin-Tiny](../experiments/training_sparse_testing_sparse/configs/resolved/47313a6e6a1f3eeb5173ac2dbf32754b3d9bba697221c794d03281c0c1c5268e.json) | 512 / 512 | 200 | 10 | 0.0006 | 88 | 456 |

The learning-rate choice is $3\times10^{-4}\sqrt{B/128}$. It is a logged study recipe, not a claim that this scaling is optimal or equivalent to the earlier batch-8 optimization trajectory. Changing batch changes the number of parameter updates, gradient noise, and potentially convergence. The evaluation batch size in this table is distinct from the latency batch sizes 1 and 8.

All groups use AdamW with betas `(0.9,0.999)`, weight decay 0.05, gradient-norm clipping at 1.0, no label smoothing, and ordinary mean cross-entropy. Biases, LayerNorm parameters, positional parameters, class tokens, and parameters with fewer than two dimensions are excluded from weight decay. No Mixup, CutMix, distillation, contrastive term, or reconstruction term is present. [Optimizer and schedule](../src/sparse_contrast/train.py#L118).

For zero-based update $t$, warmup length $T_w$, and total scheduled updates $T$, the implemented schedule is

$$
\eta_t=\eta_{\max}\frac{t+1}{T_w}\quad(t<T_w),
$$

then

$$
\eta_t=\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}2
\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right],
\qquad \eta_{\min}=10^{-6}.
$$

The last executed update is $t=T-1$, so the formula approaches the minimum rather than executing an additional update at $t=T$. The schedule derives its update counts from the actual loader length. The final smaller batch is retained; it is not padded with duplicate samples or dropped.

### Initialization and stochastic depth

Every main model starts from a fresh initialization. `initialize_matched` seeds each parameter by a hash of its parameter name and training seed. Same-shaped backbone parameters therefore start identically across matched representations even when their input stems have different channel counts. LayerNorm scales begin at one and biases at zero; multidimensional weights and positional/class parameters use truncated normal with standard deviation 0.02. This training initializer overrides constructor-only defaults, including the ViT class token's constructor initialization. [Actual initialization](../src/sparse_contrast/common.py#L67).

Stochastic-depth probabilities increase linearly from 0 to 0.1 across the 12 blocks of either architecture. Each residual branch draws one Bernoulli mask per **original image**, scaled by the inverse survival probability. In packed Swin, all rows from the same image use that image's mask. Empty images still participate in the original batch-size mask draw, avoiding a change in mask indexing caused by compaction. The two residual branches have separate draws. Evaluation disables stochastic depth. [Per-image drop path](../src/sparse_contrast/models.py#L34).

Matching parameter initialization does not claim that all training trajectories or all random draws are bitwise identical across different channel counts or hardware. Data order and augmentation are keyed separately from model RNG consumption.

### Data and augmentation

The primary protocol is `heldout_val`: 55,000 MNIST or 45,000 CIFAR-10 training images, plus a fixed 5,000-image held-out validation set with 500 examples per class. Split seed is 2026, shared across methods and training seeds. Normalization uses only the training partition.

MNIST training has no random augmentation. CIFAR-10 uses four-pixel constant-zero padding, a uniform random 32×32 crop, and random horizontal flip, all on raw `uint8` pixels **before** contrast extraction. Decisions are keyed by seed, epoch, and stable sample ID. A separate keyed epoch permutation controls sample order. Workers are configured to zero in the executed recipe; the loader does not drop the last batch. [Augmentation and sampling](../src/sparse_contrast/data.py#L218); [loader settings](../src/sparse_contrast/train.py#L33).

Full clean held-out validation runs after every epoch in evaluation mode, with no random augmentation. Primary evaluation uses the **final epoch** checkpoint. A best-validation checkpoint is diagnostic and does not replace the primary checkpoint selectively. Training stores the model, optimizer, schedule step, RNG, next epoch, and sampler boundary; resume is at an explicitly recorded epoch boundary. [Training and checkpoint loop](../src/sparse_contrast/train.py#L142).

All reported learned computation uses FP32. The frontend, ranking, support, and normalization also remain FP32. Deterministic algorithms are enabled, cuDNN benchmarking is disabled, and TF32 is disabled. These settings support reproducibility within the recorded stack; they do not establish cross-device bitwise identity. [Determinism settings](../src/sparse_contrast/common.py#L49).

### A limitation of the optional refit path

No full-training-set refit is part of the reported campaign. The source contains both `full_refit` and `full_train_refit` spellings in different paths. In particular, the trainer's validation suppression checks `full_train_refit`, while fitting/data use canonical `full_refit`. This optional path was not established end to end and must not be presented as an executed, validated protocol. It does not affect `heldout_val`, which all reported models use. A future refit requires an explicit configuration/path repair and separate validation; this documentation does not change running scientific code.

## 11. The later inference-only study

The follow-up reuses **60 exact zero-imposed-sparsity checkpoints** from the original campaign:

- 24 contrast models trained with ordinary dense execution at 0%.
- 24 contrast models trained compact at 0%.
- 12 ordinary dense raw baselines.

The first two families have different training semantics even though both labels contain “0%.” Compact 0% can discard natural-zero patches; dense 0% cannot. Their weights must not be pooled or mislabeled as one dense-trained family.

Each source checkpoint is evaluated with compact inference at sparsities 0%, 10%, …, 90%. Contrast checkpoints prune native contrast; raw checkpoints prune raw pixel magnitudes before raw mean/std normalization. The raw-pixel experiment is a separately identified intervention, not a contrast frontend. The raw checkpoints also retain 12 unchanged dense reference cases. This yields 600 compact inference cases plus 12 dense raw references: **612 cases**. No checkpoint is retrained or adapted to its test sparsity.

The adapter adds no parameters, buffers, or state-dictionary keys. `model.config` remains the original training config. A distinct immutable inference specification contains `inference_execution`, `inference_sparsity_percent`, and `sparsification_domain`. The caller validates and loads the original checkpoint strictly, while the separate case identity binds the inference intervention, original training/config/checkpoint identities, source, and normalization hash. [Adapter construction](../src/dense_sparse/adapter.py#L16).

The same fixed normalization fitted for training is reused across the entire inference sweep. Model inference explicitly resolves the requested execution rather than accidentally defaulting to the backbone's original training execution. [Adapter forward path](../src/dense_sparse/adapter.py#L70).

A dense-trained checkpoint tested compact at 0% is **not** generally the original dense evaluation: naturally empty patches now disappear. This anchor therefore needs its own evaluation. Likewise, a compact0-trained checkpoint tested at 60% is not the original separately trained compact60 model. One measures a test-time intervention on fixed weights; the other measures learning with that sparsity throughout training.

The requested comparison plots intentionally show only:

- Solid curves: separately trained compact models tested at the same original sparsity.
- Dashed curves: dense0-trained checkpoints tested compact at the new sparsity.
- A black horizontal original raw baseline, with the same checkpoint/panel remeasurement documented separately.

The full follow-up retains the compact0-trained and raw-pixel-sweep families, but those are not silently substituted into the requested dashed contrast curves.

## 12. Evaluation and efficiency endpoints

Accuracy evaluation visits every image, including empty representations. Clean test has 10,000 images per case. MNIST-C uses its 15 released corruptions at one fixed severity each. CIFAR-10-C uses 15 corruptions at all five released severities, 10,000 examples per cell. No clean counterpart supplies the support for a corrupted input, and no corruption-specific threshold, adaptation, or test-time augmentation is used.

The reported `mCE_raw` is an **unnormalized macro corruption error**:

$$
\mathrm{mCE}_{\mathrm{raw}}=\frac1{15}\sum_{c=1}^{15}
\left(\frac1{S_c}\sum_{s=1}^{S_c}\mathrm{error}_{c,s}\right),
$$

with $S_c=1$ for MNIST-C and 5 for CIFAR-10-C. It is not divided by an AlexNet reference error. Cross-entropy sums and correct/total counts are accumulated with sample weighting; unequal batches do not receive equal weight merely because they are separate batches. [Metrics](../src/sparse_contrast/metrics.py#L4).

For latency, primary comparisons use identical sample panels, seeds, batch sizes, precision, GPU model, CPU model, and thread/backend class. Measurements preserve physical device/session identities. The final comparison hardware is NVIDIA RTX A6000 with AMD EPYC 7413, four PyTorch intra-op threads, and the recorded common stack. Older timing on another GPU class is not pooled into that comparison.

The headline clean scope `gpu_raw_to_logits` starts with decoded raw `uint8` images already on the GPU and ends with GPU logits. It includes conversion to float, the online fixed frontend, stable ranking, native support, normalization, patch layout, gather/grouping, all transformer work, and the head. It excludes host-to-device transfer and argmax transfer to CPU. It does **not** start from cached contrast or cached tokens. Synchronized wall time includes host dispatch and dynamic-shape handling; separate CUDA-event times are also saved. [Measurement boundaries](../src/sparse_contrast/benchmark.py#L421).

Other saved scopes are host raw input to GPU logits; loader to CPU prediction; and a clearly labeled cached-input model-only diagnostic. The cached diagnostic excludes frontend/preparation and cannot replace the online result.

Timing uses a predetermined class-balanced panel of 1,600 original test IDs, with a private benchmark RNG seed of 20260930. The measured set takes the first `200 × batch_size` interleaved panel entries and permutes their order, giving 200 measured images at batch 1 and 1,600 at batch 8. Warmup uses its own deterministic permutation of the same predetermined panel. Methods and seeds use matching IDs and batch order; the panel is not selected by confidence, retained-token count, or observed latency. Matching a corrupted image's original ID does not give the model access to its clean counterpart. [Batch membership construction](../src/sparse_contrast/benchmark.py#L113).

Each cell/scope uses 50 warmup batches and 200 serial timed batches, followed by a separate sustained-throughput pass. Cache cleanup and peak reset occur outside the measured intervals, before the scope's warmup/reset sequence. Profiling and token diagnostics run outside primary timing. Inclusive stage-profiler intervals overlap and are not additive end-to-end measurements. [Timing and memory policy](../src/sparse_contrast/benchmark.py#L295).

The requested latency curves use the mean and sample standard deviation of the **three per-seed medians**, not a median of pooled seeds. Bands show between-seed sample SD, not a confidence interval. Points missing any of seeds 0, 1, and 2 remain missing. They are not extrapolated or connected through unmeasured levels. Batch-8 latency is milliseconds per batch of eight, not automatically milliseconds per image. Companion CSVs and provenance preserve the exact plotted definition and coverage.

## 13. What this method can and cannot establish

The comparisons separate several questions:

| Comparison | What changes | What it can inform |
|---|---|---|
| Raw dense versus contrast dense 0% | Input representation, input normalization, sometimes native channel count | Effect of replacing raw input without natural-token removal |
| Contrast dense 0% versus compact-trained 0% | Presence of naturally empty tokens during training/inference | Consequences of sparse support semantics at zero imposed pruning |
| Compact training at 0/20/40/60/80% | Input coefficient sparsity and resulting support throughout learning | Whether models can adapt to that sparse representation |
| Same compact checkpoint: compact versus dense-masked | Physical execution, with the same sparse semantics | Whether this implementation saves runtime/memory for that function |
| Fixed dense0 checkpoint: compact inference 0..90% | Test-time coefficient pruning and missing-token semantics | Sensitivity of dense-trained weights to the intervention |
| Fixed compact0 checkpoint: compact inference 0..90% | Test-time pruning beyond its training setting | Sensitivity of an already support-aware checkpoint |

None of these by itself shows that every high-pass representation, every sparse transformer kernel, or every training recipe must behave similarly. This is a fixed representation and a concrete implementation at native 28×28/32×32 resolution.

Several mechanisms are visible in the method but need measured evidence before being assigned causal responsibility for an accuracy or latency result:

- The transform changes the balance of smooth content, boundaries, and fine-scale noise.
- Grayscale removes color distinctions; global native ranking can preferentially remove smaller opponent-channel values.
- Fixed 0%-training normalization does not re-center each sparsified/corrupted input distribution.
- Coefficient zeros do not imply empty 2×2 patches, especially with three channels.
- ViT executes to the batch's maximum retained length, which can reduce arithmetic savings from variable support.
- Swin's OR merging can restore high occupancy at later stages, while exact length grouping adds dynamic host/device work.
- Dense and compact0 have different missing-token semantics; changing a dense-trained model to compact inference can hurt even at 0% imposed pruning.
- The frontend and support construction add online work that a raw dense baseline does not perform.

These are explanations of available mechanisms, not substitutes for the saved measurements. [performance_analysis.md](performance_analysis.md) connects them to the actual seed-level results and profiler/support evidence, while keeping measured facts, mathematical implications, and hypotheses distinct.
