# training_dense_testing_sparse: PDF figure index

Each PDF has its exact plot-data CSV beside it. Figures use Matplotlib and Seaborn, talk/whitegrid/serif, and savefig dpi=300. PDF lines/text remain vector.

Saved report coverage complete: **False**.

All 612 clean/OOD evaluations are complete. These 108 report figures cover accuracy, robustness and support. The saved report predates ongoing latency collection; its zero timing count is a report snapshot, not the current number of completed workers. Full follow-up latency plots remain unavailable until the measurements finish. See the overview directory for explicitly labelled partial results when available.

## cifar10 / swin_tiny

### Accuracy, robustness and loss

- [brightness_accuracy](results/cifar10/swin_tiny/brightness_accuracy.pdf) ([data](results/cifar10/swin_tiny/brightness_accuracy.csv))
- [clean_accuracy](results/cifar10/swin_tiny/clean_accuracy.pdf) ([data](results/cifar10/swin_tiny/clean_accuracy.csv))
- [clean_loss](results/cifar10/swin_tiny/clean_loss.pdf) ([data](results/cifar10/swin_tiny/clean_loss.csv))
- [contrast_accuracy](results/cifar10/swin_tiny/contrast_accuracy.pdf) ([data](results/cifar10/swin_tiny/contrast_accuracy.csv))
- [delta_clean_accuracy_from_original](results/cifar10/swin_tiny/delta_clean_accuracy_from_original.pdf) ([data](results/cifar10/swin_tiny/delta_clean_accuracy_from_original.csv))
- [delta_mCE_raw_from_original](results/cifar10/swin_tiny/delta_mCE_raw_from_original.pdf) ([data](results/cifar10/swin_tiny/delta_mCE_raw_from_original.csv))
- [fog_accuracy](results/cifar10/swin_tiny/fog_accuracy.pdf) ([data](results/cifar10/swin_tiny/fog_accuracy.csv))
- [mCE_raw](results/cifar10/swin_tiny/mCE_raw.pdf) ([data](results/cifar10/swin_tiny/mCE_raw.csv))
- [mean_corruption_loss](results/cifar10/swin_tiny/mean_corruption_loss.pdf) ([data](results/cifar10/swin_tiny/mean_corruption_loss.csv))

### Corruption breakdowns

- [corruption_severity](results/cifar10/swin_tiny/corruption_severity.pdf) ([data](results/cifar10/swin_tiny/corruption_severity.csv))
- [per_corruption_error](results/cifar10/swin_tiny/per_corruption_error.pdf) ([data](results/cifar10/swin_tiny/per_corruption_error.csv))

### Sparsity and token support

- [achieved_zero_fraction](results/cifar10/swin_tiny/support/balanced_corruption/achieved_zero_fraction.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/achieved_zero_fraction.csv))
- [natural_zero_fraction](results/cifar10/swin_tiny/support/balanced_corruption/natural_zero_fraction.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/natural_zero_fraction.csv))
- [retained_patches](results/cifar10/swin_tiny/support/balanced_corruption/retained_patches.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/retained_patches.csv))
- [spatial_zero_fraction](results/cifar10/swin_tiny/support/balanced_corruption/spatial_zero_fraction.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/spatial_zero_fraction.csv))
- [stages_0_blocks_0_active_per_image](results/cifar10/swin_tiny/support/balanced_corruption/stages_0_blocks_0_active_per_image.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/stages_0_blocks_0_active_per_image.csv))
- [stages_0_blocks_0_padding_rows](results/cifar10/swin_tiny/support/balanced_corruption/stages_0_blocks_0_padding_rows.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/stages_0_blocks_0_padding_rows.csv))
- [stages_1_blocks_0_active_per_image](results/cifar10/swin_tiny/support/balanced_corruption/stages_1_blocks_0_active_per_image.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/stages_1_blocks_0_active_per_image.csv))
- [stages_1_blocks_0_padding_rows](results/cifar10/swin_tiny/support/balanced_corruption/stages_1_blocks_0_padding_rows.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/stages_1_blocks_0_padding_rows.csv))
- [stages_2_blocks_0_active_per_image](results/cifar10/swin_tiny/support/balanced_corruption/stages_2_blocks_0_active_per_image.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/stages_2_blocks_0_active_per_image.csv))
- [stages_2_blocks_0_padding_rows](results/cifar10/swin_tiny/support/balanced_corruption/stages_2_blocks_0_padding_rows.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/stages_2_blocks_0_padding_rows.csv))
- [stages_3_blocks_0_active_per_image](results/cifar10/swin_tiny/support/balanced_corruption/stages_3_blocks_0_active_per_image.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/stages_3_blocks_0_active_per_image.csv))
- [stages_3_blocks_0_padding_rows](results/cifar10/swin_tiny/support/balanced_corruption/stages_3_blocks_0_padding_rows.pdf) ([data](results/cifar10/swin_tiny/support/balanced_corruption/stages_3_blocks_0_padding_rows.csv))
- [achieved_zero_fraction](results/cifar10/swin_tiny/support/clean/achieved_zero_fraction.pdf) ([data](results/cifar10/swin_tiny/support/clean/achieved_zero_fraction.csv))
- [natural_zero_fraction](results/cifar10/swin_tiny/support/clean/natural_zero_fraction.pdf) ([data](results/cifar10/swin_tiny/support/clean/natural_zero_fraction.csv))
- [retained_patches](results/cifar10/swin_tiny/support/clean/retained_patches.pdf) ([data](results/cifar10/swin_tiny/support/clean/retained_patches.csv))
- [spatial_zero_fraction](results/cifar10/swin_tiny/support/clean/spatial_zero_fraction.pdf) ([data](results/cifar10/swin_tiny/support/clean/spatial_zero_fraction.csv))
- [stages_0_blocks_0_active_per_image](results/cifar10/swin_tiny/support/clean/stages_0_blocks_0_active_per_image.pdf) ([data](results/cifar10/swin_tiny/support/clean/stages_0_blocks_0_active_per_image.csv))
- [stages_0_blocks_0_padding_rows](results/cifar10/swin_tiny/support/clean/stages_0_blocks_0_padding_rows.pdf) ([data](results/cifar10/swin_tiny/support/clean/stages_0_blocks_0_padding_rows.csv))
- [stages_1_blocks_0_active_per_image](results/cifar10/swin_tiny/support/clean/stages_1_blocks_0_active_per_image.pdf) ([data](results/cifar10/swin_tiny/support/clean/stages_1_blocks_0_active_per_image.csv))
- [stages_1_blocks_0_padding_rows](results/cifar10/swin_tiny/support/clean/stages_1_blocks_0_padding_rows.pdf) ([data](results/cifar10/swin_tiny/support/clean/stages_1_blocks_0_padding_rows.csv))
- [stages_2_blocks_0_active_per_image](results/cifar10/swin_tiny/support/clean/stages_2_blocks_0_active_per_image.pdf) ([data](results/cifar10/swin_tiny/support/clean/stages_2_blocks_0_active_per_image.csv))
- [stages_2_blocks_0_padding_rows](results/cifar10/swin_tiny/support/clean/stages_2_blocks_0_padding_rows.pdf) ([data](results/cifar10/swin_tiny/support/clean/stages_2_blocks_0_padding_rows.csv))
- [stages_3_blocks_0_active_per_image](results/cifar10/swin_tiny/support/clean/stages_3_blocks_0_active_per_image.pdf) ([data](results/cifar10/swin_tiny/support/clean/stages_3_blocks_0_active_per_image.csv))
- [stages_3_blocks_0_padding_rows](results/cifar10/swin_tiny/support/clean/stages_3_blocks_0_padding_rows.pdf) ([data](results/cifar10/swin_tiny/support/clean/stages_3_blocks_0_padding_rows.csv))

## cifar10 / vit_small

### Accuracy, robustness and loss

- [brightness_accuracy](results/cifar10/vit_small/brightness_accuracy.pdf) ([data](results/cifar10/vit_small/brightness_accuracy.csv))
- [clean_accuracy](results/cifar10/vit_small/clean_accuracy.pdf) ([data](results/cifar10/vit_small/clean_accuracy.csv))
- [clean_loss](results/cifar10/vit_small/clean_loss.pdf) ([data](results/cifar10/vit_small/clean_loss.csv))
- [contrast_accuracy](results/cifar10/vit_small/contrast_accuracy.pdf) ([data](results/cifar10/vit_small/contrast_accuracy.csv))
- [delta_clean_accuracy_from_original](results/cifar10/vit_small/delta_clean_accuracy_from_original.pdf) ([data](results/cifar10/vit_small/delta_clean_accuracy_from_original.csv))
- [delta_mCE_raw_from_original](results/cifar10/vit_small/delta_mCE_raw_from_original.pdf) ([data](results/cifar10/vit_small/delta_mCE_raw_from_original.csv))
- [fog_accuracy](results/cifar10/vit_small/fog_accuracy.pdf) ([data](results/cifar10/vit_small/fog_accuracy.csv))
- [mCE_raw](results/cifar10/vit_small/mCE_raw.pdf) ([data](results/cifar10/vit_small/mCE_raw.csv))
- [mean_corruption_loss](results/cifar10/vit_small/mean_corruption_loss.pdf) ([data](results/cifar10/vit_small/mean_corruption_loss.csv))

### Corruption breakdowns

- [corruption_severity](results/cifar10/vit_small/corruption_severity.pdf) ([data](results/cifar10/vit_small/corruption_severity.csv))
- [per_corruption_error](results/cifar10/vit_small/per_corruption_error.pdf) ([data](results/cifar10/vit_small/per_corruption_error.csv))

### Sparsity and token support

- [achieved_zero_fraction](results/cifar10/vit_small/support/balanced_corruption/achieved_zero_fraction.pdf) ([data](results/cifar10/vit_small/support/balanced_corruption/achieved_zero_fraction.csv))
- [natural_zero_fraction](results/cifar10/vit_small/support/balanced_corruption/natural_zero_fraction.pdf) ([data](results/cifar10/vit_small/support/balanced_corruption/natural_zero_fraction.csv))
- [padding_tokens](results/cifar10/vit_small/support/balanced_corruption/padding_tokens.pdf) ([data](results/cifar10/vit_small/support/balanced_corruption/padding_tokens.csv))
- [retained_patches](results/cifar10/vit_small/support/balanced_corruption/retained_patches.pdf) ([data](results/cifar10/vit_small/support/balanced_corruption/retained_patches.csv))
- [spatial_zero_fraction](results/cifar10/vit_small/support/balanced_corruption/spatial_zero_fraction.pdf) ([data](results/cifar10/vit_small/support/balanced_corruption/spatial_zero_fraction.csv))
- [achieved_zero_fraction](results/cifar10/vit_small/support/clean/achieved_zero_fraction.pdf) ([data](results/cifar10/vit_small/support/clean/achieved_zero_fraction.csv))
- [natural_zero_fraction](results/cifar10/vit_small/support/clean/natural_zero_fraction.pdf) ([data](results/cifar10/vit_small/support/clean/natural_zero_fraction.csv))
- [padding_tokens](results/cifar10/vit_small/support/clean/padding_tokens.pdf) ([data](results/cifar10/vit_small/support/clean/padding_tokens.csv))
- [retained_patches](results/cifar10/vit_small/support/clean/retained_patches.pdf) ([data](results/cifar10/vit_small/support/clean/retained_patches.csv))
- [spatial_zero_fraction](results/cifar10/vit_small/support/clean/spatial_zero_fraction.pdf) ([data](results/cifar10/vit_small/support/clean/spatial_zero_fraction.csv))

## mnist / swin_tiny

### Accuracy, robustness and loss

- [brightness_accuracy](results/mnist/swin_tiny/brightness_accuracy.pdf) ([data](results/mnist/swin_tiny/brightness_accuracy.csv))
- [clean_accuracy](results/mnist/swin_tiny/clean_accuracy.pdf) ([data](results/mnist/swin_tiny/clean_accuracy.csv))
- [clean_loss](results/mnist/swin_tiny/clean_loss.pdf) ([data](results/mnist/swin_tiny/clean_loss.csv))
- [delta_clean_accuracy_from_original](results/mnist/swin_tiny/delta_clean_accuracy_from_original.pdf) ([data](results/mnist/swin_tiny/delta_clean_accuracy_from_original.csv))
- [delta_mCE_raw_from_original](results/mnist/swin_tiny/delta_mCE_raw_from_original.pdf) ([data](results/mnist/swin_tiny/delta_mCE_raw_from_original.csv))
- [fog_accuracy](results/mnist/swin_tiny/fog_accuracy.pdf) ([data](results/mnist/swin_tiny/fog_accuracy.csv))
- [mCE_raw](results/mnist/swin_tiny/mCE_raw.pdf) ([data](results/mnist/swin_tiny/mCE_raw.csv))
- [mean_corruption_loss](results/mnist/swin_tiny/mean_corruption_loss.pdf) ([data](results/mnist/swin_tiny/mean_corruption_loss.csv))

### Corruption breakdowns

- [per_corruption_error](results/mnist/swin_tiny/per_corruption_error.pdf) ([data](results/mnist/swin_tiny/per_corruption_error.csv))

### Sparsity and token support

- [achieved_zero_fraction](results/mnist/swin_tiny/support/balanced_corruption/achieved_zero_fraction.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/achieved_zero_fraction.csv))
- [natural_zero_fraction](results/mnist/swin_tiny/support/balanced_corruption/natural_zero_fraction.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/natural_zero_fraction.csv))
- [retained_patches](results/mnist/swin_tiny/support/balanced_corruption/retained_patches.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/retained_patches.csv))
- [spatial_zero_fraction](results/mnist/swin_tiny/support/balanced_corruption/spatial_zero_fraction.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/spatial_zero_fraction.csv))
- [stages_0_blocks_0_active_per_image](results/mnist/swin_tiny/support/balanced_corruption/stages_0_blocks_0_active_per_image.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/stages_0_blocks_0_active_per_image.csv))
- [stages_0_blocks_0_padding_rows](results/mnist/swin_tiny/support/balanced_corruption/stages_0_blocks_0_padding_rows.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/stages_0_blocks_0_padding_rows.csv))
- [stages_1_blocks_0_active_per_image](results/mnist/swin_tiny/support/balanced_corruption/stages_1_blocks_0_active_per_image.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/stages_1_blocks_0_active_per_image.csv))
- [stages_1_blocks_0_padding_rows](results/mnist/swin_tiny/support/balanced_corruption/stages_1_blocks_0_padding_rows.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/stages_1_blocks_0_padding_rows.csv))
- [stages_2_blocks_0_active_per_image](results/mnist/swin_tiny/support/balanced_corruption/stages_2_blocks_0_active_per_image.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/stages_2_blocks_0_active_per_image.csv))
- [stages_2_blocks_0_padding_rows](results/mnist/swin_tiny/support/balanced_corruption/stages_2_blocks_0_padding_rows.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/stages_2_blocks_0_padding_rows.csv))
- [stages_3_blocks_0_active_per_image](results/mnist/swin_tiny/support/balanced_corruption/stages_3_blocks_0_active_per_image.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/stages_3_blocks_0_active_per_image.csv))
- [stages_3_blocks_0_padding_rows](results/mnist/swin_tiny/support/balanced_corruption/stages_3_blocks_0_padding_rows.pdf) ([data](results/mnist/swin_tiny/support/balanced_corruption/stages_3_blocks_0_padding_rows.csv))
- [achieved_zero_fraction](results/mnist/swin_tiny/support/clean/achieved_zero_fraction.pdf) ([data](results/mnist/swin_tiny/support/clean/achieved_zero_fraction.csv))
- [natural_zero_fraction](results/mnist/swin_tiny/support/clean/natural_zero_fraction.pdf) ([data](results/mnist/swin_tiny/support/clean/natural_zero_fraction.csv))
- [retained_patches](results/mnist/swin_tiny/support/clean/retained_patches.pdf) ([data](results/mnist/swin_tiny/support/clean/retained_patches.csv))
- [spatial_zero_fraction](results/mnist/swin_tiny/support/clean/spatial_zero_fraction.pdf) ([data](results/mnist/swin_tiny/support/clean/spatial_zero_fraction.csv))
- [stages_0_blocks_0_active_per_image](results/mnist/swin_tiny/support/clean/stages_0_blocks_0_active_per_image.pdf) ([data](results/mnist/swin_tiny/support/clean/stages_0_blocks_0_active_per_image.csv))
- [stages_0_blocks_0_padding_rows](results/mnist/swin_tiny/support/clean/stages_0_blocks_0_padding_rows.pdf) ([data](results/mnist/swin_tiny/support/clean/stages_0_blocks_0_padding_rows.csv))
- [stages_1_blocks_0_active_per_image](results/mnist/swin_tiny/support/clean/stages_1_blocks_0_active_per_image.pdf) ([data](results/mnist/swin_tiny/support/clean/stages_1_blocks_0_active_per_image.csv))
- [stages_1_blocks_0_padding_rows](results/mnist/swin_tiny/support/clean/stages_1_blocks_0_padding_rows.pdf) ([data](results/mnist/swin_tiny/support/clean/stages_1_blocks_0_padding_rows.csv))
- [stages_2_blocks_0_active_per_image](results/mnist/swin_tiny/support/clean/stages_2_blocks_0_active_per_image.pdf) ([data](results/mnist/swin_tiny/support/clean/stages_2_blocks_0_active_per_image.csv))
- [stages_2_blocks_0_padding_rows](results/mnist/swin_tiny/support/clean/stages_2_blocks_0_padding_rows.pdf) ([data](results/mnist/swin_tiny/support/clean/stages_2_blocks_0_padding_rows.csv))
- [stages_3_blocks_0_active_per_image](results/mnist/swin_tiny/support/clean/stages_3_blocks_0_active_per_image.pdf) ([data](results/mnist/swin_tiny/support/clean/stages_3_blocks_0_active_per_image.csv))
- [stages_3_blocks_0_padding_rows](results/mnist/swin_tiny/support/clean/stages_3_blocks_0_padding_rows.pdf) ([data](results/mnist/swin_tiny/support/clean/stages_3_blocks_0_padding_rows.csv))

## mnist / vit_small

### Accuracy, robustness and loss

- [brightness_accuracy](results/mnist/vit_small/brightness_accuracy.pdf) ([data](results/mnist/vit_small/brightness_accuracy.csv))
- [clean_accuracy](results/mnist/vit_small/clean_accuracy.pdf) ([data](results/mnist/vit_small/clean_accuracy.csv))
- [clean_loss](results/mnist/vit_small/clean_loss.pdf) ([data](results/mnist/vit_small/clean_loss.csv))
- [delta_clean_accuracy_from_original](results/mnist/vit_small/delta_clean_accuracy_from_original.pdf) ([data](results/mnist/vit_small/delta_clean_accuracy_from_original.csv))
- [delta_mCE_raw_from_original](results/mnist/vit_small/delta_mCE_raw_from_original.pdf) ([data](results/mnist/vit_small/delta_mCE_raw_from_original.csv))
- [fog_accuracy](results/mnist/vit_small/fog_accuracy.pdf) ([data](results/mnist/vit_small/fog_accuracy.csv))
- [mCE_raw](results/mnist/vit_small/mCE_raw.pdf) ([data](results/mnist/vit_small/mCE_raw.csv))
- [mean_corruption_loss](results/mnist/vit_small/mean_corruption_loss.pdf) ([data](results/mnist/vit_small/mean_corruption_loss.csv))

### Corruption breakdowns

- [per_corruption_error](results/mnist/vit_small/per_corruption_error.pdf) ([data](results/mnist/vit_small/per_corruption_error.csv))

### Sparsity and token support

- [achieved_zero_fraction](results/mnist/vit_small/support/balanced_corruption/achieved_zero_fraction.pdf) ([data](results/mnist/vit_small/support/balanced_corruption/achieved_zero_fraction.csv))
- [natural_zero_fraction](results/mnist/vit_small/support/balanced_corruption/natural_zero_fraction.pdf) ([data](results/mnist/vit_small/support/balanced_corruption/natural_zero_fraction.csv))
- [padding_tokens](results/mnist/vit_small/support/balanced_corruption/padding_tokens.pdf) ([data](results/mnist/vit_small/support/balanced_corruption/padding_tokens.csv))
- [retained_patches](results/mnist/vit_small/support/balanced_corruption/retained_patches.pdf) ([data](results/mnist/vit_small/support/balanced_corruption/retained_patches.csv))
- [spatial_zero_fraction](results/mnist/vit_small/support/balanced_corruption/spatial_zero_fraction.pdf) ([data](results/mnist/vit_small/support/balanced_corruption/spatial_zero_fraction.csv))
- [achieved_zero_fraction](results/mnist/vit_small/support/clean/achieved_zero_fraction.pdf) ([data](results/mnist/vit_small/support/clean/achieved_zero_fraction.csv))
- [natural_zero_fraction](results/mnist/vit_small/support/clean/natural_zero_fraction.pdf) ([data](results/mnist/vit_small/support/clean/natural_zero_fraction.csv))
- [padding_tokens](results/mnist/vit_small/support/clean/padding_tokens.pdf) ([data](results/mnist/vit_small/support/clean/padding_tokens.csv))
- [retained_patches](results/mnist/vit_small/support/clean/retained_patches.pdf) ([data](results/mnist/vit_small/support/clean/retained_patches.csv))
- [spatial_zero_fraction](results/mnist/vit_small/support/clean/spatial_zero_fraction.pdf) ([data](results/mnist/vit_small/support/clean/spatial_zero_fraction.csv))
