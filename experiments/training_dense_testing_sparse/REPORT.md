# Training dense, testing sparse

Study `training_dense_testing_sparse`; manifest `7f5681a842c6d47c878bc80e4715268f40082772e0a1cfc05d0295198f49bd66`; source `8cf4d0c1e0ce5443d5080ea397df9a8f085d1bf2c4292912676545d38533a0cb`.

Accuracy: 612/612 registered cases and 35352/35352 official cells. Timing: 0/2424 complete execution/batch workers and 0/140304 cells. All requested work complete: **False**.

Protocol `heldout_val`: MNIST Train N=55,000; CIFAR-10 Train N=45,000; validation N=5,000 each. Every official clean or corruption cell has 10,000 images. MNIST-C has 15 fixed-severity cells; CIFAR-10-C has 15 corruptions at five severities. mCE below is the unnormalized macro mean corruption error. MNIST-C has no contrast corruption, so that field is unavailable rather than zero.

No model is retrained. "Dense training" here means zero imposed coefficient sparsity; original dense execution and original compact execution at 0% remain separate checkpoint families. Dense inference raw references remain separate from compact raw-pixel 0% anchors, which can already drop naturally empty patches. Raw-pixel sparsification ranks [0,1] input values before published raw normalization; contrast sparsification ranks signed native contrast magnitudes before frozen affine normalization. No normalization statistics are refit.

[Frozen normalization references and hashes](outputs/tables/normalization.csv) include published raw constants, contrast population means/sigmas, effective guarded scales, and exact training partitions. The original training source and this inference source remain separate identities.

| Dataset / architecture | Representation / training execution | Inference execution / domain / sparsity | Clean accuracy (%) | Mean corruption error (%) | Clean loss | Brightness accuracy (%) | Contrast accuracy (%) | Fog accuracy (%) |
|---|---|---|---|---|---|---|---|---|
| cifar10 / swin_tiny | raw / dense | dense / none / 0% | 87.787 +/- 0.148 (3/3) | 31.020 +/- 0.283 (3/3) | 0.633 +/- 0.005 (3/3) | 84.237 +/- 0.177 (3/3) | 59.641 +/- 0.414 (3/3) | 73.523 +/- 0.363 (3/3) |
| cifar10 / vit_small | raw / dense | dense / none / 0% | 78.103 +/- 0.702 (3/3) | 35.560 +/- 0.177 (3/3) | 1.814 +/- 0.066 (3/3) | 71.447 +/- 0.439 (3/3) | 42.957 +/- 0.321 (3/3) | 56.517 +/- 0.931 (3/3) |
| mnist / swin_tiny | raw / dense | dense / none / 0% | 99.233 +/- 0.015 (3/3) | 24.035 +/- 0.579 (3/3) | 0.036 +/- 0.002 (3/3) | 14.863 +/- 3.567 (3/3) | not in MNIST-C | 41.177 +/- 3.315 (3/3) |
| mnist / vit_small | raw / dense | dense / none / 0% | 98.300 +/- 0.053 (3/3) | 27.104 +/- 0.167 (3/3) | 0.089 +/- 0.003 (3/3) | 44.023 +/- 3.998 (3/3) | not in MNIST-C | 42.113 +/- 4.293 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 0% | 86.763 +/- 0.248 (3/3) | 40.252 +/- 0.127 (3/3) | 0.645 +/- 0.007 (3/3) | 83.778 +/- 0.139 (3/3) | 62.301 +/- 0.646 (3/3) | 78.692 +/- 0.500 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 0% | 86.833 +/- 0.136 (3/3) | 40.097 +/- 0.386 (3/3) | 0.632 +/- 0.011 (3/3) | 83.867 +/- 0.053 (3/3) | 61.758 +/- 0.511 (3/3) | 78.549 +/- 0.474 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 0% | 83.250 +/- 0.165 (3/3) | 43.521 +/- 0.227 (3/3) | 0.683 +/- 0.025 (3/3) | 81.469 +/- 0.133 (3/3) | 59.945 +/- 0.480 (3/3) | 73.159 +/- 0.248 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 0% | 83.263 +/- 0.111 (3/3) | 43.456 +/- 0.103 (3/3) | 0.690 +/- 0.009 (3/3) | 81.387 +/- 0.285 (3/3) | 59.957 +/- 0.404 (3/3) | 73.131 +/- 0.276 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 0% | 87.760 +/- 0.151 (3/3) | 31.037 +/- 0.284 (3/3) | 0.635 +/- 0.006 (3/3) | 84.237 +/- 0.177 (3/3) | 59.641 +/- 0.414 (3/3) | 73.523 +/- 0.363 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 0% | 87.163 +/- 0.241 (3/3) | 39.138 +/- 0.264 (3/3) | 0.607 +/- 0.008 (3/3) | 84.309 +/- 0.106 (3/3) | 62.161 +/- 0.209 (3/3) | 78.581 +/- 0.268 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 0% | 87.033 +/- 0.035 (3/3) | 39.183 +/- 0.112 (3/3) | 0.615 +/- 0.009 (3/3) | 84.091 +/- 0.199 (3/3) | 62.071 +/- 0.546 (3/3) | 78.498 +/- 0.171 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 0% | 77.897 +/- 0.638 (3/3) | 48.160 +/- 0.650 (3/3) | 1.764 +/- 0.074 (3/3) | 72.284 +/- 0.255 (3/3) | 46.076 +/- 0.917 (3/3) | 62.874 +/- 0.808 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 0% | 77.963 +/- 0.264 (3/3) | 48.681 +/- 0.443 (3/3) | 1.774 +/- 0.006 (3/3) | 72.349 +/- 0.575 (3/3) | 45.461 +/- 1.052 (3/3) | 62.414 +/- 0.669 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 0% | 76.903 +/- 0.545 (3/3) | 48.475 +/- 0.534 (3/3) | 1.844 +/- 0.028 (3/3) | 74.021 +/- 0.619 (3/3) | 46.512 +/- 0.599 (3/3) | 61.184 +/- 0.578 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 0% | 76.340 +/- 1.101 (3/3) | 49.014 +/- 1.016 (3/3) | 1.821 +/- 0.046 (3/3) | 73.237 +/- 1.245 (3/3) | 46.433 +/- 0.679 (3/3) | 60.384 +/- 1.005 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 0% | 78.107 +/- 0.709 (3/3) | 35.562 +/- 0.177 (3/3) | 1.814 +/- 0.066 (3/3) | 71.447 +/- 0.439 (3/3) | 42.957 +/- 0.321 (3/3) | 56.517 +/- 0.931 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 0% | 76.163 +/- 0.840 (3/3) | 48.842 +/- 0.716 (3/3) | 1.972 +/- 0.087 (3/3) | 71.473 +/- 0.886 (3/3) | 43.435 +/- 0.536 (3/3) | 59.429 +/- 0.360 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 0% | 74.037 +/- 1.442 (3/3) | 50.633 +/- 0.733 (3/3) | 2.094 +/- 0.110 (3/3) | 69.673 +/- 1.244 (3/3) | 41.604 +/- 1.093 (3/3) | 57.063 +/- 1.420 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 0% | 98.983 +/- 0.155 (3/3) | 27.061 +/- 1.443 (3/3) | 0.046 +/- 0.006 (3/3) | 37.160 +/- 10.897 (3/3) | not in MNIST-C | 30.950 +/- 9.617 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 0% | 90.320 +/- 2.915 (3/3) | 36.028 +/- 0.406 (3/3) | 0.417 +/- 0.139 (3/3) | 63.547 +/- 6.963 (3/3) | not in MNIST-C | 57.243 +/- 8.209 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 0% | 87.047 +/- 0.975 (3/3) | 45.101 +/- 1.190 (3/3) | 0.540 +/- 0.026 (3/3) | 14.863 +/- 3.567 (3/3) | not in MNIST-C | 41.177 +/- 3.315 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 0% | 98.037 +/- 0.137 (3/3) | 38.986 +/- 0.913 (3/3) | 0.117 +/- 0.004 (3/3) | 67.763 +/- 7.251 (3/3) | not in MNIST-C | 27.570 +/- 8.234 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 0% | 51.420 +/- 1.536 (3/3) | 60.162 +/- 1.716 (3/3) | 3.917 +/- 0.087 (3/3) | 92.430 +/- 0.961 (3/3) | not in MNIST-C | 48.477 +/- 7.812 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 0% | 68.350 +/- 8.734 (3/3) | 53.748 +/- 2.942 (3/3) | 2.473 +/- 0.834 (3/3) | 44.023 +/- 3.998 (3/3) | not in MNIST-C | 42.113 +/- 4.293 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 10% | 86.787 +/- 0.194 (3/3) | 40.255 +/- 0.123 (3/3) | 0.645 +/- 0.006 (3/3) | 83.799 +/- 0.114 (3/3) | 62.301 +/- 0.649 (3/3) | 78.690 +/- 0.494 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 10% | 86.790 +/- 0.165 (3/3) | 40.101 +/- 0.391 (3/3) | 0.632 +/- 0.011 (3/3) | 83.867 +/- 0.049 (3/3) | 61.763 +/- 0.524 (3/3) | 78.553 +/- 0.477 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 10% | 83.230 +/- 0.174 (3/3) | 43.582 +/- 0.213 (3/3) | 0.684 +/- 0.023 (3/3) | 81.528 +/- 0.109 (3/3) | 59.903 +/- 0.470 (3/3) | 73.133 +/- 0.234 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 10% | 83.180 +/- 0.168 (3/3) | 43.538 +/- 0.103 (3/3) | 0.691 +/- 0.009 (3/3) | 81.357 +/- 0.217 (3/3) | 59.868 +/- 0.380 (3/3) | 73.073 +/- 0.208 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 10% | 80.997 +/- 0.316 (3/3) | 39.022 +/- 0.456 (3/3) | 1.076 +/- 0.023 (3/3) | 74.086 +/- 0.255 (3/3) | 46.550 +/- 0.424 (3/3) | 59.465 +/- 0.610 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 10% | 87.067 +/- 0.227 (3/3) | 39.181 +/- 0.276 (3/3) | 0.608 +/- 0.008 (3/3) | 84.270 +/- 0.097 (3/3) | 62.131 +/- 0.198 (3/3) | 78.525 +/- 0.230 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 10% | 86.970 +/- 0.122 (3/3) | 39.234 +/- 0.107 (3/3) | 0.617 +/- 0.009 (3/3) | 84.084 +/- 0.182 (3/3) | 62.055 +/- 0.530 (3/3) | 78.493 +/- 0.148 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 10% | 77.903 +/- 0.633 (3/3) | 48.167 +/- 0.645 (3/3) | 1.765 +/- 0.074 (3/3) | 72.266 +/- 0.281 (3/3) | 46.088 +/- 0.876 (3/3) | 62.870 +/- 0.781 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 10% | 77.947 +/- 0.284 (3/3) | 48.691 +/- 0.449 (3/3) | 1.774 +/- 0.005 (3/3) | 72.371 +/- 0.600 (3/3) | 45.447 +/- 1.066 (3/3) | 62.399 +/- 0.686 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 10% | 76.860 +/- 0.524 (3/3) | 48.619 +/- 0.549 (3/3) | 1.846 +/- 0.027 (3/3) | 74.074 +/- 0.694 (3/3) | 46.393 +/- 0.542 (3/3) | 61.061 +/- 0.552 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 10% | 76.367 +/- 1.165 (3/3) | 49.185 +/- 1.035 (3/3) | 1.822 +/- 0.050 (3/3) | 73.255 +/- 1.338 (3/3) | 46.305 +/- 0.732 (3/3) | 60.306 +/- 0.977 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 10% | 69.533 +/- 0.365 (3/3) | 44.297 +/- 0.681 (3/3) | 2.790 +/- 0.064 (3/3) | 57.365 +/- 0.894 (3/3) | 35.017 +/- 1.172 (3/3) | 46.611 +/- 0.872 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 10% | 76.180 +/- 0.899 (3/3) | 48.893 +/- 0.713 (3/3) | 1.974 +/- 0.086 (3/3) | 71.421 +/- 0.905 (3/3) | 43.433 +/- 0.544 (3/3) | 59.397 +/- 0.314 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 10% | 73.973 +/- 1.475 (3/3) | 50.695 +/- 0.738 (3/3) | 2.097 +/- 0.111 (3/3) | 69.633 +/- 1.253 (3/3) | 41.557 +/- 1.062 (3/3) | 57.023 +/- 1.465 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 10% | 98.983 +/- 0.155 (3/3) | 27.022 +/- 1.130 (3/3) | 0.046 +/- 0.006 (3/3) | 37.617 +/- 6.769 (3/3) | not in MNIST-C | 31.077 +/- 9.634 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 10% | 90.320 +/- 2.915 (3/3) | 36.295 +/- 0.467 (3/3) | 0.417 +/- 0.139 (3/3) | 59.517 +/- 5.672 (3/3) | not in MNIST-C | 57.263 +/- 8.211 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 10% | 87.047 +/- 0.975 (3/3) | 45.136 +/- 1.105 (3/3) | 0.540 +/- 0.026 (3/3) | 15.503 +/- 2.170 (3/3) | not in MNIST-C | 40.013 +/- 3.255 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 10% | 98.037 +/- 0.137 (3/3) | 38.430 +/- 0.690 (3/3) | 0.117 +/- 0.004 (3/3) | 76.073 +/- 3.801 (3/3) | not in MNIST-C | 27.600 +/- 8.237 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 10% | 51.420 +/- 1.536 (3/3) | 60.173 +/- 1.720 (3/3) | 3.917 +/- 0.087 (3/3) | 92.230 +/- 0.786 (3/3) | not in MNIST-C | 48.523 +/- 7.857 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 10% | 68.350 +/- 8.734 (3/3) | 53.827 +/- 3.340 (3/3) | 2.473 +/- 0.834 (3/3) | 43.087 +/- 9.534 (3/3) | not in MNIST-C | 41.853 +/- 3.684 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 20% | 86.743 +/- 0.220 (3/3) | 40.285 +/- 0.133 (3/3) | 0.646 +/- 0.007 (3/3) | 83.832 +/- 0.123 (3/3) | 62.267 +/- 0.632 (3/3) | 78.665 +/- 0.445 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 20% | 86.783 +/- 0.147 (3/3) | 40.133 +/- 0.399 (3/3) | 0.634 +/- 0.010 (3/3) | 83.852 +/- 0.072 (3/3) | 61.703 +/- 0.487 (3/3) | 78.501 +/- 0.508 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 20% | 83.160 +/- 0.175 (3/3) | 44.044 +/- 0.217 (3/3) | 0.696 +/- 0.023 (3/3) | 81.196 +/- 0.206 (3/3) | 59.436 +/- 0.468 (3/3) | 72.757 +/- 0.268 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 20% | 82.963 +/- 0.340 (3/3) | 43.986 +/- 0.068 (3/3) | 0.704 +/- 0.008 (3/3) | 81.045 +/- 0.351 (3/3) | 59.461 +/- 0.362 (3/3) | 72.653 +/- 0.284 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 20% | 71.313 +/- 0.527 (3/3) | 47.297 +/- 0.578 (3/3) | 1.783 +/- 0.050 (3/3) | 64.163 +/- 0.240 (3/3) | 42.310 +/- 0.553 (3/3) | 49.952 +/- 0.975 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 20% | 86.883 +/- 0.101 (3/3) | 39.469 +/- 0.268 (3/3) | 0.620 +/- 0.008 (3/3) | 84.015 +/- 0.018 (3/3) | 61.929 +/- 0.161 (3/3) | 78.271 +/- 0.191 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 20% | 86.653 +/- 0.140 (3/3) | 39.563 +/- 0.104 (3/3) | 0.627 +/- 0.008 (3/3) | 83.853 +/- 0.095 (3/3) | 61.689 +/- 0.518 (3/3) | 78.217 +/- 0.192 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 20% | 77.787 +/- 0.647 (3/3) | 48.240 +/- 0.635 (3/3) | 1.768 +/- 0.073 (3/3) | 72.190 +/- 0.248 (3/3) | 46.001 +/- 0.861 (3/3) | 62.812 +/- 0.785 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 20% | 77.930 +/- 0.303 (3/3) | 48.759 +/- 0.457 (3/3) | 1.775 +/- 0.003 (3/3) | 72.317 +/- 0.557 (3/3) | 45.364 +/- 1.076 (3/3) | 62.380 +/- 0.674 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 20% | 76.373 +/- 0.759 (3/3) | 49.327 +/- 0.501 (3/3) | 1.879 +/- 0.032 (3/3) | 73.777 +/- 0.687 (3/3) | 45.563 +/- 0.341 (3/3) | 60.485 +/- 0.551 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 20% | 75.840 +/- 1.292 (3/3) | 49.946 +/- 1.045 (3/3) | 1.856 +/- 0.068 (3/3) | 72.904 +/- 1.343 (3/3) | 45.557 +/- 0.630 (3/3) | 59.709 +/- 0.908 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 20% | 60.827 +/- 0.890 (3/3) | 51.569 +/- 1.008 (3/3) | 3.822 +/- 0.164 (3/3) | 48.869 +/- 0.861 (3/3) | 31.563 +/- 1.452 (3/3) | 40.249 +/- 1.575 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 20% | 75.650 +/- 0.968 (3/3) | 49.291 +/- 0.708 (3/3) | 2.004 +/- 0.089 (3/3) | 71.147 +/- 0.976 (3/3) | 43.226 +/- 0.509 (3/3) | 59.186 +/- 0.335 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 20% | 73.687 +/- 1.529 (3/3) | 51.078 +/- 0.722 (3/3) | 2.122 +/- 0.118 (3/3) | 69.308 +/- 1.289 (3/3) | 41.384 +/- 1.073 (3/3) | 56.786 +/- 1.383 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 20% | 98.983 +/- 0.155 (3/3) | 27.182 +/- 1.077 (3/3) | 0.046 +/- 0.006 (3/3) | 34.633 +/- 5.742 (3/3) | not in MNIST-C | 31.057 +/- 9.723 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 20% | 90.320 +/- 2.915 (3/3) | 36.455 +/- 0.548 (3/3) | 0.417 +/- 0.139 (3/3) | 58.037 +/- 4.892 (3/3) | not in MNIST-C | 57.070 +/- 8.107 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 20% | 87.047 +/- 0.975 (3/3) | 45.237 +/- 1.419 (3/3) | 0.540 +/- 0.026 (3/3) | 14.003 +/- 2.980 (3/3) | not in MNIST-C | 39.997 +/- 2.851 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 20% | 98.037 +/- 0.137 (3/3) | 38.253 +/- 0.716 (3/3) | 0.117 +/- 0.004 (3/3) | 78.737 +/- 4.404 (3/3) | not in MNIST-C | 27.787 +/- 8.249 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 20% | 51.420 +/- 1.536 (3/3) | 60.541 +/- 1.563 (3/3) | 3.917 +/- 0.087 (3/3) | 87.057 +/- 2.557 (3/3) | not in MNIST-C | 48.113 +/- 7.567 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 20% | 68.350 +/- 8.734 (3/3) | 53.533 +/- 3.431 (3/3) | 2.473 +/- 0.834 (3/3) | 46.097 +/- 8.035 (3/3) | not in MNIST-C | 43.257 +/- 1.990 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 30% | 86.643 +/- 0.205 (3/3) | 40.422 +/- 0.136 (3/3) | 0.649 +/- 0.006 (3/3) | 83.749 +/- 0.202 (3/3) | 62.085 +/- 0.654 (3/3) | 78.552 +/- 0.486 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 30% | 86.670 +/- 0.181 (3/3) | 40.275 +/- 0.402 (3/3) | 0.638 +/- 0.010 (3/3) | 83.815 +/- 0.112 (3/3) | 61.587 +/- 0.481 (3/3) | 78.417 +/- 0.481 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 30% | 81.937 +/- 0.047 (3/3) | 45.395 +/- 0.180 (3/3) | 0.749 +/- 0.021 (3/3) | 80.023 +/- 0.075 (3/3) | 57.726 +/- 0.515 (3/3) | 71.281 +/- 0.277 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 30% | 81.900 +/- 0.207 (3/3) | 45.285 +/- 0.089 (3/3) | 0.752 +/- 0.013 (3/3) | 79.751 +/- 0.381 (3/3) | 57.854 +/- 0.329 (3/3) | 71.182 +/- 0.114 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 30% | 61.033 +/- 0.801 (3/3) | 54.972 +/- 0.782 (3/3) | 2.637 +/- 0.086 (3/3) | 54.430 +/- 0.647 (3/3) | 38.422 +/- 0.896 (3/3) | 43.000 +/- 1.055 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 30% | 86.223 +/- 0.200 (3/3) | 40.292 +/- 0.285 (3/3) | 0.655 +/- 0.011 (3/3) | 83.304 +/- 0.045 (3/3) | 61.133 +/- 0.170 (3/3) | 77.393 +/- 0.170 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 30% | 86.097 +/- 0.234 (3/3) | 40.508 +/- 0.125 (3/3) | 0.666 +/- 0.006 (3/3) | 83.098 +/- 0.180 (3/3) | 60.601 +/- 0.529 (3/3) | 77.190 +/- 0.208 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 30% | 77.673 +/- 0.544 (3/3) | 48.429 +/- 0.618 (3/3) | 1.781 +/- 0.070 (3/3) | 72.171 +/- 0.257 (3/3) | 45.871 +/- 0.876 (3/3) | 62.688 +/- 0.831 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 30% | 77.753 +/- 0.304 (3/3) | 48.933 +/- 0.481 (3/3) | 1.782 +/- 0.003 (3/3) | 72.251 +/- 0.536 (3/3) | 45.269 +/- 1.067 (3/3) | 62.268 +/- 0.679 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 30% | 75.157 +/- 0.628 (3/3) | 51.010 +/- 0.422 (3/3) | 1.999 +/- 0.040 (3/3) | 72.781 +/- 0.500 (3/3) | 43.666 +/- 0.218 (3/3) | 58.626 +/- 0.353 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 30% | 74.037 +/- 1.219 (3/3) | 51.742 +/- 1.042 (3/3) | 1.989 +/- 0.061 (3/3) | 71.534 +/- 1.574 (3/3) | 43.779 +/- 0.643 (3/3) | 57.799 +/- 1.011 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 30% | 52.210 +/- 1.058 (3/3) | 58.279 +/- 1.089 (3/3) | 4.778 +/- 0.234 (3/3) | 41.711 +/- 1.065 (3/3) | 28.287 +/- 1.426 (3/3) | 34.927 +/- 1.858 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 30% | 74.647 +/- 0.926 (3/3) | 50.327 +/- 0.661 (3/3) | 2.077 +/- 0.079 (3/3) | 70.248 +/- 0.940 (3/3) | 42.645 +/- 0.400 (3/3) | 58.410 +/- 0.328 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 30% | 72.630 +/- 1.349 (3/3) | 52.133 +/- 0.699 (3/3) | 2.193 +/- 0.108 (3/3) | 68.239 +/- 1.114 (3/3) | 40.827 +/- 0.955 (3/3) | 55.996 +/- 1.210 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 30% | 98.983 +/- 0.155 (3/3) | 27.519 +/- 1.018 (3/3) | 0.046 +/- 0.006 (3/3) | 34.077 +/- 4.944 (3/3) | not in MNIST-C | 30.923 +/- 9.487 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 30% | 90.320 +/- 2.915 (3/3) | 37.321 +/- 0.466 (3/3) | 0.417 +/- 0.139 (3/3) | 50.897 +/- 6.120 (3/3) | not in MNIST-C | 55.730 +/- 7.903 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 30% | 87.047 +/- 0.975 (3/3) | 44.876 +/- 1.163 (3/3) | 0.540 +/- 0.026 (3/3) | 18.040 +/- 1.196 (3/3) | not in MNIST-C | 41.430 +/- 2.710 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 30% | 98.037 +/- 0.137 (3/3) | 38.067 +/- 0.758 (3/3) | 0.117 +/- 0.004 (3/3) | 76.627 +/- 5.812 (3/3) | not in MNIST-C | 28.517 +/- 8.617 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 30% | 51.420 +/- 1.536 (3/3) | 61.934 +/- 1.402 (3/3) | 3.917 +/- 0.087 (3/3) | 75.913 +/- 2.892 (3/3) | not in MNIST-C | 46.820 +/- 7.255 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 30% | 68.350 +/- 8.734 (3/3) | 53.558 +/- 3.629 (3/3) | 2.473 +/- 0.834 (3/3) | 46.097 +/- 9.281 (3/3) | not in MNIST-C | 43.997 +/- 2.704 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 40% | 86.353 +/- 0.265 (3/3) | 40.797 +/- 0.117 (3/3) | 0.660 +/- 0.003 (3/3) | 83.576 +/- 0.185 (3/3) | 61.737 +/- 0.670 (3/3) | 78.249 +/- 0.547 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 40% | 86.480 +/- 0.180 (3/3) | 40.678 +/- 0.419 (3/3) | 0.647 +/- 0.010 (3/3) | 83.655 +/- 0.056 (3/3) | 61.133 +/- 0.570 (3/3) | 78.013 +/- 0.505 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 40% | 79.357 +/- 0.170 (3/3) | 47.831 +/- 0.217 (3/3) | 0.870 +/- 0.030 (3/3) | 77.422 +/- 0.149 (3/3) | 54.611 +/- 0.509 (3/3) | 67.983 +/- 0.511 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 40% | 79.380 +/- 0.340 (3/3) | 47.707 +/- 0.249 (3/3) | 0.871 +/- 0.008 (3/3) | 77.254 +/- 0.340 (3/3) | 54.733 +/- 0.381 (3/3) | 67.963 +/- 0.096 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 40% | 50.900 +/- 1.107 (3/3) | 61.978 +/- 1.011 (3/3) | 3.619 +/- 0.082 (3/3) | 45.053 +/- 1.054 (3/3) | 34.633 +/- 1.411 (3/3) | 37.027 +/- 1.292 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 40% | 84.490 +/- 0.199 (3/3) | 42.065 +/- 0.362 (3/3) | 0.736 +/- 0.022 (3/3) | 81.385 +/- 0.079 (3/3) | 59.232 +/- 0.238 (3/3) | 75.155 +/- 0.254 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 40% | 84.367 +/- 0.085 (3/3) | 42.416 +/- 0.169 (3/3) | 0.748 +/- 0.010 (3/3) | 81.237 +/- 0.041 (3/3) | 58.267 +/- 0.431 (3/3) | 74.770 +/- 0.168 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 40% | 77.183 +/- 0.490 (3/3) | 48.898 +/- 0.624 (3/3) | 1.818 +/- 0.071 (3/3) | 71.801 +/- 0.351 (3/3) | 45.517 +/- 0.848 (3/3) | 62.281 +/- 0.794 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 40% | 77.160 +/- 0.282 (3/3) | 49.374 +/- 0.498 (3/3) | 1.815 +/- 0.007 (3/3) | 71.931 +/- 0.637 (3/3) | 44.951 +/- 1.031 (3/3) | 61.807 +/- 0.644 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 40% | 72.137 +/- 0.601 (3/3) | 54.134 +/- 0.254 (3/3) | 2.253 +/- 0.027 (3/3) | 69.774 +/- 0.374 (3/3) | 39.971 +/- 0.308 (3/3) | 54.392 +/- 0.290 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 40% | 70.823 +/- 1.198 (3/3) | 54.978 +/- 0.916 (3/3) | 2.263 +/- 0.057 (3/3) | 67.996 +/- 1.392 (3/3) | 40.413 +/- 0.489 (3/3) | 53.770 +/- 0.593 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 40% | 44.023 +/- 0.947 (3/3) | 64.299 +/- 0.954 (3/3) | 5.689 +/- 0.335 (3/3) | 35.588 +/- 0.882 (3/3) | 25.682 +/- 1.280 (3/3) | 30.489 +/- 2.130 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 40% | 72.130 +/- 0.923 (3/3) | 52.371 +/- 0.576 (3/3) | 2.260 +/- 0.063 (3/3) | 67.774 +/- 0.791 (3/3) | 41.242 +/- 0.446 (3/3) | 56.313 +/- 0.268 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 40% | 70.153 +/- 1.392 (3/3) | 54.278 +/- 0.653 (3/3) | 2.395 +/- 0.118 (3/3) | 65.479 +/- 1.233 (3/3) | 39.433 +/- 0.712 (3/3) | 53.885 +/- 0.893 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 40% | 98.983 +/- 0.155 (3/3) | 27.635 +/- 1.274 (3/3) | 0.046 +/- 0.006 (3/3) | 38.267 +/- 8.356 (3/3) | not in MNIST-C | 30.947 +/- 9.420 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 40% | 90.320 +/- 2.915 (3/3) | 38.514 +/- 0.745 (3/3) | 0.417 +/- 0.139 (3/3) | 46.893 +/- 5.445 (3/3) | not in MNIST-C | 51.583 +/- 7.224 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 40% | 87.047 +/- 0.975 (3/3) | 44.165 +/- 1.190 (3/3) | 0.540 +/- 0.026 (3/3) | 28.040 +/- 1.784 (3/3) | not in MNIST-C | 44.847 +/- 1.035 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 40% | 98.037 +/- 0.137 (3/3) | 37.827 +/- 0.795 (3/3) | 0.117 +/- 0.004 (3/3) | 76.160 +/- 4.556 (3/3) | not in MNIST-C | 31.187 +/- 9.252 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 40% | 51.420 +/- 1.536 (3/3) | 64.237 +/- 1.245 (3/3) | 3.917 +/- 0.087 (3/3) | 55.820 +/- 5.783 (3/3) | not in MNIST-C | 42.437 +/- 5.468 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 40% | 68.350 +/- 8.734 (3/3) | 54.088 +/- 3.634 (3/3) | 2.473 +/- 0.834 (3/3) | 47.677 +/- 10.444 (3/3) | not in MNIST-C | 44.503 +/- 3.082 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 50% | 85.673 +/- 0.177 (3/3) | 41.746 +/- 0.108 (3/3) | 0.699 +/- 0.007 (3/3) | 82.869 +/- 0.053 (3/3) | 60.899 +/- 0.693 (3/3) | 77.343 +/- 0.543 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 50% | 85.657 +/- 0.245 (3/3) | 41.685 +/- 0.510 (3/3) | 0.684 +/- 0.015 (3/3) | 82.833 +/- 0.152 (3/3) | 60.196 +/- 0.495 (3/3) | 77.154 +/- 0.589 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 50% | 75.380 +/- 0.312 (3/3) | 51.634 +/- 0.178 (3/3) | 1.077 +/- 0.040 (3/3) | 72.601 +/- 0.448 (3/3) | 49.795 +/- 0.544 (3/3) | 62.233 +/- 0.375 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 50% | 75.123 +/- 0.350 (3/3) | 51.674 +/- 0.396 (3/3) | 1.092 +/- 0.020 (3/3) | 72.499 +/- 0.424 (3/3) | 49.790 +/- 0.305 (3/3) | 62.446 +/- 0.171 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 50% | 41.663 +/- 1.063 (3/3) | 68.158 +/- 0.995 (3/3) | 4.715 +/- 0.051 (3/3) | 37.385 +/- 1.246 (3/3) | 30.448 +/- 1.112 (3/3) | 31.377 +/- 1.183 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 50% | 80.983 +/- 0.329 (3/3) | 45.206 +/- 0.369 (3/3) | 0.944 +/- 0.019 (3/3) | 77.509 +/- 0.137 (3/3) | 55.440 +/- 0.490 (3/3) | 70.449 +/- 0.417 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 50% | 80.660 +/- 0.035 (3/3) | 45.694 +/- 0.243 (3/3) | 0.949 +/- 0.013 (3/3) | 77.333 +/- 0.110 (3/3) | 54.325 +/- 0.535 (3/3) | 69.859 +/- 0.278 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 50% | 75.927 +/- 0.707 (3/3) | 49.958 +/- 0.599 (3/3) | 1.920 +/- 0.077 (3/3) | 70.613 +/- 0.485 (3/3) | 44.747 +/- 0.976 (3/3) | 61.243 +/- 0.827 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 50% | 75.680 +/- 0.382 (3/3) | 50.403 +/- 0.510 (3/3) | 1.925 +/- 0.017 (3/3) | 70.699 +/- 0.687 (3/3) | 44.070 +/- 1.036 (3/3) | 60.756 +/- 0.586 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 50% | 65.967 +/- 0.621 (3/3) | 58.853 +/- 0.125 (3/3) | 2.764 +/- 0.040 (3/3) | 63.763 +/- 0.264 (3/3) | 34.870 +/- 0.655 (3/3) | 47.867 +/- 0.801 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 50% | 64.603 +/- 0.980 (3/3) | 59.822 +/- 0.730 (3/3) | 2.822 +/- 0.072 (3/3) | 61.631 +/- 1.220 (3/3) | 35.422 +/- 0.649 (3/3) | 47.325 +/- 0.398 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 50% | 36.783 +/- 0.947 (3/3) | 69.479 +/- 0.863 (3/3) | 6.547 +/- 0.446 (3/3) | 30.458 +/- 0.957 (3/3) | 24.483 +/- 0.965 (3/3) | 26.887 +/- 1.931 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 50% | 67.310 +/- 0.770 (3/3) | 55.857 +/- 0.481 (3/3) | 2.705 +/- 0.064 (3/3) | 62.910 +/- 0.828 (3/3) | 38.741 +/- 0.543 (3/3) | 52.061 +/- 0.351 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 50% | 65.020 +/- 1.226 (3/3) | 57.709 +/- 0.604 (3/3) | 2.847 +/- 0.125 (3/3) | 60.295 +/- 1.318 (3/3) | 37.153 +/- 0.309 (3/3) | 50.085 +/- 0.402 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 50% | 98.983 +/- 0.155 (3/3) | 28.249 +/- 1.408 (3/3) | 0.046 +/- 0.006 (3/3) | 38.820 +/- 8.673 (3/3) | not in MNIST-C | 30.553 +/- 9.113 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 50% | 90.320 +/- 2.915 (3/3) | 40.536 +/- 1.138 (3/3) | 0.417 +/- 0.138 (3/3) | 43.853 +/- 4.675 (3/3) | not in MNIST-C | 44.390 +/- 6.659 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 50% | 87.047 +/- 0.975 (3/3) | 43.372 +/- 1.042 (3/3) | 0.540 +/- 0.026 (3/3) | 44.433 +/- 4.479 (3/3) | not in MNIST-C | 49.247 +/- 0.533 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 50% | 98.037 +/- 0.137 (3/3) | 36.711 +/- 0.989 (3/3) | 0.117 +/- 0.004 (3/3) | 81.660 +/- 2.828 (3/3) | not in MNIST-C | 36.117 +/- 10.141 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 50% | 51.420 +/- 1.536 (3/3) | 66.834 +/- 1.015 (3/3) | 3.917 +/- 0.087 (3/3) | 39.917 +/- 1.884 (3/3) | not in MNIST-C | 32.043 +/- 0.748 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 50% | 68.350 +/- 8.734 (3/3) | 54.477 +/- 3.771 (3/3) | 2.473 +/- 0.834 (3/3) | 49.750 +/- 10.069 (3/3) | not in MNIST-C | 44.860 +/- 3.986 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 60% | 83.527 +/- 0.144 (3/3) | 43.874 +/- 0.106 (3/3) | 0.806 +/- 0.009 (3/3) | 80.599 +/- 0.114 (3/3) | 58.916 +/- 0.858 (3/3) | 75.256 +/- 0.548 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 60% | 83.373 +/- 0.348 (3/3) | 43.943 +/- 0.571 (3/3) | 0.801 +/- 0.019 (3/3) | 80.648 +/- 0.341 (3/3) | 57.927 +/- 0.475 (3/3) | 74.801 +/- 0.715 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 60% | 68.087 +/- 0.597 (3/3) | 57.041 +/- 0.037 (3/3) | 1.480 +/- 0.035 (3/3) | 64.355 +/- 0.545 (3/3) | 43.408 +/- 0.456 (3/3) | 53.920 +/- 0.525 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 60% | 67.557 +/- 0.320 (3/3) | 57.577 +/- 0.523 (3/3) | 1.515 +/- 0.033 (3/3) | 63.832 +/- 0.635 (3/3) | 42.809 +/- 0.292 (3/3) | 53.854 +/- 0.487 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 60% | 34.260 +/- 1.355 (3/3) | 73.149 +/- 0.983 (3/3) | 5.776 +/- 0.045 (3/3) | 30.971 +/- 1.428 (3/3) | 26.717 +/- 0.921 (3/3) | 26.855 +/- 1.221 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 60% | 74.033 +/- 0.100 (3/3) | 50.396 +/- 0.485 (3/3) | 1.329 +/- 0.018 (3/3) | 69.717 +/- 0.421 (3/3) | 49.497 +/- 0.564 (3/3) | 62.461 +/- 0.727 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 60% | 73.947 +/- 0.221 (3/3) | 51.036 +/- 0.286 (3/3) | 1.330 +/- 0.034 (3/3) | 69.531 +/- 0.073 (3/3) | 48.143 +/- 0.317 (3/3) | 61.604 +/- 0.211 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 60% | 72.193 +/- 0.829 (3/3) | 52.468 +/- 0.529 (3/3) | 2.202 +/- 0.076 (3/3) | 66.803 +/- 0.787 (3/3) | 42.973 +/- 1.076 (3/3) | 58.550 +/- 0.791 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 60% | 72.273 +/- 0.257 (3/3) | 52.840 +/- 0.585 (3/3) | 2.233 +/- 0.032 (3/3) | 66.597 +/- 0.839 (3/3) | 42.277 +/- 1.080 (3/3) | 58.174 +/- 0.547 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 60% | 56.933 +/- 0.739 (3/3) | 64.969 +/- 0.348 (3/3) | 3.696 +/- 0.086 (3/3) | 54.458 +/- 0.337 (3/3) | 29.041 +/- 0.957 (3/3) | 39.336 +/- 1.345 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 60% | 54.267 +/- 1.355 (3/3) | 66.274 +/- 0.527 (3/3) | 3.805 +/- 0.088 (3/3) | 51.441 +/- 1.122 (3/3) | 29.371 +/- 0.738 (3/3) | 38.725 +/- 0.068 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 60% | 31.187 +/- 0.649 (3/3) | 73.644 +/- 0.663 (3/3) | 7.327 +/- 0.504 (3/3) | 26.558 +/- 0.481 (3/3) | 23.198 +/- 0.749 (3/3) | 24.158 +/- 1.432 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 60% | 58.873 +/- 0.869 (3/3) | 60.927 +/- 0.509 (3/3) | 3.578 +/- 0.106 (3/3) | 54.460 +/- 0.915 (3/3) | 34.579 +/- 0.614 (3/3) | 45.444 +/- 0.783 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 60% | 56.487 +/- 1.674 (3/3) | 62.780 +/- 0.477 (3/3) | 3.664 +/- 0.209 (3/3) | 51.707 +/- 1.639 (3/3) | 33.566 +/- 0.033 (3/3) | 43.764 +/- 0.286 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 60% | 98.983 +/- 0.147 (3/3) | 29.492 +/- 1.284 (3/3) | 0.046 +/- 0.006 (3/3) | 36.067 +/- 8.078 (3/3) | not in MNIST-C | 30.340 +/- 8.621 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 60% | 90.297 +/- 2.949 (3/3) | 43.213 +/- 1.343 (3/3) | 0.418 +/- 0.139 (3/3) | 41.177 +/- 5.134 (3/3) | not in MNIST-C | 37.263 +/- 6.447 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 60% | 87.047 +/- 0.975 (3/3) | 42.845 +/- 1.067 (3/3) | 0.540 +/- 0.026 (3/3) | 50.903 +/- 3.994 (3/3) | not in MNIST-C | 54.707 +/- 0.538 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 60% | 98.037 +/- 0.127 (3/3) | 36.734 +/- 1.138 (3/3) | 0.117 +/- 0.004 (3/3) | 81.783 +/- 3.216 (3/3) | not in MNIST-C | 41.877 +/- 9.457 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 60% | 51.227 +/- 1.573 (3/3) | 71.299 +/- 0.814 (3/3) | 3.935 +/- 0.094 (3/3) | 32.033 +/- 0.835 (3/3) | not in MNIST-C | 19.047 +/- 3.042 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 60% | 68.350 +/- 8.734 (3/3) | 54.491 +/- 3.867 (3/3) | 2.473 +/- 0.834 (3/3) | 49.697 +/- 9.255 (3/3) | not in MNIST-C | 44.663 +/- 5.212 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 70% | 78.517 +/- 0.429 (3/3) | 47.992 +/- 0.226 (3/3) | 1.106 +/- 0.032 (3/3) | 75.689 +/- 0.286 (3/3) | 55.202 +/- 1.159 (3/3) | 70.378 +/- 0.952 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 70% | 77.913 +/- 0.448 (3/3) | 48.390 +/- 0.624 (3/3) | 1.110 +/- 0.027 (3/3) | 75.499 +/- 0.394 (3/3) | 53.599 +/- 0.407 (3/3) | 69.437 +/- 0.829 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 70% | 55.187 +/- 0.785 (3/3) | 64.590 +/- 0.302 (3/3) | 2.251 +/- 0.021 (3/3) | 50.917 +/- 0.274 (3/3) | 34.471 +/- 0.273 (3/3) | 42.479 +/- 0.150 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 70% | 53.700 +/- 0.462 (3/3) | 65.915 +/- 0.322 (3/3) | 2.394 +/- 0.038 (3/3) | 49.163 +/- 0.482 (3/3) | 33.086 +/- 0.110 (3/3) | 41.241 +/- 0.175 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 70% | 27.763 +/- 1.062 (3/3) | 77.051 +/- 0.942 (3/3) | 6.834 +/- 0.100 (3/3) | 26.244 +/- 1.366 (3/3) | 23.720 +/- 0.592 (3/3) | 23.067 +/- 1.007 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 70% | 61.347 +/- 0.453 (3/3) | 58.566 +/- 0.683 (3/3) | 2.160 +/- 0.068 (3/3) | 56.482 +/- 0.614 (3/3) | 40.642 +/- 0.413 (3/3) | 49.743 +/- 0.592 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 70% | 61.560 +/- 0.155 (3/3) | 59.046 +/- 0.393 (3/3) | 2.117 +/- 0.047 (3/3) | 56.903 +/- 0.203 (3/3) | 39.097 +/- 0.065 (3/3) | 48.947 +/- 0.477 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 70% | 63.530 +/- 0.858 (3/3) | 58.194 +/- 0.393 (3/3) | 3.090 +/- 0.116 (3/3) | 57.835 +/- 1.067 (3/3) | 39.329 +/- 0.985 (3/3) | 52.135 +/- 0.484 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 70% | 62.323 +/- 0.950 (3/3) | 58.623 +/- 0.743 (3/3) | 3.184 +/- 0.136 (3/3) | 57.108 +/- 1.460 (3/3) | 38.391 +/- 0.975 (3/3) | 51.350 +/- 0.475 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 70% | 43.037 +/- 1.191 (3/3) | 72.142 +/- 0.435 (3/3) | 5.247 +/- 0.134 (3/3) | 41.015 +/- 0.687 (3/3) | 23.889 +/- 0.945 (3/3) | 30.723 +/- 1.460 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 70% | 39.940 +/- 1.448 (3/3) | 73.889 +/- 0.418 (3/3) | 5.521 +/- 0.237 (3/3) | 37.287 +/- 1.516 (3/3) | 23.745 +/- 0.780 (3/3) | 29.695 +/- 0.390 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 70% | 26.530 +/- 0.717 (3/3) | 76.932 +/- 0.628 (3/3) | 8.095 +/- 0.568 (3/3) | 23.365 +/- 0.817 (3/3) | 21.982 +/- 0.463 (3/3) | 21.931 +/- 0.876 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 70% | 46.830 +/- 1.187 (3/3) | 67.533 +/- 0.727 (3/3) | 4.901 +/- 0.182 (3/3) | 42.789 +/- 1.016 (3/3) | 29.461 +/- 0.990 (3/3) | 36.191 +/- 1.322 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 70% | 44.513 +/- 1.662 (3/3) | 69.276 +/- 0.526 (3/3) | 5.068 +/- 0.352 (3/3) | 40.332 +/- 2.122 (3/3) | 28.871 +/- 0.310 (3/3) | 35.381 +/- 0.229 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 70% | 98.910 +/- 0.174 (3/3) | 33.409 +/- 0.718 (3/3) | 0.048 +/- 0.007 (3/3) | 27.367 +/- 5.168 (3/3) | not in MNIST-C | 27.243 +/- 6.497 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 70% | 89.860 +/- 2.967 (3/3) | 48.134 +/- 1.427 (3/3) | 0.434 +/- 0.143 (3/3) | 33.830 +/- 2.841 (3/3) | not in MNIST-C | 32.363 +/- 4.756 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 70% | 87.060 +/- 0.960 (3/3) | 42.050 +/- 1.520 (3/3) | 0.540 +/- 0.026 (3/3) | 63.920 +/- 1.409 (3/3) | not in MNIST-C | 61.067 +/- 1.797 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 70% | 97.920 +/- 0.165 (3/3) | 40.776 +/- 1.074 (3/3) | 0.119 +/- 0.004 (3/3) | 71.007 +/- 2.768 (3/3) | not in MNIST-C | 38.820 +/- 8.217 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 70% | 44.210 +/- 2.220 (3/3) | 78.738 +/- 0.885 (3/3) | 4.641 +/- 0.154 (3/3) | 17.560 +/- 1.795 (3/3) | not in MNIST-C | 9.470 +/- 1.315 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 70% | 68.290 +/- 8.758 (3/3) | 56.049 +/- 4.228 (3/3) | 2.477 +/- 0.835 (3/3) | 49.983 +/- 8.300 (3/3) | not in MNIST-C | 44.173 +/- 6.607 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 80% | 68.930 +/- 0.515 (3/3) | 54.914 +/- 0.321 (3/3) | 1.713 +/- 0.064 (3/3) | 66.170 +/- 0.527 (3/3) | 48.210 +/- 1.027 (3/3) | 60.501 +/- 1.203 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 80% | 67.927 +/- 0.483 (3/3) | 55.924 +/- 0.773 (3/3) | 1.773 +/- 0.066 (3/3) | 65.509 +/- 0.620 (3/3) | 45.929 +/- 0.535 (3/3) | 58.606 +/- 1.361 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 80% | 35.560 +/- 0.416 (3/3) | 74.835 +/- 0.206 (3/3) | 3.924 +/- 0.053 (3/3) | 31.393 +/- 0.098 (3/3) | 21.457 +/- 0.379 (3/3) | 26.115 +/- 0.700 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 80% | 33.280 +/- 0.731 (3/3) | 76.651 +/- 0.195 (3/3) | 4.262 +/- 0.104 (3/3) | 28.808 +/- 0.897 (3/3) | 19.734 +/- 0.475 (3/3) | 24.212 +/- 0.787 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 80% | 23.093 +/- 1.242 (3/3) | 80.138 +/- 0.806 (3/3) | 7.770 +/- 0.137 (3/3) | 22.615 +/- 1.079 (3/3) | 21.128 +/- 0.649 (3/3) | 19.947 +/- 0.836 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 80% | 43.177 +/- 1.179 (3/3) | 69.753 +/- 0.932 (3/3) | 3.798 +/- 0.233 (3/3) | 38.542 +/- 1.213 (3/3) | 27.885 +/- 1.115 (3/3) | 32.588 +/- 1.304 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 80% | 43.210 +/- 0.745 (3/3) | 69.778 +/- 0.636 (3/3) | 3.643 +/- 0.095 (3/3) | 39.041 +/- 1.012 (3/3) | 26.914 +/- 0.677 (3/3) | 32.507 +/- 0.298 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 80% | 48.850 +/- 0.910 (3/3) | 67.740 +/- 0.252 (3/3) | 4.677 +/- 0.167 (3/3) | 44.316 +/- 1.077 (3/3) | 32.210 +/- 1.169 (3/3) | 39.061 +/- 0.441 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 80% | 48.210 +/- 0.858 (3/3) | 67.950 +/- 1.322 (3/3) | 4.743 +/- 0.294 (3/3) | 44.091 +/- 0.859 (3/3) | 31.250 +/- 1.370 (3/3) | 38.288 +/- 1.332 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 80% | 27.040 +/- 0.828 (3/3) | 79.871 +/- 0.400 (3/3) | 7.559 +/- 0.193 (3/3) | 25.250 +/- 0.636 (3/3) | 19.139 +/- 0.318 (3/3) | 22.144 +/- 0.610 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 80% | 24.353 +/- 0.987 (3/3) | 81.533 +/- 0.476 (3/3) | 7.835 +/- 0.289 (3/3) | 22.803 +/- 1.068 (3/3) | 17.927 +/- 0.254 (3/3) | 20.512 +/- 0.589 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 80% | 22.697 +/- 0.807 (3/3) | 79.642 +/- 0.666 (3/3) | 8.783 +/- 0.490 (3/3) | 20.864 +/- 0.854 (3/3) | 20.209 +/- 0.180 (3/3) | 20.042 +/- 0.712 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 80% | 32.830 +/- 1.358 (3/3) | 74.826 +/- 0.929 (3/3) | 6.719 +/- 0.248 (3/3) | 29.811 +/- 1.125 (3/3) | 23.663 +/- 0.975 (3/3) | 26.523 +/- 1.466 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 80% | 29.857 +/- 2.015 (3/3) | 76.614 +/- 0.838 (3/3) | 7.033 +/- 0.718 (3/3) | 27.013 +/- 2.014 (3/3) | 23.565 +/- 0.385 (3/3) | 26.402 +/- 0.656 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 80% | 98.323 +/- 0.222 (3/3) | 39.209 +/- 0.494 (3/3) | 0.076 +/- 0.011 (3/3) | 16.270 +/- 5.029 (3/3) | not in MNIST-C | 21.127 +/- 5.006 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 80% | 86.947 +/- 3.206 (3/3) | 55.652 +/- 1.750 (3/3) | 0.575 +/- 0.168 (3/3) | 15.603 +/- 0.722 (3/3) | not in MNIST-C | 21.343 +/- 0.352 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 80% | 86.397 +/- 1.096 (3/3) | 43.421 +/- 2.108 (3/3) | 0.567 +/- 0.033 (3/3) | 80.460 +/- 2.859 (3/3) | not in MNIST-C | 66.960 +/- 3.081 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 80% | 96.547 +/- 0.049 (3/3) | 47.934 +/- 0.866 (3/3) | 0.203 +/- 0.004 (3/3) | 36.077 +/- 1.472 (3/3) | not in MNIST-C | 26.983 +/- 4.990 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 80% | 23.793 +/- 1.976 (3/3) | 85.710 +/- 0.807 (3/3) | 7.639 +/- 0.582 (3/3) | 8.150 +/- 0.445 (3/3) | not in MNIST-C | 6.443 +/- 1.167 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 80% | 62.670 +/- 9.540 (3/3) | 63.733 +/- 4.781 (3/3) | 2.862 +/- 0.935 (3/3) | 45.010 +/- 7.810 (3/3) | not in MNIST-C | 43.237 +/- 8.771 (3/3) |
| cifar10 / swin_tiny | color_opponency / compact | compact / native_contrast / 90% | 45.783 +/- 1.195 (3/3) | 69.458 +/- 0.923 (3/3) | 3.555 +/- 0.211 (3/3) | 43.089 +/- 1.404 (3/3) | 31.073 +/- 1.653 (3/3) | 37.201 +/- 1.259 (3/3) |
| cifar10 / swin_tiny | color_opponency / dense | compact / native_contrast / 90% | 45.243 +/- 1.206 (3/3) | 70.464 +/- 0.809 (3/3) | 3.543 +/- 0.159 (3/3) | 42.883 +/- 0.764 (3/3) | 29.199 +/- 1.551 (3/3) | 34.765 +/- 1.726 (3/3) |
| cifar10 / swin_tiny | grayscale / compact | compact / native_contrast / 90% | 19.033 +/- 0.503 (3/3) | 84.021 +/- 0.574 (3/3) | 6.558 +/- 0.247 (3/3) | 17.129 +/- 0.608 (3/3) | 14.226 +/- 0.998 (3/3) | 14.304 +/- 0.402 (3/3) |
| cifar10 / swin_tiny | grayscale / dense | compact / native_contrast / 90% | 17.450 +/- 0.875 (3/3) | 85.537 +/- 0.362 (3/3) | 6.793 +/- 0.022 (3/3) | 15.165 +/- 0.646 (3/3) | 12.210 +/- 0.385 (3/3) | 13.291 +/- 0.259 (3/3) |
| cifar10 / swin_tiny | raw / dense | compact / raw_pixels / 90% | 18.727 +/- 0.880 (3/3) | 83.200 +/- 0.670 (3/3) | 8.670 +/- 0.166 (3/3) | 19.062 +/- 0.773 (3/3) | 17.847 +/- 0.521 (3/3) | 16.763 +/- 0.602 (3/3) |
| cifar10 / swin_tiny | single_color / compact | compact / native_contrast / 90% | 26.213 +/- 0.376 (3/3) | 79.403 +/- 0.694 (3/3) | 5.781 +/- 0.193 (3/3) | 23.354 +/- 0.864 (3/3) | 18.809 +/- 0.793 (3/3) | 19.117 +/- 0.882 (3/3) |
| cifar10 / swin_tiny | single_color / dense | compact / native_contrast / 90% | 25.420 +/- 0.945 (3/3) | 79.931 +/- 0.955 (3/3) | 5.611 +/- 0.134 (3/3) | 22.648 +/- 1.273 (3/3) | 17.331 +/- 1.768 (3/3) | 19.025 +/- 1.379 (3/3) |
| cifar10 / vit_small | color_opponency / compact | compact / native_contrast / 90% | 30.697 +/- 1.325 (3/3) | 78.247 +/- 0.708 (3/3) | 7.021 +/- 0.390 (3/3) | 28.485 +/- 0.857 (3/3) | 22.346 +/- 1.346 (3/3) | 24.287 +/- 1.865 (3/3) |
| cifar10 / vit_small | color_opponency / dense | compact / native_contrast / 90% | 29.423 +/- 1.530 (3/3) | 79.153 +/- 1.202 (3/3) | 7.240 +/- 0.519 (3/3) | 27.800 +/- 1.927 (3/3) | 20.803 +/- 1.596 (3/3) | 22.965 +/- 1.212 (3/3) |
| cifar10 / vit_small | grayscale / compact | compact / native_contrast / 90% | 14.757 +/- 1.020 (3/3) | 87.330 +/- 0.766 (3/3) | 8.551 +/- 0.766 (3/3) | 13.305 +/- 1.193 (3/3) | 12.359 +/- 0.706 (3/3) | 12.232 +/- 0.580 (3/3) |
| cifar10 / vit_small | grayscale / dense | compact / native_contrast / 90% | 13.753 +/- 0.757 (3/3) | 88.116 +/- 0.341 (3/3) | 8.656 +/- 1.138 (3/3) | 12.653 +/- 0.630 (3/3) | 11.181 +/- 0.544 (3/3) | 11.191 +/- 0.547 (3/3) |
| cifar10 / vit_small | raw / dense | compact / raw_pixels / 90% | 19.123 +/- 0.970 (3/3) | 82.152 +/- 0.705 (3/3) | 9.439 +/- 0.487 (3/3) | 18.373 +/- 0.896 (3/3) | 17.877 +/- 0.472 (3/3) | 18.152 +/- 0.600 (3/3) |
| cifar10 / vit_small | single_color / compact | compact / native_contrast / 90% | 21.207 +/- 0.691 (3/3) | 82.034 +/- 0.808 (3/3) | 7.093 +/- 0.295 (3/3) | 18.697 +/- 0.631 (3/3) | 16.513 +/- 1.776 (3/3) | 17.341 +/- 0.637 (3/3) |
| cifar10 / vit_small | single_color / dense | compact / native_contrast / 90% | 18.337 +/- 1.611 (3/3) | 83.987 +/- 0.654 (3/3) | 7.930 +/- 0.980 (3/3) | 16.263 +/- 1.162 (3/3) | 15.921 +/- 0.424 (3/3) | 15.452 +/- 0.501 (3/3) |
| mnist / swin_tiny | grayscale / compact | compact / native_contrast / 90% | 84.867 +/- 0.675 (3/3) | 51.797 +/- 0.657 (3/3) | 0.866 +/- 0.061 (3/3) | 20.613 +/- 2.081 (3/3) | not in MNIST-C | 14.653 +/- 0.474 (3/3) |
| mnist / swin_tiny | grayscale / dense | compact / native_contrast / 90% | 63.490 +/- 3.097 (3/3) | 65.226 +/- 1.325 (3/3) | 1.994 +/- 0.178 (3/3) | 17.097 +/- 1.097 (3/3) | not in MNIST-C | 13.003 +/- 0.504 (3/3) |
| mnist / swin_tiny | raw / dense | compact / raw_pixels / 90% | 68.040 +/- 3.346 (3/3) | 58.179 +/- 2.545 (3/3) | 1.613 +/- 0.227 (3/3) | 59.313 +/- 3.443 (3/3) | not in MNIST-C | 53.853 +/- 3.353 (3/3) |
| mnist / vit_small | grayscale / compact | compact / native_contrast / 90% | 79.993 +/- 0.540 (3/3) | 58.243 +/- 0.659 (3/3) | 1.411 +/- 0.025 (3/3) | 34.947 +/- 2.519 (3/3) | not in MNIST-C | 15.043 +/- 0.947 (3/3) |
| mnist / vit_small | grayscale / dense | compact / native_contrast / 90% | 10.490 +/- 1.290 (3/3) | 90.220 +/- 0.756 (3/3) | 11.145 +/- 0.168 (3/3) | 7.887 +/- 0.372 (3/3) | not in MNIST-C | 7.970 +/- 0.511 (3/3) |
| mnist / vit_small | raw / dense | compact / raw_pixels / 90% | 29.680 +/- 3.867 (3/3) | 78.498 +/- 2.257 (3/3) | 6.255 +/- 1.089 (3/3) | 24.553 +/- 1.431 (3/3) | not in MNIST-C | 32.557 +/- 8.390 (3/3) |

Values show mean +/- sample SD across the registered training seeds, with observed/expected seed counts. Incomplete means are labeled; exported curves require every registered seed at a point. No missing or failed case is removed from coverage. [All seed results](outputs/tables/accuracy_seeds.csv), [condition means and paired deltas](outputs/tables/accuracy_conditions.csv), [official cell results](outputs/tables/evaluation_cells.csv).

[Per-corruption and severity tables](outputs/tables/corruption_conditions.csv) preserve all official conditions. Brightness and fog are released corruption types, not a claim about all real-world lighting changes. Fixed diagnostic tensors, masks, IDs, logits, and display-scale provenance are stored under each case's `diagnostics/cells/`; evaluation completion requires one validated panel per official cell.

**Original training and 0% references.** [Checkpoint reference table](outputs/tables/original_references.csv) preserves the original final-epoch online training, held-out validation, clean-test, and mCE values. These are reused evidence from `training_sparse_testing_sparse`, not new training measurements. Each new seed row records differences from that checkpoint's original execution and, when available, its new compact 0% anchor. A dense-trained checkpoint evaluated compact at 0% is a new intervention. The original dense 0% result cannot be relabeled as that compact anchor. All-three-seed sign agreement in the CSV is descriptive, not a significance test.

**Measured timing and memory.** [Seed timing table](outputs/tables/timing_seeds.csv), [condition timing table](outputs/tables/timing_conditions.csv), [cell timing/memory table](outputs/tables/timing_cells.csv), [worker coverage](outputs/tables/timing_workers.csv), and [separate profiler attribution](outputs/tables/stages.csv). All four scopes remain separate: GPU-resident raw input to logits; pinned-host raw input to logits; loader input through CPU prediction; and cached-input model-only diagnostics. The cached scope excludes online preprocessing and cannot establish an online speedup. Batch latency is milliseconds per batch; amortized batch latency is not a batch-one request. Compact and dense-masked timings use the same checkpoint, support, and frozen affine normalization.

Balanced corrupted timing pools the same 200 paired repetitions from every official cell. Per-seed quantiles are computed from those raw measurements before averaging seeds. Sustained throughput is total measured images divided by summed observed throughput duration; memory is the maximum observed cell peak. Repetition spread is distinct from sample SD across training seeds. Profiler intervals are inclusive where named and are never stacked into an invented total. Hardware classes, CPU/thread configurations, physical devices, and measurement sessions are retained in tables; different classes are never pooled.

| Dataset / architecture | Hardware / GPU / CPU | Batch / scope / input | Fully paired conditions | Raw dense speedup range over all fully paired conditions |
|---|---|---|---|---|
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / cached_input_model_only_diagnostic / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / cached_input_model_only_diagnostic / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / gpu_raw_to_logits / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / gpu_raw_to_logits / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / host_raw_to_logits / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / host_raw_to_logits / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / loader_to_cpu_prediction / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / loader_to_cpu_prediction / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / cached_input_model_only_diagnostic / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / cached_input_model_only_diagnostic / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / gpu_raw_to_logits / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / gpu_raw_to_logits / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / host_raw_to_logits / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / host_raw_to_logits / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / loader_to_cpu_prediction / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / loader_to_cpu_prediction / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 1 / cached_input_model_only_diagnostic / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 1 / cached_input_model_only_diagnostic / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 1 / gpu_raw_to_logits / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 1 / gpu_raw_to_logits / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 1 / host_raw_to_logits / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 1 / host_raw_to_logits / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 1 / loader_to_cpu_prediction / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 1 / loader_to_cpu_prediction / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 8 / cached_input_model_only_diagnostic / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 8 / cached_input_model_only_diagnostic / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 8 / gpu_raw_to_logits / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 8 / gpu_raw_to_logits / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 8 / host_raw_to_logits / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 8 / host_raw_to_logits / clean | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 8 / loader_to_cpu_prediction / balanced_corruption | 0/70 | incomplete; no paired gain established |
| cifar10 / vit_small | unmeasured / unmeasured / unmeasured | 8 / loader_to_cpu_prediction / clean | 0/70 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / cached_input_model_only_diagnostic / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / cached_input_model_only_diagnostic / clean | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / gpu_raw_to_logits / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / gpu_raw_to_logits / clean | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / host_raw_to_logits / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / host_raw_to_logits / clean | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / loader_to_cpu_prediction / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 1 / loader_to_cpu_prediction / clean | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / cached_input_model_only_diagnostic / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / cached_input_model_only_diagnostic / clean | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / gpu_raw_to_logits / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / gpu_raw_to_logits / clean | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / host_raw_to_logits / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / host_raw_to_logits / clean | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / loader_to_cpu_prediction / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / swin_tiny | unmeasured / unmeasured / unmeasured | 8 / loader_to_cpu_prediction / clean | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 1 / cached_input_model_only_diagnostic / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 1 / cached_input_model_only_diagnostic / clean | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 1 / gpu_raw_to_logits / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 1 / gpu_raw_to_logits / clean | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 1 / host_raw_to_logits / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 1 / host_raw_to_logits / clean | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 1 / loader_to_cpu_prediction / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 1 / loader_to_cpu_prediction / clean | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 8 / cached_input_model_only_diagnostic / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 8 / cached_input_model_only_diagnostic / clean | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 8 / gpu_raw_to_logits / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 8 / gpu_raw_to_logits / clean | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 8 / host_raw_to_logits / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 8 / host_raw_to_logits / clean | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 8 / loader_to_cpu_prediction / balanced_corruption | 0/30 | incomplete; no paired gain established |
| mnist / vit_small | unmeasured / unmeasured / unmeasured | 8 / loader_to_cpu_prediction / clean | 0/30 | incomplete; no paired gain established |

**Token behavior.** [Saved support table](outputs/tables/token_support.csv) separates clean input and complete equally weighted corruption cells. Native coefficient zeros, empty patches, padding, Swin stage activity, and observed latency are distinct quantities. MNIST background sparsity also affects the raw-pixel sweep and prevents attributing empty-token benefits uniquely to contrast.

**Interpretation limits.** This is a follow-up on already observed benchmarks. The checkpoint set and full inference grid are registered before this study executes. No checkpoint, threshold, recipe, representation, or corruption-specific setting may be selected using clean-test or corruption-test scores. All levels and all seeds remain reportable, including worse results. Raw-pixel and native-contrast percentages have different denominators and should not be treated as the same image transformation.

Coverage and failures: [coverage.json](outputs/tables/coverage.json). Invalid evidence records: 0. An invalid artifact is excluded from numerical aggregates and remains an explicit error in coverage.

TensorBoard figures and scalars are under `outputs/tensorboard/study_summary/`. Use `tensorboard --logdir outputs/tensorboard --host 127.0.0.1 --port 6006` in this study root.

Generated figures (each includes PDF, SVG, 300-dpi PNG, and the exact plot-data CSV):
- [clean_accuracy](outputs/figures/cifar10/swin_tiny/clean_accuracy.png)
- [clean_loss](outputs/figures/cifar10/swin_tiny/clean_loss.png)
- [mCE_raw](outputs/figures/cifar10/swin_tiny/mCE_raw.png)
- [mean_corruption_loss](outputs/figures/cifar10/swin_tiny/mean_corruption_loss.png)
- [brightness_accuracy](outputs/figures/cifar10/swin_tiny/brightness_accuracy.png)
- [contrast_accuracy](outputs/figures/cifar10/swin_tiny/contrast_accuracy.png)
- [fog_accuracy](outputs/figures/cifar10/swin_tiny/fog_accuracy.png)
- [delta_clean_accuracy_from_original](outputs/figures/cifar10/swin_tiny/delta_clean_accuracy_from_original.png)
- [delta_mCE_raw_from_original](outputs/figures/cifar10/swin_tiny/delta_mCE_raw_from_original.png)
- [clean_accuracy](outputs/figures/cifar10/vit_small/clean_accuracy.png)
- [clean_loss](outputs/figures/cifar10/vit_small/clean_loss.png)
- [mCE_raw](outputs/figures/cifar10/vit_small/mCE_raw.png)
- [mean_corruption_loss](outputs/figures/cifar10/vit_small/mean_corruption_loss.png)
- [brightness_accuracy](outputs/figures/cifar10/vit_small/brightness_accuracy.png)
- [contrast_accuracy](outputs/figures/cifar10/vit_small/contrast_accuracy.png)
- [fog_accuracy](outputs/figures/cifar10/vit_small/fog_accuracy.png)
- [delta_clean_accuracy_from_original](outputs/figures/cifar10/vit_small/delta_clean_accuracy_from_original.png)
- [delta_mCE_raw_from_original](outputs/figures/cifar10/vit_small/delta_mCE_raw_from_original.png)
- [clean_accuracy](outputs/figures/mnist/swin_tiny/clean_accuracy.png)
- [clean_loss](outputs/figures/mnist/swin_tiny/clean_loss.png)
- [mCE_raw](outputs/figures/mnist/swin_tiny/mCE_raw.png)
- [mean_corruption_loss](outputs/figures/mnist/swin_tiny/mean_corruption_loss.png)
- [brightness_accuracy](outputs/figures/mnist/swin_tiny/brightness_accuracy.png)
- [fog_accuracy](outputs/figures/mnist/swin_tiny/fog_accuracy.png)
- [delta_clean_accuracy_from_original](outputs/figures/mnist/swin_tiny/delta_clean_accuracy_from_original.png)
- [delta_mCE_raw_from_original](outputs/figures/mnist/swin_tiny/delta_mCE_raw_from_original.png)
- [clean_accuracy](outputs/figures/mnist/vit_small/clean_accuracy.png)
- [clean_loss](outputs/figures/mnist/vit_small/clean_loss.png)
- [mCE_raw](outputs/figures/mnist/vit_small/mCE_raw.png)
- [mean_corruption_loss](outputs/figures/mnist/vit_small/mean_corruption_loss.png)
- [brightness_accuracy](outputs/figures/mnist/vit_small/brightness_accuracy.png)
- [fog_accuracy](outputs/figures/mnist/vit_small/fog_accuracy.png)
- [delta_clean_accuracy_from_original](outputs/figures/mnist/vit_small/delta_clean_accuracy_from_original.png)
- [delta_mCE_raw_from_original](outputs/figures/mnist/vit_small/delta_mCE_raw_from_original.png)
- [achieved_zero_fraction](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/achieved_zero_fraction.png)
- [achieved_zero_fraction](outputs/figures/cifar10/swin_tiny/support/clean/achieved_zero_fraction.png)
- [natural_zero_fraction](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/natural_zero_fraction.png)
- [natural_zero_fraction](outputs/figures/cifar10/swin_tiny/support/clean/natural_zero_fraction.png)
- [retained_patches](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/retained_patches.png)
- [retained_patches](outputs/figures/cifar10/swin_tiny/support/clean/retained_patches.png)
- [spatial_zero_fraction](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/spatial_zero_fraction.png)
- [spatial_zero_fraction](outputs/figures/cifar10/swin_tiny/support/clean/spatial_zero_fraction.png)
- [stages_0_blocks_0_active_per_image](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/stages_0_blocks_0_active_per_image.png)
- [stages_0_blocks_0_active_per_image](outputs/figures/cifar10/swin_tiny/support/clean/stages_0_blocks_0_active_per_image.png)
- [stages_0_blocks_0_padding_rows](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/stages_0_blocks_0_padding_rows.png)
- [stages_0_blocks_0_padding_rows](outputs/figures/cifar10/swin_tiny/support/clean/stages_0_blocks_0_padding_rows.png)
- [stages_1_blocks_0_active_per_image](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/stages_1_blocks_0_active_per_image.png)
- [stages_1_blocks_0_active_per_image](outputs/figures/cifar10/swin_tiny/support/clean/stages_1_blocks_0_active_per_image.png)
- [stages_1_blocks_0_padding_rows](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/stages_1_blocks_0_padding_rows.png)
- [stages_1_blocks_0_padding_rows](outputs/figures/cifar10/swin_tiny/support/clean/stages_1_blocks_0_padding_rows.png)
- [stages_2_blocks_0_active_per_image](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/stages_2_blocks_0_active_per_image.png)
- [stages_2_blocks_0_active_per_image](outputs/figures/cifar10/swin_tiny/support/clean/stages_2_blocks_0_active_per_image.png)
- [stages_2_blocks_0_padding_rows](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/stages_2_blocks_0_padding_rows.png)
- [stages_2_blocks_0_padding_rows](outputs/figures/cifar10/swin_tiny/support/clean/stages_2_blocks_0_padding_rows.png)
- [stages_3_blocks_0_active_per_image](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/stages_3_blocks_0_active_per_image.png)
- [stages_3_blocks_0_active_per_image](outputs/figures/cifar10/swin_tiny/support/clean/stages_3_blocks_0_active_per_image.png)
- [stages_3_blocks_0_padding_rows](outputs/figures/cifar10/swin_tiny/support/balanced_corruption/stages_3_blocks_0_padding_rows.png)
- [stages_3_blocks_0_padding_rows](outputs/figures/cifar10/swin_tiny/support/clean/stages_3_blocks_0_padding_rows.png)
- [per_corruption_error](outputs/figures/cifar10/swin_tiny/per_corruption_error.png)
- [corruption_severity](outputs/figures/cifar10/swin_tiny/corruption_severity.png)
- [achieved_zero_fraction](outputs/figures/cifar10/vit_small/support/balanced_corruption/achieved_zero_fraction.png)
- [achieved_zero_fraction](outputs/figures/cifar10/vit_small/support/clean/achieved_zero_fraction.png)
- [natural_zero_fraction](outputs/figures/cifar10/vit_small/support/balanced_corruption/natural_zero_fraction.png)
- [natural_zero_fraction](outputs/figures/cifar10/vit_small/support/clean/natural_zero_fraction.png)
- [padding_tokens](outputs/figures/cifar10/vit_small/support/balanced_corruption/padding_tokens.png)
- [padding_tokens](outputs/figures/cifar10/vit_small/support/clean/padding_tokens.png)
- [retained_patches](outputs/figures/cifar10/vit_small/support/balanced_corruption/retained_patches.png)
- [retained_patches](outputs/figures/cifar10/vit_small/support/clean/retained_patches.png)
- [spatial_zero_fraction](outputs/figures/cifar10/vit_small/support/balanced_corruption/spatial_zero_fraction.png)
- [spatial_zero_fraction](outputs/figures/cifar10/vit_small/support/clean/spatial_zero_fraction.png)
- [per_corruption_error](outputs/figures/cifar10/vit_small/per_corruption_error.png)
- [corruption_severity](outputs/figures/cifar10/vit_small/corruption_severity.png)
- [achieved_zero_fraction](outputs/figures/mnist/swin_tiny/support/balanced_corruption/achieved_zero_fraction.png)
- [achieved_zero_fraction](outputs/figures/mnist/swin_tiny/support/clean/achieved_zero_fraction.png)
- [natural_zero_fraction](outputs/figures/mnist/swin_tiny/support/balanced_corruption/natural_zero_fraction.png)
- [natural_zero_fraction](outputs/figures/mnist/swin_tiny/support/clean/natural_zero_fraction.png)
- [retained_patches](outputs/figures/mnist/swin_tiny/support/balanced_corruption/retained_patches.png)
- [retained_patches](outputs/figures/mnist/swin_tiny/support/clean/retained_patches.png)
- [spatial_zero_fraction](outputs/figures/mnist/swin_tiny/support/balanced_corruption/spatial_zero_fraction.png)
- [spatial_zero_fraction](outputs/figures/mnist/swin_tiny/support/clean/spatial_zero_fraction.png)
- [stages_0_blocks_0_active_per_image](outputs/figures/mnist/swin_tiny/support/balanced_corruption/stages_0_blocks_0_active_per_image.png)
- [stages_0_blocks_0_active_per_image](outputs/figures/mnist/swin_tiny/support/clean/stages_0_blocks_0_active_per_image.png)
- [stages_0_blocks_0_padding_rows](outputs/figures/mnist/swin_tiny/support/balanced_corruption/stages_0_blocks_0_padding_rows.png)
- [stages_0_blocks_0_padding_rows](outputs/figures/mnist/swin_tiny/support/clean/stages_0_blocks_0_padding_rows.png)
- [stages_1_blocks_0_active_per_image](outputs/figures/mnist/swin_tiny/support/balanced_corruption/stages_1_blocks_0_active_per_image.png)
- [stages_1_blocks_0_active_per_image](outputs/figures/mnist/swin_tiny/support/clean/stages_1_blocks_0_active_per_image.png)
- [stages_1_blocks_0_padding_rows](outputs/figures/mnist/swin_tiny/support/balanced_corruption/stages_1_blocks_0_padding_rows.png)
- [stages_1_blocks_0_padding_rows](outputs/figures/mnist/swin_tiny/support/clean/stages_1_blocks_0_padding_rows.png)
- [stages_2_blocks_0_active_per_image](outputs/figures/mnist/swin_tiny/support/balanced_corruption/stages_2_blocks_0_active_per_image.png)
- [stages_2_blocks_0_active_per_image](outputs/figures/mnist/swin_tiny/support/clean/stages_2_blocks_0_active_per_image.png)
- [stages_2_blocks_0_padding_rows](outputs/figures/mnist/swin_tiny/support/balanced_corruption/stages_2_blocks_0_padding_rows.png)
- [stages_2_blocks_0_padding_rows](outputs/figures/mnist/swin_tiny/support/clean/stages_2_blocks_0_padding_rows.png)
- [stages_3_blocks_0_active_per_image](outputs/figures/mnist/swin_tiny/support/balanced_corruption/stages_3_blocks_0_active_per_image.png)
- [stages_3_blocks_0_active_per_image](outputs/figures/mnist/swin_tiny/support/clean/stages_3_blocks_0_active_per_image.png)
- [stages_3_blocks_0_padding_rows](outputs/figures/mnist/swin_tiny/support/balanced_corruption/stages_3_blocks_0_padding_rows.png)
- [stages_3_blocks_0_padding_rows](outputs/figures/mnist/swin_tiny/support/clean/stages_3_blocks_0_padding_rows.png)
- [per_corruption_error](outputs/figures/mnist/swin_tiny/per_corruption_error.png)
- [achieved_zero_fraction](outputs/figures/mnist/vit_small/support/balanced_corruption/achieved_zero_fraction.png)
- [achieved_zero_fraction](outputs/figures/mnist/vit_small/support/clean/achieved_zero_fraction.png)
- [natural_zero_fraction](outputs/figures/mnist/vit_small/support/balanced_corruption/natural_zero_fraction.png)
- [natural_zero_fraction](outputs/figures/mnist/vit_small/support/clean/natural_zero_fraction.png)
- [padding_tokens](outputs/figures/mnist/vit_small/support/balanced_corruption/padding_tokens.png)
- [padding_tokens](outputs/figures/mnist/vit_small/support/clean/padding_tokens.png)
- [retained_patches](outputs/figures/mnist/vit_small/support/balanced_corruption/retained_patches.png)
- [retained_patches](outputs/figures/mnist/vit_small/support/clean/retained_patches.png)
- [spatial_zero_fraction](outputs/figures/mnist/vit_small/support/balanced_corruption/spatial_zero_fraction.png)
- [spatial_zero_fraction](outputs/figures/mnist/vit_small/support/clean/spatial_zero_fraction.png)
- [per_corruption_error](outputs/figures/mnist/vit_small/per_corruption_error.png)

Original frozen training/validation learning curves (reused references; no new training):
- [cifar10/swin_tiny/raw/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_raw_dense_pNone_loss.png)
- [cifar10/swin_tiny/raw/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_raw_dense_pNone_accuracy.png)
- [cifar10/vit_small/raw/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_raw_dense_pNone_loss.png)
- [cifar10/vit_small/raw/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_raw_dense_pNone_accuracy.png)
- [mnist/swin_tiny/raw/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/swin_tiny/learning_raw_dense_pNone_loss.png)
- [mnist/swin_tiny/raw/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/swin_tiny/learning_raw_dense_pNone_accuracy.png)
- [mnist/vit_small/raw/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/vit_small/learning_raw_dense_pNone_loss.png)
- [mnist/vit_small/raw/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/vit_small/learning_raw_dense_pNone_accuracy.png)
- [cifar10/swin_tiny/color_opponency/compact: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p0_loss.png)
- [cifar10/swin_tiny/color_opponency/compact: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_color_opponency_compact_p0_accuracy.png)
- [cifar10/swin_tiny/color_opponency/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_color_opponency_dense_p0_loss.png)
- [cifar10/swin_tiny/color_opponency/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_color_opponency_dense_p0_accuracy.png)
- [cifar10/swin_tiny/grayscale/compact: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p0_loss.png)
- [cifar10/swin_tiny/grayscale/compact: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_grayscale_compact_p0_accuracy.png)
- [cifar10/swin_tiny/grayscale/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_grayscale_dense_p0_loss.png)
- [cifar10/swin_tiny/grayscale/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_grayscale_dense_p0_accuracy.png)
- [cifar10/swin_tiny/single_color/compact: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p0_loss.png)
- [cifar10/swin_tiny/single_color/compact: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_single_color_compact_p0_accuracy.png)
- [cifar10/swin_tiny/single_color/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_single_color_dense_p0_loss.png)
- [cifar10/swin_tiny/single_color/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/swin_tiny/learning_single_color_dense_p0_accuracy.png)
- [cifar10/vit_small/color_opponency/compact: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p0_loss.png)
- [cifar10/vit_small/color_opponency/compact: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_color_opponency_compact_p0_accuracy.png)
- [cifar10/vit_small/color_opponency/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_color_opponency_dense_p0_loss.png)
- [cifar10/vit_small/color_opponency/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_color_opponency_dense_p0_accuracy.png)
- [cifar10/vit_small/grayscale/compact: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_grayscale_compact_p0_loss.png)
- [cifar10/vit_small/grayscale/compact: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_grayscale_compact_p0_accuracy.png)
- [cifar10/vit_small/grayscale/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_grayscale_dense_p0_loss.png)
- [cifar10/vit_small/grayscale/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_grayscale_dense_p0_accuracy.png)
- [cifar10/vit_small/single_color/compact: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_single_color_compact_p0_loss.png)
- [cifar10/vit_small/single_color/compact: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_single_color_compact_p0_accuracy.png)
- [cifar10/vit_small/single_color/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_single_color_dense_p0_loss.png)
- [cifar10/vit_small/single_color/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/cifar10/vit_small/learning_single_color_dense_p0_accuracy.png)
- [mnist/swin_tiny/grayscale/compact: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p0_loss.png)
- [mnist/swin_tiny/grayscale/compact: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/swin_tiny/learning_grayscale_compact_p0_accuracy.png)
- [mnist/swin_tiny/grayscale/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/swin_tiny/learning_grayscale_dense_p0_loss.png)
- [mnist/swin_tiny/grayscale/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/swin_tiny/learning_grayscale_dense_p0_accuracy.png)
- [mnist/vit_small/grayscale/compact: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/vit_small/learning_grayscale_compact_p0_loss.png)
- [mnist/vit_small/grayscale/compact: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/vit_small/learning_grayscale_compact_p0_accuracy.png)
- [mnist/vit_small/grayscale/dense: loss](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/vit_small/learning_grayscale_dense_p0_loss.png)
- [mnist/vit_small/grayscale/dense: accuracy](/ceph/sagnihot/projects/Lorena_Contrastive_Representation_Arch/sparse_contrast_benchmark_large_batch/outputs/figures/mnist/vit_small/learning_grayscale_dense_p0_accuracy.png)
