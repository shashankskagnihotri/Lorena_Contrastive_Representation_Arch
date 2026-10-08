# Attribution and licensing provenance

The fixed contrast frontend is derived from Shashank Agnihotri's
[`semseg_using_VLMs_Lorena`](https://github.com/shashankskagnihotri/semseg_using_VLMs_Lorena),
file `semseg/utils/blur_preprocessing.py`, commit
`73deca8c15b8d91fee29bac71b6f73c013a47d22`, SHA-256
`f7c4420a3ab2bf4036323b9b4842e66d46225a88ce6f4440101c7a190861dff8`.
The inspected source checkout contained no upstream license file or license
header. This repository preserves that attribution and does not invent or
relicense the upstream material. No blanket open-source license is asserted by
this publication; public availability alone is not a license grant.

The ViT and Swin backbones are the project's explicit native-resolution
implementations of the architecture families described in the original papers:

- [Dosovitskiy et al., An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).
- [Liu et al., Swin Transformer](https://arxiv.org/abs/2103.14030).

The project uses PyTorch, torchvision, timm, NumPy, Matplotlib, Seaborn,
TensorBoard, and other dependencies under their respective licenses. Their
implementation code is not bundled as a replacement for those packages.

MNIST, CIFAR-10, MNIST-C, and CIFAR-10-C retain their original provenance and
distribution terms. Full dataset archives are not included here. Dataset
identities, official sources, split construction, and checksums are recorded in
the experiment documentation. Published example panels are research visualizations.
