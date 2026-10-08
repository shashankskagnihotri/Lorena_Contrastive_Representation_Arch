from __future__ import annotations

import ast
from contextlib import redirect_stdout
import gzip
import hashlib
import io
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from sparse_contrast.frontend import Frontend, REPRESENTATIONS, SOURCE_SHA256, patch_support, patchify, sparsify


def source_reference(mode: str, depth: int = 1):
    """Execute the actual captured source definitions without application imports.

    Only the documented native-output boundary replaces the grayscale model
    adapter with Identity. No contrast math is reconstructed for the oracle.
    """
    path = Path(__file__).resolve().parents[1] / "provenance/original_frontend.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256
    parsed = ast.parse(path.read_text())
    names = {"calculate_weight", "create_blur_kernel", "BlurPreprocessing"}
    definitions = ast.Module(body=[node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                                  and node.name in names], type_ignores=[])
    namespace = {"np": np, "torch": torch, "nn": nn, "tensor": torch.tensor}
    exec(compile(definitions, str(path), "exec"), namespace)
    with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
        reference = namespace["BlurPreprocessing"](
            blur_bool=True, blur_depth=depth, single_color=mode == "single_color",
            color_opponency=mode == "color_opponency", channels=1 if mode == "grayscale" else 3,
            path="unused", training=False, black_white=mode == "grayscale", normalize=False,
            sparsity_threshold=0, sparsity_type="threshold", change_range=False)
    reference.change_channel_layer = nn.Identity()
    return reference.eval()


@pytest.mark.parametrize("mode", REPRESENTATIONS)
@pytest.mark.parametrize("depth", (1, 3))
def test_original_frontend_synthetic_parity(mode, depth):
    generator = torch.Generator().manual_seed(41)
    raw = torch.rand(5, 3, 12, 10, generator=generator)
    raw[0] = 0
    raw[1] = 1
    raw[2] = 0
    raw[2, :, 6, 4] = 1
    frontend = Frontend(mode, blur_depth=depth)
    expected = source_reference(mode, depth)(raw)
    actual = frontend(raw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not list(frontend.parameters())
    assert actual.min() < 0 and actual.max() > 0


def _real_inputs(dataset):
    root = Path("/ceph/sagnihot/datasets")
    if dataset == "mnist":
        raw = root / "mnist/MNIST/raw/train-images-idx3-ubyte"
        if raw.exists():
            with raw.open("rb") as stream:
                header = stream.read(16)
                buffer = stream.read(4 * 28 * 28)
        elif raw.with_suffix(".gz").exists():
            with gzip.open(str(raw) + ".gz", "rb") as stream:
                header = stream.read(16)
                buffer = stream.read(4 * 28 * 28)
        else:
            pytest.skip("Real MNIST unavailable for parity")
        assert int.from_bytes(header[:4], "big") == 2051
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(4, 1, 28, 28).copy()
    else:
        path = root / "cifar-10-batches-py/data_batch_1"
        if not path.exists():
            pytest.skip("Real CIFAR10 unavailable for parity")
        with path.open("rb") as stream:
            payload = pickle.load(stream, encoding="bytes")
        image = payload[b"data"][:4].reshape(4, 3, 32, 32).copy()
    return torch.from_numpy(image).float() / 255


@pytest.mark.parametrize("mode", REPRESENTATIONS)
@pytest.mark.parametrize("dataset", ("mnist", "cifar10"))
def test_original_frontend_real_parity(mode, dataset):
    raw = _real_inputs(dataset)
    source_input = raw.expand(-1, 3, -1, -1) if raw.shape[1] == 1 else raw
    torch.testing.assert_close(Frontend(mode)(raw), source_reference(mode)(source_input), rtol=0, atol=0)


def test_bw_source_is_equal_rgb_surround_not_bt601():
    raw = torch.zeros(1, 3, 4, 4)
    raw[0, 0, 1, 1] = 1
    actual = Frontend("grayscale")(raw)
    reference = source_reference("grayscale")(raw)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual[0, 0, 1, 1], torch.tensor((1 - 1 / 9) / 3))


def test_stable_native_threshold_ties_signs_and_per_image():
    coeff = torch.tensor([[[[1., -1., 2., -2.]]], [[[7., -7., 8., -8.]]]])
    expected = torch.tensor([[[[0., 0., 2., -2.]]], [[[0., 0., 8., -8.]]]])
    torch.testing.assert_close(sparsify(coeff, 50), expected, rtol=0, atol=0)
    assert coeff[0, 0, 0, 0] == 1  # no in-place corruption


def test_zero_percent_bypasses_sort(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("0% called ranking")
    monkeypatch.setattr(torch, "argsort", forbidden)
    coeff = torch.randn(2, 3, 4, 4)
    assert sparsify(coeff, 0) is coeff


def test_existing_zeros_count_and_native_channels_not_copies():
    coeff = torch.tensor([[[[0., 0., 1., -2.]], [[0., 0., 0., 0.]], [[0., 0., 0., 0.]]]])
    assert torch.equal(sparsify(coeff, 80), coeff)  # floor(.8*12)=9, ten natural zeros
    assert (sparsify(coeff, 95) != 0).sum() == 1  # all three native channels count
    assert torch.equal(sparsify(coeff, 100), torch.zeros_like(coeff))


def test_patch_order_and_support_preserve_raster_coordinates():
    coeff = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    patches = patchify(coeff, 2)
    assert patches[0, 0].tolist() == [0, 1, 4, 5, 16, 17, 20, 21]
    assert patches[0, 1].tolist() == [2, 3, 6, 7, 18, 19, 22, 23]
    coeff.zero_()
    coeff[0, 1, 3, 0] = -0.25
    assert patch_support(coeff).tolist() == [[False, False, True, False]]


def test_float32_under_autocast():
    raw = torch.rand(2, 3, 4, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = Frontend("color_opponency")(raw)
        sparse = sparsify(result, 20)
    assert result.dtype == sparse.dtype == torch.float32


def test_mnist_native_channels_and_opponent_degeneracy():
    raw = _real_inputs("mnist")
    single = Frontend("single_color")(raw)
    torch.testing.assert_close(single[:, 0], single[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(single[:, 1], single[:, 2], rtol=0, atol=0)
    opponent = Frontend("color_opponency")(raw)
    assert opponent[:, 1:].abs().max() < 1e-6
    assert Frontend("grayscale")(raw).shape[1] == 1


def test_invalid_contracts_fail():
    with pytest.raises(TypeError):
        Frontend("grayscale")(torch.zeros(1, 1, 4, 4, dtype=torch.uint8))
    with pytest.raises(ValueError):
        Frontend("grayscale", blur_depth=0)
    with pytest.raises(ValueError):
        patchify(torch.zeros(1, 3, 3, 3))
    with pytest.raises(ValueError):
        sparsify(torch.zeros(1, 3, 2, 2), float("nan"))
