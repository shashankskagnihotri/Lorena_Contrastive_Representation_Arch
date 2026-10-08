"""Parity on the actual allocated CUDA backend, before contrast training."""
import pytest
import torch

from sparse_contrast.frontend import Frontend, REPRESENTATIONS, sparsify
from test_frontend import source_reference, _real_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocation required")


@pytest.mark.parametrize("mode", REPRESENTATIONS)
@pytest.mark.parametrize("dataset", ("synthetic", "mnist", "cifar10"))
def test_gpu_original_frontend_exact_parity(mode, dataset):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = torch.rand(4, 3, 12, 10, generator=torch.Generator().manual_seed(45)) if dataset == "synthetic" else _real_inputs(dataset)
    raw = raw.cuda()
    source_input = raw.expand(-1, 3, -1, -1) if raw.shape[1] == 1 else raw
    source = source_reference(mode).cuda()
    frontend = Frontend(mode).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        actual = frontend(raw)
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        expected = source(source_input)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(buffer.device.type == "cuda" for buffer in frontend.buffers())
    assert not list(frontend.parameters())


def test_gpu_sparsifier_stable_native_ties_matches_cpu():
    coeff = torch.tensor([[[[0., 1., -1., 2.], [0., -2., 1., -1.]]]]).expand(3, 3, -1, -1).contiguous()
    for percent in (0, 20, 40, 60, 80):
        torch.testing.assert_close(sparsify(coeff.cuda(), percent).cpu(), sparsify(coeff, percent), rtol=0, atol=0)
