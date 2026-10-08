"""Full native architectures: semantic, gradient, official-reference parity."""
import copy

import pytest
import torch
from torch.nn import functional as F

from sparse_contrast.models import build_backbone, dense_reference, PatchMerging, SwinBlock, _image_drop


@pytest.fixture(autouse=True)
def controlled_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def parameter_reference_name(name, architecture):
    if architecture == "vit_small":
        return name.replace("patch_embed.", "patch_embed.proj.")
    if name.startswith("patch_embed."):
        return name.replace("patch_embed.", "features.0.0.")
    if name.startswith("patch_norm."):
        return name.replace("patch_norm.", "features.0.2.")
    if name.startswith("stages."):
        _, stage, block, rest = name.split(".", 3)
        rest = rest.replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.3.")
        return f"features.{2 * int(stage) + 1}.{block}.{rest}"
    if name.startswith("merges."):
        _, stage, rest = name.split(".", 2)
        return f"features.{2 * int(stage) + 2}.{rest}"
    return name


def check_parameter_gradients(first, second, name_map=lambda name: name, atol=3e-5, rtol=2e-4):
    other = dict(second.named_parameters())
    seen = set()
    for name, value in first.named_parameters():
        reference_name = name_map(name)
        reference = other[reference_name]
        seen.add(reference_name)
        assert value.grad is not None, name
        assert reference.grad is not None, reference_name
        actual = value.grad.flatten()
        expected = reference.grad.flatten()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda message: f"{name}: {message}")
    assert seen == set(other)


@pytest.mark.parametrize("architecture", ["vit_small", "swin_tiny"])
@pytest.mark.parametrize("image_size,in_channels", [(28, 1), (32, 3)])
def test_all_active_official_dense_output(architecture, image_size, in_channels):
    torch.manual_seed(601)
    model = build_backbone(architecture, image_size, in_channels, drop_path=0).eval()
    reference = dense_reference(model).eval()
    image = torch.randn(2, in_channels, image_size, image_size)
    patches = F.unfold(image, 2, stride=2).transpose(1, 2)
    support = torch.ones(patches.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        expected = reference(image)
        for execution in ("dense", "dense_masked", "compact"):
            actual = model.forward_patches(patches, support, execution=execution)
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=3e-5)


@pytest.mark.parametrize("architecture", ["vit_small", "swin_tiny"])
def test_all_active_official_parameter_gradients(architecture):
    torch.manual_seed(731)
    model = build_backbone(architecture, 28, 1, drop_path=0).eval()
    reference = dense_reference(model).eval()
    image = torch.randn(1, 1, 28, 28)
    patches = F.unfold(image, 2, stride=2).transpose(1, 2)
    support = torch.ones(patches.shape[:2], dtype=torch.bool)
    weights = torch.linspace(-1, 1, 10)
    (model.forward_patches(patches, support) * weights).sum().backward()
    (reference(image) * weights).sum().backward()
    check_parameter_gradients(model, reference, lambda name: parameter_reference_name(name, architecture))


@pytest.mark.parametrize("architecture", ["vit_small", "swin_tiny"])
@pytest.mark.parametrize("image_size", [28, 32])
def test_ragged_empty_singleton_forward_and_gradient_parity(architecture, image_size):
    torch.manual_seed(912)
    model = build_backbone(architecture, image_size, 3, drop_path=0).eval()
    # Nonzero biases expose accidental resurrection by norm/projection/merges.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.uniform_(-0.02, 0.02)
    reference = copy.deepcopy(model)
    count = (image_size // 2) ** 2
    patches = torch.randn(3, count, 12)
    support = torch.zeros(3, count, dtype=torch.bool)
    support[1, -1] = True
    support[2] = torch.rand(count) < 0.27
    compact = model.forward_patches(patches, support, execution="compact")
    masked = reference.forward_patches(patches, support, execution="dense_masked")
    torch.testing.assert_close(compact, masked, atol=5e-6, rtol=4e-5)
    weights = torch.randn(3, 10)
    (compact * weights).sum().backward()
    (masked * weights).sum().backward()
    check_parameter_gradients(model, reference)
    assert torch.isfinite(compact).all()
    if architecture == "vit_small":
        info = model.latest_telemetry
        assert info["executed_sequence_length"] == int(support.sum(1).max()) + 1
        assert info["executed_sequence_length"] < count + 1
        assert info["projected_patch_rows"] == int(support.sum())
    else:
        info = model.latest_telemetry["stages"][0]["blocks"][0]
        assert info["executed_rows"] == int(support.sum())
        assert info["padding_rows"] == 0
        assert int(info["executed_attention_pairs"]) < 3 * count * model.windows[0] ** 2


@pytest.mark.parametrize("architecture", ["vit_small", "swin_tiny"])
def test_entire_batch_empty_without_fake_tokens(architecture):
    torch.manual_seed(3)
    model = build_backbone(architecture, 28, 1, drop_path=0).eval()
    patches = torch.randn(2, 196, 4)
    support = torch.zeros(2, 196, dtype=torch.bool)
    logits = model.forward_patches(patches, support)
    expected = model.forward_patches(patches, support, execution="dense_masked")
    torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(logits).all()
    model.forward_patches(patches, support).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    if architecture == "swin_tiny":
        torch.testing.assert_close(logits, model.head.bias[None].expand(2, -1))
        assert model.latest_telemetry["stages"][0]["blocks"][0]["executed_rows"] == 0
    else:
        assert model.latest_telemetry["executed_sequence_length"] == 1


@pytest.mark.parametrize("architecture", ["vit_small", "swin_tiny"])
def test_no_cross_image_attention_or_position_renumbering(architecture):
    torch.manual_seed(11)
    model = build_backbone(architecture, 28, 1, drop_path=0).eval()
    patches = torch.randn(2, 196, 4)
    support = torch.zeros(2, 196, dtype=torch.bool)
    support[0, [0, 5, 27, 107, 195]] = True
    support[1, [3, 81]] = True
    with torch.no_grad():
        together = model.forward_patches(patches, support)
        alone = torch.cat([model.forward_patches(patches[i:i + 1], support[i:i + 1]) for i in range(2)])
    torch.testing.assert_close(together, alone, atol=5e-6, rtol=4e-5)


def test_odd_merge_parent_support_and_child_order():
    merge = PatchMerging(dim=2, grid=7).double()
    support = torch.zeros((2, 49), dtype=torch.bool)
    support[0, [0, 1, 7, 8, 48]] = True
    support[1, [6, 42]] = True
    ids = support.flatten().nonzero(as_tuple=True)[0]
    values = torch.arange(ids.numel() * 2, dtype=torch.float64).reshape(-1, 2) + 1
    dense = torch.zeros((98, 2), dtype=torch.float64).index_copy(0, ids, values).reshape(2, 49, 2)
    captured = []
    handle = merge.norm.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    compact, parents = merge.compact(values, ids, 2)
    handle.remove()
    expected, parent_support = merge.dense(dense, support)
    assert parents.tolist() == parent_support.flatten().nonzero(as_tuple=True)[0].tolist()
    torch.testing.assert_close(compact, expected.flatten(0, 1)[parents], atol=0, rtol=0)
    # Sorted children source rows are TL, TR, BL, BR; concatenation swaps TR/BL.
    torch.testing.assert_close(captured[0][0], values[[0, 2, 1, 3]].flatten())
    assert torch.equal(captured[0][1, 2:], torch.zeros(6, dtype=torch.float64))
    assert parents[1] == 15  # bottom-right odd-grid parent exists, no padded peers


def test_shifted_cyclic_boundary_does_not_connect_wrapped_edges():
    torch.manual_seed(5)
    block = SwinBlock(dim=12, heads=3, grid=8, window=4, shift=2, drop_path=0).double().eval()
    # These original corners enter the same shifted window but distinct regions.
    ids = torch.tensor([0, 63])
    assert block.window_ids[ids[0]] == block.window_ids[ids[1]]
    assert block.shift_regions[ids[0]] != block.shift_regions[ids[1]]
    values = torch.randn(2, 12, dtype=torch.float64)
    changed = values.clone()
    changed[1] += torch.arange(12, dtype=torch.float64) * 10
    with torch.no_grad():
        first = block.compact(values, ids, 1)
        second = block.compact(changed, ids, 1)
    torch.testing.assert_close(first[0], second[0], atol=1e-12, rtol=1e-12)


def test_stochastic_depth_mask_tracks_original_image_not_windows():
    ids = torch.tensor([0, 0, 0, 2, 2, 3])
    x = torch.ones(6, 5)
    torch.manual_seed(9)
    result = _image_drop(x, ids, 4, 0.4, True)
    torch.manual_seed(9)
    expected = torch.empty(4).bernoulli_(0.6).div_(0.6)
    torch.testing.assert_close(result, expected[ids, None].expand_as(x))


def test_original_positions_survive_vit_compaction():
    torch.manual_seed(51)
    model = build_backbone("vit_small", 28, 1, drop_path=0).eval()
    patches = torch.zeros(1, 196, 4)
    support = torch.zeros(1, 196, dtype=torch.bool)
    support[0, [17, 105, 195]] = True
    captured = []
    handle = model.blocks[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        model.forward_patches(patches, support)
    handle.remove()
    expected = model.patch_embed(patches[:, [17, 105, 195]]) + model.pos_embed[:, [18, 106, 196]]
    torch.testing.assert_close(captured[0][:, 1:], expected)
