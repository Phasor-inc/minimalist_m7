import pytest
import torch
import torch.nn as nn
from code.lora_adapters import LoRALinear, inject_lora_into_vlm_layers, _replace_module


class DummySelfAttn(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)


class DummyMLP(nn.Module):
    def __init__(self, dim=8, inter=16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter)
        self.up_proj = nn.Linear(dim, inter)
        self.down_proj = nn.Linear(inter, dim)


class DummyDecoderLayer(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.self_attn = DummySelfAttn(dim)
        self.mlp = DummyMLP(dim)


class DummyVLM(nn.Module):
    def __init__(self, num_layers=2, dim=8):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer(dim) for _ in range(num_layers)])


def test_lora_linear_matches_base_at_init():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_linear_base_weights_frozen():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    assert lora.base_linear.weight.requires_grad is False
    assert lora.lora_A.requires_grad is True
    assert lora.lora_B.requires_grad is True


def test_lora_linear_output_shape():
    base = nn.Linear(8, 16)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 8)
    assert lora(x).shape == (3, 16)


def test_inject_lora_into_vlm_layers_replaces_all_target_modules():
    vlm = DummyVLM(num_layers=2)
    inject_lora_into_vlm_layers(vlm, rank=4, alpha=8.0)
    for layer in vlm.layers:
        assert isinstance(layer.self_attn.q_proj, LoRALinear)
        assert isinstance(layer.self_attn.k_proj, LoRALinear)
        assert isinstance(layer.self_attn.v_proj, LoRALinear)
        assert isinstance(layer.self_attn.o_proj, LoRALinear)
        assert isinstance(layer.mlp.gate_proj, LoRALinear)
        assert isinstance(layer.mlp.up_proj, LoRALinear)
        assert isinstance(layer.mlp.down_proj, LoRALinear)


def test_inject_lora_preserves_output_at_init():
    torch.manual_seed(0)
    vlm_a = DummyVLM(num_layers=1)
    vlm_b = DummyVLM(num_layers=1)
    vlm_b.load_state_dict(vlm_a.state_dict())
    inject_lora_into_vlm_layers(vlm_b, rank=4, alpha=8.0)

    x = torch.randn(2, 8)
    out_a = vlm_a.layers[0].self_attn.q_proj(x)
    out_b = vlm_b.layers[0].self_attn.q_proj(x)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_inject_lora_only_target_modules_trainable():
    vlm = DummyVLM(num_layers=1)
    inject_lora_into_vlm_layers(vlm, rank=4, alpha=8.0)
    trainable = [n for n, p in vlm.named_parameters() if p.requires_grad]
    assert all("lora_A" in n or "lora_B" in n for n in trainable)
    assert len(trainable) == 14  # 7 target modules x 2 (lora_A, lora_B) x 1 layer


def test_lora_linear_scaling_arithmetic_with_nonzero_B():
    base = nn.Linear(8, 8)
    rank, alpha = 4, 8.0
    lora = LoRALinear(base, rank=rank, alpha=alpha)

    torch.manual_seed(1)
    lora.lora_B.data.copy_(torch.randn(8, rank))

    x = torch.randn(3, 8)
    base_out = base(x)
    expected = base_out + (alpha / rank) * (x @ lora.lora_A.T @ lora.lora_B.T)
    actual = lora(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_replace_module_raises_type_error_for_non_linear():
    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.not_linear = nn.ReLU()

    holder = Holder()
    with pytest.raises(TypeError):
        _replace_module(holder, "not_linear", rank=4, alpha=8.0)
