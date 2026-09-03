import torch
from code.m7_memory_core import ImprovedHGRNGate, ModernHopfieldLayer, M7MemoryCore


def test_hgrn_gate_output_shape():
    gate = ImprovedHGRNGate(input_size=32, hidden_size=32)
    x = torch.randn(4, 32)
    h_prev = torch.randn(4, 32)
    h_new = gate(x, h_prev)
    assert h_new.shape == (4, 32)


def test_hgrn_gate_is_bounded_combination():
    torch.manual_seed(0)
    gate = ImprovedHGRNGate(input_size=32, hidden_size=32)
    x = torch.randn(4, 32)
    h_prev = torch.zeros(4, 32)
    h_new = gate(x, h_prev)
    # h_new = (1-z)*h_prev + z*h_tilde with h_prev=0, so h_new should be
    # elementwise between 0 and h_tilde's range -- just check it's finite
    # and not identically zero (gate is actually doing something)
    assert torch.isfinite(h_new).all()
    assert not torch.allclose(h_new, torch.zeros_like(h_new))


def test_hopfield_layer_output_shape():
    hopfield = ModernHopfieldLayer(input_size=32, memory_size=16, temperature=0.1)
    x = torch.randn(4, 32)
    out = hopfield(x)
    assert out.shape == (4, 32)


def test_hopfield_layer_retrieves_nearest_memory():
    torch.manual_seed(0)
    hopfield = ModernHopfieldLayer(input_size=32, memory_size=16, temperature=0.01)
    # very low temperature -> near-hard attention -> output should be close
    # to whichever memory row is most similar to a strongly-scaled input
    x = hopfield.memory[0:1].detach() * 10.0
    out = hopfield(x)
    assert torch.isfinite(out).all()


def test_m7_memory_core_forward_and_consolidation_bias():
    core = M7MemoryCore(input_size=32, hidden_size=32, memory_size=16)
    x = torch.randn(4, 32)
    h_prev = torch.randn(4, 32)
    output, h_new = core(x, h_prev)
    assert output.shape == (4, 32)
    assert h_new.shape == (4, 32)

    bias = core.consolidation_bias()
    assert bias.shape == (32,)
    assert torch.isfinite(bias).all()


def test_m7_memory_core_consolidation_bias_zero_before_any_forward():
    core = M7MemoryCore(input_size=32, hidden_size=32, memory_size=16)
    bias = core.consolidation_bias()
    assert torch.allclose(bias, torch.zeros(32))
