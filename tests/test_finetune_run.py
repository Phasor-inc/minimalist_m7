import torch

from code.finetune_run import compute_total_loss
from code.m7_memory_core import M7MemoryCore
from code.pooled_hidden_adapter import PooledHiddenAdapter


def test_compute_total_loss_lora_only_mode_excludes_consolidation_term():
    xiaomi_loss = torch.tensor(0.5, requires_grad=True)
    total, breakdown = compute_total_loss(
        xiaomi_loss=xiaomi_loss,
        pooled_hidden=None,
        adapter=None,
        memory_core=None,
        use_m7_consolidation=False,
    )
    assert torch.allclose(total, xiaomi_loss)
    assert "consolidation" not in breakdown


def test_compute_total_loss_with_m7_consolidation_adds_real_term():
    torch.manual_seed(0)
    xiaomi_loss = torch.tensor(0.5, requires_grad=True)
    adapter = PooledHiddenAdapter(vlm_hidden_size=16, target_dim=8)
    memory_core = M7MemoryCore(input_size=8, hidden_size=8, memory_size=4)
    pooled_hidden = torch.randn(2, 16)

    # prime the memory core so consolidation_bias is non-zero BEFORE the
    # call under test, so this test exercises the "prior bias exists"
    # branch rather than the "very first call, bias is genuinely zero" edge
    projected = adapter(pooled_hidden)
    h_prev = torch.zeros(2, 8)
    memory_core.train()
    memory_core(projected, h_prev)

    total, breakdown = compute_total_loss(
        xiaomi_loss=xiaomi_loss,
        pooled_hidden=pooled_hidden,
        adapter=adapter,
        memory_core=memory_core,
        use_m7_consolidation=True,
        consolidation_weight=0.1,
    )
    assert "consolidation" in breakdown
    assert torch.isfinite(total)
    assert total.item() != xiaomi_loss.item()  # real term was added
