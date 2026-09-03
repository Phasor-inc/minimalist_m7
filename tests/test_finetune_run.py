import os
import random

import pytest
import torch
import torch.nn as nn

from code.finetune_run import (
    _ensure_data_symlink,
    _seed_everything,
    compute_total_loss,
    save_trainable_state,
    verify_trainable_params,
)
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


# ---------------------------------------------------------------------------
# _ensure_data_symlink -- pure filesystem logic, no checkpoint/GPU needed.
# ---------------------------------------------------------------------------


def test_ensure_data_symlink_creates_new_symlink(tmp_path, monkeypatch):
    target = tmp_path / "real_data"
    target.mkdir()
    workdir = tmp_path / "run_from_here"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    _ensure_data_symlink(str(target))

    link = workdir / "data"
    assert link.is_symlink()
    assert os.path.realpath(str(link)) == os.path.realpath(str(target))


def test_ensure_data_symlink_returns_early_when_already_correct(tmp_path, monkeypatch):
    target = tmp_path / "real_data"
    target.mkdir()
    workdir = tmp_path / "run_from_here"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    _ensure_data_symlink(str(target))
    _ensure_data_symlink(str(target))  # second call: matching symlink, must not raise

    link = workdir / "data"
    assert link.is_symlink()


def test_ensure_data_symlink_raises_on_mismatched_existing_symlink(tmp_path, monkeypatch):
    target = tmp_path / "real_data"
    target.mkdir()
    other_target = tmp_path / "other_data"
    other_target.mkdir()
    workdir = tmp_path / "run_from_here"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    _ensure_data_symlink(str(target))
    with pytest.raises(RuntimeError, match="different target"):
        _ensure_data_symlink(str(other_target))


def test_ensure_data_symlink_raises_on_non_symlink_path(tmp_path, monkeypatch):
    target = tmp_path / "real_data"
    target.mkdir()
    workdir = tmp_path / "run_from_here"
    workdir.mkdir()
    (workdir / "data").mkdir()  # a real directory, not a symlink
    monkeypatch.chdir(workdir)

    with pytest.raises(RuntimeError, match="not the expected symlink"):
        _ensure_data_symlink(str(target))


# ---------------------------------------------------------------------------
# verify_trainable_params -- pure nn.Module logic, no real xr1()/checkpoint
# needed. Fake models below only need a `.dit` submodule and params whose
# qualified names do/don't contain "lora_A"/"lora_B", matching exactly what
# verify_trainable_params actually inspects.
# ---------------------------------------------------------------------------


def _make_lora_like_submodule():
    holder = nn.Module()
    holder.lora_A = nn.Parameter(torch.randn(2, 4))
    holder.lora_B = nn.Parameter(torch.zeros(4, 2))
    return holder


def _make_fake_base_model(dit_trainable=False):
    model = nn.Module()
    model.dit = nn.Linear(4, 4)
    for param in model.dit.parameters():
        param.requires_grad_(dit_trainable)
    model.attn = _make_lora_like_submodule()
    return model


def test_verify_trainable_params_happy_path_returns_only_lora_params():
    model = _make_fake_base_model()
    trainable = verify_trainable_params(model)
    assert len(trainable) == 2
    trainable_ids = {id(p) for p in trainable}
    assert id(model.attn.lora_A) in trainable_ids
    assert id(model.attn.lora_B) in trainable_ids


def test_verify_trainable_params_raises_if_dit_has_trainable_params():
    model = _make_fake_base_model(dit_trainable=True)
    with pytest.raises(RuntimeError, match="DiT has trainable parameters"):
        verify_trainable_params(model)


def test_verify_trainable_params_raises_on_non_lora_trainable_base_param():
    model = _make_fake_base_model()
    model.stray = nn.Linear(4, 4)  # trainable by default, not LoRA-named
    with pytest.raises(RuntimeError, match="non-LoRA base-model params are trainable"):
        verify_trainable_params(model)


def test_verify_trainable_params_includes_adapter_and_memory_core_params():
    model = _make_fake_base_model()
    adapter = nn.Linear(4, 2)  # 2 params: weight, bias
    memory_core = nn.Module()
    memory_core.w = nn.Parameter(torch.randn(3))  # 1 param

    trainable = verify_trainable_params(model, adapter, memory_core)

    assert len(trainable) == 2 + 2 + 1
    trainable_ids = {id(p) for p in trainable}
    assert id(adapter.weight) in trainable_ids
    assert id(adapter.bias) in trainable_ids
    assert id(memory_core.w) in trainable_ids


# ---------------------------------------------------------------------------
# save_trainable_state -- pure filtering/serialization logic, no real
# xr1()/checkpoint needed.
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self):
        self.lora_rank = 4
        self.lora_alpha = 8.0
        self.checkpoint = "/fake/checkpoint/dir"
        self.vlm_hidden_size = 16
        self.adapter_target_dim = 8
        self.memory_size = 4


def _make_fake_model_for_save():
    model = nn.Module()
    layer = _make_lora_like_submodule()
    model.layer = layer
    base = nn.Linear(3, 3)
    base.weight.requires_grad_(False)
    base.bias.requires_grad_(False)
    model.base = base
    return model


def test_save_trainable_state_lora_only_filters_to_lora_params_only(tmp_path):
    model = _make_fake_model_for_save()
    args = _FakeArgs()

    out_path = save_trainable_state(str(tmp_path), model, None, None, False, args)

    assert os.path.exists(out_path)
    payload = torch.load(out_path, weights_only=True)
    assert set(payload["lora_state_dict"].keys()) == {"layer.lora_A", "layer.lora_B"}
    assert "base.weight" not in payload["lora_state_dict"]
    assert payload["use_m7_consolidation"] is False
    assert payload["lora_rank"] == args.lora_rank
    assert payload["lora_alpha"] == args.lora_alpha
    assert payload["base_checkpoint"] == os.path.abspath(args.checkpoint)
    assert "adapter_state_dict" not in payload
    assert "memory_core_state_dict" not in payload


def test_save_trainable_state_with_m7_includes_adapter_and_memory_core(tmp_path):
    model = _make_fake_model_for_save()
    adapter = nn.Linear(4, 2)
    memory_core = nn.Module()
    memory_core.w = nn.Parameter(torch.randn(3))
    args = _FakeArgs()

    out_path = save_trainable_state(str(tmp_path), model, adapter, memory_core, True, args)

    payload = torch.load(out_path, weights_only=True)
    assert payload["use_m7_consolidation"] is True
    assert set(payload["adapter_state_dict"].keys()) == {"weight", "bias"}
    assert set(payload["memory_core_state_dict"].keys()) == {"w"}
    assert payload["adapter_vlm_hidden_size"] == args.vlm_hidden_size
    assert payload["adapter_target_dim"] == args.adapter_target_dim
    assert payload["memory_core_memory_size"] == args.memory_size


def test_save_trainable_state_creates_output_dir_if_missing(tmp_path):
    model = _make_fake_model_for_save()
    args = _FakeArgs()
    output_dir = tmp_path / "nested" / "does_not_exist_yet"

    out_path = save_trainable_state(str(output_dir), model, None, None, False, args)

    assert os.path.isdir(str(output_dir))
    assert os.path.exists(out_path)


# ---------------------------------------------------------------------------
# _seed_everything -- the most load-bearing piece of pure logic for Task 9's
# ablation validity (see NOTES.md "Task 7 addendum: CUDA determinism
# investigation" for what it does and does NOT guarantee). These tests only
# check what it's actually responsible for: RNG-stream reproducibility for
# stdlib random and torch -- not CUDA kernel-level determinism, which is a
# real, separate, documented limitation, not something this function claims
# to solve.
# ---------------------------------------------------------------------------


def test_seed_everything_same_seed_reproduces_random_and_torch_draws():
    _seed_everything(123)
    random_draw_1 = random.random()
    torch_draw_1 = torch.rand(1)

    _seed_everything(123)
    random_draw_2 = random.random()
    torch_draw_2 = torch.rand(1)

    assert random_draw_1 == random_draw_2
    assert torch.equal(torch_draw_1, torch_draw_2)


def test_seed_everything_different_seed_produces_different_draws():
    """Rules out a no-op/trivially-passing implementation: different seeds
    must produce different draws, not just "same seed reproduces itself"
    (which a function that seeds nothing at all could also satisfy by
    accident if called with the exact same global RNG state each time)."""
    _seed_everything(1)
    random_draw_1 = random.random()
    torch_draw_1 = torch.rand(1)

    _seed_everything(2)
    random_draw_2 = random.random()
    torch_draw_2 = torch.rand(1)

    assert random_draw_1 != random_draw_2
    assert not torch.equal(torch_draw_1, torch_draw_2)
