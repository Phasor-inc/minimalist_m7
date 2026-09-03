import torch
from code.pooled_hidden_adapter import PooledHiddenAdapter, pool_state_token_hidden_states


def test_adapter_projects_to_target_dim():
    adapter = PooledHiddenAdapter(vlm_hidden_size=2560, target_dim=512)
    x = torch.randn(4, 2560)
    out = adapter(x)
    assert out.shape == (4, 512)


def test_pool_state_token_hidden_states_mean_pools_masked_positions():
    # hidden_states: [seq_len, hidden_size], state_token_mask: [seq_len] bool
    hidden_states = torch.tensor([
        [1.0, 1.0],
        [3.0, 3.0],
        [5.0, 5.0],
    ])
    mask = torch.tensor([True, False, True])
    pooled = pool_state_token_hidden_states(hidden_states, mask)
    assert torch.allclose(pooled, torch.tensor([3.0, 3.0]))


def test_pool_state_token_hidden_states_raises_on_empty_mask():
    hidden_states = torch.randn(3, 2)
    mask = torch.tensor([False, False, False])
    try:
        pool_state_token_hidden_states(hidden_states, mask)
        assert False, "expected ValueError on all-False mask"
    except ValueError:
        pass
