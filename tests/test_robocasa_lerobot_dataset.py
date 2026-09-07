# Copyright (C) 2026 -- this project (xr1-m7-submission), Task 9 pivot.
"""Unit tests for the pure-logic helpers in
code/robocasa_lerobot_dataset.py -- no GPU, no real checkpoint, no real
lerobot data on disk needed. Real end-to-end verification against the real
data (episode indexing, video decode, parquet read, full sample shapes) is
done separately as a smoke test against the real pod data -- see NOTES.md
"Task 9 pivot" for that evidence; these tests cover what's practical to
verify without it (matching the rest of this repo's TDD convention for
pure-logic helpers, e.g. tests/test_finetune_run.py's
_ensure_data_symlink/verify_trainable_params tests).
"""
import json
import os

import numpy as np
import pytest

from code.robocasa_lerobot_dataset import (
    OFFICIAL_ATOMIC_SEEN_TASKS,
    OFFICIAL_COMPOSITE_SEEN_TASKS,
    ROBOCASA_ACTION_DIM,
    ROBOCASA_ACTIVE_ACTION_DIMS,
    ROBOCASA_ACTIVE_STATE_DIMS,
    ROBOCASA_STATE_DIM,
    RoboCasaLerobotDataset,
    discover_task_roots,
    observation_state_to_14d,
    quat_xyzw_to_axis_angle,
)


def test_official_task_lists_have_no_overlap_and_expected_counts():
    # 18 atomic_seen + 16 composite_seen = 34, per NOTES.md's real
    # enumeration against /workspace/M7/code/m7_robocasa_eval.py.
    assert len(OFFICIAL_ATOMIC_SEEN_TASKS) == 18
    assert len(OFFICIAL_COMPOSITE_SEEN_TASKS) == 16
    assert len(set(OFFICIAL_ATOMIC_SEEN_TASKS) & set(OFFICIAL_COMPOSITE_SEEN_TASKS)) == 0
    assert len(OFFICIAL_ATOMIC_SEEN_TASKS) == len(set(OFFICIAL_ATOMIC_SEEN_TASKS))
    assert len(OFFICIAL_COMPOSITE_SEEN_TASKS) == len(set(OFFICIAL_COMPOSITE_SEEN_TASKS))


def test_quat_xyzw_to_axis_angle_identity_quaternion_is_zero():
    # Identity quaternion (xyzw = 0,0,0,1) -> zero rotation -> zero axis-angle.
    result = quat_xyzw_to_axis_angle(np.array([0.0, 0.0, 0.0, 1.0]))
    assert result.shape == (3,)
    np.testing.assert_allclose(result, np.zeros(3), atol=1e-6)


def test_quat_xyzw_to_axis_angle_known_90_degree_rotation():
    # 90-degree rotation about the z-axis: xyzw = (0, 0, sin(45deg), cos(45deg))
    half = np.pi / 4
    quaternion = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    result = quat_xyzw_to_axis_angle(quaternion)
    expected = np.array([0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_quat_xyzw_to_axis_angle_handles_near_zero_norm():
    result = quat_xyzw_to_axis_angle(np.array([0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(result, np.zeros(3))


def test_observation_state_to_14d_real_slice_layout():
    # Matches meta/modality.json's real layout verified against
    # .../CloseFridge/20250819/lerobot/meta/modality.json:
    # base_position[0:3], base_rotation quat[3:7],
    # end_effector_position_relative[7:10],
    # end_effector_rotation_relative quat[10:14], gripper_qpos[14:16].
    state16 = np.zeros(16, dtype=np.float64)
    state16[0:3] = [1.0, 2.0, 3.0]  # base_position
    state16[3:7] = [0.0, 0.0, 0.0, 1.0]  # base_rotation (identity quat)
    state16[7:10] = [4.0, 5.0, 6.0]  # end_effector_position_relative
    state16[10:14] = [0.0, 0.0, 0.0, 1.0]  # end_effector_rotation_relative (identity quat)
    state16[14:16] = [0.5, 0.6]  # gripper_qpos

    state14 = observation_state_to_14d(state16)

    assert state14.shape == (14,)
    # order: ee_pos_rel(3), ee_rot_aa(3), gripper(2), base_pos(3), base_rot_aa(3)
    np.testing.assert_allclose(state14[0:3], [4.0, 5.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(state14[3:6], np.zeros(3), atol=1e-6)  # identity quat -> zero aa
    np.testing.assert_allclose(state14[6:8], [0.5, 0.6], atol=1e-6)
    np.testing.assert_allclose(state14[8:11], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(state14[11:14], np.zeros(3), atol=1e-6)


def test_observation_state_to_14d_rejects_wrong_shape():
    with pytest.raises(ValueError):
        observation_state_to_14d(np.zeros(10))


def test_discover_task_roots_real_directory_layout(tmp_path):
    data_root = tmp_path / "pretrain"
    task_dir = data_root / "atomic" / "CloseFridge" / "20250819" / "lerobot"
    task_dir.mkdir(parents=True)
    (data_root / "composite" / "WashLettuce" / "20250720" / "lerobot").mkdir(parents=True)

    roots = discover_task_roots(str(data_root), ["CloseFridge"], ["WashLettuce"])

    assert len(roots) == 2
    assert str(task_dir) in roots


def test_discover_task_roots_raises_on_missing_task(tmp_path):
    data_root = tmp_path / "pretrain"
    (data_root / "atomic" / "CloseFridge" / "20250819" / "lerobot").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="OpenCabinet"):
        discover_task_roots(str(data_root), ["CloseFridge", "OpenCabinet"], [])


def _write_fake_episode_meta(root, episodes):
    meta_dir = os.path.join(root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as file:
        for episode in episodes:
            file.write(json.dumps(episode) + "\n")


def test_dataset_indexes_episodes_not_frames(tmp_path):
    # Real memory-safety property this loader depends on (see module
    # docstring's memory note): __init__ must scale with episode count, not
    # frame count -- verified here by giving episodes a large `length` and
    # confirming the episode index has exactly len(episodes) entries, not
    # sum(lengths).
    root = str(tmp_path / "task_a" / "date" / "lerobot")
    _write_fake_episode_meta(
        root,
        [
            {"episode_index": 0, "length": 50000, "tasks": ["Do the thing."]},
            {"episode_index": 1, "length": 50000, "tasks": ["Do the thing again."]},
        ],
    )

    dataset = RoboCasaLerobotDataset([root], action_length=30, max_samples=8, seed=42)

    assert len(dataset.episodes) == 2
    assert len(dataset.samples) == 8


def test_dataset_sample_frame_indices_are_within_episode_bounds(tmp_path):
    root = str(tmp_path / "task_a" / "date" / "lerobot")
    _write_fake_episode_meta(
        root,
        [
            {"episode_index": 0, "length": 10, "tasks": ["Do the thing."]},
        ],
    )

    dataset = RoboCasaLerobotDataset([root], action_length=30, max_samples=20, seed=42)

    for episode_index, frame_index in dataset.samples:
        assert 0 <= frame_index < dataset.episodes[episode_index]["length"]


def test_dataset_is_deterministic_given_seed(tmp_path):
    root = str(tmp_path / "task_a" / "date" / "lerobot")
    _write_fake_episode_meta(
        root,
        [
            {"episode_index": 0, "length": 200, "tasks": ["Do the thing."]},
            {"episode_index": 1, "length": 150, "tasks": ["Do another thing."]},
        ],
    )

    first = RoboCasaLerobotDataset([root], action_length=30, max_samples=25, seed=7)
    second = RoboCasaLerobotDataset([root], action_length=30, max_samples=25, seed=7)
    third = RoboCasaLerobotDataset([root], action_length=30, max_samples=25, seed=99)

    assert first.samples == second.samples
    assert first.samples != third.samples


def test_action_dim_constants_match_checkpoint_config():
    # Real, verified against checkpoints/Xiaomi-Robotics-1-RoboCasa365/
    # config.json (action_dim=60, state_dim=60) and
    # preprocessor_config.json's action_config["robocasa365"] (std>1e-5 on
    # exactly the first 12 of 60 dims) -- see NOTES.md.
    assert ROBOCASA_ACTION_DIM == 60
    assert ROBOCASA_STATE_DIM == 60
    assert ROBOCASA_ACTIVE_ACTION_DIMS == 12
    assert ROBOCASA_ACTIVE_STATE_DIMS == 14
