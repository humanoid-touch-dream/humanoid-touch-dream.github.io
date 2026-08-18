#!/usr/bin/env python3
"""Guard the copied browser contract and assets against the native prototype."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml


HERE = Path(__file__).resolve()
WEB_ROOT = HERE.parents[1]
HTD_ROOT = HERE.parents[4]
NATIVE = HTD_ROOT / "IsaacLab-Decoupled-WBC" / "sim2mujoco"
sys.path.insert(0, str(NATIVE))

from htd_controller import (  # noqa: E402
    ACTION_DIM,
    ACTION_SCALE,
    ARM_RATE_LIMIT,
    ARM_STAND_HEIGHT,
    COMMAND_MAX,
    COMMAND_MIN,
    CONTROL_DECIMATION,
    CONTROL_DT,
    DEFAULT_JOINT_POS,
    EFFORT_LIMIT,
    JOINT_NAMES,
    KD,
    KP,
    NEUTRAL_COMMAND,
    OBS_DIM,
    SIM_DT,
)
from validate_mujoco import PRESETS  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    metadata = json.loads((WEB_ROOT / "public/assets/contract.json").read_text())
    assert metadata["sim_dt"] == SIM_DT
    assert metadata["control_dt"] == CONTROL_DT
    assert metadata["control_decimation"] == CONTROL_DECIMATION
    assert metadata["observation_dim"] == OBS_DIM
    assert metadata["action_dim"] == ACTION_DIM
    assert metadata["action_scale"] == ACTION_SCALE
    assert metadata["arm_rate_limit"] == ARM_RATE_LIMIT
    assert metadata["arm_stand_height"] == ARM_STAND_HEIGHT
    assert metadata["joint_names"] == list(JOINT_NAMES)
    np.testing.assert_allclose(metadata["default_joint_pos"], DEFAULT_JOINT_POS, rtol=0, atol=0)
    np.testing.assert_allclose(metadata["kp"], KP, rtol=0, atol=0)
    np.testing.assert_allclose(metadata["kd"], KD, rtol=0, atol=0)
    np.testing.assert_allclose(metadata["effort_limit"], EFFORT_LIMIT, rtol=0, atol=0)
    np.testing.assert_allclose(metadata["neutral_command"], NEUTRAL_COMMAND, rtol=0, atol=5e-8)
    np.testing.assert_allclose([item["min"] for item in metadata["commands"]], COMMAND_MIN, rtol=0, atol=2e-7)
    np.testing.assert_allclose([item["max"] for item in metadata["commands"]], COMMAND_MAX, rtol=0, atol=2e-7)

    training_cfg_path = HTD_ROOT / "IsaacLab-Decoupled-WBC" / metadata["training_command_source"]
    training_cfg = yaml.load(training_cfg_path.read_text(), Loader=yaml.UnsafeLoader)
    ranges = training_cfg["commands"]["ranges"]
    training_keys = ("lin_vel_x", "lin_vel_y", "ang_vel_z", "body_height", "body_roll", "body_pitch", "body_yaw")
    for spec, key in zip(metadata["commands"], training_keys, strict=True):
        np.testing.assert_allclose(
            [spec["training"]["min"], spec["training"]["max"]], ranges[key], rtol=0, atol=0,
        )
    curriculum = training_cfg["commands"]["curriculum"]
    assert metadata["training_curriculum"]["start_iter"] == curriculum["body_height"]["start_iter"]
    assert metadata["training_curriculum"]["end_iter"] == curriculum["body_height"]["end_iter"]
    for metadata_key, curriculum_key in (
        ("body_height_start", "body_height"),
        ("body_roll_start", "body_roll"),
        ("body_pitch_start", "body_pitch"),
        ("body_yaw_start", "body_yaw"),
    ):
        np.testing.assert_allclose(
            metadata["training_curriculum"][metadata_key], curriculum[curriculum_key]["start_range"], rtol=0, atol=0,
        )
    for spec, curriculum_key in zip(metadata["commands"][3:], ("body_height", "body_roll", "body_pitch", "body_yaw"), strict=True):
        np.testing.assert_allclose(
            [spec["training"]["start_min"], spec["training"]["start_max"]],
            curriculum[curriculum_key]["start_range"], rtol=0, atol=0,
        )

    browser_presets = {item["key"]: item["command"] for item in metadata["safe_presets"]}
    assert set(browser_presets) == set(PRESETS)
    for name, command in PRESETS.items():
        np.testing.assert_allclose(browser_presets[name], command, rtol=0, atol=5e-8)

    arm = np.load(NATIVE / "assets/arm_coupling.npz", allow_pickle=False)
    for name in ("gain", "bias", "swing", "lo", "hi"):
        np.testing.assert_allclose(metadata["arm_coupling"][name], arm[name], rtol=0, atol=0)

    copied = WEB_ROOT / "public/assets"
    for relative in ("scene.xml", "g1_29dof.xml"):
        source = NATIVE / f"assets/g1/{relative}"
        assert digest(copied / relative) == digest(source), relative
    assert digest(copied / "policy.onnx") == metadata["policy"]["onnx_sha256"]
    source_meshes = NATIVE / "assets/g1/meshes"
    assert set(metadata["mesh_files"]) == {path.name for path in source_meshes.glob("*.STL")}
    for name in metadata["mesh_files"]:
        assert digest(copied / "meshes" / name) == digest(source_meshes / name), name

    print("native/browser contract, 38 copied model assets, and declared browser policy match")


if __name__ == "__main__":
    main()
