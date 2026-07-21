"""Experiment-local AMP, depth, and reward terms."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from src.amp.state import KEY_BODY_NAMES, build_state_with
from src.student_api import load_student_function

_ROBOT_CFG = SceneEntityCfg("robot")


def amp_observation(
  env: ManagerBasedRlEnv,
  student_path: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return the policy state used by the AMP discriminator, shape ``[B, 83]``."""
  robot: Entity = env.scene[asset_cfg.name]
  root_pos = robot.data.root_link_pos_w
  root_quat = robot.data.root_link_quat_w
  heading_quat = yaw_quat(root_quat)
  key_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids]
  relative_w = key_pos_w - root_pos.unsqueeze(1)
  heading = heading_quat.unsqueeze(1).expand(-1, len(KEY_BODY_NAMES), -1)
  relative_yaw = quat_apply_inverse(heading, relative_w)
  pelvis_height = (root_pos[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)
  builder = load_student_function(student_path, "build_amp_state")
  return build_state_with(
    builder,
    robot.data.joint_pos,
    robot.data.joint_vel,
    pelvis_height,
    robot.data.projected_gravity_b,
    quat_apply_inverse(heading_quat, robot.data.root_link_lin_vel_w),
    quat_apply_inverse(heading_quat, robot.data.root_link_ang_vel_w),
    relative_yaw,
  )


def normalized_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  student_path: str,
) -> torch.Tensor:
  """Return student-normalized depth in ``[B, 1, 60, 80]`` layout."""
  sensor: CameraSensor = env.scene[sensor_name]
  depth = sensor.data.depth
  if depth is None:
    raise RuntimeError(f"Camera sensor {sensor_name!r} did not produce depth")
  depth_chw = depth.permute(0, 3, 1, 2)
  normalize = load_student_function(student_path, "normalize_depth")
  normalized = normalize(depth_chw)
  expected = (env.num_envs, 1, 60, 80)
  if tuple(normalized.shape) != expected:
    raise ValueError(
      f"normalize_depth() returned {normalized.shape}; expected {expected}"
    )
  if not torch.isfinite(normalized).all():
    raise ValueError("normalize_depth() returned NaN or Inf")
  return normalized


def rough_tracking_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  student_path: str,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Evaluate the student's velocity-error reward formula."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  if not isinstance(command, torch.Tensor):
    raise TypeError(f"Command {command_name!r} must be a tensor")
  linear_error = torch.norm(
    command[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1
  )
  angular_error = torch.abs(command[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
  reward_fn = load_student_function(student_path, "rough_task_reward")
  reward = reward_fn(linear_error, angular_error)
  if reward.shape != linear_error.shape or not torch.isfinite(reward).all():
    raise ValueError("rough_task_reward() must return a finite [B] tensor")
  return reward


def student_smoothness_penalty(
  env: ManagerBasedRlEnv,
  student_path: str,
) -> torch.Tensor:
  """Evaluate the student's action second-difference penalty."""
  penalty_fn = load_student_function(student_path, "smoothness_penalty")
  penalty = penalty_fn(
    env.action_manager.action,
    env.action_manager.prev_action,
    env.action_manager.prev_prev_action,
  )
  expected = (env.num_envs,)
  if tuple(penalty.shape) != expected or not torch.isfinite(penalty).all():
    raise ValueError("smoothness_penalty() must return a finite [B] tensor")
  return penalty
