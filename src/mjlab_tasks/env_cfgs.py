"""Native Unitree G1 rough-walk environment with AMP and optional depth."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.amp.state import KEY_BODY_NAMES

from .observations import (
  amp_observation,
  normalized_depth,
  rough_tracking_reward,
  student_smoothness_penalty,
)
from .rough_curriculum import easy_to_hard_commands
from .safe_events import reset_joints_by_offset_batched

ObservationMode = Literal["height", "depth"]
EXP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_PATH = EXP_ROOT / "student.py"
DEFAULT_MOTION_PATH = EXP_ROOT / "assets" / "motions" / "G1_walk_50hz.npz"


def course_g1_rough_walk_env_cfg(
  observation_mode: ObservationMode = "height",
  play: bool = False,
  student_path: str | Path = DEFAULT_STUDENT_PATH,
  *,
  walk_focus: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  """Return a G1 rough-terrain cfg while preserving the 29-D native action."""
  if observation_mode not in ("height", "depth"):
    raise ValueError(f"Unknown observation mode: {observation_mode}")
  student_path = str(Path(student_path).resolve())
  cfg = unitree_g1_rough_env_cfg(play=play)
  cfg.scene.num_envs = 32 if play else 4096
  cfg.scene.extent = 1.8
  cfg.episode_length_s = 12.0
  cfg.seed = 7
  cfg.auto_reset = False
  cfg.scale_rewards_by_dt = True
  cfg.events["reset_robot_joints"].func = reset_joints_by_offset_batched

  if cfg.scene.terrain is not None:
    cfg.scene.terrain.max_init_terrain_level = 1
    generator = cfg.scene.terrain.terrain_generator
    if generator is not None:
      generator.num_rows = 6
      generator.curriculum = not play

  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      cast(Any, sensor).debug_vis = False

  amp_cfg = SceneEntityCfg("robot", body_names=KEY_BODY_NAMES)
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={
      "state": ObservationTermCfg(
        func=amp_observation,
        params={"student_path": student_path, "asset_cfg": amp_cfg},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  if observation_mode == "depth":
    actor_terms = cfg.observations["actor"].terms
    # Student actor must not see privileged height. Keep a frozen-teacher view
    # and a dense height target for optional reconstruction distillation.
    teacher_terms = dict(actor_terms)
    cfg.observations["teacher"] = ObservationGroupCfg(
      terms=teacher_terms,
      concatenate_terms=True,
      enable_corruption=False,
    )
    # Prefer critic height (no observation noise) as reconstruction target.
    critic_height = cfg.observations["critic"].terms.get("height_scan")
    height_scan_term = critic_height or actor_terms["height_scan"]
    cfg.observations["height_target"] = ObservationGroupCfg(
      terms={"height_scan": height_scan_term},
      concatenate_terms=True,
      enable_corruption=False,
    )
    actor_terms.pop("height_scan", None)
    depth_sensor = CameraSensorCfg(
      name="depth",
      camera_name=None,
      parent_body="robot/torso_link",
      pos=(0.18, 0.0, 0.12),
      quat=(0.5, -0.5, 0.5, -0.5),
      fovy=70.0,
      width=80,
      height=60,
      data_types=("depth",),
      enabled_geom_groups=(0,),
      use_shadows=False,
      use_textures=False,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_sensor,)
    cfg.observations["depth"] = ObservationGroupCfg(
      terms={
        "depth": ObservationTermCfg(
          func=normalized_depth,
          params={"sensor_name": "depth", "student_path": student_path},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )

  cfg.rewards.pop("track_linear_velocity", None)
  cfg.rewards.pop("track_angular_velocity", None)
  cfg.rewards["rough_task"] = RewardTermCfg(
    func=rough_tracking_reward,
    weight=3.8,
    params={"command_name": "twist", "student_path": student_path},
  )
  cfg.rewards["student_smoothness"] = RewardTermCfg(
    func=student_smoothness_penalty,
    weight=-0.05,
    params={"student_path": student_path},
  )
  cfg.rewards["upright"].weight = 1.4
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_slip"].weight = -0.18
  cfg.rewards["soft_landing"].weight = -2.0e-5

  # Depth (and early walk debugging) should not over-reward safe standing.
  # Keep scale_rewards_by_dt=True; raise tracking relative to posture terms.
  if walk_focus is None:
    walk_focus = observation_mode == "depth"
  if walk_focus:
    cfg.rewards["rough_task"].weight = 8.0
    cfg.rewards["student_smoothness"].weight = -0.01
    cfg.rewards["upright"].weight = 0.7
    if "pose" in cfg.rewards:
      cfg.rewards["pose"].weight = 0.45
    if "action_rate_l2" in cfg.rewards:
      cfg.rewards["action_rate_l2"].weight = -0.04

  twist_cmd = cfg.commands["twist"]
  if not isinstance(twist_cmd, UniformVelocityCommandCfg):
    raise TypeError("Native G1 rough cfg must expose UniformVelocityCommandCfg")
  twist_cmd.ranges.lin_vel_x = (0.0, 0.6)
  twist_cmd.ranges.lin_vel_y = (-0.15, 0.15)
  twist_cmd.ranges.ang_vel_z = (-0.25, 0.25)
  twist_cmd.rel_standing_envs = 0.05
  twist_cmd.rel_forward_envs = 0.55
  twist_cmd.debug_vis = False

  if not play:
    cfg.curriculum["course_command_schedule"] = CurriculumTermCfg(
      func=easy_to_hard_commands,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (0.0, 0.5),
            "lin_vel_y": (-0.1, 0.1),
            "ang_vel_z": (-0.2, 0.2),
          },
          {
            "step": 4_800,
            "lin_vel_x": (0.1, 0.8),
            "lin_vel_y": (-0.2, 0.2),
            "ang_vel_z": (-0.35, 0.35),
          },
          {
            "step": 9_600,
            "lin_vel_x": (0.15, 1.1),
            "lin_vel_y": (-0.3, 0.3),
            "ang_vel_z": (-0.5, 0.5),
          },
        ],
      },
    )
  return cfg


def course_g1_rough_traversal_env_cfg(
  observation_mode: ObservationMode = "height",
  student_path: str | Path = DEFAULT_STUDENT_PATH,
) -> ManagerBasedRlEnvCfg:
  """Held-out 6 m forward traversal configuration used by evaluation."""
  cfg = course_g1_rough_walk_env_cfg(
    observation_mode=observation_mode,
    play=True,
    student_path=student_path,
  )
  cfg.episode_length_s = 12.0
  command = cfg.commands["twist"]
  if not isinstance(command, UniformVelocityCommandCfg):
    raise TypeError("Expected UniformVelocityCommandCfg")
  command.resampling_time_range = (1.0e9, 1.0e9)
  command.heading_command = False
  command.ranges.heading = None
  command.ranges.lin_vel_x = (0.6, 0.6)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  command.rel_standing_envs = 0.0
  command.rel_heading_envs = 0.0
  command.rel_forward_envs = 1.0
  return cfg
