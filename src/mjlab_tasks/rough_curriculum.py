"""Curriculum helpers for the rough terrain lab."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired
    
import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: NotRequired[tuple[float, float]]
  lin_vel_y: NotRequired[tuple[float, float]]
  ang_vel_z: NotRequired[tuple[float, float]]


def easy_to_hard_commands(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  """Gradually increases commanded walking speed and turning range."""
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  if command_term is None:
    raise RuntimeError(f"Command term {command_name!r} is not registered")
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  active_stage = 0
  for index, stage in enumerate(velocity_stages):
    if env.common_step_counter >= int(stage["step"]):
      active_stage = index
      if "lin_vel_x" in stage:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "stage": torch.tensor(float(active_stage), device=env.device),
    "lin_vel_x_max": torch.tensor(float(cfg.ranges.lin_vel_x[1]), device=env.device),
    "ang_vel_z_max": torch.tensor(float(cfg.ranges.ang_vel_z[1]), device=env.device),
  }
