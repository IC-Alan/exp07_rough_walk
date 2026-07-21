"""Student formulas for Experiment 07.

This file contains the small, framework-independent pieces used by the
rough-terrain G1 AMP-PPO experiment.  The surrounding runtime supplies the
simulation, replay buffer, optimizer, checkpoints, and policy networks.
"""

from __future__ import annotations

import torch


def build_amp_state(
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  pelvis_height: torch.Tensor,
  projected_gravity: torch.Tensor,
  base_lin_vel_yaw: torch.Tensor,
  base_ang_vel_yaw: torch.Tensor,
  key_body_pos_pelvis: torch.Tensor,
) -> torch.Tensor:
  """Build one 83-D AMP state.

  Concatenate, in this exact order, ``q[29]``, ``qdot[29]``, pelvis height
  ``[1]``, projected gravity ``[3]``, yaw-local base linear/angular velocity
  ``[3] + [3]``, and five pelvis-relative key-body positions ``[5, 3]``.

  All inputs have leading batch dimensions.  Return shape: ``[..., 83]``.
  """
  # Flatten only the five key-body xyz coordinates.  ``flatten(-2)`` preserves
  # any leading batch dimensions, so this also works for unbatched motion data.
  key_body_flat = key_body_pos_pelvis.flatten(start_dim=-2)
  return torch.cat(
    (
      joint_pos,
      joint_vel,
      pelvis_height,
      projected_gravity,
      base_lin_vel_yaw,
      base_ang_vel_yaw,
      key_body_flat,
    ),
    dim=-1,
  )


def least_squares_discriminator_loss(
  expert_output: torch.Tensor,
  policy_output: torch.Tensor,
  gradient_penalty: torch.Tensor,
) -> torch.Tensor:
  """Return the scalar LSGAN discriminator loss.

  Use expert target ``+1``, policy target ``-1``, and gradient-penalty weight
  10: ``0.5 * (mean((D_e - 1)^2) + mean((D_p + 1)^2)) + 10 * gp``.
  """
  expert_loss = torch.mean((expert_output - 1.0).square())
  policy_loss = torch.mean((policy_output + 1.0).square())
  return 0.5 * (expert_loss + policy_loss) + 10.0 * gradient_penalty


def style_reward(discriminator_output: torch.Tensor) -> torch.Tensor:
  """Map discriminator output to AMP style reward with unchanged shape.

  ``r_style = clamp(1 - 0.25 * (D - 1)^2, 0, 1)``.
  """
  reward = 1.0 - 0.25 * (discriminator_output - 1.0).square()
  return torch.clamp(reward, min=0.0, max=1.0)


def normalize_depth(depth_m: torch.Tensor) -> torch.Tensor:
  """Clip metric depth to ``[0.1, 5.0]`` and map it linearly to ``[0, 1]``.

  Input and output shape: ``[B, 1, 60, 80]``.
  """
  minimum_depth = 0.1
  maximum_depth = 5.0
  clipped = torch.clamp(depth_m, min=minimum_depth, max=maximum_depth)
  return (clipped - minimum_depth) / (maximum_depth - minimum_depth)


def rough_task_reward(
  linear_velocity_error: torch.Tensor,
  angular_velocity_error: torch.Tensor,
) -> torch.Tensor:
  """Return the per-environment rough-walk tracking reward.

  Define ``r_lin = exp(-(e_lin / 0.45)^2)`` and
  ``r_ang = exp(-(e_ang / 0.35)^2)``, then return
  ``0.6 * r_lin + 0.4 * r_ang``.  Return shape: ``[B]``.
  """
  linear_reward = torch.exp(-torch.square(linear_velocity_error / 0.45))
  angular_reward = torch.exp(-torch.square(angular_velocity_error / 0.35))
  return 0.6 * linear_reward + 0.4 * angular_reward


def smoothness_penalty(
  action: torch.Tensor,
  previous_action: torch.Tensor,
  previous_previous_action: torch.Tensor,
) -> torch.Tensor:
  """Return mean absolute second action difference for each environment.

  ``mean(abs(a_t - 2 a_(t-1) + a_(t-2)), dim=-1)``.  Inputs are ``[B, 29]``
  and the return shape is ``[B]``.
  """
  second_difference = action - 2.0 * previous_action + previous_previous_action
  return torch.mean(torch.abs(second_difference), dim=-1)
