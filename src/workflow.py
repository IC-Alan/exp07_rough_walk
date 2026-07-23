"""Notebook-facing training, evaluation, visualization, and export tools."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import imageio.v3 as iio
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.runner import MjlabOnPolicyRunner

from src.amp import ManualResetAmpVecEnvWrapper, MjlabDistillationRunner
from src.mjlab_tasks.env_cfgs import (
  course_g1_rough_traversal_env_cfg,
  course_g1_rough_walk_env_cfg,
)
from src.mjlab_tasks.rl_cfg import (
  course_g1_amp_ppo_runner_cfg,
  course_g1_distill_runner_cfg,
  course_g1_distill_finetune_runner_cfg,
)
from src.paths import EXP_ROOT

ObservationMode = Literal["height", "depth"]
DEFAULT_STUDENT = EXP_ROOT / "student.py"
LOAD_CFG = {
  "actor": True,
  "critic": True,
  "optimizer": False,
  "iteration": False,
  "rnd": False,
}
DISTILL_TEACHER_LOAD_CFG = {
  "teacher": True,
  "student": False,
  "optimizer": False,
  "iteration": False,
}

MODEL_HEIGHT = """from __future__ import annotations

from collections.abc import Mapping
import torch


class ExportedPolicy:
  observation_keys = ("actor",)

  def __init__(self, path: str, device: str) -> None:
    self.device = torch.device(device)
    self.module = torch.jit.load(path, map_location=self.device).eval()

  def predict(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
      return self.module(obs["actor"].to(self.device))


def load_policy(policy_path: str, device: str = "cpu") -> ExportedPolicy:
  return ExportedPolicy(policy_path, device)
"""

MODEL_DEPTH = """from __future__ import annotations

from collections.abc import Mapping
import torch


class ExportedPolicy:
  observation_keys = ("actor", "depth")

  def __init__(self, path: str, device: str) -> None:
    self.device = torch.device(device)
    self.module = torch.jit.load(path, map_location=self.device).eval()

  def predict(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
      actor = obs["actor"].to(self.device)
      depth = obs["depth"].to(self.device)
      return self.module(actor, [depth])


def load_policy(policy_path: str, device: str = "cpu") -> ExportedPolicy:
  return ExportedPolicy(policy_path, device)
"""


def _student_path(path: str | Path | None) -> Path:
  return Path(path or DEFAULT_STUDENT).resolve()


def _tensor_observation(observations: Mapping[str, Any], key: str) -> torch.Tensor:
  value = observations[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Observation group {key!r} must be concatenated")
  return value


def latest_checkpoint(root: str | Path | None = None) -> Path:
  """Return the newest training checkpoint under the experiment outputs."""
  search_root = Path(root or EXP_ROOT / "outputs" / "rsl_rl")
  candidates = [path for path in search_root.rglob("model_*.pt") if path.is_file()]
  if not candidates:
    raise FileNotFoundError(f"No model_*.pt checkpoint under {search_root}")
  return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def smoke(
  mode: ObservationMode = "height",
  *,
  num_envs: int = 32,
  steps: int = 16,
  device: str = "cuda:0",
  student_file: str | Path | None = None,
  force_termination: bool = True,
) -> dict[str, Any]:
  """Run the short reset/step check used in the notebook."""
  cfg = course_g1_rough_walk_env_cfg(mode, student_path=_student_path(student_file))
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  resets = 0
  try:
    observations, _ = env.reset()
    amp = _tensor_observation(observations, "amp")
    if amp.shape != (num_envs, 83):
      raise RuntimeError(f"Unexpected AMP shape: {amp.shape}")
    depth_shape = None
    if mode == "depth":
      depth = _tensor_observation(observations, "depth")
      depth_shape = tuple(depth.shape)
      if depth_shape != (num_envs, 1, 60, 80):
        raise RuntimeError(f"Unexpected depth shape: {depth_shape}")
      if not torch.isfinite(depth).all() or not torch.any(depth != 0):
        raise RuntimeError("Depth observation must be finite and non-empty")
    for step in range(steps):
      if force_termination and step == steps // 2:
        env.episode_length_buf[:] = env.max_episode_length
      action = torch.zeros(num_envs, 29, device=device)
      _, reward, terminated, truncated, _ = env.step(action)
      if not torch.isfinite(reward).all():
        raise RuntimeError("Non-finite reward")
      done = terminated | truncated
      if done.any():
        env.reset(env_ids=done.nonzero(as_tuple=False).squeeze(-1))
        resets += int(done.sum())
  finally:
    env.close()
  return {
    "mode": mode,
    "num_envs": num_envs,
    "steps": steps,
    "manual_resets": resets,
    "amp_shape": (num_envs, 83),
    "depth_shape": depth_shape,
  }


def train(
  mode: ObservationMode = "height",
  *,
  num_envs: int = 4096,
  iterations: int = 600,
  steps_per_env: int = 24,
  device: str = "cuda:0",
  seed: int = 7,
  student_file: str | Path | None = None,
) -> Path:
  """Train AMP-PPO and return the run directory containing checkpoints."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"Requested {device}, but CUDA is not available")
  student_path = _student_path(student_file)
  env_cfg = course_g1_rough_walk_env_cfg(mode, student_path=student_path)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  agent_cfg = course_g1_amp_ppo_runner_cfg(mode, student_path=student_path)
  agent_cfg.max_iterations = iterations
  agent_cfg.num_steps_per_env = steps_per_env
  agent_cfg.save_interval = max(1, min(50, iterations))
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = EXP_ROOT / "outputs" / "rsl_rl" / agent_cfg.experiment_name / timestamp
  log_dir.mkdir(parents=True, exist_ok=True)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(
    wrapped, asdict(agent_cfg), log_dir=str(log_dir), device=device
  )
  try:
    runner.learn(num_learning_iterations=iterations)
  finally:
    wrapped.close()
  return log_dir


def _asdict_train_cfg(cfg: Any) -> dict[str, Any]:
  """Convert nested runner dataclasses into plain dicts for RSL-RL."""
  data = asdict(cfg)
  return data


def train_distill(
  teacher_checkpoint: str | Path,
  *,
  num_envs: int = 128,
  iterations: int = 400,
  steps_per_env: int = 24,
  device: str = "cuda:0",
  seed: int = 7,
  student_file: str | Path | None = None,
  teacher_mix_decay_iters: int | None = None,
  height_loss_coef: float = 0.2,
  student_init_std: float = 0.1,
) -> Path:
  """DAgger-distill a depth student from a frozen height teacher checkpoint."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"Requested {device}, but CUDA is not available")
  student_path = _student_path(student_file)
  env_cfg = course_g1_rough_walk_env_cfg(
    "depth", student_path=student_path, walk_focus=True
  )
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  agent_cfg = course_g1_distill_runner_cfg(
    student_path=student_path,
    teacher_mix_decay_iters=teacher_mix_decay_iters or max(1, iterations),
    height_loss_coef=height_loss_coef,
    student_init_std=student_init_std,
  )
  agent_cfg.max_iterations = iterations
  agent_cfg.num_steps_per_env = steps_per_env
  agent_cfg.save_interval = max(1, min(50, iterations))
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = EXP_ROOT / "outputs" / "rsl_rl" / agent_cfg.experiment_name / timestamp
  log_dir.mkdir(parents=True, exist_ok=True)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  train_cfg = _asdict_train_cfg(agent_cfg)
  runner = MjlabDistillationRunner(
    wrapped, train_cfg, log_dir=str(log_dir), device=device
  )
  runner.load(str(teacher_checkpoint), load_cfg=DISTILL_TEACHER_LOAD_CFG)
  try:
    runner.learn(num_learning_iterations=iterations)
  finally:
    wrapped.close()
  return log_dir


def finetune_from_distill(
  distill_checkpoint: str | Path,
  *,
  num_envs: int = 128,
  iterations: int = 150,
  steps_per_env: int = 24,
  device: str = "cuda:0",
  seed: int = 7,
  student_file: str | Path | None = None,
  amp_reward_scale: float = 0.0,
  distill_coef: float = 1.0,
  distill_coef_end: float = 0.05,
  distill_decay_iters: int | None = None,
) -> Path:
  """PPO-finetune a distilled depth student with teacher distillation regularisation.

  The frozen height teacher (loaded from *distill_checkpoint*) provides a
  distillation loss ``distill_coef * MSE(student_mean, teacher_mean)`` after
  every PPO update.  The coefficient decays from *distill_coef* to
  *distill_coef_end* over *distill_decay_iters* iterations, preventing PPO
  from pulling the student back to a standing local optimum.

  Args:
    distill_checkpoint: Path to a DAgger distillation checkpoint that
        contains both ``actor_state_dict`` (student) and
        ``teacher_state_dict`` (frozen height teacher).
    num_envs: Parallel simulation environments.
    iterations: PPO update iterations.
    steps_per_env: Rollout steps per env per iteration.
    device: Torch device string.
    seed: RNG seed.
    student_file: Optional override for student.py path.
    amp_reward_scale: AMP style-reward weight (0 = pure PPO).
    distill_coef: Initial distillation loss coefficient λ_start.
    distill_coef_end: Final distillation loss coefficient λ_end.
    distill_decay_iters: Iterations to decay λ over (defaults to
        ``max(1, iterations)``).
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"Requested {device}, but CUDA is not available")
  student_path = _student_path(student_file)
  env_cfg = course_g1_rough_walk_env_cfg(
    "depth", student_path=student_path, walk_focus=True
  )
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  decay = distill_decay_iters if distill_decay_iters is not None else max(1, iterations)
  agent_cfg = course_g1_distill_finetune_runner_cfg(
    student_path=student_path,
    iterations=iterations,
    distill_coef=distill_coef,
    distill_coef_end=distill_coef_end,
    distill_decay_iters=decay,
    amp_reward_scale=amp_reward_scale,
  )
  agent_cfg.num_steps_per_env = steps_per_env
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = (
    EXP_ROOT / "outputs" / "rsl_rl" / f"{agent_cfg.experiment_name}_distill_ft" / timestamp
  )
  log_dir.mkdir(parents=True, exist_ok=True)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(
    wrapped, asdict(agent_cfg), log_dir=str(log_dir), device=device
  )
  # Load checkpoint: actor (student) weights + teacher weights for distillation.
  loaded = torch.load(str(distill_checkpoint), map_location=device, weights_only=False)
  actor_sd = loaded.get("actor_state_dict") or loaded.get("student_state_dict")
  if actor_sd is None:
    raise KeyError(
      f"Distill checkpoint {distill_checkpoint} has no student/actor state_dict"
    )
  missing, unexpected = runner.alg._raw_actor.load_state_dict(actor_sd, strict=False)
  if missing:
    print(f"finetune actor missing keys: {missing}")
  if unexpected:
    print(f"finetune actor unexpected keys: {unexpected}")
  # Wire the frozen teacher into AmpPPOWithDistill for regularisation.
  teacher_sd = loaded.get("teacher_state_dict")
  if teacher_sd is not None and hasattr(runner.alg, "set_teacher_for_distill"):
    from rsl_rl.utils import resolve_callable
    from src.mjlab_tasks.rl_cfg import course_g1_distill_runner_cfg
    tmp_cfg = course_g1_distill_runner_cfg(student_path=student_path)
    teacher_cls = resolve_callable(tmp_cfg.teacher.class_name)
    obs_dummy = wrapped.get_observations().to(device)
    teacher_model = teacher_cls(
      obs_dummy,
      {"teacher": ("teacher",)},
      "teacher",
      wrapped.num_actions,
      hidden_dims=tmp_cfg.teacher.hidden_dims,
      activation=tmp_cfg.teacher.activation,
      obs_normalization=tmp_cfg.teacher.obs_normalization,
    ).to(device)
    if "std" in teacher_sd:
      teacher_sd["distribution.std_param"] = teacher_sd.pop("std")
    if "log_std" in teacher_sd:
      teacher_sd["distribution.log_std_param"] = teacher_sd.pop("log_std")
    teacher_model.load_state_dict(teacher_sd, strict=False)
    runner.alg.set_teacher_for_distill(teacher_model)
    print("Distillation teacher loaded for Phase-3 regularisation.")
  else:
    print(
      "Warning: no teacher_state_dict in checkpoint or algorithm does not support "
      "distillation regularisation — running plain PPO finetune."
    )
  try:
    runner.learn(num_learning_iterations=iterations)
  finally:
    wrapped.close()
  return log_dir


def eval_teacher_on_depth(
  teacher_checkpoint: str | Path,
  *,
  num_envs: int = 16,
  steps: int = 300,
  device: str = "cuda:0",
  student_file: str | Path | None = None,
) -> dict[str, float]:
  """Roll out a height teacher on the depth env using privileged teacher obs."""
  student_path = _student_path(student_file)
  env_cfg = course_g1_rough_traversal_env_cfg("depth", student_path)
  env_cfg.scene.num_envs = num_envs
  agent_cfg = course_g1_distill_runner_cfg(student_path=student_path)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env)
  runner = MjlabDistillationRunner(wrapped, _asdict_train_cfg(agent_cfg), device=device)
  runner.load(str(teacher_checkpoint), load_cfg=DISTILL_TEACHER_LOAD_CFG)
  teacher = runner.alg.teacher
  teacher.eval()
  observations = wrapped.get_observations().to(device)
  start = env.scene["robot"].data.root_link_pos_w[:, :2].clone()
  max_distance = torch.zeros(num_envs, device=device)
  try:
    with torch.inference_mode():
      for _ in range(steps):
        action = teacher(observations, stochastic_output=False)
        observations, _, dones, _ = wrapped.step(action)
        robot = env.scene["robot"]
        distance = torch.norm(robot.data.root_link_pos_w[:, :2] - start, dim=-1)
        max_distance = torch.maximum(max_distance, distance)
        done_mask = dones.bool()
        if done_mask.any():
          start[done_mask] = robot.data.root_link_pos_w[done_mask, :2]
  finally:
    wrapped.close()
  progress = torch.clamp(max_distance / 6.0, 0.0, 1.0)
  return {
    "traversal_progress": float(progress.mean()),
    "traversal_success": float((max_distance >= 6.0).float().mean()),
    "max_distance_m": float(max_distance.mean()),
  }


def _inference_runner(
  checkpoint: str | Path,
  mode: ObservationMode,
  *,
  num_envs: int,
  device: str,
  student_file: str | Path | None,
  render_mode: str | None = None,
) -> tuple[ManagerBasedRlEnv, ManualResetAmpVecEnvWrapper, MjlabOnPolicyRunner]:
  student_path = _student_path(student_file)
  env_cfg = course_g1_rough_traversal_env_cfg(mode, student_path)
  env_cfg.scene.num_envs = num_envs
  agent_cfg = course_g1_amp_ppo_runner_cfg(mode, student_path)
  env = ManagerBasedRlEnv(env_cfg, device=device, render_mode=render_mode)
  wrapped = ManualResetAmpVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  # Distillation checkpoints only carry student/actor weights; PPO checkpoints
  # carry critic + discriminator as well.  Detect and load accordingly.
  loaded = torch.load(str(checkpoint), map_location=device, weights_only=False)
  is_distill = "student_state_dict" in loaded or (
    "actor_state_dict" in loaded and "critic_state_dict" not in loaded
  )
  if is_distill:
    actor_sd = loaded.get("actor_state_dict") or loaded.get("student_state_dict")
    missing, unexpected = runner.alg._raw_actor.load_state_dict(actor_sd, strict=False)
    if missing:
      print(f"[evaluate] actor missing keys: {missing}")
    if unexpected:
      print(f"[evaluate] actor unexpected keys: {unexpected}")
  else:
    runner.load(str(checkpoint), load_cfg=LOAD_CFG)
  return env, wrapped, runner


def evaluate(
  checkpoint: str | Path,
  mode: ObservationMode = "height",
  *,
  num_envs: int = 32,
  steps: int = 600,
  device: str = "cuda:0",
  student_file: str | Path | None = None,
) -> dict[str, float]:
  """Measure tracking, traversal progress, and second-order action change."""
  env, wrapped, runner = _inference_runner(
    checkpoint,
    mode,
    num_envs=num_envs,
    device=device,
    student_file=student_file,
  )
  policy = runner.get_inference_policy(device=device)
  observations = wrapped.get_observations().to(device)
  start = env.scene["robot"].data.root_link_pos_w[:, :2].clone()
  max_distance = torch.zeros(num_envs, device=device)
  linear_errors: list[torch.Tensor] = []
  angular_errors: list[torch.Tensor] = []
  smoothness: list[torch.Tensor] = []
  previous = torch.zeros(num_envs, 29, device=device)
  previous_previous = previous.clone()
  try:
    with torch.inference_mode():
      for _ in range(steps):
        action = policy(observations)
        observations, _, dones, _ = wrapped.step(action)
        command = env.command_manager.get_command("twist")
        if not isinstance(command, torch.Tensor):
          raise TypeError("Command 'twist' must be a tensor")
        robot = env.scene["robot"]
        linear_errors.append(
          torch.norm(command[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1)
        )
        angular_errors.append(
          torch.abs(command[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
        )
        smoothness.append(
          torch.mean(torch.abs(action - 2 * previous + previous_previous), dim=-1)
        )
        distance = torch.norm(robot.data.root_link_pos_w[:, :2] - start, dim=-1)
        max_distance = torch.maximum(max_distance, distance)
        done_mask = dones.bool()
        if done_mask.any():
          start[done_mask] = robot.data.root_link_pos_w[done_mask, :2]
        previous_previous = previous
        previous = action
  finally:
    wrapped.close()
  progress = torch.clamp(max_distance / 6.0, 0.0, 1.0)
  return {
    "linear_velocity_error": float(torch.cat(linear_errors).mean()),
    "angular_velocity_error": float(torch.cat(angular_errors).mean()),
    "traversal_progress": float(progress.mean()),
    "traversal_success": float((max_distance >= 6.0).float().mean()),
    "smoothness": float(torch.cat(smoothness).mean()),
  }


def record_video(
  checkpoint: str | Path,
  mode: ObservationMode = "height",
  *,
  frames: int = 150,
  device: str = "cuda:0",
  student_file: str | Path | None = None,
  output: str | Path | None = None,
) -> Path:
  """Record a physical rollout MP4 for inline notebook display."""
  if frames < 150:
    raise ValueError("Evaluation videos must contain at least 150 frames")
  env, wrapped, runner = _inference_runner(
    checkpoint,
    mode,
    num_envs=1,
    device=device,
    student_file=student_file,
    render_mode="rgb_array",
  )
  policy = runner.get_inference_policy(device)
  observations = wrapped.get_observations().to(device)
  images: list[np.ndarray] = []
  try:
    with torch.inference_mode():
      for _ in range(frames):
        observations, _, _, _ = wrapped.step(policy(observations))
        frame = env.render()
        if frame is None:
          raise RuntimeError("mjlab offscreen renderer returned no frame")
        images.append(np.asarray(frame).copy())
  finally:
    wrapped.close()
  output_path = Path(output or EXP_ROOT / "outputs" / "evaluation.mp4")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  iio.imwrite(output_path, np.stack(images), fps=50, codec="libx264")
  return output_path


def prepare_submission(
  checkpoint: str | Path,
  mode: ObservationMode = "height",
  *,
  device: str = "cuda:0",
  student_file: str | Path | None = None,
  output_dir: str | Path | None = None,
) -> Path:
  """Prepare the strict policy.pt, model.py, student.py grading folder."""
  student_path = _student_path(student_file)
  cfg = course_g1_rough_walk_env_cfg(mode, play=True, student_path=student_path)
  cfg.scene.num_envs = 1
  agent_cfg = course_g1_amp_ppo_runner_cfg(mode, student_path)
  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg=LOAD_CFG)
  build_dir = Path(output_dir or EXP_ROOT / "outputs" / "submission")
  if build_dir.exists():
    shutil.rmtree(build_dir)
  build_dir.mkdir(parents=True)
  try:
    runner.export_policy_to_jit(str(build_dir), filename="policy.pt")
  finally:
    wrapped.close()
  (build_dir / "model.py").write_text(
    MODEL_DEPTH if mode == "depth" else MODEL_HEIGHT, encoding="utf-8"
  )
  shutil.copy2(student_path, build_dir / "student.py")
  return build_dir


def plot_training_design(cfg: Any):
  """Visualize reward weights and the three-stage command curriculum."""
  import matplotlib.pyplot as plt

  rewards = {
    name: float(term.weight)
    for name, term in cfg.rewards.items()
    if float(term.weight) != 0.0
  }
  curriculum = cfg.curriculum["course_command_schedule"].params["velocity_stages"]
  figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
  names = list(rewards)
  values = [rewards[name] for name in names]
  axes[0].barh(
    names, values, color=["#2878b5" if value > 0 else "#d1495b" for value in values]
  )
  axes[0].axvline(0.0, color="#222222", linewidth=0.8)
  axes[0].set_title("Reward weights")
  axes[0].grid(axis="x", alpha=0.2)
  steps = [stage["step"] for stage in curriculum]
  for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    maxima = [max(abs(value) for value in stage[key]) for stage in curriculum]
    axes[1].step(steps, maxima, where="post", marker="o", label=key)
  axes[1].set_title("Command curriculum")
  axes[1].set_xlabel("global step")
  axes[1].legend()
  axes[1].grid(alpha=0.2)
  figure.tight_layout()
  return figure


__all__ = [
  "evaluate",
  "eval_teacher_on_depth",
  "finetune_from_distill",
  "latest_checkpoint",
  "plot_training_design",
  "prepare_submission",
  "record_video",
  "smoke",
  "train",
  "train_distill",
]
