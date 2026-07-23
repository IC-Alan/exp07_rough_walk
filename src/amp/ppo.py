"""RSL-RL PPO extended only with AMP reward and discriminator updates."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from src.student_api import load_student_function

from .motion import MotionDataset
from .replay import ReplayBuffer
from .state import AMP_STATE_DIM, AMP_TRANSITION_DIM, build_transition


def _tensor_observation(obs: TensorDict, key: str) -> torch.Tensor:
  value = obs.get(key)
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Observation {key!r} must be a tensor")
  return value


class AmpDiscriminator(nn.Sequential):
  """The fixed ``166 -> 256 -> 128 -> 1`` AMP discriminator."""

  def __init__(self) -> None:
    super().__init__(
      nn.Linear(AMP_TRANSITION_DIM, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 1),
    )


class AmpPPO(PPO):
  """Standard RSL-RL PPO plus a two-step LSGAN discriminator update."""

  def __init__(
    self,
    *args,
    motion_path: str,
    student_path: str,
    step_dt: float,
    amp_observation_key: str = "amp",
    amp_reward_scale: float = 0.5,
    discriminator_learning_rate: float = 1.0e-4,
    discriminator_batch_size: int = 256,
    discriminator_updates: int = 2,
    replay_capacity: int = 50_000,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if step_dt <= 0.0:
      raise ValueError("step_dt must be positive")
    self.amp_observation_key = amp_observation_key
    self.amp_reward_scale = amp_reward_scale
    self.step_dt = step_dt
    self.discriminator_batch_size = discriminator_batch_size
    self.discriminator_updates = discriminator_updates
    self.discriminator = AmpDiscriminator().to(self.device)
    self.discriminator_optimizer = torch.optim.Adam(
      self.discriminator.parameters(), lr=discriminator_learning_rate
    )
    self.motion = MotionDataset(motion_path, student_path, device=self.device)
    self.replay = ReplayBuffer(
      capacity=replay_capacity,
      feature_dim=AMP_TRANSITION_DIM,
      device=self.device,
    )
    self._style_reward = load_student_function(student_path, "style_reward")
    self._discriminator_loss = load_student_function(
      student_path, "least_squares_discriminator_loss"
    )
    self._current_amp_state: torch.Tensor | None = None

  def act(self, obs: TensorDict) -> torch.Tensor:
    amp_state = _tensor_observation(obs, self.amp_observation_key)
    if amp_state.ndim != 2 or amp_state.shape[-1] != AMP_STATE_DIM:
      raise ValueError(
        f"Observation '{self.amp_observation_key}' must be [B, 83], "
        f"got {tuple(amp_state.shape)}"
      )
    self._current_amp_state = amp_state.detach().clone()
    return super().act(obs)

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    if self._current_amp_state is None:
      raise RuntimeError("AmpPPO.process_env_step() called before act()")
    next_amp = _tensor_observation(obs, self.amp_observation_key).detach().clone()
    terminal_amp = extras.get("terminal_amp_observation")
    if terminal_amp is not None:
      if not isinstance(terminal_amp, torch.Tensor):
        raise TypeError("terminal_amp_observation must be a tensor")
      done_mask = dones.bool()
      next_amp[done_mask] = terminal_amp.to(self.device)[done_mask]
    transition = build_transition(self._current_amp_state, next_amp)
    with torch.no_grad():
      discriminator_output = self.discriminator(transition).squeeze(-1)
      style = self._style_reward(discriminator_output)
    if style.shape != rewards.shape or not torch.isfinite(style).all():
      raise ValueError(
        "style_reward() must return a finite tensor matching rewards; "
        f"got {tuple(style.shape)} and {tuple(rewards.shape)}"
      )
    shaped_rewards = rewards + self.amp_reward_scale * self.step_dt * style
    self.replay.add(transition)
    super().process_env_step(obs, shaped_rewards, dones, extras)

  def update(self) -> dict[str, float]:
    losses = super().update()
    if len(self.replay) == 0:
      losses["discriminator"] = 0.0
      losses["gradient_penalty"] = 0.0
      return losses
    discriminator_loss = 0.0
    gradient_penalty = 0.0
    for _ in range(self.discriminator_updates):
      disc, penalty = self._update_discriminator()
      discriminator_loss += disc
      gradient_penalty += penalty
    divisor = max(1, self.discriminator_updates)
    losses["discriminator"] = discriminator_loss / divisor
    losses["gradient_penalty"] = gradient_penalty / divisor
    return losses

  def _update_discriminator(self) -> tuple[float, float]:
    expert = self.motion.sample_transitions(self.discriminator_batch_size)
    policy = self.replay.sample(self.discriminator_batch_size)
    expert_for_grad = expert.detach().requires_grad_(True)
    expert_output = self.discriminator(expert_for_grad).squeeze(-1)
    policy_output = self.discriminator(policy).squeeze(-1)
    gradient = torch.autograd.grad(
      outputs=expert_output.sum(),
      inputs=expert_for_grad,
      create_graph=True,
    )[0]
    penalty = ((gradient.norm(2, dim=-1) - 1.0) ** 2).mean()
    loss = self._discriminator_loss(expert_output, policy_output, penalty)
    if loss.ndim != 0 or not torch.isfinite(loss):
      raise ValueError("least_squares_discriminator_loss() must return a finite scalar")
    self.discriminator_optimizer.zero_grad()
    loss.backward()
    self.discriminator_optimizer.step()
    return float(loss.detach()), float(penalty.detach())

  def train_mode(self) -> None:
    super().train_mode()
    self.discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.discriminator.eval()

  def save(self) -> dict:
    saved = super().save()
    saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
    saved["amp_discriminator_optimizer_state_dict"] = (
      self.discriminator_optimizer.state_dict()
    )
    return saved

  def load(self, loaded_dict: Mapping, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(dict(loaded_dict), load_cfg, strict)
    self.discriminator.load_state_dict(
      loaded_dict["amp_discriminator_state_dict"], strict=strict
    )
    if load_cfg is None or load_cfg.get("optimizer", True):
      self.discriminator_optimizer.load_state_dict(
        loaded_dict["amp_discriminator_optimizer_state_dict"]
      )
    return load_iteration


# ---------------------------------------------------------------------------
# Phase-3 helper: PPO + AMP + distillation regularisation
# ---------------------------------------------------------------------------

_DISTILL_SKIP_KEYS = frozenset({"critic", "amp", "teacher", "height_target"})


class AmpPPOWithDistill(AmpPPO):
  """AmpPPO with an optional teacher distillation regularisation pass.

  After every standard PPO + AMP update a single extra gradient step adds::

      distill_coef * MSE(student_mean, teacher_mean.detach())

  to the actor.  The coefficient decays linearly from *distill_coef* to
  *distill_coef_end* over *distill_decay_iters* calls to ``update()``.

  Usage::

      alg = AmpPPOWithDistill(...)
      alg.set_teacher_for_distill(frozen_teacher_model)
  """

  def __init__(
    self,
    *args,
    teacher_obs_key: str = "teacher",
    distill_coef: float = 1.0,
    distill_coef_end: float = 0.05,
    distill_decay_iters: int = 100,
    distill_batch_size: int = 512,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.teacher_obs_key = teacher_obs_key
    self._distill_coef_start = float(distill_coef)
    self._distill_coef_end = float(distill_coef_end)
    self._distill_decay_iters = max(1, int(distill_decay_iters))
    self._distill_batch_size = int(distill_batch_size)
    self.distill_coef = float(distill_coef)
    self._distill_iter = 0
    self._teacher_for_distill: nn.Module | None = None
    # Buffers accumulated during process_env_step for the distillation pass
    self._dbuf_teacher: list[torch.Tensor] = []
    self._dbuf_actor: list[dict[str, torch.Tensor]] = []

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def set_teacher_for_distill(self, teacher: nn.Module) -> None:
    """Freeze *teacher* and register it as the distillation supervisor."""
    self._teacher_for_distill = teacher.eval()
    for p in self._teacher_for_distill.parameters():
      p.requires_grad_(False)

  # ------------------------------------------------------------------
  # Override hooks
  # ------------------------------------------------------------------

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict,
  ) -> None:
    """Accumulate teacher obs for the later distillation pass."""
    if self._teacher_for_distill is not None:
      teacher_obs = obs.get(self.teacher_obs_key)
      if teacher_obs is not None:
        self._dbuf_teacher.append(teacher_obs.detach().clone())
        actor_dict = {
          k: obs[k].detach().clone()
          for k in obs.keys()
          if k not in _DISTILL_SKIP_KEYS
        }
        self._dbuf_actor.append(actor_dict)
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    """PPO + AMP update followed by a distillation regularisation step."""
    loss_dict = super().update()
    distill_val = 0.0
    if (
      self._teacher_for_distill is not None
      and self.distill_coef > 0.0
      and self._dbuf_teacher
    ):
      distill_val = self._distill_pass()
    # Anneal coefficient and clear buffers
    self._distill_iter += 1
    progress = min(1.0, self._distill_iter / self._distill_decay_iters)
    self.distill_coef = self._distill_coef_start + (
      self._distill_coef_end - self._distill_coef_start
    ) * progress
    self._dbuf_teacher.clear()
    self._dbuf_actor.clear()
    loss_dict["distill"] = distill_val
    loss_dict["distill_coef"] = self.distill_coef
    return loss_dict

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _distill_pass(self) -> float:
    """One gradient step: MSE(student_mean, teacher_mean) on a random sub-batch."""
    import torch.nn.functional as F

    teacher_all = torch.cat(self._dbuf_teacher, dim=0).to(self.device)
    keys = list(self._dbuf_actor[0].keys())
    actor_all = {k: torch.cat([d[k] for d in self._dbuf_actor], dim=0).to(self.device) for k in keys}

    N = teacher_all.shape[0]
    bs = min(self._distill_batch_size, N)
    idx = torch.randperm(N, device=self.device)[:bs]

    teacher_in = TensorDict(
      {self.teacher_obs_key: teacher_all[idx]}, batch_size=[bs]
    )
    actor_in = TensorDict({k: actor_all[k][idx] for k in keys}, batch_size=[bs])

    with torch.no_grad():
      t_mean = self._teacher_for_distill(teacher_in, stochastic_output=False)

    s_mean = self._raw_actor(actor_in, stochastic_output=False)
    distill_loss = F.mse_loss(s_mean, t_mean)

    self.optimizer.zero_grad()
    (self.distill_coef * distill_loss).backward()
    nn.utils.clip_grad_norm_(self._raw_actor.parameters(), self.max_grad_norm)
    self.optimizer.step()
    return float(distill_loss.detach())
