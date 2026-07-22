"""Height-teacher to depth-student DAgger distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer
from tensordict import TensorDict


class DaggerDistillation(Distillation):
  """Student-teacher distillation with DAgger action mixing.

  Environment actions are a mixture of teacher mean and student sample:

  ``a = beta * mu_teacher + (1 - beta) * a_student``

  while the supervised target is always the teacher action mean.  Optional
  depth-to-height reconstruction uses a linear head on the student latent.
  """

  def __init__(
    self,
    student: MLPModel,
    teacher: MLPModel,
    storage: RolloutStorage,
    *,
    teacher_mix_start: float = 1.0,
    teacher_mix_end: float = 0.0,
    teacher_mix_decay_iters: int = 300,
    height_loss_coef: float = 0.2,
    height_target_key: str = "height_target",
    student_init_std: float = 0.1,
    num_learning_epochs: int = 1,
    gradient_length: int = 15,
    learning_rate: float = 1e-3,
    max_grad_norm: float | None = None,
    loss_type: str = "mse",
    optimizer: str = "adam",
    device: str = "cpu",
    multi_gpu_cfg: dict | None = None,
    **kwargs,
  ) -> None:
    del kwargs
    super().__init__(
      student,
      teacher,
      storage,
      num_learning_epochs=num_learning_epochs,
      gradient_length=gradient_length,
      learning_rate=learning_rate,
      max_grad_norm=max_grad_norm,
      loss_type=loss_type,
      optimizer=optimizer,
      device=device,
      multi_gpu_cfg=multi_gpu_cfg,
    )
    self.teacher_mix_start = float(teacher_mix_start)
    self.teacher_mix_end = float(teacher_mix_end)
    self.teacher_mix_decay_iters = max(1, int(teacher_mix_decay_iters))
    self.height_loss_coef = float(height_loss_coef)
    self.height_target_key = height_target_key
    self.beta = self.teacher_mix_start
    self.height_head: nn.Linear | None = None
    self._height_latent_dim = int(self.student._get_latent_dim())  # noqa: SLF001
    self._set_student_std(student_init_std)
    self._freeze_teacher()
    params = list(self.student.parameters())
    self.optimizer = resolve_optimizer(optimizer)(params, lr=learning_rate)

  def _freeze_teacher(self) -> None:
    self.teacher.eval()
    for parameter in self.teacher.parameters():
      parameter.requires_grad_(False)

  def _set_student_std(self, init_std: float) -> None:
    dist = getattr(self.student, "distribution", None)
    if dist is None:
      return
    value = float(init_std)
    with torch.no_grad():
      if hasattr(dist, "std_param"):
        dist.std_param.fill_(value)
      elif hasattr(dist, "log_std_param"):
        dist.log_std_param.fill_(torch.log(torch.tensor(value, device=self.device)))

  def _ensure_height_head(self, height_dim: int) -> nn.Linear:
    if self.height_head is None:
      self.height_head = nn.Linear(self._height_latent_dim, height_dim).to(self.device)
      self.optimizer.add_param_group({"params": list(self.height_head.parameters())})
    return self.height_head

  def set_iteration(self, iteration: int) -> None:
    """Linearly anneal teacher mixing ratio over the requested iterations."""
    progress = min(1.0, max(0.0, float(iteration) / float(self.teacher_mix_decay_iters)))
    self.beta = self.teacher_mix_start + (
      self.teacher_mix_end - self.teacher_mix_start
    ) * progress

  def act(self, obs: TensorDict) -> torch.Tensor:
    with torch.no_grad():
      teacher_mean = self.teacher(obs, stochastic_output=False)
    student_action = self.student(obs, stochastic_output=True).detach()
    mixed = self.beta * teacher_mean + (1.0 - self.beta) * student_action
    self.transition.actions = mixed
    self.transition.privileged_actions = teacher_mean.detach()
    self.transition.observations = obs
    return mixed

  def update(self) -> dict[str, float]:
    self.num_updates += 1
    mean_behavior_loss = 0.0
    mean_height_loss = 0.0
    loss = 0.0
    cnt = 0
    height_cnt = 0

    for _ in range(self.num_learning_epochs):
      self.student.reset(hidden_state=self.last_hidden_states[0])
      self.teacher.reset(hidden_state=self.last_hidden_states[1])
      self.student.detach_hidden_state()
      for batch in self.storage.generator():
        student_mean = self.student(batch.observations, stochastic_output=False)
        behavior_loss = self.loss_fn(student_mean, batch.privileged_actions)
        step_loss = behavior_loss
        if self.height_loss_coef > 0.0 and self.height_target_key in batch.observations:
          target = batch.observations[self.height_target_key]
          if target.ndim > 2:
            target = target.flatten(start_dim=1)
          head = self._ensure_height_head(int(target.shape[-1]))
          latent = self.student.get_latent(batch.observations)
          pred = head(latent)
          height_loss = F.smooth_l1_loss(pred, target)
          step_loss = step_loss + self.height_loss_coef * height_loss
          mean_height_loss += float(height_loss.detach())
          height_cnt += 1

        loss = loss + step_loss
        mean_behavior_loss += float(behavior_loss.detach())
        cnt += 1

        if cnt % self.gradient_length == 0:
          self.optimizer.zero_grad()
          loss.backward()
          if self.is_multi_gpu:
            self.reduce_parameters()
          if self.max_grad_norm:
            params = list(self.student.parameters())
            if self.height_head is not None:
              params += list(self.height_head.parameters())
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
          self.optimizer.step()
          self.student.detach_hidden_state()
          loss = 0.0

        self.student.reset(batch.dones.view(-1))
        self.teacher.reset(batch.dones.view(-1))
        self.student.detach_hidden_state(batch.dones.view(-1))

    mean_behavior_loss /= max(1, cnt)
    self.storage.clear()
    self.last_hidden_states = (
      self.student.get_hidden_state(),
      self.teacher.get_hidden_state(),
    )
    self.student.detach_hidden_state()
    out = {
      "behavior": mean_behavior_loss,
      "teacher_mix_beta": float(self.beta),
    }
    if height_cnt > 0:
      out["height_recon"] = mean_height_loss / height_cnt
    return out

  def save(self) -> dict:
    saved = super().save()
    if self.height_head is not None:
      saved["height_head_state_dict"] = self.height_head.state_dict()
    saved["dagger_beta"] = self.beta
    # Allow later PPO finetune loaders to treat student as actor.
    saved["actor_state_dict"] = self._raw_student.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    if "height_head_state_dict" in loaded_dict and self.height_loss_coef > 0.0:
      state = loaded_dict["height_head_state_dict"]
      weight = state.get("weight")
      if weight is not None:
        head = self._ensure_height_head(int(weight.shape[0]))
        head.load_state_dict(state, strict=strict)
    if "dagger_beta" in loaded_dict:
      self.beta = float(loaded_dict["dagger_beta"])
    self._freeze_teacher()
    return load_iteration

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> DaggerDistillation:
    alg_class: type[DaggerDistillation] = resolve_callable(
      cfg["algorithm"].pop("class_name")
    )  # type: ignore
    student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
    teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore

    default_sets = ["student", "teacher"]
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    if cfg["algorithm"].get("rnd_cfg") is not None and "rnd_cfg" in cfg["algorithm"]:
      if cfg["algorithm"]["rnd_cfg"] is not None:
        raise ValueError("RND is not compatible with distillation.")
    cfg["algorithm"]["rnd_cfg"] = None
    if cfg["algorithm"].get("symmetry_cfg") is not None:
      raise ValueError("Symmetry is not compatible with distillation.")
    cfg["algorithm"]["symmetry_cfg"] = None

    student = student_class(
      obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]
    ).to(device)
    teacher = teacher_class(
      obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]
    ).to(device)
    print(f"Student Model: {student}")
    print(f"Teacher Model: {teacher}")

    storage = RolloutStorage(
      "distillation",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
    )
    alg: DaggerDistillation = alg_class(
      student,
      teacher,
      storage,
      device=device,
      multi_gpu_cfg=cfg["multi_gpu"],
      **cfg["algorithm"],
    )
    alg.compile(cfg.get("torch_compile_mode"))
    return alg
