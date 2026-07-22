"""Distillation runner with mjlab checkpoint compatibility helpers."""

from __future__ import annotations

import os
import time

import torch
from rsl_rl.runners.distillation_runner import DistillationRunner
from rsl_rl.utils import check_nan

from src.amp.dagger import DaggerDistillation


class MjlabDistillationRunner(DistillationRunner):
  """DistillationRunner that strips unused model fields and anneals DAgger beta."""

  alg: DaggerDistillation

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    for key in ("student", "teacher"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    train_cfg.setdefault("algorithm", {})
    train_cfg["algorithm"].setdefault("rnd_cfg", None)
    train_cfg["algorithm"].setdefault("symmetry_cfg", None)
    super().__init__(env, train_cfg, log_dir, device)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if not self.alg.teacher_loaded:
      raise ValueError(
        "Teacher model parameters not loaded. Pass teacher_checkpoint to train_distill()."
      )
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      if hasattr(self.alg, "set_iteration"):
        self.alg.set_iteration(it)
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras, None)
        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      action_std = None
      policy = self.alg.get_policy()
      if getattr(policy, "distribution", None) is not None:
        try:
          action_std = policy.output_std
        except Exception:
          action_std = None
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=getattr(self.alg, "learning_rate", 0.0),
        action_std=action_std,
        rnd_weight=None,
      )
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg.get("upload_model"):
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    # Height PPO checkpoints expose actor_state_dict for the teacher.
    if "actor_state_dict" in loaded_dict and "teacher_state_dict" not in loaded_dict:
      if load_cfg is None:
        load_cfg = {
          "teacher": True,
          "student": False,
          "optimizer": False,
          "iteration": False,
        }
      # Migrate rsl-rl 4.x actor keys if present.
      actor_sd = loaded_dict.get("actor_state_dict", {})
      if "std" in actor_sd:
        actor_sd["distribution.std_param"] = actor_sd.pop("std")
      if "log_std" in actor_sd:
        actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]
    infos = loaded_dict.get("infos") or {}
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
