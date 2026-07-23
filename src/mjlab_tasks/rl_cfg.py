"""AMP-PPO and height/depth actor configurations for Experiment 07."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .env_cfgs import DEFAULT_MOTION_PATH, DEFAULT_STUDENT_PATH, ObservationMode

_VISION_MODEL = "CNNModel"
_VISION_CNN = {
  "output_channels": [16, 32, 64],
  "kernel_size": [5, 3, 3],
  "stride": [2, 2, 1],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "avg",
}
_VISION_CNN_LEGACY = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax": True,
  "spatial_softmax_temperature": 1.0,
}


@dataclass
class AmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Configuration fields consumed by the experiment-local :class:`AmpPPO`."""

  class_name: str = "src.amp.ppo:AmpPPO"
  motion_path: str = str(DEFAULT_MOTION_PATH)
  student_path: str = str(DEFAULT_STUDENT_PATH)
  step_dt: float = 0.02
  amp_observation_key: str = "amp"
  amp_reward_scale: float = 0.5
  discriminator_learning_rate: float = 1.0e-4
  discriminator_batch_size: int = 256
  discriminator_updates: int = 2
  replay_capacity: int = 50_000


def course_g1_amp_ppo_runner_cfg(
  observation_mode: ObservationMode = "height",
  student_path: str | Path = DEFAULT_STUDENT_PATH,
  motion_path: str | Path = DEFAULT_MOTION_PATH,
) -> RslRlOnPolicyRunnerCfg:
  """Return standard PPO with only the algorithm class replaced by AmpPPO."""
  obs_groups: dict[str, tuple[str, ...]]
  if observation_mode == "depth":
    actor = RslRlModelCfg(
      hidden_dims=(256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg=_VISION_CNN,
      class_name=_VISION_MODEL,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.2,
        "std_type": "scalar",
      },
    )
    obs_groups = {"actor": ("actor", "depth"), "critic": ("critic",)}
  else:
    actor = RslRlModelCfg(
      hidden_dims=(256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
    )
    obs_groups = {"actor": ("actor",), "critic": ("critic",)}

  algorithm = AmpPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=2,
    num_mini_batches=2,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.02,
    max_grad_norm=1.0,
    motion_path=str(Path(motion_path).resolve()),
    student_path=str(Path(student_path).resolve()),
  )
  return RslRlOnPolicyRunnerCfg(
    actor=actor,
    critic=RslRlModelCfg(
      hidden_dims=(256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=algorithm,
    obs_groups=obs_groups,
    experiment_name=f"exp07_rough_amp_{observation_mode}",
    run_name="course",
    logger="tensorboard",
    upload_model=False,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=600,
  )


@dataclass
class DistillationAlgorithmCfg:
  """Config fields for :class:`src.amp.dagger.DaggerDistillation`."""

  class_name: str = "src.amp.dagger:DaggerDistillation"
  num_learning_epochs: int = 2
  gradient_length: int = 15
  learning_rate: float = 1.0e-3
  max_grad_norm: float = 1.0
  loss_type: str = "mse"
  optimizer: str = "adam"
  teacher_mix_start: float = 1.0
  teacher_mix_end: float = 0.0
  teacher_mix_decay_iters: int = 300
  height_loss_coef: float = 0.2
  height_target_key: str = "height_target"
  student_init_std: float = 0.1


@dataclass
class RslRlDistillationRunnerCfg:
  """Minimal runner cfg matching RSL-RL DistillationRunner fields."""

  class_name: str = "src.amp.runner:MjlabDistillationRunner"
  seed: int = 42
  num_steps_per_env: int = 24
  max_iterations: int = 400
  obs_groups: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]
  save_interval: int = 50
  experiment_name: str = "exp07_rough_distill_depth"
  run_name: str = "dagger"
  logger: str = "tensorboard"
  wandb_project: str = "mjlab"
  wandb_tags: tuple[str, ...] = ()
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  clip_actions: float | None = None
  upload_model: bool = False
  student: RslRlModelCfg = None  # type: ignore[assignment]
  teacher: RslRlModelCfg = None  # type: ignore[assignment]
  algorithm: DistillationAlgorithmCfg = None  # type: ignore[assignment]


def course_g1_distill_runner_cfg(
  student_path: str | Path = DEFAULT_STUDENT_PATH,
  *,
  teacher_mix_decay_iters: int = 300,
  height_loss_coef: float = 0.2,
  student_init_std: float = 0.1,
) -> RslRlDistillationRunnerCfg:
  """Height teacher + depth student DAgger configuration."""
  del student_path  # kept for API symmetry with PPO cfg
  student = RslRlModelCfg(
    hidden_dims=(256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg=dict(_VISION_CNN),
    class_name=_VISION_MODEL,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": student_init_std,
      "std_type": "scalar",
    },
  )
  teacher = RslRlModelCfg(
    hidden_dims=(256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.1,
      "std_type": "scalar",
    },
  )
  algorithm = DistillationAlgorithmCfg(
    teacher_mix_decay_iters=teacher_mix_decay_iters,
    height_loss_coef=height_loss_coef,
    student_init_std=student_init_std,
  )
  return RslRlDistillationRunnerCfg(
    student=student,
    teacher=teacher,
    algorithm=algorithm,
    obs_groups={
      "student": ("actor", "depth"),
      "teacher": ("teacher",),
    },
    experiment_name="exp07_rough_distill_depth",
    run_name="dagger",
    logger="tensorboard",
    upload_model=False,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=400,
  )


@dataclass
class AmpPpoWithDistillAlgorithmCfg(AmpPpoAlgorithmCfg):
  """Algorithm config for :class:`src.amp.ppo.AmpPPOWithDistill`.

  Adds distillation regularisation fields on top of the standard AmpPPO
  hyperparameters.  The teacher model is loaded at runtime by
  :func:`finetune_from_distill`; this config only carries the hyper-params.
  """

  class_name: str = "src.amp.ppo:AmpPPOWithDistill"
  teacher_obs_key: str = "teacher"
  distill_coef: float = 1.0
  distill_coef_end: float = 0.05
  distill_decay_iters: int = 100
  distill_batch_size: int = 512


def course_g1_distill_finetune_runner_cfg(
  student_path: str | Path = DEFAULT_STUDENT_PATH,
  motion_path: str | Path = DEFAULT_MOTION_PATH,
  *,
  iterations: int = 150,
  distill_coef: float = 1.0,
  distill_coef_end: float = 0.05,
  distill_decay_iters: int = 100,
  amp_reward_scale: float = 0.0,
  student_init_std: float = 0.15,
) -> RslRlOnPolicyRunnerCfg:
  """Depth-student PPO finetune that keeps a frozen teacher for regularisation.

  The algorithm is :class:`AmpPPOWithDistill`: after every standard PPO + AMP
  update it adds ``distill_coef * MSE(student_mean, teacher_mean)`` to the
  actor.  The coefficient decays linearly over *distill_decay_iters* updates.

  The *teacher* obs group is included in ``obs_groups`` so the rollout buffer
  stores the privileged height-scan obs required by the distillation pass.

  Args:
    student_path: Path to student.py formula file.
    motion_path: Path to G1 expert motion NPZ.
    iterations: Number of PPO update iterations.
    distill_coef: Initial distillation loss weight (λ_start).
    distill_coef_end: Final distillation loss weight (λ_end).
    distill_decay_iters: Iterations over which λ decays from start to end.
    amp_reward_scale: AMP style reward scale (default 0 = pure PPO).
    student_init_std: Initial action std for the depth actor.
  """
  actor = RslRlModelCfg(
    hidden_dims=(256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg=dict(_VISION_CNN),
    class_name=_VISION_MODEL,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": student_init_std,
      "std_type": "scalar",
    },
  )
  algorithm = AmpPpoWithDistillAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=2,
    num_mini_batches=2,
    learning_rate=3.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    motion_path=str(Path(motion_path).resolve()),
    student_path=str(Path(student_path).resolve()),
    amp_reward_scale=amp_reward_scale,
    distill_coef=distill_coef,
    distill_coef_end=distill_coef_end,
    distill_decay_iters=distill_decay_iters,
  )
  # Include "teacher" obs group so the rollout buffer stores height-scan obs
  # that the distillation pass needs for teacher forward passes.
  obs_groups = {
    "actor": ("actor", "depth"),
    "critic": ("critic",),
    "teacher": ("teacher",),
  }
  return RslRlOnPolicyRunnerCfg(
    actor=actor,
    critic=RslRlModelCfg(
      hidden_dims=(256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=algorithm,
    obs_groups=obs_groups,
    experiment_name="exp07_rough_amp_depth_finetune",
    run_name="distill_ft",
    logger="tensorboard",
    upload_model=False,
    save_interval=max(1, min(50, iterations)),
    num_steps_per_env=24,
    max_iterations=iterations,
  )

