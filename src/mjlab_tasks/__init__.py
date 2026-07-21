"""Register the independent Experiment 07 height and depth tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import course_g1_rough_walk_env_cfg
from .rl_cfg import course_g1_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Course-G1-Rough-AMP-Height",
  env_cfg=course_g1_rough_walk_env_cfg("height"),
  play_env_cfg=course_g1_rough_walk_env_cfg("height", play=True),
  rl_cfg=course_g1_amp_ppo_runner_cfg("height"),
)

register_mjlab_task(
  task_id="Course-G1-Rough-AMP-Depth",
  env_cfg=course_g1_rough_walk_env_cfg("depth"),
  play_env_cfg=course_g1_rough_walk_env_cfg("depth", play=True),
  rl_cfg=course_g1_amp_ppo_runner_cfg("depth"),
)
