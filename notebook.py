#!/usr/bin/env python
# coding: utf-8

# # Exp07 Rough Walk
# 
# 本实验在粗糙地形上训练带 AMP 风格先验的 G1 速度策略，并比较 height scan 与可选
# depth observation。使用 Conda `summer` kernel，并在该环境中用 pip 安装
# `mjlab==1.5.0`。提交文件夹只包含
# `policy.pt`、`model.py`、`student.py`。
# 
# 满分 100：提交与有限动作 20；速度跟踪 20；6 m 穿越 20；AMP 10；depth 10；
# 平滑度 20。`G1_walk_50hz.npz` 源自 humanoid_amp 固定提交并按 BSD-3-Clause
# 使用，完整哈希和关节顺序见 `assets/motions/manifest.json`，许可文本见
# `assets/motions/NOTICE.txt`。

# In[1]:



import importlib.util
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

#import matplotlib.pyplot as plt
import torch
from IPython.display import Video, display
from tqdm.auto import tqdm


def find_exp_root(name: str) -> Path:
  for candidate in (Path.cwd(), *Path.cwd().parents):
    if candidate.name == name:
      return candidate
    nested = candidate / name
    if nested.is_dir():
      return nested
  raise FileNotFoundError(name)


def load_student(path: Path):
  spec = importlib.util.spec_from_file_location("active_student", path)
  if spec is None or spec.loader is None:
    raise ImportError(path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


#assert Path(sys.prefix).name == "summer", "请切换到 Conda summer kernel"
assert version("mjlab") == "1.5.0"

EXP_ROOT = find_exp_root("exp07_rough_walk")
COURSE_ROOT = EXP_ROOT.parent
STUDENT_FILE = EXP_ROOT / "student.py"
MODE = "height"  # 可改为 "depth"
sys.path.insert(0, str(EXP_ROOT))

from src import workflow  # noqa: E402
from src.mjlab_tasks.env_cfgs import (  # noqa: E402
  course_g1_rough_walk_env_cfg,
)

cfg = course_g1_rough_walk_env_cfg(MODE, student_path=STUDENT_FILE)
assert cfg.auto_reset is False and set(cfg.actions) == {"joint_pos"}
print("环境初始化完成：29 维动作，83 维 AMP state，mode=", MODE)


# ## AMP 状态、判别器与风格奖励
# 
# 单状态拼接 29 维关节位置、29 维关节速度、pelvis 高度、projected gravity、
# yaw-local base 线/角速度以及 5 个关键 body 的 pelvis-relative position，共 83 维；
# 相邻状态组成 166 维判别器输入。
# 
# 完成 `build_amp_state()`、LSGAN
# $L_D=\frac12[(D(s_E)-1)^2+(D(s_\pi)+1)^2]+10L_{gp}$，以及
# $r_{style}=\operatorname{clip}(1-0.25(D-1)^2,0,1)$。

# In[2]:


s = load_student(STUDENT_FILE)
parts = (
  torch.zeros(2, 29), torch.ones(2, 29), torch.ones(2, 1),
  torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3),
  torch.zeros(2, 5, 3),
)
state = s.build_amp_state(*parts)
assert state.shape == (2, 83) and torch.isfinite(state).all()
print("代码检查通过；build_amp_state 对应最终 AMP 10 分的一部分")
loss = s.least_squares_discriminator_loss(
  torch.ones(8), -torch.ones(8), torch.tensor(0.0)
)
torch.testing.assert_close(loss, torch.tensor(0.0))
print("代码检查通过；LSGAN 公式用于训练稳定性")
torch.testing.assert_close(
  s.style_reward(torch.tensor([1.0, 3.0])), torch.tensor([1.0, 0.0])
)
print("代码检查通过；style_reward 对应最终 AMP 10 分的一部分")


# ## Depth、任务奖励与平滑度
# 
# Depth 裁剪到 $[0.1,5.0]$ m 后归一化；任务奖励在零误差时为 1，并随线速度和角速度
# 误差下降。平滑项使用二阶差分
# $p_t=\operatorname{mean}(|a_t-2a_{t-1}+a_{t-2}|)$。完成剩余三个函数。

# In[3]:


s = load_student(STUDENT_FILE)
depth = s.normalize_depth(torch.tensor([[[[0.0, 0.1, 5.0, 8.0]]]]))
torch.testing.assert_close(depth, torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]]]))
print("代码检查通过；normalize_depth 对应可选 depth 10 分")
torch.testing.assert_close(
  s.rough_task_reward(torch.zeros(4), torch.zeros(4)), torch.ones(4)
)
print("代码检查通过；rough_task_reward 对应任务 40 分")
actions = torch.ones(3, 29)
torch.testing.assert_close(
  s.smoothness_penalty(actions, actions, actions), torch.zeros(3)
)
print("代码检查通过；smoothness_penalty 对应平滑度 20 分")


# ## 环境设计与 smoke
# 
# Actor 仅接收 proprioception、command 与 height/depth；critic 保留 privileged
# observation。命令课程逐步增加速度与转向范围。下面先显示奖励和课程，再运行
# 32-env、16-step、强制终止与手动 reset 检查。

# In[4]:


# %%time
# display(workflow.plot_training_design(cfg))
# smoke_result = workflow.smoke(
#   MODE, num_envs=32, steps=16, device="cuda:0",
#   student_file=STUDENT_FILE, force_termination=True,
# )
# display(smoke_result)


# ## AMP-PPO 训练
# 
# 标准 PPO 每次更新额外进行两次 AMP discriminator 更新。Height 默认 4096 env；
# depth 先以 32 env 检查显存。训练结束后 workflow 会从 outputs 中选择最新 checkpoint。

# In[ ]:




# In[ ]:


# %%time
# CHECKPOINT = workflow.latest_checkpoint()
# metrics = workflow.evaluate(
#   CHECKPOINT, MODE, num_envs=32, steps=600,
#   device="cuda:0", student_file=STUDENT_FILE,
# )
# display(metrics)
# figure, axis = plt.subplots(figsize=(8, 3.5))
# axis.bar(metrics, metrics.values(), color="#2878b5")
# axis.set_title("Exp07 held-out evaluation")
# axis.grid(axis="y", alpha=0.2)
# display(figure)
# for checkpoint in tqdm([CHECKPOINT], desc="录制 150 帧视频"):
#   video_path = workflow.record_video(
#     checkpoint, MODE, frames=150, device="cuda:0", student_file=STUDENT_FILE
#   )
# display(Video(str(video_path), embed=True))


# ## 训练结果视频

# In[ ]:


import numpy as np
import imageio.v3 as iio
from src.workflow import _inference_runner


def record_multi_terrain(
    checkpoint, mode="height", *,
    frames_per_terrain=600, num_terrains=4,
    device="cuda:0", student_file=None, output=None,
):
    """每种地形录 frames_per_terrain 帧（12s @ 50fps），展示多种地形适应能力"""
    env, wrapped, runner = _inference_runner(
        checkpoint, mode, num_envs=1, device=device,
        student_file=student_file, render_mode="rgb_array",
    )
    policy = runner.get_inference_policy(device)
    observations = wrapped.get_observations().to(device)
    images = []
    total_frames = frames_per_terrain * num_terrains
    try:
        with torch.inference_mode():
            for i in range(total_frames):
                # 每段地形开始时 reset，随机化地形
                if i > 0 and i % frames_per_terrain == 0:
                    env.reset(env_ids=torch.tensor([0], device=device))
                    observations = wrapped.get_observations().to(device)

                observations, _, dones, _ = wrapped.step(policy(observations))

                # 摔倒也 reset 到新地形
                if dones.item():
                    observations = wrapped.get_observations().to(device)

                frame = env.render()
                if frame is None:
                    raise RuntimeError("mjlab offscreen renderer returned no frame")
                images.append(np.asarray(frame).copy())
    finally:
        wrapped.close()

    output_path = Path(output or EXP_ROOT / "outputs" / "multi_terrain.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, np.stack(images), fps=50, codec="libx264")
    return output_path


# 使用 height checkpoint
CHECKPOINT = Path(
    "/home/alan/Desktop/rl/hw/exp07_rough_walk/outputs/rsl_rl/"
    "exp07_rough_amp_height/2026-07-23_20-55-52/model_2999.pt"
)
print(f"Checkpoint: {CHECKPOINT}")

video_path = record_multi_terrain(
    CHECKPOINT, MODE, frames_per_terrain=600, num_terrains=4,
    device="cuda:0", student_file=STUDENT_FILE,
)
print(f"Video: {video_path}")
display(Video(str(video_path), embed=True))


# ## Height Teacher → Depth Student DAgger 蒸馏
# 
# 三阶段训练：
# 1. **阶段 0（验证）**：Height teacher 在 depth 环境中控制，验证动力学正常。
# 2. **阶段 1+2（DAgger）**：冻结 teacher，纯动作蒸馏 + 在线 DAgger，辅以 depth→height 重建监督。
#    执行动作由 teacher/student 按 env 级 Bernoulli(β) 选择，β 从 1.0 线性降至 0.0。
#    损失：`L = MSE(μ_student, μ_teacher) + 0.2 * SmoothL1(ĥ_depth, h_scan)`
# 3. **阶段 3（微调）**：恢复 PPO + AMP，蒸馏权重逐渐衰减。

# ### 阶段 0：Teacher 在 Depth 环境的验证
# 
# 用 height teacher 直接控制 depth 模式的仿真环境（teacher 仍使用 privileged `height_scan`）。
# 若 `max_distance_m` ≥ 2 m，则环境正常，可进行蒸馏；若 teacher 也无法行走，
# 说明 depth 环境本身（相机仿真开销、控制频率）存在问题，需先修复环境。

# In[ ]:


# ### 阶段 1+2：在线 DAgger 蒸馏（纯动作蒸馏 → β 衰减）
# 
# - 前期（β≈1）：几乎所有 env 由 teacher 控制，student 安全模仿；
# - 后期（β→0）：student 接管所有 env，teacher 仅提供监督标签；
# - 辅助损失：depth encoder 同时预测 `height_scan`（系数 0.2），
#   使 CNN 直接学习局部地形表征。

# In[ ]:
# ### 蒸馏后评估：student 单独运行

# In[ ]:


# ### 阶段 3：PPO + AMP 联合微调
# 
# Student 已具备基本步态后，恢复环境奖励和 AMP 风格奖励。
# AMP 权重降低至 0.2（避免判别器扭曲已习得的步态），蒸馏初始噪声 0.15。
# 若 DAgger 结束时 student 已稳定行走，本阶段仅微调速度跟踪精度；
# 若 student 仍不稳定，可先延长 DAgger iterations 再进入本阶段。

# In[ ]:




# ### Depth 模型最终评估与视频

# In[ ]:


# get_ipython().run_cell_magic('time', '', 'depth_metrics = workflow.evaluate(\n    DEPTH_CKPT,\n    "depth",\n    num_envs=32,\n    steps=600,\n    device="cuda:0",\n    student_file=STUDENT_FILE,\n)\ndisplay(depth_metrics)\nfigure, axis = plt.subplots(figsize=(8, 3.5))\naxis.bar(list(depth_metrics.keys()), list(depth_metrics.values()), color="#2878b5")\naxis.set_title("Depth policy final evaluation")\naxis.grid(axis="y", alpha=0.2)\ndisplay(figure)\n')


# # In[ ]:


# get_ipython().run_cell_magic('time', '', '# 600 frames @ 50 fps ≈ 12 s; multi_terrain cycles terrain types/difficulty\n# every ~frames_per_terrain steps so stairs/slopes/flat all appear.\ndepth_video = workflow.record_video(\n    DEPTH_CKPT, "depth",\n    frames=2400,\n    device="cuda:0",\n    student_file=STUDENT_FILE,\n    multi_terrain=True,\n    terrain_rows=4,\n    frames_per_terrain=600,  # ~1.6 s per terrain patch\n)\ndisplay(Video(str(depth_video), embed=True))\n')


# # ### Depth 模型提交

# # In[ ]:


# get_ipython().run_cell_magic('time', '', '# DISTILL_CKPT (DAgger) > DEPTH_CKPT (finetune) on all metrics:\n# linear_err 0.226 vs 0.418, angular_err 0.581 vs 0.846, smooth 0.086 vs 0.121\n# Use DISTILL_CKPT as primary submission.\n# Switch to DEPTH_CKPT only if it gets a better grade.\nsubmission = workflow.prepare_submission(\n    DISTILL_CKPT, "depth", device="cuda:0", student_file=STUDENT_FILE\n)\nprint("submission:", submission)\n# subprocess.run(\n#   [sys.executable, str(COURSE_ROOT / "grading_toolkit" / "grade.py"),\n#    str(submission), "--task", "exp07", "--device", "cuda:0"],\n#   cwd=COURSE_ROOT,\n#   check=True,\n# )\n')


# # In[ ]:


# # %%time
# # submission = workflow.prepare_submission(
# #   CHECKPOINT, MODE, device="cuda:0", student_file=STUDENT_FILE
# # )
# # print("submission:", submission)
# # subprocess.run(
# #   [sys.executable, str(COURSE_ROOT / "grading_toolkit" / "grade.py"),
# #    str(submission), "--task", "exp07", "--device", "cuda:0"],
# #   cwd=COURSE_ROOT,
# #   check=True,
# # )

