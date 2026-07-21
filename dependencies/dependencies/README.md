# 本地依赖安装说明

`dependencies/` 存放实验四、实验七和实验八需要用到的本地源码依赖。不要只把该目录加入 `PYTHONPATH`；请先在当前 Python 环境中以 editable mode 安装，这样 notebook、脚本和命令行入口会使用同一套本地源码。

## 安装步骤

在实训包根目录执行：

```bash
conda activate summer
python -m pip install -U pip
python -m pip install -e dependencies/mujoco_warp
python -m pip install -e dependencies/mjlab
python -m pip install -e dependencies/terrain-generator
```

安装顺序建议保持如上：

- `mujoco_warp` 是本地 MuJoCo Warp 依赖。
- `mjlab` 依赖 MuJoCo、MuJoCo Warp、PyTorch、rsl-rl 等组件，实验四和实验七使用它加载 G1 locomotion 任务。
- `terrain-generator` 提供地形生成、地图读取和路线规划接口，实验八使用它构建 navigation 环境。

## 对应实验

- `exp04_flat_walk/`: 需要 `mujoco_warp` 和 `mjlab`。
- `exp07_rough_curr_walk/`: 需要 `mujoco_warp` 和 `mjlab`。
- `exp08_navigation_train/`: 需要 `terrain-generator`；如果使用统一环境，也可以直接安装上面三项。
- `exp08_navigation_grade/`: 评分脚本读取同样的 navigation 地图格式，也需要 `terrain-generator`。

完成 editable install 后，再打开对应实验文件夹中的 `notebook.ipynb` 或运行 README 中的脚本命令。
