# terrain-generator

`terrain-generator` 是一套独立的、纯程序化的地形生成与可视化接口。

当前实现目标：

- 不依赖 OpenUSD
- 使用程序化代码直接生成地形几何
- 网格保存为 `.obj`
- 元数据保存为 `.npz`
- 使用 Open3D 做交互式可视化（推荐），pyrender / matplotlib 作为备选
- 提供出生点提取、距离查询和路径重建能力

目前已经落地的地形类型：

- `door`
- `forest`
- `platform_gap`
- `pile`
- `pyramid_stairs`
- `stakes`

其中 `pile` 支持“中心高台 + 梅花桩路径”的程序化生成，并提供 `cross_route`、`full_grid`、`custom_mask` 三种路径模式。
`pyramid_stairs` 会在同一个 terrain family 内随机生成上楼梯或下楼梯；`platform_gap` 生成平台/坑洞结构；`stakes` 生成可踩桩点地形。

此外，新包现在也已经包含一套通用 `mesh parts -> MeshTile -> WFC` 管线，可以把：

- 旧式 height map / platform / wall / capsule / box tile
- 新的程序化 terrain tile

统一放进 WFC 流程。

## 1. 安装

### 1.1 基础安装

如果你只需要：

- 地形几何生成
- `.obj` 导出
- `.npz` 加载/保存
- 可视化（推荐安装 Open3D，见下方）

可以直接安装基础依赖：

```bash
cd terrain-generator
pip install -e .
```

基础安装会同时安装 `rtree`。当前版本的 `trimesh` 射线查询依赖它来构建空间索引；
`MeshTerrain` 的地形高度采样、WFC tile 边界采样和导航图生成都会走这条路径。

如果你的环境之前已经装过旧版本依赖，建议补跑一次：

```bash
cd terrain-generator
pip install -e . --upgrade
```

安装后会提供命令行入口：

```bash
terrain-generator --help
```

### 1.2 推荐安装 Open3D（可视化 + SDF）

Open3D 同时用于两个功能：

1. **可视化**：交互式窗口和高质量离屏渲染（headless MP4 视频）
2. **SDF 自动计算**：`MeshTerrain` 在没有显式传入 `sdf` 时自动计算 SDF

```bash
cd terrain-generator
pip install -e .[sdf]
# 或直接安装 open3d
pip install open3d
```

> **注意**: `open3d` 目前不提供 Python 3.13 的预编译 wheel。如果使用 Python ≥3.13，
> 请跳过 Open3D 安装，改用 `--renderer matplotlib` 做可视化，
> 以及 `--bundle-mode light` 在代码中显式传入 `sdf` 数据。

说明：

- `open3d` 是推荐的可视化后端，提供 better 光影效果和离屏渲染
- `pyrender` 作为备选交互可视化（部分环境 headless 模式可能不可用）
- `matplotlib` 始终可作为最终 fallback
- 不存在 OpenUSD 依赖链

## 2. 顶层接口

包顶层导出位于 [generator/__init__.py](generator/__init__.py)。

### 2.1 核心对象

- `MeshTerrain`
- `MeshTerrainCfg`
- `SDFArray`
- `NavDistance`

### 2.2 程序化地形

- `DoorTerrainCfg`
- `ForestTerrainCfg`
- `PlatformGapTerrainCfg`
- `PileTerrainCfg`
- `ProceduralTerrainResult`
- `PyramidStairsTerrainCfg`
- `StakesTerrainCfg`
- `generate_door_terrain(cfg)`
- `generate_forest_terrain(cfg)`
- `generate_platform_gap_terrain(cfg)`
- `generate_pile_terrain(cfg)`
- `generate_pyramid_stairs_terrain(cfg)`
- `generate_stakes_terrain(cfg)`

### 2.3 导航与可视化

- `calc_spawnable_locations_on_terrain(mesh, ...)`
- `calc_spawnable_locations_with_sdf(mesh, sdf_array, ...)`
- `compute_distance_matrix(mesh, ...)`
- `find_route(mesh, start_xy, goal_xy, ...)`
- `build_scene(mesh, ...)`
- `visualize_mesh(mesh, ...)`

## 3. 快速开始

### 3.0 直接使用 CLI

生成一个程序化 `pile` 地形：

```bash
terrain-generator procedural pile --output outputs/pile_cli --bundle-mode light
```

生成一个默认的 WFC 场景。当前默认集合会包含所有已接入并启用 WFC 的 terrain family：
`door / forest / platform_gap / pile / pyramid_stairs / stakes`

```bash
terrain-generator wfc-scene --output outputs/wfc_cli --shape 4 4 --bundle-mode light
```

可视化地形（交互式或生成 MP4 视频）：

```bash
# 交互式 Open3D 可视化（推荐）
python examples/visualize_terrain.py --mode pile --renderer open3d

# 生成高质量 MP4 视频
python examples/visualize_terrain.py --headless --mode wfc --shape 4 4 \
    --renderer open3d --duration 10 --fps 30 --output sweep.mp4

# 自定义 WFC terrain 集合
python examples/visualize_terrain.py --headless --mode wfc --shape 4 4 \
    --terrains pyramid_stairs,platform_gap,stakes,pile --renderer open3d --output terrain_sweep.mp4
```

说明：

- `light` 模式默认不强制依赖 `open3d`
- 会输出 `mesh.obj` 和 `terrain.npz`
- 如果已安装 `.[sdf]`，可改用 `--bundle-mode full`
- CLI 默认 WFC terrain 集合当前是 `door,forest,platform_gap,pile,pyramid_stairs,stakes`
- WFC 网格建议 ≤ 6×6（更大网格可能因回溯过多而超时）
- `--renderer` 支持 `open3d`（推荐）、`matplotlib`（fallback）、`pyrender`、`auto`

### 3.1 生成一个 `pile` 地形

```python
from generator import PileTerrainCfg, generate_pile_terrain

cfg = PileTerrainCfg(
	size=(10.0, 10.0),
	platform_half=1.0,
	platform_height=0.25,
	route_mode="cross_route",
	route_line_count=3,
)

result = generate_pile_terrain(cfg)

print(result.mesh)
print(result.origin)
print(result.metadata["pillar_count"])
```

返回值 `result` 的类型是 `ProceduralTerrainResult`，包含：

- `mesh`: 完整地形网格
- `origin`: 推荐出生点
- `terrain_mesh`: 当前与 `mesh` 相同，后续可扩展为“仅地形本体”
- `metadata`: 程序化生成时的附加信息

### 3.2 直接可视化

```python
from generator import PileTerrainCfg, generate_pile_terrain

result = generate_pile_terrain(PileTerrainCfg())
result.visualize(show=True)
```

### 3.3 保存为 `.obj + .npz`

如果已安装 `.[sdf]`：

```python
from generator import PileTerrainCfg, generate_pile_terrain

result = generate_pile_terrain(PileTerrainCfg())
result.save("outputs/pile_demo")
```

输出目录默认包含：

- `mesh.obj`
- `terrain.npz`

如果你没有安装 `.[sdf]`，则需要显式提供 `sdf` 和距离矩阵等数据，或者只使用几何生成与可视化，不走自动 SDF 保存路径。

## 4. `ProceduralTerrainResult` 接口

实现文件位于 [generator/terrains/base.py](generator/terrains/base.py)。

### 4.1 `to_mesh_terrain(**kwargs)`

将程序化结果包装成 `MeshTerrain`。

```python
terrain = result.to_mesh_terrain()
```

常见覆盖参数：

- `sdf`
- `distance_matrix`
- `distance_shape`
- `height_map_resolution`
- `graph_ratio`
- `height_cost_threshold`
- `min_traversable_height`
- `pit_inset_radius`
- `obstacle_inflation_radius`

### 4.2 `save(output_dir, **kwargs)`

将结果直接保存为 `mesh.obj + terrain.npz`。

```python
result.save("outputs/forest_case")
```

### 4.3 `visualize(...)`

支持把点、颜色、目标点和路线一起叠加到场景中：

```python
result.visualize(
	points=points,
	color_values=values,
	goal_pos=goal,
	route_points=route,
	show=True,
)
```

## 5. 程序化地形接口

### 5.1 `DoorTerrainCfg`

用途：生成环状墙段与门口结构的地形。

关键参数：

- `size`: 地形平面大小 `(x, y)`
- `ground_thickness`: 地面厚度
- `wall_thickness`: 墙厚
- `wall_height`: 墙高
- `door_width`: 门宽
- `wall_length`: 每段墙弧对应的近似长度
- `ring_spacing`: 环之间的间距
- `min_radius`: 最内层环半径

调用方式：

```python
from generator import DoorTerrainCfg, generate_door_terrain

cfg = DoorTerrainCfg(size=(8.0, 8.0), ring_spacing=1.2, min_radius=1.2)
result = generate_door_terrain(cfg)
```

元数据：

- `ring_count`
- `door_count`

### 5.2 `ForestTerrainCfg`

用途：生成树干障碍密布、中心区域保留通行空间的地形。

关键参数：

- `size`: 地形大小
- `min_gap`: 树干最小间距
- `density`: 树干密度因子
- `boundary_margin`: 边界留白
- `center_safe_radius`: 中心安全区半径
- `trunk_height_range`: 树干高度范围
- `trunk_radius_range`: 树干半径范围
- `seed`: 随机种子；默认随机，传入固定 seed 可复现

调用方式：

```python
from generator import ForestTerrainCfg, generate_forest_terrain

cfg = ForestTerrainCfg(size=(10.0, 10.0), center_safe_radius=1.5, seed=7)
result = generate_forest_terrain(cfg)
```

元数据：

- `tree_positions`
- `tree_count`

### 5.3 `PyramidStairsTerrainCfg`

用途：生成正向/反向金字塔楼梯。默认 `direction="random"`，每个 tile 约 50% 概率为上楼梯，50% 概率为下楼梯。

关键参数：

- `step_count_range`: 台阶层数范围
- `step_width_range`: 每层环带宽度范围
- `step_height_range`: 台阶高度范围
- `platform_half_range`: 中心平台半宽范围
- `direction`: `"random" | "up" | "down"`
- `seed`: 随机种子；默认随机，传入固定 seed 可复现

调用方式：

```python
from generator import PyramidStairsTerrainCfg, generate_pyramid_stairs_terrain

cfg = PyramidStairsTerrainCfg(size=(8.0, 8.0), seed=4)
result = generate_pyramid_stairs_terrain(cfg)
```

### 5.4 `PlatformGapTerrainCfg`

用途：生成外部平地、中心平台和中间 gap/坑洞。gap 底部会被标记为 `void`，规划器不会把坑底当作可通行地面。

关键参数：

- `gap_depth`: gap 下挖深度，默认与 `PileTerrainCfg.ground_depth` 一致为 5m
- `gap_width_range`: 平台边缘到外部地面边缘的 gap 宽度范围，默认约 0.25m
- `gap_half_range`: gap 外边缘半宽范围；仅当 `gap_width_range=None` 时直接使用
- `platform_half_range`: 中心平台半宽范围
- `seed`: 随机种子；默认随机，传入固定 seed 可复现

调用方式：

```python
from generator import PlatformGapTerrainCfg, generate_platform_gap_terrain

cfg = PlatformGapTerrainCfg(size=(8.0, 8.0), seed=3)
result = generate_platform_gap_terrain(cfg)
```

### 5.5 `PileTerrainCfg`

用途：生成中心高台 + 梅花桩路径地形。

关键参数：

- `size`: 地形大小
- `ground_depth`: 外围下挖深度
- `platform_half`: 中心平台半宽
- `platform_height`: 平台相对高度
- `pillar_radius`: 桩半径，默认 0.3m
- `pillar_spacing`: 桩间距
- `route_mode`: `"full_grid" | "cross_route" | "custom_mask"`
- `route_line_count`: `cross_route` 模式下保留的中心横/竖线路数
- `custom_mask`: `custom_mask` 模式下的格点保留掩码

调用方式：

```python
from generator import PileTerrainCfg, generate_pile_terrain

cfg = PileTerrainCfg(
	size=(8.0, 8.0),
	route_mode="cross_route",
	route_line_count=3,
)
result = generate_pile_terrain(cfg)
```

元数据：

- `pillar_positions`
- `pillar_count`
- `route_mode`
- `platform_height`

`route_mode` 语义：

- `full_grid`: 保留完整桩阵
- `cross_route`: 只保留中心若干条横线和竖线，形成十字型梅花桩路径
- `custom_mask`: 使用自定义布桩掩码

### 5.6 `StakesTerrainCfg`

用途：生成中心平台向四周扩展的可踩桩点地形。该地形使用 `foothold_graph` 规划，不把桩当普通 obstacle 绕开。

关键参数：

- `void_depth`: 桩之间的下挖深度
- `center_platform_half`: 中心平台半宽
- `stake_radius`: 桩半径
- `stake_spacing`: 桩间距
- `route_line_count`: 中心横/竖保留线路数
- `xy_jitter_ratio`: 桩点随机扰动比例

调用方式：

```python
from generator import StakesTerrainCfg, generate_stakes_terrain

cfg = StakesTerrainCfg(size=(8.0, 8.0), route_line_count=3, seed=7)
result = generate_stakes_terrain(cfg)
```

## 6. `MeshTerrain` 接口

实现位于 [generator/mesh_terrain.py](generator/mesh_terrain.py)。

### 6.1 构造

支持三种构造方式：

```python
from generator import MeshTerrain, MeshTerrainCfg

# 1. 从配置对象构造
terrain = MeshTerrain(MeshTerrainCfg(...))

# 2. 从字典构造
terrain = MeshTerrain({...})

# 3. 从保存目录或 terrain.npz 直接加载
terrain = MeshTerrain("outputs/pile_demo")
terrain = MeshTerrain("outputs/pile_demo/terrain.npz")
```

### 6.2 主要方法

#### `get_sdf(points)`

查询一批点的 SDF 值：

```python
values = terrain.get_sdf(points)
```

输入：

- `points`: `N x 3`

#### `get_distance(points, goal_pos)`

查询到目标点的导航距离：

```python
dist = terrain.get_distance(points_xy, goal_xy)
```

输入：

- `points`: `N x 2`
- `goal_pos`: `M x 2`

#### `get_route(start_pos, goal_pos)`

使用单一安全规划器重建一条路径。默认行为会：

- 禁止用对角线抄近路穿过障碍角点
- 将低于 `min_traversable_height` 的低洼区域视为绝对不可走
- 对高于地面的墙体 / 门框类障碍做栅格化与膨胀缓冲
- 在 `pile` 这类地形上返回按真实落脚点离散化的安全路径点

在 WFC 组合场景中，示例路由现在采用双层规划：

- 高层只在 tile grid 上走 4 邻接，不允许跨 tile 斜跳
- 低层进入某个 sub terrain 后，会根据实际入点 / 出点做局部路径规划
- sub terrain 内部允许使用安全对角线，不再强制经过 tile 中心 waypoint
- tile 生成阶段会保存规划 metadata，例如边界可通行 anchors，便于后续层级规划复用

调用方式：

```python
route = terrain.get_route(start_pos, goal_pos)
```

返回值：

- `K x 3` 的路径点序列

#### `save(output_dir)`

保存当前地形：

```python
terrain.save("outputs/case_a")
```

默认输出：

- `mesh.obj`
- `terrain.npz`

### 6.3 `.npz` 内容

当前 `terrain.npz` 中会保存：

- `cfg_json`
- `mesh_path`
- `origin`
- `mesh_dim`
- `sdf_array`
- `sdf_center`
- `sdf_resolution`
- `distance_matrix`
- `distance_shape`
- `distance_center`
- `spawnable_locations`

## 7. 出生点与导航接口

实现位于 [generator/nav_utils.py](generator/nav_utils.py)。

### 7.1 出生点提取

```python
from generator import calc_spawnable_locations_on_terrain

spawnable = calc_spawnable_locations_on_terrain(mesh, resolution=0.2)
```

如果已有 SDF，可进一步过滤：

```python
from generator import calc_spawnable_locations_with_sdf

spawnable = calc_spawnable_locations_with_sdf(mesh, sdf_array)
```

### 7.2 距离矩阵计算

```python
from generator import compute_distance_matrix

dist_matrix, shape, center = compute_distance_matrix(mesh)
```

距离矩阵与路径重建现在基于同一套安全可通行规则，不再存在“距离可达但实际 route 会穿越危险区域”的双轨语义。

### 7.3 路径重建

```python
from generator import find_route

route = find_route(mesh, start_xy=[1.0, 1.0], goal_xy=[7.0, 7.0])
```

常用导航参数：

- `height_threshold`: 允许的相邻安全落脚点最大高度差
- `min_traversable_height`: 最低地表高度阈值，低于该值的区域绝对不可走
- `pit_inset_radius`: 对低洼区域额外收缩的栅格半径，用于避免贴着坑边走
- `obstacle_inflation_radius`: 对墙体 / 门框等高障碍的膨胀半径，用于给门框和墙角留出安全边距
- `use_diagonal=False`: 默认关闭，避免在窄通道和角点处走危险对角捷径

## 8. 可视化接口

实现位于 [generator/visualization.py](generator/visualization.py)。

可视化后端优先级（`auto` 模式）：

1. **Open3D**（推荐）—— 高质量交互式渲染 + 离屏 MP4 视频
2. **pyrender** —— 备选交互式渲染
3. **matplotlib** —— 最终 fallback，支持交互式窗口和动画导出

### 8.1 `visualize_mesh(...)`

```python
from generator import visualize_mesh

visualize_mesh(
	mesh,
	points=points,
	color_values=values,
	goal_pos=goal,
	route_points=route,
	show=True,
)
```

支持叠加：

- 网格本体（地形高度颜色映射）
- 点云 / 出生点
- 标量颜色映射
- 目标点 marker
- 路线 polyline

当 `show=True` 时，自动优选 Open3D 交互式窗口；若 Open3D 不可用则 fallback 到 pyrender。

### 8.2 `draw_open3d(...)` · `render_open3d_frame_sequence(...)`

Open3D 专用接口：

```python
from generator.visualization import draw_open3d, render_open3d_frame_sequence

# 交互式窗口
draw_open3d(mesh, points=points, show_ground=True, show_axes=True,
            grid_meta=grid_meta)

# 离屏渲染为 MP4 视频
render_open3d_frame_sequence(
    mesh, output_path="sweep.mp4",
    width=1600, height=896, fps=30,
    elev_seq=elev_seq, azim_seq=azim_seq,
    points=points, show_ground=True, show_axes=True,
    grid_meta=grid_meta,
)
```

### 8.3 `build_scene(...)` · `render_scene_offscreen(...)`

pyrender 场景构建与离屏渲染：

```python
from generator import build_scene, render_scene_offscreen

scene = build_scene(mesh, points=points, goal_pos=goal, route_points=route)
frame = render_scene_offscreen(scene, width=1600, height=900)
```

### 8.4 `build_open3d_scene_items(...)`

构建 Open3D 场景元素列表（geometry + material），供自定义渲染管线使用：

```python
from generator.visualization import build_open3d_scene_items

items = build_open3d_scene_items(
    mesh, points=points, show_ground=True,
    show_axes=True, grid_meta=grid_meta,
)
# items: list of (name, o3d_geometry, material_record)
```

## 9. 一个完整示例

下面给出一条推荐链路：生成 `pile`，保存，重新加载，提取路线，再做可视化。

```python
import numpy as np

from generator import (
	MeshTerrain,
	PileTerrainCfg,
	generate_pile_terrain,
)

result = generate_pile_terrain(
	PileTerrainCfg(
		size=(8.0, 8.0),
		route_mode="cross_route",
		route_line_count=3,
	)
)

# 需要安装 pip install -e .[sdf]
result.save("outputs/pile_case")

terrain = MeshTerrain("outputs/pile_case")

start = np.array([1.0, 1.0, 0.0], dtype=np.float32)
goal = np.array([7.0, 7.0, 0.0], dtype=np.float32)
route = terrain.get_route(start, goal)

result.visualize(
	points=terrain.cfg.spawnable_locations,
	goal_pos=goal,
	route_points=route,
	show=True,
)
```

## 10. 当前限制

### 10.1 Open3D 依赖说明

Open3D 同时用于：
- **可视化**（交互式窗口 + headless MP4 视频）—— 推荐后端
- **SDF 自动计算**（可选 `.[sdf]` 扩展）

若未安装 Open3D：
- 可视化 fallback：`pyrender`（交互式）→ `matplotlib`（视频/交互）
- SDF 需手动传入或使用 `--bundle-mode light`

> **Python 3.13 用户**: `open3d` 暂无 Python 3.13 预编译 wheel。
> 可使用 `--renderer matplotlib` 做可视化，`--bundle-mode light` 绕过 SDF 自动计算。

### 10.2 WFC 网格规模限制

WFC 求解器在大网格上可能因回溯爆炸导致超时：

| 网格 | 格子数 | 状态 |
|------|--------|------|
| 4×4 | 16 | ✅ 秒级 |
| 5×5 | 25 | ✅ < 1 min |
| 6×6 | 36 | ✅ < 1 min |
| 7×7 | 49 | ⚠️ 可能超时 |
| ≥8×8 | ≥64 | ❌ 不建议 |

建议 CLI 和测试使用 4×4 到 6×6 的网格。

### 10.3 `WFC` 仅迁入了核心层

当前新包已经包含：

- `WFC` 核心求解器
- `Tile / ArrayTile / MeshTile` 数据结构
- `ConnectionManager`
- 程序化 `terrain -> MeshTile -> WFC` 适配层

其中 `door / forest / platform_gap / pile / pyramid_stairs / stakes` 现在都可以先对齐到统一 tile 尺寸，再作为 `MeshTile` 参与 WFC。

同时新包现在已经支持：

- 运行 WFC
- 把 WFC wave 直接合并成完整 scene mesh
- 导出完整 scene 的 `.obj` 和 `.npz`

但旧包里的以下能力还没有完整迁入：

- mesh-part 组合体系
- pattern/config 体系
- 旧 examples 里的完整 WFC 生成流水线

## 11. 相关文件

- [generator/__init__.py](generator/__init__.py)
- [generator/mesh_terrain.py](generator/mesh_terrain.py)
- [generator/cli.py](generator/cli.py)
- [generator/presets.py](generator/presets.py)
- [generator/nav_utils.py](generator/nav_utils.py)
- [generator/visualization.py](generator/visualization.py)
- [generator/terrains/base.py](generator/terrains/base.py)
- [generator/terrains/door.py](generator/terrains/door.py)
- [generator/terrains/forest.py](generator/terrains/forest.py)
- [generator/terrains/pile.py](generator/terrains/pile.py)
- [generator/terrains/platform_gap.py](generator/terrains/platform_gap.py)
- [generator/terrains/pyramid_stairs.py](generator/terrains/pyramid_stairs.py)
- [generator/terrains/stakes.py](generator/terrains/stakes.py)
- [examples/generate_procedural.py](examples/generate_procedural.py)
- [examples/generate_wfc_scene.py](examples/generate_wfc_scene.py)
- [tests/test_smoke.py](tests/test_smoke.py)
- [tests/test_procedural_terrains.py](tests/test_procedural_terrains.py)
