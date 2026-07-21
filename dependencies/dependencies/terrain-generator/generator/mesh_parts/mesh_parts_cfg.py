from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np

from ..terrains.base import ProceduralTerrainResult


@dataclass
class MeshPartsCfg:
    name: str = "mesh"
    dim: tuple[float, float, float] = (2.0, 2.0, 2.0)
    floor_thickness: float = 0.1
    minimal_triangles: bool = True
    weight: float = 1.0
    rotations: tuple[Literal[90, 180, 270], ...] = ()
    flips: tuple[Literal["x", "y"], ...] = ()
    height_offset: float = 0.0
    edge_array: Optional[np.ndarray] = None
    use_generator: bool = True
    load_from_cache: bool = True


@dataclass
class WallPartsCfg(MeshPartsCfg):
    wall_thickness: float = 0.4
    wall_height: float = 3.0
    wall_edges: tuple[str, ...] = ()
    wall_type: str = "wall"
    create_door: bool = False
    door_width: float = 0.8
    door_height: float = 1.5
    door_direction: str = ""


@dataclass
class StairMeshPartsCfg(MeshPartsCfg):
    @dataclass
    class Stair(MeshPartsCfg):
        step_width: float = 1.0
        step_depth: float = 0.3
        n_steps: int = 5
        total_height: float = 1.0
        height_offset: float = 0.0
        stair_type: str = "standard"
        add_residual_side_up: bool = True
        add_rail: bool = False
        direction: str = "up"
        attach_side: str = "left"

    stairs: tuple[Stair, ...] = (Stair(),)
    wall: Optional[WallPartsCfg] = None


@dataclass
class PlatformMeshPartsCfg(MeshPartsCfg):
    array: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float32))
    z_dim_array: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float32))
    arrays: Optional[tuple[np.ndarray, ...]] = None
    z_dim_arrays: Optional[tuple[np.ndarray, ...]] = None
    add_floor: bool = True
    use_z_dim_array: bool = False
    wall: Optional[WallPartsCfg] = None


@dataclass
class HeightMapMeshPartsCfg(MeshPartsCfg):
    height_map: np.ndarray = field(default_factory=lambda: np.ones((10, 10), dtype=np.float32))
    add_floor: bool = True
    vertical_scale: float = 1.0
    slope_threshold: float = 4.0
    fill_borders: bool = True
    simplify: bool = True
    target_num_faces: int = 500

    def __post_init__(self):
        self.horizontal_scale = self.dim[0] / self.height_map.shape[0]


@dataclass
class MeshPattern:
    dim: tuple[float, float, float] = (2.0, 2.0, 2.0)
    mesh_parts: tuple[MeshPartsCfg, ...] = field(default_factory=lambda: (MeshPartsCfg(),))


@dataclass
class OverhangingMeshPartsCfg(MeshPartsCfg):
    connection_array: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float32))
    height_array: Optional[np.ndarray] = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float32))
    mesh: Any = None
    obstacle_type: str = "wall"


@dataclass
class WallMeshPartsCfg(OverhangingMeshPartsCfg):
    wall_thickness: float = 0.4
    wall_height: float = 3.0
    create_door: bool = False
    door_width: float = 0.8
    door_height: float = 1.5


@dataclass
class OverhangingBoxesPartsCfg(OverhangingMeshPartsCfg):
    gap_mean: float = 0.8
    gap_std: float = 0.2
    box_height: float = 0.5
    box_grid_n: int = 6


@dataclass
class FloatingBoxesPartsCfg(OverhangingMeshPartsCfg):
    n_boxes: int = 5
    box_dim_min: tuple[float, float, float] = (0.1, 0.1, 0.1)
    box_dim_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    roll_pitch_range: tuple[float, float] = (0.0, np.pi / 3)
    yaw_range: tuple[float, float] = (0.0, 2 * np.pi)
    min_height: float = 0.5
    max_height: float = 1.0


@dataclass
class CapsuleMeshPartsCfg(MeshPartsCfg):
    add_floor: bool = True
    radii: tuple[float, ...] = ()
    heights: tuple[float, ...] = ()
    transformations: tuple[np.ndarray, ...] = ()


@dataclass
class BoxMeshPartsCfg(MeshPartsCfg):
    add_floor: bool = True
    box_dims: tuple[tuple[float, float, float], ...] = ()
    transformations: tuple[np.ndarray, ...] = ()


@dataclass
class CombinedMeshPartsCfg(MeshPartsCfg):
    add_floor: bool = True
    cfgs: tuple[MeshPartsCfg, ...] = ()


@dataclass
class ProceduralMeshPartsCfg(MeshPartsCfg):
    terrain_cfg: Any = None
    terrain_generator: Optional[Callable[[Any], ProceduralTerrainResult]] = None
