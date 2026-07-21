from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import torch
import trimesh

from .mesh_utils import compute_sdf, export_mesh_obj
from .nav_utils import (
    calc_spawnable_locations_on_terrain,
    calc_spawnable_locations_with_sdf,
    compute_distance_matrix,
    find_route,
    find_strict_route,
    has_strict_route,
)
from .terrain_state import TerrainMember, TerrainPortal, TerrainState, ensure_terrain_state
from .terrain_bundle import load_terrain_bundle_cfg, save_terrain_bundle
from .terrain_fields import NavDistance, SDFArray


_DEFAULT_ROUTE_OPTION = object()


@dataclass
class MeshTerrainCfg:
    bundle_schema_version: int = 1
    mesh_path: Optional[str] = None
    mesh: Optional[trimesh.Trimesh] = None
    sdf: Optional[np.ndarray] = None
    distance_matrix: Optional[np.ndarray] = None
    distance_shape: Optional[tuple[int, int]] = None
    spawnable_locations: Optional[np.ndarray] = None
    terrain_state: TerrainState | dict | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_dim: tuple[float, float, float] = (2.0, 2.0, 2.0)
    sdf_resolution: float = 0.1
    sdf_threshold: float = 0.4
    sdf_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sdf_max_value: float = 1000.0
    height_offset: float = 0.5
    height_map_resolution: float = 0.1
    distance_center: tuple[float, float] = (0.0, 0.0)
    graph_ratio: int = 4
    height_cost_threshold: float = 0.4
    min_traversable_height: float = -0.2
    pit_inset_radius: int = 0
    obstacle_inflation_radius: int = 1
    ceiling_clearance_threshold: float = 0.5
    route_sample_spacing: float | None = 0.2
    route_smoothing_iterations: int = 1
    auto_compute_sdf: bool = True
    auto_compute_distance: bool = True


class MeshTerrain:
    def __init__(self, cfg: MeshTerrainCfg | dict | str, device: str = "cpu"):
        self.root_dir = Path.cwd()
        if isinstance(cfg, str):
            cfg_path = Path(cfg)
            if cfg_path.is_dir():
                cfg_path = cfg_path / "terrain.npz"
            self.root_dir = cfg_path.parent
            self.cfg = MeshTerrainCfg(**load_terrain_bundle_cfg(cfg_path))
        elif isinstance(cfg, dict):
            self.cfg = MeshTerrainCfg(**cfg)
        elif isinstance(cfg, MeshTerrainCfg):
            self.cfg = cfg
        else:
            raise ValueError("cfg must be MeshTerrainCfg, dict, or path to a saved terrain bundle")

        self.cfg.terrain_state = ensure_terrain_state(self.cfg.terrain_state)

        self.mesh = self.cfg.mesh
        if self.mesh is None:
            if self.cfg.mesh_path is not None:
                self.mesh = trimesh.load(self.root_dir / self.cfg.mesh_path, process=False)
                self.cfg.mesh_dim = tuple(self.mesh.bounding_box.extents.tolist())
            else:
                self.mesh = trimesh.creation.box([1.0, 1.0, 1.0])

        self.sdf = self._build_sdf(device)

        if self.cfg.distance_matrix is None and self.cfg.auto_compute_distance:
            dist_matrix, shape, center = compute_distance_matrix(
                self.mesh,
                self.cfg.graph_ratio,
                height_threshold=self.cfg.height_cost_threshold,
                height_map_resolution=self.cfg.height_map_resolution,
                min_traversable_height=self.cfg.min_traversable_height,
                pit_inset_radius=self.cfg.pit_inset_radius,
                obstacle_inflation_radius=self.cfg.obstacle_inflation_radius,
                ceiling_clearance_threshold=self.cfg.ceiling_clearance_threshold,
                terrain_state=self.cfg.terrain_state,
            )
        else:
            dist_matrix = self.cfg.distance_matrix
            shape = self.cfg.distance_shape or (1, 1)
            center = self.cfg.distance_center
        self.nav_distance = NavDistance(
            dist_matrix,
            shape,
            np.asarray(center, dtype=np.float32),
            self.cfg.height_map_resolution * self.cfg.graph_ratio,
            device=device,
        )
        self.device = torch.device(device)
        self.terrain_state = self.cfg.terrain_state

    def _build_sdf(self, device: str) -> SDFArray | None:
        if self.cfg.sdf is not None:
            return SDFArray(
                self.cfg.sdf,
                np.asarray(self.cfg.sdf_center, dtype=np.float32),
                self.cfg.sdf_resolution,
                max_value=self.cfg.sdf_max_value,
                device=device,
            )

        if not self.cfg.auto_compute_sdf:
            return None

        try:
            sdf_array = compute_sdf(self.mesh, self.cfg.mesh_dim, self.cfg.sdf_resolution)
        except ImportError:
            warnings.warn(
                "SDF backend is unavailable; continuing without SDF. Install the `sdf` extra to enable SDF features.",
                stacklevel=2,
            )
            return None

        return SDFArray(
            sdf_array,
            np.asarray(self.cfg.sdf_center, dtype=np.float32),
            self.cfg.sdf_resolution,
            max_value=self.cfg.sdf_max_value,
            device=device,
        )

    @property
    def has_sdf(self) -> bool:
        return self.sdf is not None

    def to(self, device: str | torch.device):
        device = torch.device(device)
        if self.sdf is not None:
            self.sdf.to(device)
        self.nav_distance.to(device)
        self.device = device
        return self

    def transform(self, transformation: np.ndarray | torch.Tensor):
        if isinstance(transformation, torch.Tensor):
            transformation = transformation.detach().cpu().numpy()
        self.mesh.apply_transform(transformation)
        if self.sdf is not None:
            self.sdf.transform(transformation)
        self.nav_distance.transform(transformation)

    def translate(self, translation: np.ndarray | torch.Tensor):
        if isinstance(translation, torch.Tensor):
            translation = translation.detach().cpu().numpy()
        transform = np.eye(4)
        transform[:3, 3] = translation
        self.transform(transform)

    def get_sdf(self, points: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.sdf is None:
            raise RuntimeError("SDF is unavailable for this terrain instance")
        return self.sdf.get_sdf(points)

    def get_distance(self, points: np.ndarray | torch.Tensor, goal_pos: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self.nav_distance.get_distance(points, goal_pos)

    def get_route(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        use_diagonal: bool = False,
        route_sample_spacing: float | None | object = _DEFAULT_ROUTE_OPTION,
        route_smoothing_iterations: int | None = None,
    ) -> np.ndarray:
        sample_spacing = (
            self.cfg.route_sample_spacing
            if route_sample_spacing is _DEFAULT_ROUTE_OPTION
            else route_sample_spacing
        )
        return find_route(
            self.mesh,
            start_pos[:2],
            goal_pos[:2],
            graph_ratio=self.cfg.graph_ratio,
            height_threshold=self.cfg.height_cost_threshold,
            height_map_resolution=self.cfg.height_map_resolution,
            use_diagonal=use_diagonal,
            min_traversable_height=self.cfg.min_traversable_height,
            pit_inset_radius=self.cfg.pit_inset_radius,
            obstacle_inflation_radius=self.cfg.obstacle_inflation_radius,
            ceiling_clearance_threshold=self.cfg.ceiling_clearance_threshold,
            terrain_state=self.terrain_state,
            route_sample_spacing=sample_spacing,
            route_smoothing_iterations=(
                self.cfg.route_smoothing_iterations
                if route_smoothing_iterations is None
                else route_smoothing_iterations
            ),
        )

    def has_strict_route(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                         use_diagonal: bool = False) -> bool:
        return has_strict_route(
            self.mesh,
            start_pos[:2],
            goal_pos[:2],
            graph_ratio=self.cfg.graph_ratio,
            height_threshold=self.cfg.height_cost_threshold,
            height_map_resolution=self.cfg.height_map_resolution,
            use_diagonal=use_diagonal,
            min_traversable_height=self.cfg.min_traversable_height,
            pit_inset_radius=self.cfg.pit_inset_radius,
            obstacle_inflation_radius=self.cfg.obstacle_inflation_radius,
            ceiling_clearance_threshold=self.cfg.ceiling_clearance_threshold,
            terrain_state=self.terrain_state,
        )

    def get_strict_route(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        use_diagonal: bool = False,
        route_sample_spacing: float | None | object = _DEFAULT_ROUTE_OPTION,
        route_smoothing_iterations: int | None = None,
    ) -> np.ndarray:
        sample_spacing = (
            self.cfg.route_sample_spacing
            if route_sample_spacing is _DEFAULT_ROUTE_OPTION
            else route_sample_spacing
        )
        return find_strict_route(
            self.mesh,
            start_pos[:2],
            goal_pos[:2],
            graph_ratio=self.cfg.graph_ratio,
            height_threshold=self.cfg.height_cost_threshold,
            height_map_resolution=self.cfg.height_map_resolution,
            use_diagonal=use_diagonal,
            min_traversable_height=self.cfg.min_traversable_height,
            pit_inset_radius=self.cfg.pit_inset_radius,
            obstacle_inflation_radius=self.cfg.obstacle_inflation_radius,
            ceiling_clearance_threshold=self.cfg.ceiling_clearance_threshold,
            terrain_state=self.terrain_state,
            route_sample_spacing=sample_spacing,
            route_smoothing_iterations=(
                self.cfg.route_smoothing_iterations
                if route_smoothing_iterations is None
                else route_smoothing_iterations
            ),
        )

    def get_traversability_map(self) -> np.ndarray:
        """Return a 2D boolean traversability map for this terrain.

        True = flat ground (traversable), False = steep slope / obstacle (blocked).
        Resolution is (height_array_shape // graph_ratio).
        This serves as the per-terrain 2D connectivity graph for high-level planning.
        """
        from .nav_utils import compute_traversability_map
        return compute_traversability_map(
            self.mesh,
            graph_ratio=self.cfg.graph_ratio,
            height_threshold=self.cfg.height_cost_threshold,
            height_map_resolution=self.cfg.height_map_resolution,
            min_traversable_height=self.cfg.min_traversable_height,
            pit_inset_radius=self.cfg.pit_inset_radius,
            obstacle_inflation_radius=self.cfg.obstacle_inflation_radius,
            ceiling_clearance_threshold=self.cfg.ceiling_clearance_threshold,
            terrain_state=self.terrain_state,
        )

    def get_members(self) -> list[TerrainMember]:
        return list(self.terrain_state.members)

    def get_portals(self) -> list[TerrainPortal]:
        return list(self.terrain_state.portals)

    def save(self, output_dir: str | Path, mesh_name: str = "mesh.obj", metadata_name: str = "terrain.npz") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        export_mesh_obj(self.mesh, output_dir / mesh_name)
        self.cfg.mesh_path = mesh_name
        self.cfg.mesh_dim = tuple(self.mesh.bounding_box.extents.tolist())
        self.cfg.terrain_state = self.terrain_state

        if self.cfg.spawnable_locations is None:
            if self.sdf is not None:
                self.cfg.spawnable_locations = calc_spawnable_locations_with_sdf(
                    self.mesh,
                    self.sdf.array.detach().cpu().numpy(),
                    height_offset=self.cfg.height_offset,
                    sdf_resolution=self.cfg.sdf_resolution,
                    sdf_threshold=self.cfg.sdf_threshold,
                )
            else:
                self.cfg.spawnable_locations = calc_spawnable_locations_on_terrain(
                    self.mesh,
                    resolution=self.cfg.height_map_resolution,
                )

        if self.sdf is None:
            self.cfg.auto_compute_sdf = False

        return save_terrain_bundle(
            output_dir=output_dir,
            metadata_name=metadata_name,
            cfg=self.cfg,
            sdf_array=np.array([], dtype=np.float32) if self.sdf is None else self.sdf.array.detach().cpu().numpy(),
            sdf_center=np.asarray(self.cfg.sdf_center, dtype=np.float32) if self.sdf is None else self.sdf.center.detach().cpu().numpy(),
            sdf_resolution=self.cfg.sdf_resolution if self.sdf is None else self.sdf.resolution,
            sdf_max_value=self.cfg.sdf_max_value if self.sdf is None else self.sdf.max_value,
            distance_matrix=self.nav_distance.matrix.detach().cpu().numpy(),
            distance_shape=self.nav_distance.shape,
            distance_center=self.nav_distance.center.detach().cpu().numpy(),
            distance_resolution=self.nav_distance.resolution,
            distance_max_value=self.nav_distance.max_value,
            spawnable_locations=np.asarray(self.cfg.spawnable_locations, dtype=np.float32),
        )
