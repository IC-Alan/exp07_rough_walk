from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from ..mesh_terrain import MeshTerrain, MeshTerrainCfg
from ..nav_utils import compute_boundary_anchors
from ..terrain_state import TerrainState, ensure_terrain_state
from ..visualization import visualize_mesh


def ground_box(size: tuple[float, float], thickness: float) -> trimesh.Trimesh:
    size_x, size_y = size
    return trimesh.creation.box(
        extents=[size_x, size_y, thickness],
        transform=trimesh.transformations.translation_matrix([size_x / 2.0, size_y / 2.0, -thickness / 2.0]),
    )


def centered_boundary_edge_constraints(
    size: tuple[float, float],
    half_span: float,
    *,
    z: float = 0.0,
) -> dict[str, object]:
    size_x, size_y = float(size[0]), float(size[1])
    center_x = size_x * 0.5
    center_y = size_y * 0.5
    span_x = float(min(max(half_span, 0.0), size_x * 0.5))
    span_y = float(min(max(half_span, 0.0), size_y * 0.5))
    return {
        "mode": "edge",
        "boundaries": {
            "left": [{"start": (0.0, center_y - span_y, z), "end": (0.0, center_y + span_y, z)}],
            "right": [{"start": (size_x, center_y - span_y, z), "end": (size_x, center_y + span_y, z)}],
            "down": [{"start": (center_x - span_x, 0.0, z), "end": (center_x + span_x, 0.0, z)}],
            "up": [{"start": (center_x - span_x, size_y, z), "end": (center_x + span_x, size_y, z)}],
        },
    }


@dataclass
class ProceduralTerrainResult:
    mesh: trimesh.Trimesh
    origin: np.ndarray
    terrain_mesh: trimesh.Trimesh | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    terrain_state: TerrainState | dict[str, Any] | None = None

    def __post_init__(self):
        self.terrain_state = ensure_terrain_state(self.terrain_state)
        state_metadata = dict(self.terrain_state.metadata)
        state_planning = dict(state_metadata.get("planning", {}))
        planning_metadata = dict(state_planning)
        planning_metadata.update(dict(self.metadata.get("planning", {})))
        if "boundary_anchors" not in planning_metadata:
            planning_metadata["boundary_anchors"] = compute_boundary_anchors(
                self.mesh,
                terrain_state=self.terrain_state,
            )
        self.metadata["planning"] = planning_metadata
        state_metadata["planning"] = planning_metadata
        self.terrain_state.metadata = state_metadata

    def to_mesh_terrain(self, **kwargs: Any) -> MeshTerrain:
        cfg = MeshTerrainCfg(
            mesh=self.mesh,
            mesh_dim=tuple(self.mesh.bounding_box.extents.tolist()),
            origin=tuple(self.origin.tolist()),
            terrain_state=self.terrain_state,
        )
        for key, value in kwargs.items():
            setattr(cfg, key, value)
        return MeshTerrain(cfg)

    def save(self, output_dir: str | Path, **kwargs: Any):
        return self.to_mesh_terrain(**kwargs).save(output_dir)

    def visualize(
        self,
        points: np.ndarray | None = None,
        color_values: np.ndarray | None = None,
        goal_pos: np.ndarray | None = None,
        route_points: np.ndarray | None = None,
        show: bool = True,
    ):
        return visualize_mesh(
            self.mesh,
            points=points,
            color_values=color_values,
            goal_pos=goal_pos,
            route_points=route_points,
            show=show,
        )


__all__ = ["ProceduralTerrainResult", "centered_boundary_edge_constraints", "ground_box"]
