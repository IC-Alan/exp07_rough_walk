from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from ..mesh_utils import merge_meshes
from ..terrain_state import TerrainPortal, TerrainState, make_box_member
from ..utils import random_seed
from .base import ProceduralTerrainResult, centered_boundary_edge_constraints, ground_box


@dataclass
class ForestTerrainCfg:
    size: tuple[float, float] = (10.0, 10.0)
    ground_thickness: float = 0.1
    min_gap: float = 1.2
    density: float = 0.28
    boundary_margin: float = 0.8
    center_safe_radius: float = 1.25
    trunk_height_range: tuple[float, float] = (1.5, 4.0)
    trunk_radius_range: tuple[float, float] = (0.1, 0.3)
    max_trees: int = 200
    seed: int = field(default_factory=random_seed)
    trunk_sections: int = 12


def generate_forest_terrain(cfg: ForestTerrainCfg) -> ProceduralTerrainResult:
    rng = np.random.default_rng(cfg.seed)
    size_x, size_y = cfg.size
    center = np.array([size_x / 2.0, size_y / 2.0], dtype=np.float32)
    origin = np.array([center[0], center[1], 0.3], dtype=np.float32)

    meshes = [ground_box(cfg.size, cfg.ground_thickness)]
    members = [
        make_box_member(
            "ground",
            kind="ground_plane",
            role="ground",
            bounds=meshes[0].bounds,
            traversable=True,
            boundary_contacts=("up", "down", "left", "right"),
            params={"ground_thickness": cfg.ground_thickness},
        )
    ]
    tree_positions: list[np.ndarray] = []
    area = size_x * size_y
    num_trees = min(cfg.max_trees, int(area * cfg.density / max(cfg.min_gap * cfg.min_gap, 1e-6)))

    for _ in range(num_trees):
        placed = False
        for _attempt in range(100):
            x = rng.uniform(cfg.boundary_margin, size_x - cfg.boundary_margin)
            y = rng.uniform(cfg.boundary_margin, size_y - cfg.boundary_margin)
            if np.linalg.norm(np.array([x, y]) - center) < cfg.center_safe_radius:
                continue
            if any(np.linalg.norm(np.array([x, y]) - existing[:2]) < cfg.min_gap for existing in tree_positions):
                continue

            trunk_height = rng.uniform(*cfg.trunk_height_range)
            trunk_radius = rng.uniform(*cfg.trunk_radius_range)
            trunk = trimesh.creation.cylinder(radius=trunk_radius, height=trunk_height, sections=cfg.trunk_sections)
            trunk.apply_transform(trimesh.transformations.translation_matrix([x, y, trunk_height / 2.0]))
            meshes.append(trunk)
            tree_positions.append(np.array([x, y, trunk_height], dtype=np.float32))
            members.append(
                make_box_member(
                    f"tree_{len(tree_positions) - 1}",
                    kind="tree_trunk",
                    role="obstacle",
                    bounds=trunk.bounds,
                    traversable=False,
                    params={"radius": trunk_radius, "height": trunk_height},
                )
            )
            placed = True
            break
        if not placed:
            continue

    mesh = merge_meshes(meshes, minimal_triangles=False)
    return ProceduralTerrainResult(
        mesh=mesh,
        terrain_mesh=mesh,
        origin=origin,
        terrain_state=TerrainState(
            members=members,
            portals=[
                TerrainPortal(
                    name="center_safe_zone",
                    boundary="interior",
                    center=(float(center[0]), float(center[1]), 0.0),
                    span=(2.0 * cfg.center_safe_radius, 2.0 * cfg.center_safe_radius, max(cfg.trunk_height_range)),
                    normal=(0.0, 0.0, 1.0),
                )
            ],
            metadata={"generator": "forest"},
        ),
        metadata={
            "tree_positions": np.asarray(tree_positions, dtype=np.float32),
            "tree_count": len(tree_positions),
            "planning": {
                "route_constraints": centered_boundary_edge_constraints(
                    cfg.size,
                    min(max(cfg.center_safe_radius, cfg.min_gap), min(size_x, size_y) * 0.25),
                )
            },
        },
    )


__all__ = ["ForestTerrainCfg", "generate_forest_terrain"]
