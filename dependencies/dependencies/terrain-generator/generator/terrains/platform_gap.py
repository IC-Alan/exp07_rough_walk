from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mesh_utils import merge_meshes
from ..route_graph import build_route_graph, route_graph_edge, route_graph_node
from ..terrain_state import TerrainPortal, TerrainState
from ._box_utils import add_box, rect_ring_bounds, ring_bounds
from .base import ProceduralTerrainResult, centered_boundary_edge_constraints
from ..utils import random_seed


@dataclass
class PlatformGapTerrainCfg:
    size: tuple[float, float] = (10.0, 10.0)
    ground_thickness: float = 0.1
    gap_depth: float = 5.0
    gap_half_range: tuple[float, float] = (2.0, 3.0)
    gap_width_range: tuple[float, float] | None = (0.25, 0.25)
    platform_half_range: tuple[float, float] = (0.8, 1.3)
    seed: int = field(default_factory=random_seed)


def _build_platform_gap_route_graph(
    *,
    center_x: float,
    center_y: float,
    size_x: float,
    size_y: float,
    gap_half: float,
    platform_half: float,
) -> dict[str, object]:
    left_ground_lip_x = center_x - gap_half
    right_ground_lip_x = center_x + gap_half
    down_ground_lip_y = center_y - gap_half
    up_ground_lip_y = center_y + gap_half
    left_platform_lip_x = center_x - platform_half
    right_platform_lip_x = center_x + platform_half
    down_platform_lip_y = center_y - platform_half
    up_platform_lip_y = center_y + platform_half
    points = {
        "left": (center_x - size_x * 0.5, center_y, 0.0),
        "right": (center_x + size_x * 0.5, center_y, 0.0),
        "down": (center_x, center_y - size_y * 0.5, 0.0),
        "up": (center_x, center_y + size_y * 0.5, 0.0),
        "left_ground_lip": (left_ground_lip_x, center_y, 0.0),
        "right_ground_lip": (right_ground_lip_x, center_y, 0.0),
        "down_ground_lip": (center_x, down_ground_lip_y, 0.0),
        "up_ground_lip": (center_x, up_ground_lip_y, 0.0),
        "left_platform_lip": (left_platform_lip_x, center_y, 0.0),
        "right_platform_lip": (right_platform_lip_x, center_y, 0.0),
        "down_platform_lip": (center_x, down_platform_lip_y, 0.0),
        "up_platform_lip": (center_x, up_platform_lip_y, 0.0),
        "center_platform": (center_x, center_y, 0.0),
    }
    nodes = [
        route_graph_node(
            node_id,
            point,
            "boundary" if node_id in {"left", "right", "down", "up"} else "platform" if "platform" in node_id else "lip",
        )
        for node_id, point in points.items()
    ]
    edges = [
        route_graph_edge("left", "left_ground_lip"),
        route_graph_edge("left_ground_lip", "left_platform_lip"),
        route_graph_edge("left_platform_lip", "center_platform"),
        route_graph_edge("right", "right_ground_lip"),
        route_graph_edge("right_ground_lip", "right_platform_lip"),
        route_graph_edge("right_platform_lip", "center_platform"),
        route_graph_edge("down", "down_ground_lip"),
        route_graph_edge("down_ground_lip", "down_platform_lip"),
        route_graph_edge("down_platform_lip", "center_platform"),
        route_graph_edge("up", "up_ground_lip"),
        route_graph_edge("up_ground_lip", "up_platform_lip"),
        route_graph_edge("up_platform_lip", "center_platform"),
    ]
    boundaries = {
        "left": ["left"],
        "right": ["right"],
        "up": ["up"],
        "down": ["down"],
    }
    return build_route_graph(nodes, edges, boundaries)


def generate_platform_gap_terrain(cfg: PlatformGapTerrainCfg) -> ProceduralTerrainResult:
    rng = np.random.default_rng(cfg.seed)
    size_x, size_y = (float(cfg.size[0]), float(cfg.size[1]))
    center_x, center_y = size_x * 0.5, size_y * 0.5
    outer_half = min(size_x, size_y) * 0.5
    max_gap_half = max(0.6, outer_half - 0.5)
    min_gap_width = 0.05
    if cfg.gap_width_range is None:
        gap_half = max(0.6, min(float(rng.uniform(*cfg.gap_half_range)), max_gap_half))
        platform_half = max(0.2, min(float(rng.uniform(*cfg.platform_half_range)), gap_half - min_gap_width))
    else:
        gap_width = max(min_gap_width, float(rng.uniform(*cfg.gap_width_range)))
        max_platform_half = max(0.2, max_gap_half - gap_width)
        platform_half = max(0.2, min(float(rng.uniform(*cfg.platform_half_range)), max_platform_half))
        gap_width = max(min_gap_width, min(gap_width, max_gap_half - platform_half))
        gap_half = platform_half + gap_width
    z_min = -float(cfg.gap_depth)

    meshes = []
    members = []
    for part_id, bounds in enumerate(rect_ring_bounds(center_x, center_y, size_x * 0.5, size_y * 0.5, gap_half, gap_half, -cfg.ground_thickness, 0.0)):
        add_box(
            meshes,
            members,
            f"outer_ground_{part_id}",
            kind="ground_plane",
            role="ground",
            bounds=bounds,
            traversable=True,
            boundary_contacts=("up", "down", "left", "right"),
        )

    for part_id, bounds in enumerate(ring_bounds(center_x, center_y, gap_half, platform_half, z_min, z_min + cfg.ground_thickness)):
        add_box(
            meshes,
            members,
            f"gap_floor_{part_id}",
            kind="gap_floor",
            role="void",
            bounds=bounds,
            traversable=False,
            params={"gap_depth": cfg.gap_depth},
        )

    add_box(
        meshes,
        members,
        "center_platform",
        kind="platform",
        role="platform",
        bounds=(
            center_x - platform_half,
            center_x + platform_half,
            center_y - platform_half,
            center_y + platform_half,
            -cfg.ground_thickness,
            0.0,
        ),
        traversable=True,
        params={"platform_half": platform_half},
    )

    mesh = merge_meshes(meshes, minimal_triangles=False)
    planning_metadata = {
        "height_limits": {"min": -0.2, "max": float(cfg.ground_thickness)},
        "route_constraints": centered_boundary_edge_constraints(cfg.size, min(gap_half * 0.35, outer_half * 0.25)),
        "route_graph": _build_platform_gap_route_graph(
            center_x=center_x,
            center_y=center_y,
            size_x=size_x,
            size_y=size_y,
            gap_half=gap_half,
            platform_half=platform_half,
        ),
    }
    return ProceduralTerrainResult(
        mesh=mesh,
        terrain_mesh=mesh,
        origin=np.array([center_x, center_y, 0.3], dtype=np.float32),
        terrain_state=TerrainState(
            members=members,
            portals=[
                TerrainPortal(
                    name="center_platform_top",
                    boundary="interior",
                    center=(center_x, center_y, 0.0),
                    span=(2.0 * platform_half, 2.0 * platform_half, cfg.ground_thickness),
                    normal=(0.0, 0.0, 1.0),
                )
            ],
            connectivity={"gap_half": gap_half, "platform_half": platform_half},
            metadata={"generator": "platform_gap", "planning": planning_metadata},
        ),
        metadata={
            "gap_half": gap_half,
            "platform_half": platform_half,
            "gap_width": gap_half - platform_half,
            "gap_depth": cfg.gap_depth,
            "planning": planning_metadata,
        },
    )


__all__ = ["PlatformGapTerrainCfg", "generate_platform_gap_terrain"]
