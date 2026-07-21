from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..mesh_utils import merge_meshes
from ..route_graph import build_route_graph, route_graph_edge, route_graph_node
from ..terrain_state import TerrainPortal, TerrainState
from ..utils import random_seed
from ._box_utils import add_box, rect_ring_bounds
from .base import ProceduralTerrainResult, centered_boundary_edge_constraints


@dataclass
class PyramidStairsTerrainCfg:
    size: tuple[float, float] = (10.0, 10.0)
    ground_thickness: float = 0.1
    step_count_range: tuple[int, int] = (4, 7)
    step_width_range: tuple[float, float] = (0.45, 0.75)
    step_height_range: tuple[float, float] = (0.08, 0.16)
    platform_half_range: tuple[float, float] = (0.7, 1.2)
    direction: Literal["random", "up", "down"] = "random"
    seed: int = field(default_factory=random_seed)


def _sample_direction(rng: np.random.Generator, direction: str) -> str:
    if direction in {"up", "down"}:
        return direction
    if direction != "random":
        raise ValueError("direction must be 'random', 'up', or 'down'")
    return "up" if float(rng.random()) < 0.5 else "down"


def _build_pyramid_route_graph(
    *,
    center_x: float,
    center_y: float,
    step_count: int,
    step_width: float,
    step_height: float,
    platform_half: float,
    outer_half_x: float,
    outer_half_y: float,
    direction: str,
) -> dict[str, object]:
    sign = 1.0 if direction == "up" else -1.0
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    boundaries: dict[str, list[str]] = {}

    side_vectors = {
        "left": np.array([-1.0, 0.0], dtype=np.float32),
        "right": np.array([1.0, 0.0], dtype=np.float32),
        "down": np.array([0.0, -1.0], dtype=np.float32),
        "up": np.array([0.0, 1.0], dtype=np.float32),
    }

    for level in range(step_count + 1):
        top_z = sign * level * step_height
        center_id = f"center_{level}"
        nodes.append(route_graph_node(center_id, (center_x, center_y, top_z), "center"))
        if level > 0:
            edges.append(route_graph_edge(f"center_{level - 1}", center_id))

        half_x = platform_half + max(step_count - level, 0) * step_width
        half_y = platform_half + max(step_count - level, 0) * step_width
        side_offsets = {
            "left": np.array([-half_x, 0.0], dtype=np.float32),
            "right": np.array([half_x, 0.0], dtype=np.float32),
            "down": np.array([0.0, -half_y], dtype=np.float32),
            "up": np.array([0.0, half_y], dtype=np.float32),
        }
        for side, offset in side_offsets.items():
            node_id = f"{side}_{level}"
            point_xy = np.array([center_x, center_y], dtype=np.float32) + offset
            nodes.append(route_graph_node(node_id, (point_xy[0], point_xy[1], top_z), "boundary" if level == 0 else "step"))
            edges.append(route_graph_edge(center_id, node_id))
            if level > 0:
                edges.append(route_graph_edge(f"{side}_{level - 1}", node_id))
            if level == 0:
                boundaries.setdefault(side, []).append(node_id)

    return build_route_graph(nodes, edges, boundaries)


def generate_pyramid_stairs_terrain(cfg: PyramidStairsTerrainCfg) -> ProceduralTerrainResult:
    rng = np.random.default_rng(cfg.seed)
    size_x, size_y = (float(cfg.size[0]), float(cfg.size[1]))
    center_x, center_y = size_x * 0.5, size_y * 0.5
    direction = _sample_direction(rng, cfg.direction)
    sign = 1.0 if direction == "up" else -1.0

    step_count = int(rng.integers(int(cfg.step_count_range[0]), int(cfg.step_count_range[1]) + 1))
    outer_half_x = size_x * 0.5
    outer_half_y = size_y * 0.5
    max_outer_half = min(outer_half_x, outer_half_y)
    platform_half = min(float(rng.uniform(*cfg.platform_half_range)), max(0.2, max_outer_half - 0.5))
    step_width = min(
        float(rng.uniform(*cfg.step_width_range)),
        max(0.05, (max_outer_half - platform_half) / max(step_count, 1)),
    )
    step_height = float(rng.uniform(*cfg.step_height_range))
    min_top = min(0.0, sign * step_count * step_height)
    max_top = max(0.0, sign * step_count * step_height)
    z_min = min_top - float(cfg.ground_thickness)

    meshes = []
    members = []
    half_x = outer_half_x
    half_y = outer_half_y
    for level in range(step_count):
        inner_half_x = max(platform_half, half_x - step_width)
        inner_half_y = max(platform_half, half_y - step_width)
        top_z = sign * level * step_height
        for part_id, bounds in enumerate(rect_ring_bounds(center_x, center_y, half_x, half_y, inner_half_x, inner_half_y, z_min, top_z)):
            add_box(
                meshes,
                members,
                f"step_{level}_{part_id}",
                kind="pyramid_step",
                role="platform",
                bounds=bounds,
                traversable=True,
                boundary_contacts=("up", "down", "left", "right") if level == 0 else (),
                params={"level": level, "top_z": top_z, "direction": direction},
            )
        half_x = inner_half_x
        half_y = inner_half_y

    center_top = sign * step_count * step_height
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
            z_min,
            center_top,
        ),
        traversable=True,
        params={"platform_half": platform_half, "direction": direction},
    )

    mesh = merge_meshes(meshes, minimal_triangles=False)
    route_half_span = min(max(platform_half, step_width), min(size_x, size_y) * 0.25)
    planning_metadata = {
        "height_limits": {"min": float(min_top - 0.05), "max": float(max_top + cfg.ground_thickness)},
        "route_constraints": centered_boundary_edge_constraints(cfg.size, route_half_span),
        "route_graph": _build_pyramid_route_graph(
            center_x=center_x,
            center_y=center_y,
            step_count=step_count,
            step_width=step_width,
            step_height=step_height,
            platform_half=platform_half,
            outer_half_x=outer_half_x,
            outer_half_y=outer_half_y,
            direction=direction,
        ),
    }
    return ProceduralTerrainResult(
        mesh=mesh,
        terrain_mesh=mesh,
        origin=np.array([center_x, center_y, max(0.3, max_top + 0.2)], dtype=np.float32),
        terrain_state=TerrainState(
            members=members,
            portals=[
                TerrainPortal(
                    name="center_platform_top",
                    boundary="interior",
                    center=(center_x, center_y, center_top),
                    span=(2.0 * platform_half, 2.0 * platform_half, step_height),
                    normal=(0.0, 0.0, 1.0),
                )
            ],
            connectivity={"direction": direction, "step_count": step_count},
            metadata={"generator": "pyramid_stairs", "planning": planning_metadata},
        ),
        metadata={
            "direction": direction,
            "step_count": step_count,
            "step_width": step_width,
            "step_height": step_height,
            "platform_half": platform_half,
            "planning": planning_metadata,
        },
    )


__all__ = ["PyramidStairsTerrainCfg", "generate_pyramid_stairs_terrain"]
