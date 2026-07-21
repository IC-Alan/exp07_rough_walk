from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import trimesh

from ..mesh_utils import merge_meshes
from ..route_graph import build_support_route_graph
from ..terrain_state import TerrainPortal, TerrainState, make_box_member
from ..utils import random_seed
from ._box_utils import add_box
from .base import ProceduralTerrainResult


@dataclass
class StakesTerrainCfg:
    size: tuple[float, float] = (10.0, 10.0)
    base_thickness: float = 0.2
    void_depth: float = 2.0
    boundary_wall_thickness: float = 0.15
    center_platform_half: float = 0.9
    stake_radius: float = 0.13
    stake_height: float = 0.0
    stake_spacing: float = 0.55
    route_line_count: int = 3
    xy_jitter_ratio: float = 0.12
    stake_sections: int = 16
    route_mode: Literal["cross", "full_grid"] = "cross"
    seed: int = field(default_factory=random_seed)


def _cross_axes(center: float, spacing: float, line_count: int) -> np.ndarray:
    count = max(1, int(line_count))
    offsets = (np.arange(count, dtype=np.float32) - (count - 1) * 0.5) * float(spacing)
    return center + offsets


def _stake_axis_positions(length: float, spacing: float, margin: float) -> np.ndarray:
    span = max(0.0, float(length) - 2.0 * float(margin))
    count = max(2, int(np.floor(span / max(float(spacing), 1e-6))) + 1)
    return np.linspace(float(margin), float(length) - float(margin), count, dtype=np.float32)


def _stake_candidate_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    cross_x: np.ndarray,
    cross_y: np.ndarray,
    route_mode: str,
) -> list[tuple[float, float]]:
    if route_mode == "full_grid":
        candidates = [(float(x), float(y)) for x in xs for y in ys]
    elif route_mode == "cross":
        candidates = [(float(x), float(y)) for x in xs for y in cross_y]
        candidates.extend((float(x), float(y)) for x in cross_x for y in ys)
    else:
        raise ValueError("route_mode must be 'cross' or 'full_grid'")

    deduped: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in candidates:
        key = (round(x, 5), round(y, 5))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((x, y))
    return deduped


def _stake_route_constraints(points: np.ndarray, tolerance: float) -> dict[str, object]:
    if points.size == 0:
        return {"mode": "edge_points", "boundaries": {}}
    boundaries: dict[str, list[dict[str, tuple[float, float, float]]]] = {side: [] for side in ("up", "down", "left", "right")}
    center_y = float(np.median(points[:, 1]))
    center_x = float(np.median(points[:, 0]))
    horizontal = points[np.abs(points[:, 1] - center_y) <= tolerance]
    vertical = points[np.abs(points[:, 0] - center_x) <= tolerance]
    if horizontal.size:
        boundaries["left"].append({"point": tuple(horizontal[int(np.argmin(horizontal[:, 0]))].tolist())})
        boundaries["right"].append({"point": tuple(horizontal[int(np.argmax(horizontal[:, 0]))].tolist())})
    if vertical.size:
        boundaries["down"].append({"point": tuple(vertical[int(np.argmin(vertical[:, 1]))].tolist())})
        boundaries["up"].append({"point": tuple(vertical[int(np.argmax(vertical[:, 1]))].tolist())})
    return {"mode": "edge_points", "boundaries": boundaries}


def _constraint_boundary_points(route_constraints: dict[str, object]) -> dict[str, list[tuple[float, float, float]]]:
    boundaries = route_constraints.get("boundaries", {})
    if not isinstance(boundaries, dict):
        return {}
    boundary_points: dict[str, list[tuple[float, float, float]]] = {}
    for side, records in boundaries.items():
        points: list[tuple[float, float, float]] = []
        for record in records:
            point = record.get("point")
            if point is None:
                continue
            points.append(tuple(float(value) for value in point))
        if points:
            boundary_points[str(side)] = points
    return boundary_points


def generate_stakes_terrain(cfg: StakesTerrainCfg) -> ProceduralTerrainResult:
    rng = np.random.default_rng(cfg.seed)
    size_x, size_y = (float(cfg.size[0]), float(cfg.size[1]))
    center_x, center_y = size_x * 0.5, size_y * 0.5
    z_bottom = -float(cfg.void_depth)

    meshes = []
    members = []
    add_box(
        meshes,
        members,
        "base",
        kind="void_base",
        role="void",
        bounds=(0.0, size_x, 0.0, size_y, z_bottom, z_bottom + cfg.base_thickness),
        traversable=False,
        params={"void_depth": cfg.void_depth},
    )
    wall = float(cfg.boundary_wall_thickness)
    add_box(meshes, members, "wall_down", kind="boundary_wall", role="wall", bounds=(0.0, size_x, 0.0, wall, z_bottom, 0.0), traversable=False, boundary_contacts=("down",))
    add_box(meshes, members, "wall_up", kind="boundary_wall", role="wall", bounds=(0.0, size_x, size_y - wall, size_y, z_bottom, 0.0), traversable=False, boundary_contacts=("up",))
    add_box(meshes, members, "wall_left", kind="boundary_wall", role="wall", bounds=(0.0, wall, 0.0, size_y, z_bottom, 0.0), traversable=False, boundary_contacts=("left",))
    add_box(meshes, members, "wall_right", kind="boundary_wall", role="wall", bounds=(size_x - wall, size_x, 0.0, size_y, z_bottom, 0.0), traversable=False, boundary_contacts=("right",))
    add_box(
        meshes,
        members,
        "center_platform",
        kind="platform",
        role="platform",
        bounds=(
            center_x - cfg.center_platform_half,
            center_x + cfg.center_platform_half,
            center_y - cfg.center_platform_half,
            center_y + cfg.center_platform_half,
            z_bottom,
            cfg.stake_height,
        ),
        traversable=True,
        params={"platform_half": cfg.center_platform_half},
    )

    margin = wall + cfg.stake_radius
    xs = _stake_axis_positions(size_x, cfg.stake_spacing, margin)
    ys = _stake_axis_positions(size_y, cfg.stake_spacing, margin)
    cross_x = _cross_axes(center_x, cfg.stake_spacing, cfg.route_line_count)
    cross_y = _cross_axes(center_y, cfg.stake_spacing, cfg.route_line_count)
    top_points: list[np.ndarray] = []
    for x, y in _stake_candidate_xy(xs, ys, cross_x, cross_y, cfg.route_mode):
        if abs(float(x) - center_x) <= cfg.center_platform_half and abs(float(y) - center_y) <= cfg.center_platform_half:
            continue
        jitter = cfg.stake_spacing * cfg.xy_jitter_ratio
        px = float(np.clip(x + rng.uniform(-jitter, jitter), margin, size_x - margin))
        py = float(np.clip(y + rng.uniform(-jitter, jitter), margin, size_y - margin))
        stake = trimesh.creation.cylinder(radius=cfg.stake_radius, height=cfg.void_depth + cfg.stake_height, sections=cfg.stake_sections)
        stake.apply_transform(trimesh.transformations.translation_matrix([px, py, (z_bottom + cfg.stake_height) * 0.5]))
        meshes.append(stake)
        top_points.append(np.array([px, py, cfg.stake_height], dtype=np.float32))
        members.append(
            make_box_member(
                f"stake_{len(top_points) - 1}",
                kind="stake",
                role="support",
                bounds=stake.bounds,
                traversable=False,
                params={"radius": cfg.stake_radius, "route_mode": cfg.route_mode},
            )
        )

    mesh = merge_meshes(meshes, minimal_triangles=False)
    points = np.asarray(top_points, dtype=np.float32)
    route_constraints = _stake_route_constraints(points, tolerance=max(1e-3, cfg.stake_spacing * 0.5))
    planning_metadata = {
        "height_limits": {"min": -0.2, "max": float(cfg.stake_height + 0.3)},
        "foothold_graph": {
            "enabled": True,
            "support_kinds": ("stake",),
            "max_step_distance": float(cfg.stake_spacing * 1.45),
            "max_height_delta": 0.35,
        },
        "route_constraints": route_constraints,
        "route_graph": build_support_route_graph(
            points,
            boundary_points=_constraint_boundary_points(route_constraints),
            support_kind="stake",
            max_step_distance=float(cfg.stake_spacing * 1.45),
            max_height_delta=0.35,
            platform_bounds=(
                center_x - cfg.center_platform_half,
                center_x + cfg.center_platform_half,
                center_y - cfg.center_platform_half,
                center_y + cfg.center_platform_half,
                float(cfg.stake_height),
            ),
        ),
    }
    return ProceduralTerrainResult(
        mesh=mesh,
        terrain_mesh=mesh,
        origin=np.array([center_x, center_y, max(0.3, cfg.stake_height + 0.3)], dtype=np.float32),
        terrain_state=TerrainState(
            members=members,
            portals=[
                TerrainPortal(
                    name="center_platform_top",
                    boundary="interior",
                    center=(center_x, center_y, cfg.stake_height),
                    span=(2.0 * cfg.center_platform_half, 2.0 * cfg.center_platform_half, cfg.void_depth),
                    normal=(0.0, 0.0, 1.0),
                )
            ],
            connectivity={"route_mode": cfg.route_mode, "stake_count": len(top_points)},
            metadata={"generator": "stakes", "planning": planning_metadata},
        ),
        metadata={
            "stake_positions": points,
            "stake_count": len(top_points),
            "route_mode": cfg.route_mode,
            "planning": planning_metadata,
        },
    )


__all__ = ["StakesTerrainCfg", "generate_stakes_terrain"]
