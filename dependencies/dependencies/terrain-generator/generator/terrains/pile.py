from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh

from ..mesh_utils import merge_meshes
from ..route_graph import build_route_graph, build_support_route_graph, route_graph_edge, route_graph_node
from ..terrain_state import TerrainPortal, TerrainState, make_box_member
from .base import ProceduralTerrainResult


@dataclass
class PileTerrainCfg:
    size: tuple[float, float] = (10.0, 10.0)
    base_thickness: float = 0.2
    ground_depth: float = 5.0
    skirt_thickness: float = 0.15
    platform_half: float = 1.0
    platform_height: float = 0.0
    pillar_radius: float = 0.3
    pillar_spacing: float = 0.45
    pillar_clearance_between: float = 0.05
    pillar_sections: int = 24
    route_mode: Literal["full_grid", "cross_route", "custom_mask"] = "cross_route"
    route_line_count: int = 3
    route_line_tolerance_ratio: float = 0.6
    custom_mask: np.ndarray | None = None


def _create_boundary_skirt(size_x: float, size_y: float, skirt_thickness: float, skirt_height: float) -> list[trimesh.Trimesh]:
    center_x, center_y = size_x / 2.0, size_y / 2.0
    skirt_z = -skirt_height / 2.0
    half_skirt = skirt_thickness / 2.0
    return [
        trimesh.creation.box(
            extents=[size_x, skirt_thickness, skirt_height],
            transform=trimesh.transformations.translation_matrix([center_x, half_skirt, skirt_z]),
        ),
        trimesh.creation.box(
            extents=[size_x, skirt_thickness, skirt_height],
            transform=trimesh.transformations.translation_matrix([center_x, size_y - half_skirt, skirt_z]),
        ),
        trimesh.creation.box(
            extents=[skirt_thickness, size_y, skirt_height],
            transform=trimesh.transformations.translation_matrix([half_skirt, center_y, skirt_z]),
        ),
        trimesh.creation.box(
            extents=[skirt_thickness, size_y, skirt_height],
            transform=trimesh.transformations.translation_matrix([size_x - half_skirt, center_y, skirt_z]),
        ),
    ]


def _generate_lattice_positions(
    size_x: float,
    size_y: float,
    dx: float,
    dy: float,
    margin_x: float,
    margin_y: float,
) -> list[list[np.ndarray]]:
    rows: list[list[np.ndarray]] = []
    y = margin_y
    row_idx = 0
    while y <= size_y - margin_y + 1e-9:
        x_start = margin_x + (dx / 2.0 if row_idx % 2 == 1 else 0.0)
        row_positions: list[np.ndarray] = []
        x = x_start
        while x <= size_x - margin_x + 1e-9:
            row_positions.append(np.array([x, y], dtype=np.float32))
            x += dx
        rows.append(row_positions)
        y += dy
        row_idx += 1
    return rows


def _should_keep_pillar(
    cfg: PileTerrainCfg,
    x: float,
    y: float,
    row_id: int,
    col_id: int,
    xs: np.ndarray,
    ys: np.ndarray,
    tolerance: float,
) -> bool:
    if cfg.route_mode == "full_grid":
        return True
    if cfg.route_mode == "cross_route":
        return bool(np.any(np.abs(xs - x) < tolerance) or np.any(np.abs(ys - y) < tolerance))
    return bool(cfg.custom_mask[row_id, col_id])


def _cluster_axis(values: np.ndarray, tolerance: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32)
    sorted_values = np.sort(values.astype(np.float32))
    groups: list[list[float]] = [[float(sorted_values[0])]]
    for value in sorted_values[1:]:
        if abs(float(value) - float(np.mean(groups[-1]))) <= tolerance:
            groups[-1].append(float(value))
        else:
            groups.append([float(value)])
    return np.asarray([float(np.mean(group)) for group in groups], dtype=np.float32)


def _build_pile_route_constraints(selected_positions: np.ndarray, tolerance: float) -> dict[str, object]:
    if selected_positions.size == 0:
        return {"mode": "edge_points", "boundaries": {}}

    boundaries: dict[str, list[dict[str, tuple[float, float, float]]]] = {side: [] for side in ("up", "down", "left", "right")}
    y_clusters = _cluster_axis(selected_positions[:, 1], tolerance)
    for y_value in y_clusters:
        row_points = selected_positions[np.abs(selected_positions[:, 1] - y_value) <= tolerance]
        left_point = row_points[int(np.argmin(row_points[:, 0]))]
        right_point = row_points[int(np.argmax(row_points[:, 0]))]
        boundaries["left"].append({"point": tuple(left_point.astype(np.float32).tolist())})
        boundaries["right"].append({"point": tuple(right_point.astype(np.float32).tolist())})

    x_clusters = _cluster_axis(selected_positions[:, 0], tolerance)
    for x_value in x_clusters:
        col_points = selected_positions[np.abs(selected_positions[:, 0] - x_value) <= tolerance]
        down_point = col_points[int(np.argmin(col_points[:, 1]))]
        up_point = col_points[int(np.argmax(col_points[:, 1]))]
        boundaries["down"].append({"point": tuple(down_point.astype(np.float32).tolist())})
        boundaries["up"].append({"point": tuple(up_point.astype(np.float32).tolist())})

    for side in boundaries:
        key_axis = 1 if side in {"left", "right"} else 0
        boundaries[side].sort(key=lambda record: float(record["point"][key_axis]))

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


def _build_pile_cross_route_graph(
    support_points: np.ndarray,
    *,
    support_kind: str,
    platform_bounds: tuple[float, float, float, float, float],
    axis_tolerance: float,
) -> dict[str, object]:
    support_array = np.asarray(support_points, dtype=np.float32).reshape((-1, 3))
    x_min, x_max, y_min, y_max, top_z = (float(value) for value in platform_bounds)
    platform_center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, top_z], dtype=np.float32)

    nodes = [route_graph_node(f"support_{index}", point, support_kind) for index, point in enumerate(support_array)]
    nodes.append(route_graph_node("platform_center", platform_center, "platform"))
    edges: list[dict[str, object]] = []
    boundaries: dict[str, list[str]] = {side: [] for side in ("left", "right", "down", "up")}
    contact_ids: dict[tuple[float, float, float], str] = {}

    def support_id(index: int) -> str:
        return f"support_{index}"

    def add_edge(a: str, b: str) -> None:
        point_a = platform_center if a == "platform_center" else None
        point_b = platform_center if b == "platform_center" else None
        if point_a is None:
            point_a = support_array[int(a.rsplit("_", 1)[1])] if a.startswith("support_") else np.asarray(contact_points[a], dtype=np.float32)
        if point_b is None:
            point_b = support_array[int(b.rsplit("_", 1)[1])] if b.startswith("support_") else np.asarray(contact_points[b], dtype=np.float32)
        edges.append(route_graph_edge(a, b, float(np.linalg.norm(point_a - point_b))))

    contact_points: dict[str, np.ndarray] = {}

    def platform_contact(point: np.ndarray) -> str:
        contact_point = np.asarray(point, dtype=np.float32).reshape(3)
        key = tuple(np.round(contact_point, 5).astype(float).tolist())
        if key in contact_ids:
            return contact_ids[key]
        node_id = f"platform_contact_{len(contact_ids)}"
        contact_ids[key] = node_id
        contact_points[node_id] = contact_point
        nodes.append(route_graph_node(node_id, contact_point, "platform"))
        add_edge(node_id, "platform_center")
        return node_id

    def connect_ordered(indices: list[int]) -> None:
        for first, second in zip(indices[:-1], indices[1:]):
            add_edge(support_id(first), support_id(second))

    if support_array.size == 0:
        return build_route_graph(nodes, edges, boundaries)

    tolerance = max(float(axis_tolerance), 1e-6)

    for y_value in _cluster_axis(support_array[:, 1], tolerance):
        row_indices = np.flatnonzero(np.abs(support_array[:, 1] - float(y_value)) <= tolerance).astype(int).tolist()
        row_indices.sort(key=lambda index: float(support_array[index, 0]))
        y_on_platform = y_min - tolerance <= float(y_value) <= y_max + tolerance
        if not y_on_platform:
            continue

        left_indices = [index for index in row_indices if float(support_array[index, 0]) < x_min - 1e-6]
        right_indices = [index for index in row_indices if float(support_array[index, 0]) > x_max + 1e-6]
        connect_ordered(left_indices)
        connect_ordered(right_indices)

        contact_y = float(np.clip(float(y_value), y_min, y_max))
        if left_indices:
            boundaries["left"].append(support_id(left_indices[0]))
            contact_id = platform_contact(np.array([x_min, contact_y, top_z], dtype=np.float32))
            add_edge(support_id(left_indices[-1]), contact_id)
        if right_indices:
            boundaries["right"].append(support_id(right_indices[-1]))
            contact_id = platform_contact(np.array([x_max, contact_y, top_z], dtype=np.float32))
            add_edge(support_id(right_indices[0]), contact_id)

    for x_value in _cluster_axis(support_array[:, 0], tolerance):
        col_indices = np.flatnonzero(np.abs(support_array[:, 0] - float(x_value)) <= tolerance).astype(int).tolist()
        col_indices.sort(key=lambda index: float(support_array[index, 1]))
        x_on_platform = x_min - tolerance <= float(x_value) <= x_max + tolerance
        if not x_on_platform:
            continue

        down_indices = [index for index in col_indices if float(support_array[index, 1]) < y_min - 1e-6]
        up_indices = [index for index in col_indices if float(support_array[index, 1]) > y_max + 1e-6]
        connect_ordered(down_indices)
        connect_ordered(up_indices)

        contact_x = float(np.clip(float(x_value), x_min, x_max))
        if down_indices:
            boundaries["down"].append(support_id(down_indices[0]))
            contact_id = platform_contact(np.array([contact_x, y_min, top_z], dtype=np.float32))
            add_edge(support_id(down_indices[-1]), contact_id)
        if up_indices:
            boundaries["up"].append(support_id(up_indices[-1]))
            contact_id = platform_contact(np.array([contact_x, y_max, top_z], dtype=np.float32))
            add_edge(support_id(up_indices[0]), contact_id)

    return build_route_graph(nodes, edges, boundaries)


def generate_pile_terrain(cfg: PileTerrainCfg) -> ProceduralTerrainResult:
    size_x, size_y = cfg.size
    center_x, center_y = size_x / 2.0, size_y / 2.0
    origin = np.array([center_x, center_y, max(0.3, cfg.platform_height)], dtype=np.float32)

    pillar_spacing = max(cfg.pillar_spacing, 2.0 * cfg.pillar_radius + cfg.pillar_clearance_between)
    dx = pillar_spacing
    dy = pillar_spacing * np.sqrt(3.0) / 2.0
    margin_x = cfg.pillar_radius + 1e-4
    margin_y = cfg.pillar_radius + 1e-4

    meshes: list[trimesh.Trimesh] = [
        trimesh.creation.box(
            extents=[size_x, size_y, cfg.base_thickness],
            transform=trimesh.transformations.translation_matrix(
                [center_x, center_y, -cfg.ground_depth + cfg.base_thickness / 2.0]
            ),
        )
    ]
    members = [
        make_box_member(
            "base",
            kind="ground_base",
            role="void",
            bounds=meshes[0].bounds,
            traversable=False,
            params={"base_thickness": cfg.base_thickness, "ground_depth": cfg.ground_depth},
        )
    ]
    skirts = _create_boundary_skirt(size_x, size_y, cfg.skirt_thickness, cfg.ground_depth)
    meshes.extend(skirts)
    skirt_contacts = [("up",), ("down",), ("left",), ("right",)]
    for index, (skirt, contacts) in enumerate(zip(skirts, skirt_contacts)):
        members.append(
            make_box_member(
                f"boundary_skirt_{index}",
                kind="boundary_skirt",
                role="wall",
                bounds=skirt.bounds,
                traversable=False,
                boundary_contacts=contacts,
                params={"skirt_thickness": cfg.skirt_thickness, "ground_depth": cfg.ground_depth},
            )
        )

    platform = trimesh.creation.box(
        extents=[2.0 * cfg.platform_half, 2.0 * cfg.platform_half, cfg.ground_depth + cfg.platform_height],
        transform=trimesh.transformations.translation_matrix(
            [center_x, center_y, (-cfg.ground_depth + cfg.platform_height) / 2.0]
        ),
    )
    meshes.append(platform)
    members.append(
        make_box_member(
            "center_platform",
            kind="platform",
            role="platform",
            bounds=platform.bounds,
            traversable=True,
            params={"platform_half": cfg.platform_half, "platform_height": cfg.platform_height},
        )
    )

    candidate_rows = _generate_lattice_positions(size_x, size_y, dx, dy, margin_x, margin_y)
    if cfg.route_mode == "custom_mask" and cfg.custom_mask is None:
        raise ValueError("custom_mask must be provided when route_mode='custom_mask'")

    max_cols = max((len(row) for row in candidate_rows), default=0)
    if cfg.custom_mask is not None:
        if cfg.custom_mask.shape[0] != len(candidate_rows) or cfg.custom_mask.shape[1] < max_cols:
            raise ValueError("custom_mask shape must match the generated pile lattice shape")

    route_line_count = max(1, int(cfg.route_line_count))
    cluster_half = pillar_spacing * (route_line_count - 1) * 0.5
    xs = center_x + np.linspace(-cluster_half, cluster_half, num=route_line_count)
    ys = center_y + np.linspace(-cluster_half, cluster_half, num=route_line_count)
    tolerance = pillar_spacing * cfg.route_line_tolerance_ratio

    selected_positions: list[np.ndarray] = []
    for row_id, row_positions in enumerate(candidate_rows):
        for col_id, pos in enumerate(row_positions):
            x, y = float(pos[0]), float(pos[1])
            inside_platform = abs(x - center_x) <= cfg.platform_half and abs(y - center_y) <= cfg.platform_half
            if inside_platform:
                continue
            if not _should_keep_pillar(cfg, x, y, row_id, col_id, xs, ys, tolerance):
                continue

            pillar = trimesh.creation.cylinder(
                radius=cfg.pillar_radius,
                height=cfg.ground_depth,
                sections=cfg.pillar_sections,
            )
            pillar.apply_transform(trimesh.transformations.translation_matrix([x, y, -cfg.ground_depth / 2.0]))
            meshes.append(pillar)
            selected_positions.append(np.array([x, y, 0.0], dtype=np.float32))
            members.append(
                make_box_member(
                    f"pillar_{len(selected_positions) - 1}",
                    kind="pillar",
                    role="support",
                    bounds=pillar.bounds,
                    traversable=False,
                    params={"radius": cfg.pillar_radius, "route_mode": cfg.route_mode},
                    metadata={"lattice": (row_id, col_id)},
                )
            )

    mesh = merge_meshes(meshes, minimal_triangles=False)
    selected_positions_array = np.asarray(selected_positions, dtype=np.float32)
    route_constraints = _build_pile_route_constraints(
        selected_positions_array,
        tolerance=max(1e-3, pillar_spacing * 0.25),
    )
    platform_bounds = (
        center_x - cfg.platform_half,
        center_x + cfg.platform_half,
        center_y - cfg.platform_half,
        center_y + cfg.platform_half,
        float(cfg.platform_height),
    )
    route_graph = (
        _build_pile_cross_route_graph(
            selected_positions_array,
            support_kind="pillar",
            platform_bounds=platform_bounds,
            axis_tolerance=max(1e-3, pillar_spacing * 0.25),
        )
        if cfg.route_mode == "cross_route"
        else build_support_route_graph(
            selected_positions_array,
            boundary_points=_constraint_boundary_points(route_constraints),
            support_kind="pillar",
            max_step_distance=float(pillar_spacing * 1.35),
            max_height_delta=0.35,
            platform_bounds=platform_bounds,
        )
    )
    planning_metadata = {
        "height_limits": {
            "min": -0.2,
            "max": float(max(cfg.platform_height, 0.0) + cfg.ground_depth * 0.25),
        },
        "foothold_graph": {
            "max_step_distance": float(pillar_spacing * 1.35),
            "max_height_delta": 0.35,
        },
        "route_constraints": route_constraints,
        "route_graph": route_graph,
    }
    return ProceduralTerrainResult(
        mesh=mesh,
        terrain_mesh=mesh,
        origin=origin,
        terrain_state=TerrainState(
            members=members,
            portals=[
                TerrainPortal(
                    name="center_platform_top",
                    boundary="interior",
                    center=(float(center_x), float(center_y), float(cfg.platform_height)),
                    span=(2.0 * cfg.platform_half, 2.0 * cfg.platform_half, cfg.ground_depth + cfg.platform_height),
                    normal=(0.0, 0.0, 1.0),
                )
            ],
            connectivity={"route_mode": cfg.route_mode, "route_line_count": int(cfg.route_line_count)},
            metadata={
                "generator": "pile",
                "planning": planning_metadata,
            },
        ),
        metadata={
            "pillar_positions": selected_positions_array,
            "pillar_count": len(selected_positions),
            "route_mode": cfg.route_mode,
            "platform_height": cfg.platform_height,
            "planning": planning_metadata,
        },
    )


__all__ = ["PileTerrainCfg", "generate_pile_terrain"]
