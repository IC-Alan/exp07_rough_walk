from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Optional

import networkx as nx
import numpy as np
import trimesh
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from .mesh_utils import get_heights_from_mesh
from .route_graph import nearest_route_graph_node, route_graph_nodes_by_id, route_graph_shortest_path
from .terrain_state import ensure_terrain_state
from .visualization import visualize_mesh


def _require_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for debug image visualization") from exc
    return plt


def _height_sample_grid(
    bounds: np.ndarray,
    resolution: float,
    border_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    b_min = np.min(bounds, axis=0)
    b_max = np.max(bounds, axis=0)
    span_x = max(0.0, float(b_max[0] - b_min[0] - 2.0 * border_offset))
    span_y = max(0.0, float(b_max[1] - b_min[1] - 2.0 * border_offset))
    n_cols = max(2, int(np.floor(span_x / resolution)) + 1)
    n_rows = max(2, int(np.floor(span_y / resolution)) + 1)
    x = b_min[0] + border_offset + np.arange(n_cols, dtype=np.float32) * resolution
    y = b_min[1] + border_offset + np.arange(n_rows, dtype=np.float32) * resolution
    x = np.minimum(x, b_max[0] - border_offset)
    y = np.minimum(y, b_max[1] - border_offset)
    return x, y, b_min, b_max, (b_min + b_max) * 0.5


def get_height_array_of_mesh_with_resolution(
    mesh: trimesh.Trimesh,
    resolution: float = 0.4,
    border_offset: float = 0.0,
    return_points: bool = False,
):
    bbox = mesh.bounding_box.bounds
    x, y, b_min, b_max, center = _height_sample_grid(bbox, resolution, border_offset)
    dim = np.array([b_max[0] - b_min[0], b_max[1] - b_min[1], b_max[2] - b_min[2]])

    xv, yv = np.meshgrid(x, y)
    origins = np.stack([xv.flatten(), yv.flatten(), np.ones_like(xv.flatten()) * dim[2] * 2], axis=-1)
    heights = get_heights_from_mesh(mesh, origins)
    array = heights.reshape(len(y), len(x))
    origins[:, 2] = heights
    if return_points:
        return array, center[:2], origins
    return array, center[:2]


def _collect_ignored_height_members(terrain_state, ignored_roles: tuple[str, ...]) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    if terrain_state is None or len(ignored_roles) == 0:
        return []

    state = ensure_terrain_state(terrain_state)
    ignored = []
    ignored_role_set = set(ignored_roles)
    for member in state.members:
        if member.role not in ignored_role_set:
            continue
        center = np.asarray(member.center, dtype=np.float32)
        half_extents = np.asarray(member.extents, dtype=np.float32) * 0.5
        yaw_rad = float(np.deg2rad(member.yaw_deg))
        ignored.append((center, half_extents, float(np.cos(yaw_rad)), float(np.sin(yaw_rad))))
    return ignored


def _point_inside_member_volume(
    point: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    cos_yaw: float,
    sin_yaw: float,
    *,
    xy_padding: float = 1e-4,
    z_padding: float = 1e-4,
) -> bool:
    dx = float(point[0] - center[0])
    dy = float(point[1] - center[1])
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    return bool(
        abs(local_x) <= float(half_extents[0]) + xy_padding
        and abs(local_y) <= float(half_extents[1]) + xy_padding
        and abs(float(point[2] - center[2])) <= float(half_extents[2]) + z_padding
    )


def _get_navigation_heights_from_mesh(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    *,
    terrain_state=None,
    ignored_roles: tuple[str, ...] = ("ceiling",),
) -> np.ndarray:
    ignored_members = _collect_ignored_height_members(terrain_state, ignored_roles)
    if len(ignored_members) == 0:
        return get_heights_from_mesh(mesh, origins)

    if origins.shape[1] == 2:
        origins = np.concatenate([origins, np.ones((origins.shape[0], 1), dtype=origins.dtype) * 100.0], axis=1)

    vectors = np.stack(
        [np.zeros_like(origins[:, 0]), np.zeros_like(origins[:, 1]), -np.ones_like(origins[:, 2])],
        axis=-1,
    )
    points, index_ray, _ = mesh.ray.intersects_location(origins, vectors, multiple_hits=True)
    heights = np.zeros((origins.shape[0],), dtype=np.float32)
    if len(points) == 0:
        return heights

    order = np.lexsort((-points[:, 2], index_ray))
    resolved = np.zeros(origins.shape[0], dtype=bool)
    for hit_index in order:
        ray_index = int(index_ray[hit_index])
        if resolved[ray_index]:
            continue

        point = points[hit_index]
        if any(
            _point_inside_member_volume(point, center, half_extents, cos_yaw, sin_yaw)
            for center, half_extents, cos_yaw, sin_yaw in ignored_members
        ):
            continue

        heights[ray_index] = float(point[2])
        resolved[ray_index] = True

    return heights


def get_navigation_height_array_of_mesh_with_resolution(
    mesh: trimesh.Trimesh,
    resolution: float = 0.4,
    border_offset: float = 0.0,
    return_points: bool = False,
    *,
    terrain_state=None,
    ignored_roles: tuple[str, ...] = ("ceiling",),
):
    bbox = mesh.bounding_box.bounds
    x, y, b_min, b_max, center = _height_sample_grid(bbox, resolution, border_offset)
    dim = np.array([b_max[0] - b_min[0], b_max[1] - b_min[1], b_max[2] - b_min[2]])

    xv, yv = np.meshgrid(x, y)
    origins = np.stack([xv.flatten(), yv.flatten(), np.ones_like(xv.flatten()) * dim[2] * 2], axis=-1)
    heights = _get_navigation_heights_from_mesh(
        mesh,
        origins,
        terrain_state=terrain_state,
        ignored_roles=ignored_roles,
    )
    array = heights.reshape(len(y), len(x))
    origins[:, 2] = heights
    if return_points:
        return array, center[:2], origins
    return array, center[:2]


def _member_xy_mask(
    center: np.ndarray,
    half_extents: np.ndarray,
    cos_yaw: float,
    sin_yaw: float,
    xx: np.ndarray,
    yy: np.ndarray,
    padding: float,
) -> np.ndarray:
    dx = xx - center[0]
    dy = yy - center[1]
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    return (np.abs(local_x) <= half_extents[0] + padding) & (np.abs(local_y) <= half_extents[1] + padding)


def calc_spawnable_locations_on_terrain(
    mesh: trimesh.Trimesh,
    filter_size: tuple[int, int] = (5, 5),
    spawnable_threshold: float = 0.1,
    border_offset: float = 1.0,
    resolution: float = 0.4,
    visualize: bool = False,
) -> np.ndarray:
    array, _, origins = get_height_array_of_mesh_with_resolution(
        mesh,
        resolution=resolution,
        border_offset=border_offset,
        return_points=True,
    )

    if visualize:
        plt = _require_matplotlib_pyplot()
        plt.imshow(array)
        plt.colorbar()
        plt.show()

    flat_filter = np.ones(filter_size, dtype=np.float32) * -1.0
    flat_filter[filter_size[0] // 2, filter_size[1] // 2] = flat_filter.size - 1
    filtered_img = ndimage.convolve(array, flat_filter, mode="nearest")

    if visualize:
        plt = _require_matplotlib_pyplot()
        plt.imshow(filtered_img)
        plt.colorbar()
        plt.show()
        plt.imshow(np.abs(filtered_img) <= spawnable_threshold)
        plt.colorbar()
        plt.show()

    spawnable_idx = np.argwhere(np.abs(filtered_img) <= spawnable_threshold)
    spawnable_locations = origins.reshape((filtered_img.shape[0], filtered_img.shape[1], 3))[
        spawnable_idx[:, 0], spawnable_idx[:, 1]
    ]
    spawnable_locations[:, 2] = array[spawnable_idx[:, 0], spawnable_idx[:, 1]]
    return spawnable_locations


def get_sdf_of_points(points: np.ndarray, sdf_array: np.ndarray, center: np.ndarray, resolution: float) -> np.ndarray:
    point = points - center
    point = np.round(point / resolution).astype(int)
    point += np.array(sdf_array.shape) // 2

    is_valid = np.logical_and.reduce(
        [
            point[:, 0] >= 0,
            point[:, 0] < sdf_array.shape[0],
            point[:, 1] >= 0,
            point[:, 1] < sdf_array.shape[1],
            point[:, 2] >= 0,
            point[:, 2] < sdf_array.shape[2],
        ]
    )

    point[:, 0] = np.clip(point[:, 0], 0, sdf_array.shape[0] - 1)
    point[:, 1] = np.clip(point[:, 1], 0, sdf_array.shape[1] - 1)
    point[:, 2] = np.clip(point[:, 2], 0, sdf_array.shape[2] - 1)

    sdf = np.ones(point.shape[0], dtype=np.float32) * 1000.0
    sdf[is_valid] = sdf_array[point[is_valid, 0], point[is_valid, 1], point[is_valid, 2]]
    return sdf


def filter_spawnable_locations_with_sdf(
    spawnable_locations: np.ndarray,
    sdf_array: np.ndarray,
    height_offset: float = 0.5,
    sdf_resolution: float = 0.1,
    sdf_threshold: float = 0.2,
) -> np.ndarray:
    query_locations = spawnable_locations.copy()
    query_locations[:, 2] += height_offset
    sdf_values = get_sdf_of_points(query_locations, sdf_array, np.array([0.0, 0.0, 0.0]), sdf_resolution)
    return spawnable_locations[sdf_values > sdf_threshold]


def calc_spawnable_locations_with_sdf(
    terrain_mesh: trimesh.Trimesh,
    sdf_array: np.ndarray,
    visualize: bool = False,
    height_offset: float = 0.5,
    sdf_resolution: float = 0.1,
    sdf_threshold: float = 0.4,
) -> np.ndarray:
    spawnable_locations = calc_spawnable_locations_on_terrain(terrain_mesh, visualize=visualize)
    return filter_spawnable_locations_with_sdf(
        spawnable_locations,
        sdf_array,
        height_offset=height_offset,
        sdf_resolution=sdf_resolution,
        sdf_threshold=sdf_threshold,
    )


def locations_to_graph(positions: np.ndarray, threshold: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    for index, pos in enumerate(positions):
        graph.add_node(index, pos=pos)

    distances = np.sqrt(((positions[:, np.newaxis] - positions) ** 2).sum(axis=2))
    edges = np.transpose(np.where(distances < threshold))
    graph.add_edges_from(edges)
    return graph


def visualize_mesh_and_graphs(
    mesh: trimesh.Trimesh,
    points: nx.Graph | np.ndarray,
    color_values: Optional[np.ndarray] = None,
    goal_pos: Optional[np.ndarray] = None,
    route_points: Optional[np.ndarray] = None,
    show: bool = True,
):
    if isinstance(points, nx.Graph):
        point_attrs = nx.get_node_attributes(points, "pos")
        points = np.array(list(point_attrs.values()))
    return visualize_mesh(
        mesh,
        points=points,
        color_values=color_values,
        goal_pos=goal_pos,
        route_points=route_points,
        show=show,
    )


def create_2d_graph_from_height_array(
    height_array: np.ndarray,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    invalid_cost: float = 1000.0,
    use_diagonal: bool = False,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_mask: np.ndarray | None = None,
) -> nx.Graph:
    del invalid_cost
    graph, _, _ = _build_navigation_graph_from_height_array(
        height_array,
        graph_ratio=graph_ratio,
        height_threshold=height_threshold,
        use_diagonal=use_diagonal,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_mask=obstacle_mask,
    )
    return graph


def _graph_shape_from_height_array(height_array: np.ndarray, graph_ratio: int) -> tuple[int, int]:
    graph_shape = tuple((np.array(height_array.shape) // graph_ratio).astype(int).tolist())
    if graph_shape[0] <= 0 or graph_shape[1] <= 0:
        raise ValueError(
            f"graph_ratio={graph_ratio} is too large for height array shape {tuple(height_array.shape)}"
        )
    return graph_shape


def _coarsen_height_array(height_array: np.ndarray, graph_ratio: int, reducer) -> np.ndarray:
    graph_shape = _graph_shape_from_height_array(height_array, graph_ratio)
    trimmed = height_array[: graph_shape[0] * graph_ratio, : graph_shape[1] * graph_ratio]
    blocks = trimmed.reshape(graph_shape[0], graph_ratio, graph_shape[1], graph_ratio)
    return reducer(blocks, axis=(1, 3))


def _coarsen_boolean_mask(mask: np.ndarray, graph_ratio: int, reducer=np.any) -> np.ndarray:
    graph_shape = _graph_shape_from_height_array(mask.astype(np.uint8), graph_ratio)
    trimmed = mask[: graph_shape[0] * graph_ratio, : graph_shape[1] * graph_ratio]
    blocks = trimmed.reshape(graph_shape[0], graph_ratio, graph_shape[1], graph_ratio)
    return reducer(blocks, axis=(1, 3)).astype(bool)


def _compute_navigation_heights(height_array: np.ndarray, graph_ratio: int) -> np.ndarray:
    return _coarsen_height_array(height_array, graph_ratio, np.max).astype(np.float32)


def _navigation_height_limits(terrain_state) -> tuple[float | None, float | None]:
    if terrain_state is None:
        return None, None

    state = ensure_terrain_state(terrain_state)
    planning = state.metadata.get("planning", {})
    if not isinstance(planning, dict):
        return None, None
    limits = planning.get("height_limits", {})
    if not isinstance(limits, dict):
        return None, None

    min_height = limits.get("min")
    max_height = limits.get("max")
    return (
        None if min_height is None else float(min_height),
        None if max_height is None else float(max_height),
    )


def _apply_navigation_height_limits(height_array: np.ndarray, terrain_state) -> np.ndarray:
    min_height, max_height = _navigation_height_limits(terrain_state)
    if min_height is None and max_height is None:
        return height_array

    limited = np.asarray(height_array, dtype=np.float32).copy()
    if min_height is not None:
        limited[limited < min_height] = -np.inf
    if max_height is not None:
        limited[limited > max_height] = np.inf
    return limited


def _effective_min_traversable_height(min_traversable_height: float, terrain_state) -> float:
    min_height, _ = _navigation_height_limits(terrain_state)
    if min_height is None:
        return float(min_traversable_height)
    return float(min(min_traversable_height, min_height))


def _close_xy_mask(
    mask: np.ndarray,
    *,
    gap_fill_iterations: int = 1,
    safe_margin_iterations: int = 1,
) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)

    structure = ndimage.generate_binary_structure(2, 2)
    closed = np.asarray(mask, dtype=bool)
    if gap_fill_iterations > 0:
        closed = ndimage.binary_dilation(closed, structure=structure, iterations=int(gap_fill_iterations))
        closed = ndimage.binary_erosion(closed, structure=structure, iterations=int(gap_fill_iterations), border_value=1)
    if safe_margin_iterations > 0:
        closed = ndimage.binary_dilation(closed, structure=structure, iterations=int(safe_margin_iterations))
    return closed.astype(bool)


def _rasterize_semantic_obstacles_fine(
    bounds: np.ndarray,
    height_shape: tuple[int, int],
    resolution: float,
    terrain_state,
    obstacle_inflation_radius: int,
    navigation_height_array: np.ndarray | None = None,
    ceiling_clearance_threshold: float = 0.5,
) -> np.ndarray:
    if terrain_state is None:
        return np.zeros(height_shape, dtype=bool)

    state = ensure_terrain_state(terrain_state)
    mask = np.zeros(height_shape, dtype=bool)
    b_min = np.asarray(bounds[0], dtype=np.float32)
    blocked_roles = {"wall", "obstacle", "ceiling"}

    for member in state.members:
        if member.traversable or member.role not in blocked_roles:
            continue

        center = np.asarray(member.center, dtype=np.float32)
        extents = np.asarray(member.extents, dtype=np.float32)
        half_extents = extents * 0.5
        yaw_rad = float(np.deg2rad(member.yaw_deg))
        cos_yaw = float(np.cos(yaw_rad))
        sin_yaw = float(np.sin(yaw_rad))

        corners_local = np.array(
            [
                [-half_extents[0], -half_extents[1]],
                [-half_extents[0], half_extents[1]],
                [half_extents[0], -half_extents[1]],
                [half_extents[0], half_extents[1]],
            ],
            dtype=np.float32,
        )
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
        corners_world = corners_local @ rotation.T + center[:2]
        xy_min = corners_world.min(axis=0)
        xy_max = corners_world.max(axis=0)

        row_start = int(np.floor((xy_min[1] - b_min[1]) / resolution))
        row_end = int(np.ceil((xy_max[1] - b_min[1]) / resolution))
        col_start = int(np.floor((xy_min[0] - b_min[0]) / resolution))
        col_end = int(np.ceil((xy_max[0] - b_min[0]) / resolution))

        if row_end < 0 or col_end < 0 or row_start >= height_shape[0] or col_start >= height_shape[1]:
            continue

        row_start = max(row_start, 0)
        col_start = max(col_start, 0)
        row_end = min(row_end, height_shape[0] - 1)
        col_end = min(col_end, height_shape[1] - 1)

        row_coords = b_min[1] + np.arange(row_start, row_end + 1, dtype=np.float32) * resolution
        col_coords = b_min[0] + np.arange(col_start, col_end + 1, dtype=np.float32) * resolution
        yy, xx = np.meshgrid(row_coords, col_coords, indexing="ij")
        local_mask = _member_xy_mask(
            center,
            half_extents,
            cos_yaw,
            sin_yaw,
            xx,
            yy,
            resolution * 0.5,
        )
        if member.role == "ceiling" and navigation_height_array is not None:
            ceiling_bottom = float(center[2] - half_extents[2])
            clearance = ceiling_bottom - navigation_height_array[row_start : row_end + 1, col_start : col_end + 1]
            local_mask &= clearance < float(ceiling_clearance_threshold)
        mask[row_start : row_end + 1, col_start : col_end + 1] |= local_mask

    if obstacle_inflation_radius > 0 and np.any(mask):
        mask = ndimage.binary_dilation(
            mask,
            structure=ndimage.generate_binary_structure(2, 2),
            iterations=obstacle_inflation_radius,
        )
    return mask.astype(bool)


def _build_fine_navigation_masks(
    height_array: np.ndarray,
    bounds: np.ndarray,
    resolution: float,
    terrain_state,
    min_traversable_height: float,
    pit_inset_radius: int,
    obstacle_inflation_radius: int,
    ceiling_clearance_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(height_array)
    low_void_mask = _close_xy_mask(
        height_array < float(min_traversable_height),
        gap_fill_iterations=1,
        safe_margin_iterations=1,
    )
    obstacle_mask = _rasterize_semantic_obstacles_fine(
        bounds,
        height_array.shape,
        resolution,
        terrain_state,
        obstacle_inflation_radius,
        navigation_height_array=height_array,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
    )
    obstacle_mask |= ~finite_mask
    walkable_mask = finite_mask & ~low_void_mask
    if pit_inset_radius > 0 and np.any(low_void_mask):
        unsafe_low_margin = ndimage.binary_dilation(
            low_void_mask,
            structure=ndimage.generate_binary_structure(2, 1),
            iterations=pit_inset_radius,
        )
        walkable_mask &= ~unsafe_low_margin
    walkable_mask &= ~obstacle_mask
    return walkable_mask.astype(bool), low_void_mask.astype(bool), obstacle_mask.astype(bool)


@dataclass(frozen=True)
class _NavigationGrid:
    height_array: np.ndarray
    center: np.ndarray
    bounds: np.ndarray
    resolution: float
    walkable_mask: np.ndarray
    low_void_mask: np.ndarray
    obstacle_mask: np.ndarray

    def cell_for(self, point_xy: np.ndarray) -> tuple[int, int]:
        return _world_to_height_cell(point_xy, self.bounds, self.height_array, self.resolution)

    def nearest_walkable_cell(self, point_xy: np.ndarray) -> tuple[int, int] | None:
        return _nearest_walkable_height_cell(self.walkable_mask, self.cell_for(point_xy))

    def point_for(self, cell: tuple[int, int]) -> np.ndarray:
        return _height_cell_to_world_point(cell, self.bounds, self.height_array, self.resolution)


def _build_navigation_grid(
    mesh: trimesh.Trimesh,
    *,
    height_map_resolution: float,
    min_traversable_height: float,
    pit_inset_radius: int,
    obstacle_inflation_radius: int,
    ceiling_clearance_threshold: float,
    terrain_state=None,
) -> _NavigationGrid:
    height_array, center = get_navigation_height_array_of_mesh_with_resolution(
        mesh,
        resolution=height_map_resolution,
        terrain_state=terrain_state,
    )
    height_array = _apply_navigation_height_limits(height_array, terrain_state)
    bounds = mesh.bounding_box.bounds
    effective_min_height = _effective_min_traversable_height(min_traversable_height, terrain_state)
    walkable_mask, low_void_mask, obstacle_mask = _build_fine_navigation_masks(
        height_array,
        bounds,
        height_map_resolution,
        terrain_state,
        min_traversable_height=effective_min_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_inflation_radius=obstacle_inflation_radius,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
    )
    return _NavigationGrid(
        height_array=height_array,
        center=np.asarray(center, dtype=np.float32),
        bounds=bounds,
        resolution=height_map_resolution,
        walkable_mask=walkable_mask,
        low_void_mask=low_void_mask,
        obstacle_mask=obstacle_mask,
    )


def _world_to_height_cell(
    point_xy: np.ndarray,
    bounds: np.ndarray,
    height_array: np.ndarray,
    resolution: float,
) -> tuple[int, int]:
    b_min = bounds[0]
    row = int(np.round((point_xy[1] - b_min[1]) / resolution))
    col = int(np.round((point_xy[0] - b_min[0]) / resolution))
    row = int(np.clip(row, 0, height_array.shape[0] - 1))
    col = int(np.clip(col, 0, height_array.shape[1] - 1))
    return row, col


def _nearest_walkable_height_cell(walkable_mask: np.ndarray, target: tuple[int, int]) -> tuple[int, int] | None:
    if walkable_mask[target]:
        return target

    max_radius = max(walkable_mask.shape)
    row0, col0 = target
    for radius in range(1, max_radius + 1):
        row_start = max(0, row0 - radius)
        row_end = min(walkable_mask.shape[0] - 1, row0 + radius)
        col_start = max(0, col0 - radius)
        col_end = min(walkable_mask.shape[1] - 1, col0 + radius)
        best_cell: tuple[int, int] | None = None
        best_distance: int | None = None

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                if not walkable_mask[row, col]:
                    continue
                distance = abs(row - row0) + abs(col - col0)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_cell = (row, col)

        if best_cell is not None:
            return best_cell
    return None


def _height_cell_to_world_point(
    cell: tuple[int, int],
    bounds: np.ndarray,
    height_array: np.ndarray,
    resolution: float,
) -> np.ndarray:
    row, col = cell
    x = bounds[0, 0] + col * resolution
    y = bounds[0, 1] + row * resolution
    z = float(height_array[row, col])
    return np.array([x, y, z], dtype=np.float32)


def _extract_boundary_anchor_cells(
    walkable_mask: np.ndarray,
    bounds: np.ndarray,
    resolution: float,
    side: str,
) -> list[tuple[int, int]]:
    if side not in {"up", "down", "left", "right"}:
        raise ValueError(f"Unsupported boundary side: {side}")

    row_indices, col_indices = np.where(walkable_mask)
    if row_indices.size == 0:
        return []

    selected: list[tuple[int, int]] = []
    max_depth = max(2, int(round(0.75 / max(resolution, 1e-6))))

    if side in {"up", "down"}:
        edge_axis = row_indices
        sweep_axis = col_indices
        extreme = edge_axis.max() if side == "up" else edge_axis.min()
    else:
        edge_axis = col_indices
        sweep_axis = row_indices
        extreme = edge_axis.min() if side == "left" else edge_axis.max()

    within_band = np.abs(edge_axis - extreme) <= max_depth
    candidate_rows = row_indices[within_band]
    candidate_cols = col_indices[within_band]
    if candidate_rows.size == 0:
        candidate_rows = row_indices
        candidate_cols = col_indices

    if side in {"up", "down"}:
        cells = sorted({(int(r), int(c)) for r, c in zip(candidate_rows, candidate_cols)}, key=lambda item: (item[1], item[0]))
    else:
        cells = sorted({(int(r), int(c)) for r, c in zip(candidate_rows, candidate_cols)}, key=lambda item: (item[0], item[1]))

    for cell in cells:
        if not selected:
            selected.append(cell)
            continue
        if side in {"up", "down"}:
            if abs(cell[1] - selected[-1][1]) >= max(1, int(round(0.5 / max(resolution, 1e-6)))):
                selected.append(cell)
        else:
            if abs(cell[0] - selected[-1][0]) >= max(1, int(round(0.5 / max(resolution, 1e-6)))):
                selected.append(cell)

    if not selected and cells:
        selected.append(cells[len(cells) // 2])
    return selected


def compute_boundary_anchors(
    mesh: trimesh.Trimesh,
    *,
    height_map_resolution: float = 0.1,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_inflation_radius: int = 1,
    ceiling_clearance_threshold: float = 0.5,
    terrain_state=None,
) -> dict[str, list[dict[str, tuple[float, float, float] | tuple[int, int]]]]:
    grid = _build_navigation_grid(
        mesh,
        height_map_resolution=height_map_resolution,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_inflation_radius=obstacle_inflation_radius,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
        terrain_state=terrain_state,
    )

    anchors: dict[str, list[dict[str, tuple[float, float, float] | tuple[int, int]]]] = {}
    for side in ("up", "down", "left", "right"):
        anchors[side] = []
        for cell in _extract_boundary_anchor_cells(grid.walkable_mask, grid.bounds, grid.resolution, side):
            world_point = grid.point_for(cell)
            anchors[side].append({"cell": (int(cell[0]), int(cell[1])), "point": tuple(world_point.tolist())})
    return anchors


def _iter_height_neighbors(
    node: tuple[int, int],
    height_array: np.ndarray,
    walkable_mask: np.ndarray,
    low_void_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    height_threshold: float,
    use_diagonal: bool,
):
    del low_void_mask
    del obstacle_mask
    row, col = node
    start_height = float(height_array[row, col])
    cardinal_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    diagonal_offsets = ((1, 1), (1, -1), (-1, 1), (-1, -1)) if use_diagonal else ()

    for row_delta, col_delta in cardinal_offsets:
        next_row = row + row_delta
        next_col = col + col_delta
        if next_row < 0 or next_row >= walkable_mask.shape[0] or next_col < 0 or next_col >= walkable_mask.shape[1]:
            continue
        if walkable_mask[next_row, next_col]:
            next_height = float(height_array[next_row, next_col])
            if abs(next_height - start_height) <= height_threshold:
                yield (next_row, next_col), 1.0

    for row_delta, col_delta in diagonal_offsets:
        next_row = row + row_delta
        next_col = col + col_delta
        if next_row < 0 or next_row >= walkable_mask.shape[0] or next_col < 0 or next_col >= walkable_mask.shape[1]:
            continue
        if not walkable_mask[next_row, next_col]:
            continue
        if not walkable_mask[row, next_col] or not walkable_mask[next_row, col]:
            continue

        next_height = float(height_array[next_row, next_col])
        side_a_height = float(height_array[row, next_col])
        side_b_height = float(height_array[next_row, col])
        if (
            abs(next_height - start_height) <= height_threshold
            and abs(side_a_height - start_height) <= height_threshold
            and abs(side_b_height - start_height) <= height_threshold
        ):
            yield (next_row, next_col), float(np.sqrt(2.0))


def _find_height_path(
    height_array: np.ndarray,
    walkable_mask: np.ndarray,
    low_void_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    height_threshold: float,
    use_diagonal: bool,
) -> list[tuple[int, int]] | None:
    if start_cell == goal_cell:
        return [start_cell]

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start_cell))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
    closed: set[tuple[int, int]] = set()

    def heuristic(cell: tuple[int, int]) -> float:
        return float(abs(cell[0] - goal_cell[0]) + abs(cell[1] - goal_cell[1]))

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)
        for neighbor, step_cost in _iter_height_neighbors(
            current,
            height_array,
            walkable_mask,
            low_void_mask,
            obstacle_mask,
            height_threshold,
            use_diagonal,
        ):
            tentative_cost = current_cost + step_cost
            if tentative_cost >= g_score.get(neighbor, np.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_cost
            priority = tentative_cost + heuristic(neighbor)
            heapq.heappush(open_heap, (priority, tentative_cost, neighbor))

    return None


def _compute_navigation_mask(
    navigation_heights: np.ndarray,
    min_traversable_height: float,
    pit_inset_radius: int = 0,
    obstacle_mask: np.ndarray | None = None,
) -> np.ndarray:
    finite_mask = np.isfinite(navigation_heights)
    pit_mask = _close_xy_mask(
        ~finite_mask | (navigation_heights < float(min_traversable_height)),
        gap_fill_iterations=1,
        safe_margin_iterations=1,
    )
    safe_mask = finite_mask & ~pit_mask

    if pit_inset_radius > 0 and np.any(pit_mask):
        pit_mask = ndimage.binary_dilation(
            pit_mask,
            structure=ndimage.generate_binary_structure(2, 1),
            iterations=pit_inset_radius,
        )
        safe_mask &= ~pit_mask

    if obstacle_mask is not None:
        if obstacle_mask.shape != safe_mask.shape:
            raise ValueError(
                f"Obstacle mask shape {obstacle_mask.shape} does not match graph shape {safe_mask.shape}"
            )
        safe_mask &= ~obstacle_mask.astype(bool)

    return safe_mask.astype(bool)


def _rasterize_semantic_obstacles(
    bounds: np.ndarray,
    graph_shape: tuple[int, int],
    resolution: float,
    graph_ratio: int,
    terrain_state,
    obstacle_inflation_radius: int,
    navigation_height_array: np.ndarray | None = None,
    ceiling_clearance_threshold: float = 0.5,
) -> np.ndarray:
    fine_mask = _rasterize_semantic_obstacles_fine(
        bounds,
        (graph_shape[0] * graph_ratio, graph_shape[1] * graph_ratio),
        resolution,
        terrain_state,
        obstacle_inflation_radius,
        navigation_height_array=navigation_height_array,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
    )
    return _coarsen_boolean_mask(fine_mask, graph_ratio, reducer=np.any)


def _flatten_graph_node(node: tuple[int, int], graph_shape: tuple[int, int]) -> int:
    return int(node[0] * graph_shape[1] + node[1])


def _diagonal_move_is_safe(
    edge_start: tuple[int, int],
    edge_end: tuple[int, int],
    navigation_heights: np.ndarray,
    safe_mask: np.ndarray,
    height_threshold: float,
) -> bool:
    row_delta = edge_end[0] - edge_start[0]
    col_delta = edge_end[1] - edge_start[1]
    if abs(row_delta) != 1 or abs(col_delta) != 1:
        return False

    side_a = (edge_start[0], edge_end[1])
    side_b = (edge_end[0], edge_start[1])
    if not safe_mask[side_a] or not safe_mask[side_b]:
        return False

    start_height = float(navigation_heights[edge_start])
    end_height = float(navigation_heights[edge_end])
    side_a_height = float(navigation_heights[side_a])
    side_b_height = float(navigation_heights[side_b])

    return bool(
        abs(end_height - start_height) <= height_threshold
        and abs(side_a_height - start_height) <= height_threshold
        and abs(side_b_height - start_height) <= height_threshold
    )


def is_height_transition_traversable(
    edge_start: tuple[int, int],
    edge_end: tuple[int, int],
    height_array: np.ndarray,
    ratio: int,
    height_threshold: float = 0.4,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_mask: np.ndarray | None = None,
) -> bool:
    navigation_heights = _compute_navigation_heights(height_array, ratio)
    safe_mask = _compute_navigation_mask(
        navigation_heights,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_mask=obstacle_mask,
    )

    if edge_start not in np.ndindex(safe_mask.shape) or edge_end not in np.ndindex(safe_mask.shape):
        return False
    if not safe_mask[edge_start] or not safe_mask[edge_end]:
        return False

    row_delta = edge_end[0] - edge_start[0]
    col_delta = edge_end[1] - edge_start[1]
    if abs(row_delta) + abs(col_delta) == 1:
        return bool(abs(float(navigation_heights[edge_end]) - float(navigation_heights[edge_start])) <= height_threshold)
    if abs(row_delta) == 1 and abs(col_delta) == 1:
        return _diagonal_move_is_safe(edge_start, edge_end, navigation_heights, safe_mask, height_threshold)
    return False


def _build_navigation_graph_from_height_array(
    height_array: np.ndarray,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    use_diagonal: bool = False,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_mask: np.ndarray | None = None,
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    navigation_heights = _compute_navigation_heights(height_array, graph_ratio)
    safe_mask = _compute_navigation_mask(
        navigation_heights,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_mask=obstacle_mask,
    )

    graph = nx.Graph()
    graph_shape = navigation_heights.shape
    for row in range(graph_shape[0]):
        for col in range(graph_shape[1]):
            if safe_mask[row, col]:
                graph.add_node((row, col), height=float(navigation_heights[row, col]))

    cardinal_offsets = ((1, 0), (0, 1))
    diagonal_offsets = ((1, 1), (1, -1)) if use_diagonal else ()
    for row in range(graph_shape[0]):
        for col in range(graph_shape[1]):
            if not safe_mask[row, col]:
                continue

            for row_delta, col_delta in cardinal_offsets:
                next_row = row + row_delta
                next_col = col + col_delta
                if next_row >= graph_shape[0] or next_col >= graph_shape[1] or not safe_mask[next_row, next_col]:
                    continue

                if abs(float(navigation_heights[next_row, next_col]) - float(navigation_heights[row, col])) > height_threshold:
                    continue

                graph.add_edge((row, col), (next_row, next_col), weight=1.0)

            for row_delta, col_delta in diagonal_offsets:
                next_row = row + row_delta
                next_col = col + col_delta
                if next_row < 0 or next_row >= graph_shape[0] or next_col < 0 or next_col >= graph_shape[1]:
                    continue
                if not safe_mask[next_row, next_col]:
                    continue
                if not _diagonal_move_is_safe(
                    (row, col),
                    (next_row, next_col),
                    navigation_heights,
                    safe_mask,
                    height_threshold,
                ):
                    continue

                graph.add_edge((row, col), (next_row, next_col), weight=float(np.sqrt(2.0)))

    return graph, navigation_heights, safe_mask


def create_strict_2d_graph_from_height_array(
    height_array: np.ndarray,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    use_diagonal: bool = False,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_mask: np.ndarray | None = None,
) -> nx.Graph:
    graph, _, _ = _build_navigation_graph_from_height_array(
        height_array,
        graph_ratio=graph_ratio,
        height_threshold=height_threshold,
        use_diagonal=use_diagonal,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_mask=obstacle_mask,
    )
    return graph


def distance_matrix_from_graph(graph: nx.Graph, graph_shape: tuple[int, int] | None = None) -> np.ndarray:
    if graph_shape is None:
        adjacency = nx.adjacency_matrix(graph, weight="weight")
        graph_matrix = csr_matrix(adjacency)
        distances, _ = shortest_path(csgraph=graph_matrix, directed=False, return_predecessors=True)
        return distances

    flat_size = graph_shape[0] * graph_shape[1]
    full_distances = np.full((flat_size, flat_size), np.inf, dtype=np.float32)
    node_list = sorted(graph.nodes())
    if not node_list:
        return full_distances

    adjacency = nx.adjacency_matrix(graph, nodelist=node_list, weight="weight")
    graph_matrix = csr_matrix(adjacency)
    distances, _ = shortest_path(csgraph=graph_matrix, directed=False, return_predecessors=True)
    flat_indices = np.asarray([_flatten_graph_node(node, graph_shape) for node in node_list], dtype=np.int64)
    full_distances[np.ix_(flat_indices, flat_indices)] = distances.astype(np.float32)
    return full_distances


@dataclass(frozen=True)
class _FootholdNode:
    point: tuple[float, float, float]
    kind: str

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.point, dtype=np.float32)


def _has_foothold_graph(terrain_state) -> bool:
    if terrain_state is None:
        return False
    state = ensure_terrain_state(terrain_state)
    planning = state.metadata.get("planning", {})
    graph_cfg = planning.get("foothold_graph", {}) if isinstance(planning, dict) else {}
    if isinstance(graph_cfg, dict) and bool(graph_cfg.get("enabled", False)):
        return True
    if state.metadata.get("generator") == "pile":
        return True
    kinds = {member.kind for member in state.members}
    return "pillar" in kinds and "platform" in kinds


def _get_planning_route_graph(terrain_state) -> dict | None:
    if terrain_state is None:
        return None
    state = ensure_terrain_state(terrain_state)
    planning = state.metadata.get("planning", {})
    if not isinstance(planning, dict):
        return None
    route_graph = planning.get("route_graph")
    if not isinstance(route_graph, dict):
        return None
    return route_graph


def _find_route_graph_route(terrain_state, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    route_graph = _get_planning_route_graph(terrain_state)
    if route_graph is None:
        raise nx.NetworkXNoPath("Terrain does not declare a route graph")
    if not route_graph.get("nodes") or not route_graph.get("edges"):
        raise nx.NetworkXNoPath("Terrain route graph has no nodes or edges")

    start = np.asarray(start_xy, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
    if start.shape[0] < 2 or goal.shape[0] < 2:
        raise ValueError("Route endpoints must include x/y coordinates")
    start_point = np.array([start[0], start[1], start[2] if start.shape[0] >= 3 else 0.0], dtype=np.float32)
    goal_point = np.array([goal[0], goal[1], goal[2] if goal.shape[0] >= 3 else 0.0], dtype=np.float32)

    try:
        start_id, _ = nearest_route_graph_node(route_graph, start_point)
        goal_id, _ = nearest_route_graph_node(route_graph, goal_point)
        path_node_ids = route_graph_shortest_path(route_graph, start_id, goal_id)
    except ValueError as exc:
        raise nx.NetworkXNoPath(str(exc)) from exc

    nodes_by_id = route_graph_nodes_by_id(route_graph)
    route_points: list[np.ndarray] = []
    for node_id in path_node_ids:
        point = nodes_by_id[node_id]["point"].astype(np.float32)
        if not route_points or not np.allclose(route_points[-1], point, atol=1e-5):
            route_points.append(point)
    return np.stack(route_points, axis=0).astype(np.float32)


def _member_top_z(member) -> float:
    center = np.asarray(member.center, dtype=np.float32)
    extents = np.asarray(member.extents, dtype=np.float32)
    return float(center[2] + extents[2] * 0.5)


def _member_local_xy(point_xy: np.ndarray, member) -> np.ndarray:
    center = np.asarray(member.center, dtype=np.float32)
    yaw_rad = float(np.deg2rad(member.yaw_deg))
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    dx = float(point_xy[0] - center[0])
    dy = float(point_xy[1] - center[1])
    return np.array([cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy], dtype=np.float32)


def _member_world_xy(local_xy: np.ndarray, member) -> np.ndarray:
    center = np.asarray(member.center, dtype=np.float32)
    yaw_rad = float(np.deg2rad(member.yaw_deg))
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    x = cos_yaw * float(local_xy[0]) - sin_yaw * float(local_xy[1]) + float(center[0])
    y = sin_yaw * float(local_xy[0]) + cos_yaw * float(local_xy[1]) + float(center[1])
    return np.array([x, y], dtype=np.float32)


def _point_inside_member_xy(point_xy: np.ndarray, member, padding: float = 1e-4) -> bool:
    local_xy = _member_local_xy(point_xy, member)
    half_extents = np.asarray(member.extents, dtype=np.float32)[:2] * 0.5
    return bool(np.all(np.abs(local_xy) <= half_extents + float(padding)))


def _closest_member_xy(point_xy: np.ndarray, member) -> np.ndarray:
    local_xy = _member_local_xy(point_xy, member)
    half_extents = np.asarray(member.extents, dtype=np.float32)[:2] * 0.5
    clamped = np.clip(local_xy, -half_extents, half_extents)
    return _member_world_xy(clamped, member)


def _node_on_platform(node: _FootholdNode) -> bool:
    return node.kind in {"platform", "endpoint_platform"}


def _dedupe_nodes(nodes: list[_FootholdNode]) -> list[_FootholdNode]:
    deduped: list[_FootholdNode] = []
    seen: set[tuple[float, float, float, str]] = set()
    for node in nodes:
        key = (*np.round(node.array, 5).astype(float).tolist(), node.kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(node)
    return deduped


def _nearest_node(point_xy: np.ndarray, nodes: list[_FootholdNode], kinds: set[str] | None = None) -> tuple[_FootholdNode, float]:
    candidates = nodes if kinds is None else [node for node in nodes if node.kind in kinds]
    if not candidates:
        raise nx.NetworkXNoPath("Foothold terrain has no matching support nodes")
    best = min(candidates, key=lambda node: float(np.linalg.norm(node.array[:2] - point_xy[:2])))
    return best, float(np.linalg.norm(best.array[:2] - point_xy[:2]))


def _foothold_endpoint_node(
    point_xy: np.ndarray,
    platform_members: list,
    support_nodes: list[_FootholdNode],
    max_step_distance: float,
    label: str,
) -> _FootholdNode:
    for platform in platform_members:
        if _point_inside_member_xy(point_xy, platform):
            point = np.array([point_xy[0], point_xy[1], _member_top_z(platform)], dtype=np.float32)
            return _FootholdNode(tuple(point.tolist()), "endpoint_platform")

    nearest, distance = _nearest_node(point_xy, support_nodes)
    if distance > max_step_distance:
        raise nx.NetworkXNoPath(f"Foothold {label} is not within one step of a support")
    return _FootholdNode(tuple(nearest.array.tolist()), f"endpoint_{label}")


def _collect_foothold_nodes(
    terrain_state,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> tuple[list[_FootholdNode], int, int, float, float]:
    state = ensure_terrain_state(terrain_state)
    planning = state.metadata.get("planning", {})
    graph_cfg = planning.get("foothold_graph", {}) if isinstance(planning, dict) else {}
    max_step_distance = float(graph_cfg.get("max_step_distance", 0.65))
    max_height_delta = float(graph_cfg.get("max_height_delta", 0.4))
    support_kinds = {str(kind) for kind in graph_cfg.get("support_kinds", ("pillar",))}

    platform_members = [member for member in state.members if member.kind == "platform" or member.role == "platform"]
    support_nodes: list[_FootholdNode] = []
    for member in state.members:
        if member.kind not in support_kinds:
            continue
        center = np.asarray(member.center, dtype=np.float32)
        point = np.array([center[0], center[1], _member_top_z(member)], dtype=np.float32)
        support_nodes.append(_FootholdNode(tuple(point.tolist()), member.kind))

    if not platform_members or not support_nodes:
        raise nx.NetworkXNoPath("Foothold terrain requires platform and support footholds")

    start_node = _foothold_endpoint_node(start_xy, platform_members, support_nodes, max_step_distance, "start")
    goal_node = _foothold_endpoint_node(goal_xy, platform_members, support_nodes, max_step_distance, "goal")
    nodes = [start_node, goal_node]
    nodes.extend(support_nodes)

    platform_nodes: list[_FootholdNode] = []
    for platform in platform_members:
        center = np.asarray(platform.center, dtype=np.float32)
        top_z = _member_top_z(platform)
        platform_nodes.append(_FootholdNode((float(center[0]), float(center[1]), top_z), "platform"))
        for point_xy in (start_xy, goal_xy):
            if _point_inside_member_xy(point_xy, platform):
                point = np.array([point_xy[0], point_xy[1], top_z], dtype=np.float32)
                platform_nodes.append(_FootholdNode(tuple(point.tolist()), "platform"))
        for support in support_nodes:
            closest_xy = _closest_member_xy(support.array[:2], platform)
            if float(np.linalg.norm(closest_xy - support.array[:2])) <= max_step_distance:
                point = np.array([closest_xy[0], closest_xy[1], top_z], dtype=np.float32)
                platform_nodes.append(_FootholdNode(tuple(point.tolist()), "platform"))

    nodes.extend(_dedupe_nodes(platform_nodes))
    return nodes, 0, 1, max_step_distance, max_height_delta


def _foothold_edge_cost(
    node_a: _FootholdNode,
    node_b: _FootholdNode,
    max_step_distance: float,
    max_height_delta: float,
) -> float | None:
    point_a = node_a.array
    point_b = node_b.array
    horizontal_distance = float(np.linalg.norm(point_a[:2] - point_b[:2]))
    height_delta = abs(float(point_a[2] - point_b[2]))

    if _node_on_platform(node_a) and _node_on_platform(node_b):
        return float(np.linalg.norm(point_a - point_b))

    if horizontal_distance > max_step_distance or height_delta > max_height_delta:
        return None
    return float(np.linalg.norm(point_a - point_b))


def _find_foothold_route(terrain_state, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    nodes, start_idx, goal_idx, max_step_distance, max_height_delta = _collect_foothold_nodes(
        terrain_state,
        np.asarray(start_xy, dtype=np.float32),
        np.asarray(goal_xy, dtype=np.float32),
    )

    open_heap: list[tuple[float, float, int]] = [(0.0, 0.0, start_idx)]
    came_from: dict[int, int] = {}
    g_score: dict[int, float] = {start_idx: 0.0}
    closed: set[int] = set()
    goal_point = nodes[goal_idx].array

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return np.stack([nodes[index].array for index in path], axis=0).astype(np.float32)

        closed.add(current)
        for neighbor, neighbor_node in enumerate(nodes):
            if neighbor == current or neighbor in closed:
                continue
            edge_cost = _foothold_edge_cost(nodes[current], neighbor_node, max_step_distance, max_height_delta)
            if edge_cost is None:
                continue
            tentative_cost = current_cost + edge_cost
            if tentative_cost >= g_score.get(neighbor, np.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_cost
            heuristic = float(np.linalg.norm(neighbor_node.array - goal_point))
            heapq.heappush(open_heap, (tentative_cost + heuristic, tentative_cost, neighbor))

    raise nx.NetworkXNoPath("No foothold route between requested endpoints")


def _resample_route_xy(route: np.ndarray, spacing: float | None) -> np.ndarray:
    if spacing is None or spacing <= 0.0 or route.shape[0] < 2:
        return route.astype(np.float32)

    diffs = np.diff(route, axis=0)
    segment_lengths = np.linalg.norm(diffs[:, :2], axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 1e-8:
        return route[:1].astype(np.float32)

    sample_distances = np.arange(0.0, total_length, float(spacing), dtype=np.float64)
    if sample_distances.size == 0 or not np.isclose(sample_distances[-1], total_length):
        sample_distances = np.append(sample_distances, total_length)

    cum_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    indices = np.searchsorted(cum_lengths[1:], sample_distances, side="right")
    indices = np.clip(indices, 0, route.shape[0] - 2)

    sampled = np.empty((sample_distances.shape[0], 3), dtype=np.float32)
    for output_index, (distance, segment_index) in enumerate(zip(sample_distances, indices)):
        denom = max(float(segment_lengths[segment_index]), 1e-8)
        t = float(np.clip((distance - cum_lengths[segment_index]) / denom, 0.0, 1.0))
        sampled[output_index] = route[segment_index] + t * diffs[segment_index]
    return sampled


def _smooth_route_xy(route: np.ndarray, iterations: int) -> np.ndarray:
    smoothed = np.asarray(route, dtype=np.float32)
    for _ in range(max(0, int(iterations))):
        if smoothed.shape[0] < 3:
            break
        points = [smoothed[0]]
        for start, end in zip(smoothed[:-1], smoothed[1:]):
            points.append(0.75 * start + 0.25 * end)
            points.append(0.25 * start + 0.75 * end)
        points.append(smoothed[-1])
        smoothed = np.asarray(points, dtype=np.float32)
    return smoothed


def _project_route_to_grid(
    route: np.ndarray,
    grid: _NavigationGrid,
    *,
    height_threshold: float,
) -> np.ndarray | None:
    projected = np.asarray(route, dtype=np.float32).copy()
    for index, point in enumerate(projected):
        cell = grid.cell_for(point[:2])
        if not grid.walkable_mask[cell]:
            return None
        projected[index, 2] = float(grid.height_array[cell])

    for start, end in zip(projected[:-1], projected[1:]):
        distance = float(np.linalg.norm(end[:2] - start[:2]))
        sample_count = max(2, int(np.ceil(distance / max(grid.resolution, 1e-6))) + 1)
        previous_height: float | None = None
        for t in np.linspace(0.0, 1.0, sample_count):
            point_xy = start[:2] + float(t) * (end[:2] - start[:2])
            cell = grid.cell_for(point_xy)
            if not grid.walkable_mask[cell]:
                return None
            height = float(grid.height_array[cell])
            if previous_height is not None and abs(height - previous_height) > height_threshold:
                return None
            previous_height = height
    return projected


def _postprocess_grid_route(
    route: np.ndarray,
    grid: _NavigationGrid,
    *,
    sample_spacing: float | None,
    smoothing_iterations: int,
    height_threshold: float,
) -> np.ndarray:
    if (sample_spacing is None or sample_spacing <= 0.0) and smoothing_iterations <= 0:
        return route.astype(np.float32)

    resampled = _project_route_to_grid(
        _resample_route_xy(route, sample_spacing),
        grid,
        height_threshold=height_threshold,
    )
    if smoothing_iterations <= 0:
        return route.astype(np.float32) if resampled is None else resampled

    smoothed = _resample_route_xy(_smooth_route_xy(route, smoothing_iterations), sample_spacing)
    projected = _project_route_to_grid(smoothed, grid, height_threshold=height_threshold)
    if projected is not None:
        return projected
    return route.astype(np.float32) if resampled is None else resampled


def compute_distance_matrix(
    mesh: trimesh.Trimesh,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    invalid_cost: float = 1000.0,
    height_map_resolution: float = 0.1,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_inflation_radius: int = 1,
    ceiling_clearance_threshold: float = 0.5,
    terrain_state=None,
    use_diagonal: bool = False,
):
    del invalid_cost
    height_array, center = get_navigation_height_array_of_mesh_with_resolution(
        mesh,
        resolution=height_map_resolution,
        terrain_state=terrain_state,
    )
    height_array = _apply_navigation_height_limits(height_array, terrain_state)
    effective_min_height = _effective_min_traversable_height(min_traversable_height, terrain_state)
    graph_shape = _graph_shape_from_height_array(height_array, graph_ratio)
    obstacle_mask = _rasterize_semantic_obstacles(
        mesh.bounding_box.bounds,
        graph_shape,
        height_map_resolution,
        graph_ratio,
        terrain_state,
        obstacle_inflation_radius,
        navigation_height_array=height_array,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
    )
    graph, _, _ = _build_navigation_graph_from_height_array(
        height_array,
        graph_ratio=graph_ratio,
        height_threshold=height_threshold,
        use_diagonal=use_diagonal,
        min_traversable_height=effective_min_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_mask=obstacle_mask,
    )
    dist_matrix = distance_matrix_from_graph(graph, graph_shape)
    shape = graph_shape
    return dist_matrix, shape, center


def compute_strict_distance_matrix(
    mesh: trimesh.Trimesh,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    height_map_resolution: float = 0.1,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_inflation_radius: int = 1,
    ceiling_clearance_threshold: float = 0.5,
    terrain_state=None,
    use_diagonal: bool = False,
):
    return compute_distance_matrix(
        mesh,
        graph_ratio=graph_ratio,
        height_threshold=height_threshold,
        height_map_resolution=height_map_resolution,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_inflation_radius=obstacle_inflation_radius,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
        terrain_state=terrain_state,
        use_diagonal=use_diagonal,
    )


def height_map_cost(
    edge_start: tuple[int, int],
    edge_end: tuple[int, int],
    height_array: np.ndarray,
    ratio: int,
    height_threshold: float = 0.4,
    invalid_cost: float = 1000.0,
) -> float:
    distance = float(np.linalg.norm(np.asarray(edge_end) - np.asarray(edge_start)))
    if is_height_transition_traversable(
        edge_start,
        edge_end,
        height_array,
        ratio,
        height_threshold=height_threshold,
    ):
        return distance
    return distance + float(invalid_cost)


def find_route(
    mesh: trimesh.Trimesh,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    invalid_cost: float = 1000.0,
    height_map_resolution: float = 0.1,
    use_diagonal: bool = False,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_inflation_radius: int = 1,
    ceiling_clearance_threshold: float = 0.5,
    terrain_state=None,
    route_sample_spacing: float | None = 0.2,
    route_smoothing_iterations: int = 1,
) -> np.ndarray:
    del invalid_cost
    if _get_planning_route_graph(terrain_state) is not None:
        return _find_route_graph_route(terrain_state, start_xy, goal_xy)

    if _has_foothold_graph(terrain_state):
        return _find_foothold_route(terrain_state, start_xy, goal_xy)

    grid = _build_navigation_grid(
        mesh,
        height_map_resolution=height_map_resolution,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_inflation_radius=obstacle_inflation_radius,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
        terrain_state=terrain_state,
    )
    start_cell = grid.nearest_walkable_cell(np.asarray(start_xy, dtype=np.float32))
    goal_cell = grid.nearest_walkable_cell(np.asarray(goal_xy, dtype=np.float32))
    if start_cell is None or goal_cell is None:
        raise nx.NetworkXNoPath("No safe navigation cells are available near the requested endpoints")

    path_cells = _find_height_path(
        grid.height_array,
        grid.walkable_mask,
        grid.low_void_mask,
        grid.obstacle_mask,
        start_cell,
        goal_cell,
        height_threshold,
        use_diagonal,
    )
    if path_cells is None:
        raise nx.NetworkXNoPath(f"No path between {start_cell} and {goal_cell}.")

    route = np.stack(
        [grid.point_for(cell) for cell in path_cells],
        axis=0,
    )
    return _postprocess_grid_route(
        route,
        grid,
        sample_spacing=route_sample_spacing,
        smoothing_iterations=route_smoothing_iterations,
        height_threshold=height_threshold,
    )


def has_strict_route(
    mesh: trimesh.Trimesh,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    height_map_resolution: float = 0.1,
    use_diagonal: bool = False,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_inflation_radius: int = 1,
    ceiling_clearance_threshold: float = 0.5,
    terrain_state=None,
) -> bool:
    if _get_planning_route_graph(terrain_state) is not None:
        try:
            _find_route_graph_route(terrain_state, start_xy, goal_xy)
        except nx.NetworkXNoPath:
            return False
        return True

    if _has_foothold_graph(terrain_state):
        try:
            _find_foothold_route(terrain_state, start_xy, goal_xy)
        except nx.NetworkXNoPath:
            return False
        return True

    grid = _build_navigation_grid(
        mesh,
        height_map_resolution=height_map_resolution,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_inflation_radius=obstacle_inflation_radius,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
        terrain_state=terrain_state,
    )
    start_cell = grid.nearest_walkable_cell(np.asarray(start_xy, dtype=np.float32))
    goal_cell = grid.nearest_walkable_cell(np.asarray(goal_xy, dtype=np.float32))
    if start_cell is None or goal_cell is None:
        return False

    path_cells = _find_height_path(
        grid.height_array,
        grid.walkable_mask,
        grid.low_void_mask,
        obstacle_mask=grid.obstacle_mask,
        start_cell=start_cell,
        goal_cell=goal_cell,
        height_threshold=height_threshold,
        use_diagonal=use_diagonal,
    )
    return path_cells is not None


def find_strict_route(
    mesh: trimesh.Trimesh,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    height_map_resolution: float = 0.1,
    use_diagonal: bool = False,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_inflation_radius: int = 1,
    ceiling_clearance_threshold: float = 0.5,
    terrain_state=None,
    route_sample_spacing: float | None = 0.2,
    route_smoothing_iterations: int = 1,
) -> np.ndarray:
    return find_route(
        mesh,
        start_xy,
        goal_xy,
        graph_ratio=graph_ratio,
        height_threshold=height_threshold,
        height_map_resolution=height_map_resolution,
        use_diagonal=use_diagonal,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_inflation_radius=obstacle_inflation_radius,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
        terrain_state=terrain_state,
        route_sample_spacing=route_sample_spacing,
        route_smoothing_iterations=route_smoothing_iterations,
    )


def visualize_distance(
    mesh: trimesh.Trimesh,
    height_array: np.ndarray,
    distance_matrix: np.ndarray,
    graph_ratio: int,
    goal_pos: tuple[int, int],
    height_array_resolution: float = 0.1,
    show: bool = True,
):
    distance_shape = (np.array(height_array.shape) // graph_ratio).astype(int)
    grid_x, grid_y = np.meshgrid(np.arange(distance_shape[0]), np.arange(distance_shape[1]), indexing="ij")
    grid_z = height_array[grid_x * graph_ratio, grid_y * graph_ratio]
    points = np.stack(
        [
            mesh.bounds[0, 0] + grid_y.flatten() * graph_ratio * height_array_resolution,
            mesh.bounds[0, 1] + grid_x.flatten() * graph_ratio * height_array_resolution,
            grid_z.flatten(),
        ],
        axis=1,
    )
    goal_idx = goal_pos[0] * distance_shape[0] + goal_pos[1]
    distances = distance_matrix[goal_idx, :]
    return visualize_mesh(mesh, points=points, color_values=distances, show=show)


def compute_traversability_map(
    mesh: trimesh.Trimesh,
    graph_ratio: int = 4,
    height_threshold: float = 0.4,
    height_map_resolution: float = 0.1,
    fill_holes: bool = True,
    min_traversable_height: float = -0.2,
    pit_inset_radius: int = 0,
    obstacle_inflation_radius: int = 1,
    ceiling_clearance_threshold: float = 0.5,
    terrain_state=None,
) -> np.ndarray:
    del height_threshold
    del fill_holes
    grid = _build_navigation_grid(
        mesh,
        height_map_resolution=height_map_resolution,
        min_traversable_height=min_traversable_height,
        pit_inset_radius=pit_inset_radius,
        obstacle_inflation_radius=obstacle_inflation_radius,
        ceiling_clearance_threshold=ceiling_clearance_threshold,
        terrain_state=terrain_state,
    )
    return _coarsen_boolean_mask(grid.walkable_mask, graph_ratio, reducer=np.any)



__all__ = [
    "calc_spawnable_locations_on_terrain",
    "calc_spawnable_locations_with_sdf",
    "compute_distance_matrix",
    "compute_strict_distance_matrix",
    "compute_traversability_map",
    "create_2d_graph_from_height_array",
    "create_strict_2d_graph_from_height_array",
    "distance_matrix_from_graph",
    "find_route",
    "find_strict_route",
    "get_height_array_of_mesh_with_resolution",
    "get_navigation_height_array_of_mesh_with_resolution",
    "has_strict_route",
    "height_map_cost",
    "is_height_transition_traversable",
    "locations_to_graph",
    "visualize_distance",
    "visualize_mesh_and_graphs",
]
