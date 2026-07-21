from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh

from ..mesh_terrain import MeshTerrain, MeshTerrainCfg
from ..mesh_utils import export_mesh_obj, merge_meshes
from ..nav_utils import compute_boundary_anchors
from ..route_graph import build_route_graph, nearest_route_graph_node, route_graph_nodes_by_id, route_graph_shortest_path
from ..terrain_state import TerrainState, ensure_terrain_state, merge_terrain_states, transform_terrain_state
from ..visualization import visualize_mesh
from .procedural_tiles import ProceduralWFCTileSpec, build_wfc_solver_from_tiles, create_procedural_wfc_tiles, expand_tiles_for_wfc
from .tiles import MeshTile


_FALLBACK_ANCHOR_PENALTY_SCALE = 0.35


def _route_length(route: np.ndarray) -> float:
    if route.shape[0] <= 1:
        return 0.0
    return float(np.linalg.norm(np.diff(route[:, :3], axis=0), axis=1).sum())


def _concat_route_segments(segments: Sequence[np.ndarray]) -> np.ndarray:
    merged: np.ndarray | None = None
    for segment in segments:
        if segment.shape[0] == 0:
            continue
        if merged is None:
            merged = segment.copy()
            continue
        if np.allclose(merged[-1], segment[0], atol=1e-5):
            merged = np.vstack([merged, segment[1:]])
        else:
            merged = np.vstack([merged, segment])
    if merged is None:
        return np.zeros((0, 3), dtype=np.float32)
    return merged.astype(np.float32)


def _connect_boundary_points(start_point: np.ndarray, goal_point: np.ndarray, resolution: float) -> np.ndarray:
    del resolution
    start = np.asarray(start_point, dtype=np.float32)
    goal = np.asarray(goal_point, dtype=np.float32)
    if np.allclose(start, goal, atol=1e-5):
        return start.reshape(1, 3).astype(np.float32)
    return np.stack([start, goal], axis=0).astype(np.float32)


def _translate_route_graph(route_graph: dict[str, Any], offset: np.ndarray) -> dict[str, Any]:
    if not isinstance(route_graph, dict):
        return {"nodes": [], "edges": [], "boundaries": {}}
    offset_vec = np.asarray(offset, dtype=np.float32)
    translated_nodes: list[dict[str, Any]] = []
    for node in route_graph.get("nodes", []):
        translated_node = dict(node)
        translated_node["point"] = tuple((np.asarray(node["point"], dtype=np.float32) + offset_vec).tolist())
        translated_nodes.append(translated_node)
    return build_route_graph(
        translated_nodes,
        route_graph.get("edges", []),
        route_graph.get("boundaries", {}),
    )


def _sample_constraint_edge_points(start: np.ndarray, end: np.ndarray, resolution: float) -> list[np.ndarray]:
    start_point = np.asarray(start, dtype=np.float32)
    end_point = np.asarray(end, dtype=np.float32)
    length = float(np.linalg.norm(end_point[:2] - start_point[:2]))
    if length <= 1e-6:
        return [start_point.astype(np.float32)]
    sample_spacing = max(0.5, resolution * 4.0)
    sample_count = max(3, int(np.ceil(length / sample_spacing)) + 1)
    return [point.astype(np.float32) for point in np.linspace(start_point, end_point, num=sample_count, dtype=np.float32)]


def _constraint_points_for_side(
    route_constraints: dict[str, Any],
    side: str,
    *,
    resolution: float,
) -> list[np.ndarray]:
    mode = str(route_constraints.get("mode", "edge_points"))
    boundaries = route_constraints.get("boundaries", {})
    if not isinstance(boundaries, dict):
        return []
    records = boundaries.get(side, [])
    if isinstance(records, dict):
        records = [records]

    points: list[np.ndarray] = []
    if mode == "edge_points":
        for record in records:
            if "point" not in record:
                continue
            points.append(np.asarray(record["point"], dtype=np.float32))
        return points

    if mode == "edge":
        for record in records:
            if "start" not in record or "end" not in record:
                continue
            points.extend(_sample_constraint_edge_points(record["start"], record["end"], resolution))
    return points


def _dedupe_anchor_records(records: list[dict[str, Any]], *, atol: float = 1e-4) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for record in records:
        point = np.asarray(record["point"], dtype=np.float32)
        if any(np.linalg.norm(point[:2] - np.asarray(existing["point"], dtype=np.float32)[:2]) <= atol for existing in deduped):
            continue
        new_record = {"point": point.astype(np.float32), "penalty": float(record.get("penalty", 0.0))}
        if "node_id" in record:
            new_record["node_id"] = str(record["node_id"])
        deduped.append(new_record)
    return deduped


def _preferred_anchor_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = [record for record in records if float(record.get("penalty", 0.0)) <= 1e-6]
    return preferred if preferred else records


def _validate_common_tile_dim(tiles: Sequence[MeshTile]) -> tuple[float, float, float]:
    if len(tiles) == 0:
        raise ValueError("At least one tile is required")
    reference = np.asarray(tiles[0].mesh_dim, dtype=np.float32)
    for tile in tiles[1:]:
        current = np.asarray(tile.mesh_dim, dtype=np.float32)
        if not np.allclose(current, reference):
            raise ValueError("All tiles must share the same mesh_dim to build a WFC scene")
    return tuple(reference.tolist())


@dataclass
class WFCSceneResult:
    wave: np.ndarray
    tile_names: list[str]
    mesh: trimesh.Trimesh
    tile_dim: tuple[float, float, float]
    placements: np.ndarray
    tile_meshes: dict[str, trimesh.Trimesh] = field(default_factory=dict)
    tile_terrain_states: dict[str, TerrainState] = field(default_factory=dict)
    tile_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    terrain_state: TerrainState = field(default_factory=TerrainState)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save_mesh(self, output_dir: str | Path, mesh_name: str = "mesh.obj") -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return export_mesh_obj(self.mesh, output_dir / mesh_name)

    def to_mesh_terrain(self, origin: tuple[float, float, float] = (0.0, 0.0, 0.0), **kwargs: Any) -> MeshTerrain:
        cfg = MeshTerrainCfg(
            mesh=self.mesh,
            mesh_dim=tuple(self.mesh.bounding_box.extents.tolist()),
            origin=origin,
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

    def _tile_key(self, row: int, col: int) -> str:
        return f"tile_{row}_{col}"

    def _tile_offset(self, row: int, col: int) -> np.ndarray:
        shift = np.asarray(self.metadata.get("scene_center_shift", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        return np.array([col * self.tile_dim[0], -row * self.tile_dim[1], 0.0], dtype=np.float32) - shift

    def _grid_from_world_xy(self, point_xy: np.ndarray) -> tuple[int, int]:
        shift = np.asarray(self.metadata.get("scene_center_shift", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        scene_xy = np.asarray(point_xy, dtype=np.float32) + shift[:2]
        col = int(np.clip(np.floor((scene_xy[0] + self.tile_dim[0] * 0.5) / self.tile_dim[0]), 0, self.wave.shape[1] - 1))
        row = int(np.clip(np.floor((-scene_xy[1] + self.tile_dim[1] * 0.5) / self.tile_dim[1]), 0, self.wave.shape[0] - 1))
        return row, col

    def _tile_has_foothold_graph(self, row: int, col: int) -> bool:
        tile_name = self.tile_names[int(self.wave[row, col])]
        state = self.tile_terrain_states.get(tile_name)
        if state is None:
            return False
        planning = state.metadata.get("planning", {})
        route_graph = planning.get("route_graph", {}) if isinstance(planning, dict) else {}
        if isinstance(route_graph, dict) and route_graph.get("nodes") and route_graph.get("edges"):
            return True
        graph_cfg = planning.get("foothold_graph", {}) if isinstance(planning, dict) else {}
        if isinstance(graph_cfg, dict) and bool(graph_cfg.get("enabled", False)):
            return True
        if state.metadata.get("generator") == "pile":
            return True
        kinds = {member.kind for member in state.members}
        return "pillar" in kinds and "platform" in kinds

    def _tile_route_graph(self, row: int, col: int) -> dict[str, Any] | None:
        tile_name = self.tile_names[int(self.wave[row, col])]
        state = self.tile_terrain_states.get(tile_name)
        if state is None:
            return None
        planning = state.metadata.get("planning", {})
        if not isinstance(planning, dict):
            return None
        route_graph = planning.get("route_graph")
        if not isinstance(route_graph, dict):
            return None
        if not route_graph.get("nodes") or not route_graph.get("edges"):
            return None
        return route_graph

    def _tile_world_route_graph(self, row: int, col: int) -> dict[str, Any] | None:
        route_graph = self._tile_route_graph(row, col)
        if route_graph is None:
            return None
        return _translate_route_graph(route_graph, self._tile_offset(row, col))

    def _get_tile_terrain(self, row: int, col: int, *, height_map_resolution: float = 0.1) -> MeshTerrain:
        tile_name = self.tile_names[int(self.wave[row, col])]
        return MeshTerrain(
            MeshTerrainCfg(
                mesh=self.tile_meshes[tile_name].copy(),
                terrain_state=self.tile_terrain_states.get(tile_name),
                auto_compute_sdf=False,
                auto_compute_distance=False,
                distance_matrix=np.zeros((1, 1), dtype=np.float32),
                distance_shape=(1, 1),
                height_map_resolution=height_map_resolution,
            )
        )

    def _fallback_anchor_penalty(self) -> float:
        return float(max(self.tile_dim[0], self.tile_dim[1]) * _FALLBACK_ANCHOR_PENALTY_SCALE)

    def _tile_anchor_candidate_records(
        self,
        row: int,
        col: int,
        side: str,
        *,
        height_map_resolution: float = 0.1,
    ) -> list[dict[str, Any]]:
        tile_name = self.tile_names[int(self.wave[row, col])]
        terrain_state = self.tile_terrain_states.get(tile_name)
        planning = dict(getattr(terrain_state, "metadata", {}).get("planning", {}))
        offset = self._tile_offset(row, col)

        route_graph = self._tile_route_graph(row, col)
        if route_graph is not None:
            nodes_by_id = route_graph_nodes_by_id(route_graph)
            boundary_ids = route_graph.get("boundaries", {}).get(side, [])
            records = []
            for node_id in boundary_ids:
                node = nodes_by_id.get(str(node_id))
                if node is None:
                    continue
                records.append(
                    {
                        "point": (node["point"] + offset).astype(np.float32),
                        "penalty": 0.0,
                        "node_id": str(node_id),
                    }
                )
            return _dedupe_anchor_records(records)

        records: list[dict[str, Any]] = []
        route_constraints = planning.get("route_constraints")
        if isinstance(route_constraints, dict):
            constrained_points = _constraint_points_for_side(
                route_constraints,
                side,
                resolution=height_map_resolution,
            )
            records.extend(
                {"point": (point + offset).astype(np.float32), "penalty": 0.0}
                for point in constrained_points
            )

        anchor_planning = planning
        if not anchor_planning:
            anchor_planning = dict(self.tile_metadata.get(tile_name, {})).get("planning", {})
        anchors = anchor_planning.get("boundary_anchors")
        if anchors is None:
            anchors = compute_boundary_anchors(
                self.tile_meshes[tile_name],
                height_map_resolution=height_map_resolution,
                terrain_state=terrain_state,
            )

        fallback_penalty = self._fallback_anchor_penalty() if records else 0.0
        records.extend(
            {
                "point": (np.asarray(record["point"], dtype=np.float32) + offset).astype(np.float32),
                "penalty": fallback_penalty,
            }
            for record in anchors.get(side, [])
        )
        return _dedupe_anchor_records(records)

    def _tile_anchor_candidates(
        self,
        row: int,
        col: int,
        side: str,
        *,
        height_map_resolution: float = 0.1,
    ) -> list[np.ndarray]:
        records = self._tile_anchor_candidate_records(
            row,
            col,
            side,
            height_map_resolution=height_map_resolution,
        )
        active_records = _preferred_anchor_records(records)
        return [np.asarray(record["point"], dtype=np.float32) for record in active_records]

    def _tile_anchor_pairs(
        self,
        current_rc: tuple[int, int],
        next_rc: tuple[int, int],
        *,
        height_map_resolution: float = 0.1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        row, col = current_rc
        next_row, next_col = next_rc
        row_delta = next_row - row
        col_delta = next_col - col
        if abs(row_delta) + abs(col_delta) != 1:
            raise ValueError("High-level route must move only in 4-neighborhood between tiles")

        if row_delta == 1:
            current_side, next_side = "down", "up"
        elif row_delta == -1:
            current_side, next_side = "up", "down"
        elif col_delta == 1:
            current_side, next_side = "right", "left"
        else:
            current_side, next_side = "left", "right"

        return (
            _preferred_anchor_records(
                self._tile_anchor_candidate_records(row, col, current_side, height_map_resolution=height_map_resolution)
            ),
            _preferred_anchor_records(
                self._tile_anchor_candidate_records(next_row, next_col, next_side, height_map_resolution=height_map_resolution)
            ),
        )

    def _tile_path_between_points(
        self,
        row: int,
        col: int,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        *,
        height_map_resolution: float = 0.1,
        use_diagonal: bool = True,
        start_node_id: str | None = None,
        goal_node_id: str | None = None,
    ) -> np.ndarray:
        route_graph = self._tile_route_graph(row, col)
        terrain = self._get_tile_terrain(row, col, height_map_resolution=height_map_resolution)
        offset = self._tile_offset(row, col)
        local_start = np.asarray(start_point, dtype=np.float32) - offset
        local_goal = np.asarray(goal_point, dtype=np.float32) - offset
        if route_graph is None:
            local_route = terrain.get_route(local_start, local_goal, use_diagonal=use_diagonal)
            world_route = local_route.copy()
            world_route[:, :3] += offset
            return world_route.astype(np.float32)

        nodes_by_id = route_graph_nodes_by_id(route_graph)
        if start_node_id is None:
            start_node_id, _ = nearest_route_graph_node(route_graph, local_start)
        elif start_node_id not in nodes_by_id:
            raise ValueError(f"Unknown route-graph start node: {start_node_id}")
        if goal_node_id is None:
            goal_node_id, _ = nearest_route_graph_node(route_graph, local_goal)
        elif goal_node_id not in nodes_by_id:
            raise ValueError(f"Unknown route-graph goal node: {goal_node_id}")

        path_node_ids = route_graph_shortest_path(route_graph, start_node_id, goal_node_id)
        route_points: list[np.ndarray] = [local_start.astype(np.float32)]
        snapped_start = nodes_by_id[start_node_id]["point"].astype(np.float32)
        if not np.allclose(route_points[-1], snapped_start, atol=1e-5):
            route_points.append(snapped_start)
        for node_id in path_node_ids[1:]:
            point = nodes_by_id[node_id]["point"].astype(np.float32)
            if np.allclose(route_points[-1], point, atol=1e-5):
                continue
            route_points.append(point)
        snapped_goal = nodes_by_id[goal_node_id]["point"].astype(np.float32)
        if not np.allclose(route_points[-1], snapped_goal, atol=1e-5):
            route_points.append(snapped_goal)
        if not np.allclose(route_points[-1], local_goal, atol=1e-5):
            route_points.append(local_goal.astype(np.float32))
        world_route = np.stack(route_points, axis=0).astype(np.float32)
        world_route[:, :3] += offset
        return world_route

    def _repair_foothold_route_segments(
        self,
        route: np.ndarray,
        *,
        height_map_resolution: float = 0.1,
    ) -> np.ndarray:
        if route.shape[0] < 2:
            return route.astype(np.float32)

        segments: list[np.ndarray] = []
        index = 0
        while index < route.shape[0]:
            row, col = self._grid_from_world_xy(route[index, :2])
            if not self._tile_has_foothold_graph(row, col):
                segments.append(route[index : index + 1])
                index += 1
                continue

            end = index + 1
            while end < route.shape[0] and self._grid_from_world_xy(route[end, :2]) == (row, col):
                end += 1

            start_point = route[index]
            goal_point = route[end - 1]
            if end - index >= 2:
                segments.append(
                    self._tile_path_between_points(
                        row,
                        col,
                        start_point,
                        goal_point,
                        height_map_resolution=height_map_resolution,
                        use_diagonal=True,
                    )
                )
            else:
                segments.append(route[index:end])
            index = end

        return _concat_route_segments(segments)

    def _solve_anchor_sequence(
        self,
        tile_path: Sequence[tuple[int, int]],
        start_point: np.ndarray,
        goal_point: np.ndarray,
        *,
        height_map_resolution: float = 0.1,
    ) -> list[dict[str, Any]]:
        if len(tile_path) < 2:
            return []

        layers: list[dict[tuple[int, int], dict[str, Any]]] = [
            {(-1, -1): {"cost": 0.0, "point": np.asarray(start_point, dtype=np.float32), "prev": None}}
        ]

        for edge_index in range(len(tile_path) - 1):
            current_rc = tile_path[edge_index]
            next_rc = tile_path[edge_index + 1]
            current_anchors, next_anchors = self._tile_anchor_pairs(
                current_rc,
                next_rc,
                height_map_resolution=height_map_resolution,
            )
            if not current_anchors or not next_anchors:
                raise ValueError(f"Missing boundary anchors for tile transition {current_rc} -> {next_rc}")

            next_layer: dict[tuple[int, int], dict[str, Any]] = {}
            for state_key, state in layers[-1].items():
                entry_point = state["point"]
                entry_node_id = state.get("node_id")
                for current_anchor_index, current_anchor_record in enumerate(current_anchors):
                    current_anchor = np.asarray(current_anchor_record["point"], dtype=np.float32)
                    try:
                        local_route = self._tile_path_between_points(
                            current_rc[0],
                            current_rc[1],
                            entry_point,
                            current_anchor,
                            height_map_resolution=height_map_resolution,
                            use_diagonal=True,
                            start_node_id=entry_node_id,
                            goal_node_id=current_anchor_record.get("node_id"),
                        )
                    except Exception:
                        continue

                    local_cost = _route_length(local_route) + float(current_anchor_record.get("penalty", 0.0))
                    for next_anchor_index, next_anchor_record in enumerate(next_anchors):
                        next_anchor = np.asarray(next_anchor_record["point"], dtype=np.float32)
                        bridge_route = _connect_boundary_points(current_anchor, next_anchor, height_map_resolution)
                        total_cost = (
                            state["cost"]
                            + local_cost
                            + _route_length(bridge_route)
                            + float(next_anchor_record.get("penalty", 0.0))
                        )
                        key = (edge_index + 1, next_anchor_index)
                        existing = next_layer.get(key)
                        if existing is not None and total_cost >= existing["cost"]:
                            continue
                        next_layer[key] = {
                            "cost": total_cost,
                            "point": next_anchor,
                            "node_id": next_anchor_record.get("node_id"),
                            "prev": state_key,
                            "prev_layer": edge_index,
                            "edge_index": edge_index,
                            "current_rc": current_rc,
                            "next_rc": next_rc,
                            "local_route": local_route,
                            "current_anchor": current_anchor,
                            "bridge_route": bridge_route,
                        }

            if not next_layer:
                raise ValueError(f"No feasible local transition for tile {current_rc} -> {next_rc}")
            layers.append(next_layer)

        final_tile = tile_path[-1]
        best_state_key = None
        best_state = None
        best_cost = None
        best_final_route = None
        for state_key, state in layers[-1].items():
            try:
                final_route = self._tile_path_between_points(
                    final_tile[0],
                    final_tile[1],
                    state["point"],
                    np.asarray(goal_point, dtype=np.float32),
                    height_map_resolution=height_map_resolution,
                    use_diagonal=True,
                    start_node_id=state.get("node_id"),
                )
            except Exception:
                continue

            total_cost = state["cost"] + _route_length(final_route)
            if best_cost is None or total_cost < best_cost:
                best_cost = total_cost
                best_state_key = state_key
                best_state = state
                best_final_route = final_route

        if best_state_key is None or best_state is None or best_final_route is None:
            raise ValueError(f"No feasible final tile route for {final_tile}")

        ordered_states: list[dict[str, Any]] = []
        current_key = best_state_key
        layer_index = len(layers) - 1
        while layer_index > 0:
            state = layers[layer_index][current_key]
            ordered_states.append(state)
            current_key = state["prev"]
            layer_index -= 1
        ordered_states.reverse()
        ordered_states.append({"final_route": best_final_route})
        return ordered_states

    def _high_level_tile_path(self, start_rc: tuple[int, int], goal_rc: tuple[int, int]) -> list[tuple[int, int]]:
        for path in self._iter_high_level_tile_paths(start_rc, goal_rc, max_paths=1):
            return path
        raise ValueError(f"No high-level tile path between {start_rc} and {goal_rc}")

    def _iter_high_level_tile_paths(
        self,
        start_rc: tuple[int, int],
        goal_rc: tuple[int, int],
        *,
        max_paths: int = 32,
    ):
        if start_rc == goal_rc:
            yield [start_rc]
            return

        def heuristic(cell: tuple[int, int]) -> float:
            return float(abs(cell[0] - goal_rc[0]) + abs(cell[1] - goal_rc[1]))

        max_nodes = int(self.wave.shape[0] * self.wave.shape[1])
        open_heap: list[tuple[float, float, tuple[int, int], tuple[tuple[int, int], ...]]] = []
        heapq.heappush(open_heap, (heuristic(start_rc), 0.0, start_rc, (start_rc,)))
        yielded = 0

        while open_heap:
            _, current_cost, current, path = heapq.heappop(open_heap)
            if current == goal_rc:
                yield list(path)
                yielded += 1
                if yielded >= max_paths:
                    return
                continue
            if len(path) >= max_nodes:
                continue

            row, col = current
            for next_row, next_col in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
                if next_row < 0 or next_row >= self.wave.shape[0] or next_col < 0 or next_col >= self.wave.shape[1]:
                    continue
                neighbor = (next_row, next_col)
                if neighbor in path:
                    continue
                tentative = current_cost + 1.0
                heapq.heappush(open_heap, (tentative + heuristic(neighbor), tentative, neighbor, (*path, neighbor)))

    def plan_route(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        *,
        height_map_resolution: float = 0.1,
    ) -> np.ndarray:
        start = np.asarray(start_pos, dtype=np.float32)
        goal = np.asarray(goal_pos, dtype=np.float32)
        start_rc = self._grid_from_world_xy(start[:2])
        goal_rc = self._grid_from_world_xy(goal[:2])

        if start_rc == goal_rc:
            route = self._tile_path_between_points(
                start_rc[0],
                start_rc[1],
                start,
                goal,
                height_map_resolution=height_map_resolution,
                use_diagonal=True,
            )
            return self._repair_foothold_route_segments(route, height_map_resolution=height_map_resolution)

        last_error: Exception | None = None
        state_sequence = None
        for tile_path in self._iter_high_level_tile_paths(start_rc, goal_rc):
            try:
                state_sequence = self._solve_anchor_sequence(
                    tile_path,
                    start,
                    goal,
                    height_map_resolution=height_map_resolution,
                )
                break
            except Exception as exc:
                last_error = exc
        if state_sequence is None:
            raise ValueError(f"No feasible WFC tile route between {start_rc} and {goal_rc}") from last_error

        segments: list[np.ndarray] = []
        for state in state_sequence[:-1]:
            segments.append(state["local_route"])
            segments.append(state["bridge_route"])
        segments.append(state_sequence[-1]["final_route"])

        merged = _concat_route_segments(segments)
        return self._repair_foothold_route_segments(merged, height_map_resolution=height_map_resolution)


def compose_wave_mesh(
    wave: np.ndarray,
    tile_names: Sequence[str],
    tiles: Sequence[MeshTile],
    center_scene: bool = True,
) -> WFCSceneResult:
    if wave.ndim != 2:
        raise ValueError("Only 2D wave composition is currently supported")

    tile_dim = _validate_common_tile_dim(tiles)
    tiles_by_name = {tile.name: tile for tile in tiles}
    tile_meshes = {tile.name: tile.get_mesh().copy() for tile in tiles}
    tile_terrain_states = {tile.name: ensure_terrain_state(tile.terrain_state) for tile in tiles}
    tile_metadata = {tile.name: dict(getattr(tile, "metadata", {})) for tile in tiles}

    meshes: list[trimesh.Trimesh] = []
    placements: list[np.ndarray] = []
    state_instances: list[TerrainState] = []
    tile_records: list[dict[str, Any]] = []

    for y in range(wave.shape[0]):
        for x in range(wave.shape[1]):
            tile_name = tile_names[int(wave[y, x])]
            tile = tiles_by_name[tile_name]
            mesh = tile.get_mesh().copy()
            offset = np.array([x * tile_dim[0], -y * tile_dim[1], 0.0], dtype=np.float32)
            mesh.apply_translation(offset)
            transform = np.eye(4, dtype=np.float32)
            transform[:3, 3] = offset
            meshes.append(mesh)
            placements.append(offset)
            state_instances.append(
                transform_terrain_state(
                    tile.terrain_state,
                    transform,
                    prefix=f"tile_{y}_{x}",
                    instance_metadata={"tile_name": tile_name, "grid": (y, x)},
                )
            )
            tile_records.append({"grid": (y, x), "tile_name": tile_name})

    scene_mesh = merge_meshes(meshes, minimal_triangles=False)
    placements_array = np.stack(placements, axis=0) if placements else np.zeros((0, 3), dtype=np.float32)
    scene_state = merge_terrain_states(state_instances, metadata={"tile_count": len(tile_records)})

    if center_scene and len(scene_mesh.vertices) > 0:
        bbox = scene_mesh.bounding_box.bounds
        center = np.mean(bbox, axis=0)
        center[2] = 0.0
        scene_mesh.apply_translation(-center)
        placements_array = placements_array - center
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = -center
        scene_state = transform_terrain_state(scene_state, transform)
    else:
        center = np.zeros(3, dtype=np.float32)

    return WFCSceneResult(
        wave=wave,
        tile_names=list(tile_names),
        mesh=scene_mesh,
        tile_dim=tile_dim,
        placements=placements_array,
        tile_meshes=tile_meshes,
        tile_terrain_states=tile_terrain_states,
        tile_metadata=tile_metadata,
        terrain_state=scene_state,
        metadata={"tile_records": tile_records, "scene_center_shift": center},
    )


def run_wfc_scene_with_tiles(
    shape: Sequence[int],
    tiles: Sequence[MeshTile],
    dimensions: int = 2,
    seed: int | None = None,
    observation_mode: str = "weighted",
    init_tiles: Sequence[tuple[str, tuple[int, ...]]] = (),
    max_steps: int = 1000,
    center_scene: bool = True,
) -> tuple[WFCSceneResult, Any]:
    expanded_tiles = expand_tiles_for_wfc(tiles)
    solver = build_wfc_solver_from_tiles(shape, expanded_tiles, dimensions=dimensions, seed=seed, observation_mode=observation_mode)
    wave = solver.run(list(init_tiles), max_steps=max_steps)
    scene = compose_wave_mesh(wave, solver.names, expanded_tiles, center_scene=center_scene)
    return scene, solver


def run_wfc_scene_with_procedural_specs(
    shape: Sequence[int],
    specs: Sequence[ProceduralWFCTileSpec],
    tile_size: tuple[float, float] | None = None,
    tile_height: float | None = None,
    dimensions: int = 2,
    seed: int | None = None,
    observation_mode: str = "weighted",
    init_tiles: Sequence[tuple[str, tuple[int, ...]]] = (),
    max_steps: int = 1000,
    center_scene: bool = True,
) -> tuple[WFCSceneResult, Any]:
    tiles = create_procedural_wfc_tiles(specs, tile_size=tile_size, tile_height=tile_height)
    return run_wfc_scene_with_tiles(
        shape=shape,
        tiles=tiles,
        dimensions=dimensions,
        seed=seed,
        observation_mode=observation_mode,
        init_tiles=init_tiles,
        max_steps=max_steps,
        center_scene=center_scene,
    )


__all__ = [
    "WFCSceneResult",
    "compose_wave_mesh",
    "run_wfc_scene_with_procedural_specs",
    "run_wfc_scene_with_tiles",
]
