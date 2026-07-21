from __future__ import annotations

import heapq
from typing import Any, Iterable, Sequence

import numpy as np


def route_graph_node(node_id: str, point: Sequence[float], kind: str) -> dict[str, Any]:
    xyz = np.asarray(point, dtype=np.float32).reshape(3)
    return {
        "id": str(node_id),
        "point": tuple(xyz.tolist()),
        "kind": str(kind),
    }


def route_graph_edge(a: str, b: str, cost: float | None = None) -> dict[str, Any]:
    edge: dict[str, Any] = {"a": str(a), "b": str(b)}
    if cost is not None:
        edge["cost"] = float(cost)
    return edge


def _dedupe_boundary_ids(node_ids: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for node_id in node_ids:
        key = str(node_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def build_route_graph(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
    boundaries: dict[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    normalized_nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for node in nodes:
        node_id = str(node["id"])
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        normalized_nodes.append(
            route_graph_node(
                node_id=node_id,
                point=node["point"],
                kind=str(node.get("kind", "waypoint")),
            )
        )

    node_ids = {node["id"] for node in normalized_nodes}
    best_edges: dict[tuple[str, str], float | None] = {}
    for edge in edges:
        a = str(edge["a"])
        b = str(edge["b"])
        if a == b or a not in node_ids or b not in node_ids:
            continue
        key = tuple(sorted((a, b)))
        cost = None if edge.get("cost") is None else float(edge["cost"])
        if key not in best_edges:
            best_edges[key] = cost
            continue
        existing = best_edges[key]
        if existing is None:
            if cost is not None:
                best_edges[key] = cost
            continue
        if cost is not None and cost < existing:
            best_edges[key] = cost

    normalized_edges = [route_graph_edge(a, b, cost) for (a, b), cost in best_edges.items()]

    normalized_boundaries: dict[str, list[str]] = {}
    if boundaries is not None:
        for side, node_list in boundaries.items():
            normalized_ids = [node_id for node_id in _dedupe_boundary_ids(node_list) if node_id in node_ids]
            if normalized_ids:
                normalized_boundaries[str(side)] = normalized_ids

    return {
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "boundaries": normalized_boundaries,
    }


def route_graph_nodes_by_id(route_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in route_graph.get("nodes", []):
        node_id = str(node["id"])
        nodes_by_id[node_id] = {
            "id": node_id,
            "point": np.asarray(node["point"], dtype=np.float32),
            "kind": str(node.get("kind", "waypoint")),
        }
    return nodes_by_id


def nearest_route_graph_node(
    route_graph: dict[str, Any],
    point: Sequence[float],
    *,
    allowed_ids: Sequence[str] | None = None,
) -> tuple[str, float]:
    nodes_by_id = route_graph_nodes_by_id(route_graph)
    candidate_ids = list(nodes_by_id.keys()) if allowed_ids is None else [str(node_id) for node_id in allowed_ids if str(node_id) in nodes_by_id]
    if not candidate_ids:
        raise ValueError("Route graph has no eligible nodes for nearest-node lookup")
    query = np.asarray(point, dtype=np.float32).reshape(3)
    best_id = min(candidate_ids, key=lambda node_id: float(np.linalg.norm(nodes_by_id[node_id]["point"] - query)))
    best_distance = float(np.linalg.norm(nodes_by_id[best_id]["point"] - query))
    return best_id, best_distance


def route_graph_shortest_path(route_graph: dict[str, Any], start_id: str, goal_id: str) -> list[str]:
    nodes_by_id = route_graph_nodes_by_id(route_graph)
    start = str(start_id)
    goal = str(goal_id)
    if start not in nodes_by_id or goal not in nodes_by_id:
        raise ValueError(f"Route graph path endpoints must exist: {start!r} -> {goal!r}")
    if start == goal:
        return [start]

    adjacency: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in nodes_by_id}
    for edge in route_graph.get("edges", []):
        a = str(edge["a"])
        b = str(edge["b"])
        if a not in nodes_by_id or b not in nodes_by_id:
            continue
        cost = edge.get("cost")
        if cost is None:
            cost_value = float(np.linalg.norm(nodes_by_id[a]["point"] - nodes_by_id[b]["point"]))
        else:
            cost_value = float(cost)
        adjacency[a].append((b, cost_value))
        adjacency[b].append((a, cost_value))

    goal_point = nodes_by_id[goal]["point"]

    def heuristic(node_id: str) -> float:
        return float(np.linalg.norm(nodes_by_id[node_id]["point"] - goal_point))

    open_heap: list[tuple[float, float, str]] = [(heuristic(start), 0.0, start)]
    came_from: dict[str, str] = {}
    g_score: dict[str, float] = {start: 0.0}
    closed: set[str] = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)
        for neighbor, edge_cost in adjacency.get(current, ()):
            if neighbor in closed:
                continue
            tentative = current_cost + float(edge_cost)
            if tentative >= g_score.get(neighbor, np.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(neighbor), tentative, neighbor))

    raise ValueError(f"No route-graph path between {start!r} and {goal!r}")


def build_support_route_graph(
    support_points: np.ndarray,
    *,
    boundary_points: dict[str, Sequence[Sequence[float]]],
    support_kind: str,
    max_step_distance: float,
    max_height_delta: float,
    platform_bounds: tuple[float, float, float, float, float] | None = None,
) -> dict[str, Any]:
    support_array = np.asarray(support_points, dtype=np.float32).reshape((-1, 3))
    nodes = [
        route_graph_node(f"support_{index}", point, support_kind)
        for index, point in enumerate(support_array)
    ]

    if platform_bounds is not None:
        x_min, x_max, y_min, y_max, top_z = (float(value) for value in platform_bounds)
        nodes.append(route_graph_node("platform_center", ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, top_z), "platform"))

        seen_platform_contacts: set[tuple[float, float, float]] = set()
        for index, support in enumerate(support_array):
            closest_xy = np.array(
                [
                    np.clip(float(support[0]), x_min, x_max),
                    np.clip(float(support[1]), y_min, y_max),
                ],
                dtype=np.float32,
            )
            if float(np.linalg.norm(closest_xy - support[:2])) > float(max_step_distance):
                continue
            contact_point = np.array([closest_xy[0], closest_xy[1], top_z], dtype=np.float32)
            key = tuple(np.round(contact_point, 5).astype(float).tolist())
            if key in seen_platform_contacts:
                continue
            seen_platform_contacts.add(key)
            nodes.append(route_graph_node(f"platform_contact_{index}", contact_point, "platform"))

    nodes_by_id = route_graph_nodes_by_id({"nodes": nodes})
    node_ids = list(nodes_by_id.keys())
    edges: list[dict[str, Any]] = []
    for index, node_id in enumerate(node_ids):
        point_a = nodes_by_id[node_id]["point"]
        kind_a = nodes_by_id[node_id]["kind"]
        for other_id in node_ids[index + 1 :]:
            point_b = nodes_by_id[other_id]["point"]
            kind_b = nodes_by_id[other_id]["kind"]
            horizontal_distance = float(np.linalg.norm(point_a[:2] - point_b[:2]))
            height_delta = abs(float(point_a[2] - point_b[2]))

            if kind_a == "platform" and kind_b == "platform":
                edges.append(route_graph_edge(node_id, other_id, float(np.linalg.norm(point_a - point_b))))
                continue
            if horizontal_distance > float(max_step_distance) or height_delta > float(max_height_delta):
                continue
            edges.append(route_graph_edge(node_id, other_id, float(np.linalg.norm(point_a - point_b))))

    support_node_ids = [node_id for node_id in node_ids if nodes_by_id[node_id]["kind"] == support_kind]
    boundaries: dict[str, list[str]] = {}
    for side, points in boundary_points.items():
        mapped_ids: list[str] = []
        for point in points:
            node_id, _ = nearest_route_graph_node(
                {"nodes": [nodes_by_id[candidate] for candidate in support_node_ids], "edges": []},
                point,
            )
            mapped_ids.append(node_id)
        if mapped_ids:
            boundaries[side] = _dedupe_boundary_ids(mapped_ids)

    return build_route_graph(nodes, edges, boundaries)


__all__ = [
    "build_route_graph",
    "build_support_route_graph",
    "nearest_route_graph_node",
    "route_graph_edge",
    "route_graph_node",
    "route_graph_nodes_by_id",
    "route_graph_shortest_path",
]
