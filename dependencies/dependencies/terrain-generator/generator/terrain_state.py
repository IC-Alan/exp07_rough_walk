from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

import numpy as np


MemberRole = Literal["ground", "wall", "obstacle", "portal", "ceiling", "platform", "support", "void"]


@dataclass
class TerrainPortal:
    name: str
    boundary: Literal["up", "down", "left", "right", "interior", "none"]
    center: tuple[float, float, float]
    span: tuple[float, float, float]
    normal: tuple[float, float, float]
    traversable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerrainMember:
    member_id: str
    kind: str
    role: MemberRole
    center: tuple[float, float, float]
    extents: tuple[float, float, float]
    yaw_deg: float = 0.0
    traversable: bool = False
    boundary_contacts: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerrainState:
    schema_version: int = 1
    members: list[TerrainMember] = field(default_factory=list)
    portals: list[TerrainPortal] = field(default_factory=list)
    connectivity: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TerrainState":
        if not data:
            return cls()
        members = [TerrainMember(**member) for member in data.get("members", [])]
        portals = [TerrainPortal(**portal) for portal in data.get("portals", [])]
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            members=members,
            portals=portals,
            connectivity=dict(data.get("connectivity", {})),
            metadata=dict(data.get("metadata", {})),
        )


_BOUNDARY_VECTORS: dict[str, np.ndarray] = {
    "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "down": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
}


def ensure_terrain_state(value: TerrainState | dict[str, Any] | None) -> TerrainState:
    if value is None:
        return TerrainState()
    if isinstance(value, TerrainState):
        return value
    if isinstance(value, dict):
        return TerrainState.from_dict(value)
    raise TypeError(f"Unsupported terrain_state type: {type(value)!r}")


def terrain_state_to_serializable_dict(value: TerrainState | dict[str, Any] | None) -> dict[str, Any]:
    state = ensure_terrain_state(value)
    return state.to_dict()


def _as_transform_matrix(transformation: np.ndarray) -> np.ndarray:
    transform = np.asarray(transformation, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform matrix, got {transform.shape}")
    return transform


def _transform_point(point: tuple[float, float, float], transformation: np.ndarray) -> tuple[float, float, float]:
    point_h = np.ones(4, dtype=np.float32)
    point_h[:3] = np.asarray(point, dtype=np.float32)
    transformed = transformation @ point_h
    return tuple(transformed[:3].astype(np.float32).tolist())


def _transform_vector(vector: tuple[float, float, float], transformation: np.ndarray) -> tuple[float, float, float]:
    transformed = transformation[:3, :3] @ np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(transformed)
    if norm > 1e-6:
        transformed = transformed / norm
    return tuple(transformed.astype(np.float32).tolist())


def _transform_boundary_contact(contact: str, transformation: np.ndarray) -> str:
    if contact not in _BOUNDARY_VECTORS:
        return contact
    transformed = transformation[:3, :3] @ _BOUNDARY_VECTORS[contact]
    axis = int(np.argmax(np.abs(transformed[:2])))
    if axis == 0:
        return "right" if transformed[0] >= 0.0 else "left"
    return "up" if transformed[1] >= 0.0 else "down"


def _transform_yaw_deg(yaw_deg: float, transformation: np.ndarray) -> float:
    yaw_rad = np.deg2rad(float(yaw_deg))
    heading = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=np.float32)
    transformed = transformation[:3, :3] @ heading
    if np.linalg.norm(transformed[:2]) <= 1e-6:
        return float(yaw_deg)
    return float(np.rad2deg(np.arctan2(transformed[1], transformed[0])))


def _transform_planning_records(
    records: list[dict[str, Any]],
    transformation: np.ndarray,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    transformed_records: list[dict[str, Any]] = []
    for record in records:
        new_record = dict(record)
        if mode == "edge_points" and "point" in record:
            new_record["point"] = _transform_point(tuple(record["point"]), transformation)
        elif mode == "edge":
            if "start" in record:
                new_record["start"] = _transform_point(tuple(record["start"]), transformation)
            if "end" in record:
                new_record["end"] = _transform_point(tuple(record["end"]), transformation)
        if "cell" in new_record:
            del new_record["cell"]
        transformed_records.append(new_record)
    return transformed_records


def _transform_planning_metadata(planning: dict[str, Any], transformation: np.ndarray) -> dict[str, Any]:
    transformed_planning = dict(planning)

    anchors = planning.get("boundary_anchors")
    if isinstance(anchors, dict):
        transformed_anchors: dict[str, list[dict[str, Any]]] = {}
        for side, records in anchors.items():
            transformed_side = _transform_boundary_contact(side, transformation)
            transformed_anchors.setdefault(transformed_side, []).extend(
                _transform_planning_records(list(records), transformation, mode="edge_points")
            )
        transformed_planning["boundary_anchors"] = transformed_anchors

    route_constraints = planning.get("route_constraints")
    if isinstance(route_constraints, dict):
        transformed_constraints = dict(route_constraints)
        mode = str(route_constraints.get("mode", "edge_points"))
        boundaries = route_constraints.get("boundaries", {})
        transformed_boundaries: dict[str, list[dict[str, Any]]] = {}
        if isinstance(boundaries, dict):
            for side, records in boundaries.items():
                transformed_side = _transform_boundary_contact(side, transformation)
                normalized_records = records if isinstance(records, list) else [records]
                transformed_boundaries.setdefault(transformed_side, []).extend(
                    _transform_planning_records(list(normalized_records), transformation, mode=mode)
                )
        transformed_constraints["boundaries"] = transformed_boundaries
        transformed_planning["route_constraints"] = transformed_constraints

    route_graph = planning.get("route_graph")
    if isinstance(route_graph, dict):
        transformed_graph = dict(route_graph)
        nodes = route_graph.get("nodes", [])
        transformed_nodes: list[dict[str, Any]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            transformed_node = dict(node)
            if "point" in node:
                transformed_node["point"] = _transform_point(tuple(node["point"]), transformation)
            transformed_nodes.append(transformed_node)
        transformed_graph["nodes"] = transformed_nodes

        boundaries = route_graph.get("boundaries", {})
        transformed_boundaries: dict[str, list[str]] = {}
        if isinstance(boundaries, dict):
            for side, node_ids in boundaries.items():
                transformed_side = _transform_boundary_contact(side, transformation)
                normalized_ids = [str(node_id) for node_id in node_ids]
                transformed_boundaries.setdefault(transformed_side, []).extend(normalized_ids)
        transformed_graph["boundaries"] = transformed_boundaries
        transformed_planning["route_graph"] = transformed_graph

    height_limits = planning.get("height_limits")
    if isinstance(height_limits, dict):
        transformed_limits = dict(height_limits)
        transformed_values: list[float] = []
        for key in ("min", "max"):
            value = height_limits.get(key)
            if value is None:
                continue
            transformed_z = _transform_point((0.0, 0.0, float(value)), transformation)[2]
            transformed_limits[key] = float(transformed_z)
            transformed_values.append(float(transformed_z))
        if len(transformed_values) == 2 and transformed_limits["min"] > transformed_limits["max"]:
            transformed_limits["min"], transformed_limits["max"] = transformed_limits["max"], transformed_limits["min"]
        transformed_planning["height_limits"] = transformed_limits

    return transformed_planning


def transform_terrain_state(
    value: TerrainState | dict[str, Any] | None,
    transformation: np.ndarray,
    *,
    prefix: str | None = None,
    instance_metadata: dict[str, Any] | None = None,
) -> TerrainState:
    state = ensure_terrain_state(value)
    transform = _as_transform_matrix(transformation)

    members = []
    for member in state.members:
        member_metadata = dict(member.metadata)
        if instance_metadata is not None:
            member_metadata["instance"] = dict(instance_metadata)
        member_id = f"{prefix}/{member.member_id}" if prefix else member.member_id
        members.append(
            TerrainMember(
                member_id=member_id,
                kind=member.kind,
                role=member.role,
                center=_transform_point(member.center, transform),
                extents=member.extents,
                yaw_deg=_transform_yaw_deg(member.yaw_deg, transform),
                traversable=member.traversable,
                boundary_contacts=tuple(_transform_boundary_contact(contact, transform) for contact in member.boundary_contacts),
                params=dict(member.params),
                metadata=member_metadata,
            )
        )

    portals = []
    for portal in state.portals:
        portal_metadata = dict(portal.metadata)
        if instance_metadata is not None:
            portal_metadata["instance"] = dict(instance_metadata)
        portal_name = f"{prefix}/{portal.name}" if prefix else portal.name
        portals.append(
            TerrainPortal(
                name=portal_name,
                boundary=_transform_boundary_contact(portal.boundary, transform),
                center=_transform_point(portal.center, transform),
                span=portal.span,
                normal=_transform_vector(portal.normal, transform),
                traversable=portal.traversable,
                metadata=portal_metadata,
            )
        )

    metadata = dict(state.metadata)
    planning = metadata.get("planning")
    if isinstance(planning, dict):
        metadata["planning"] = _transform_planning_metadata(planning, transform)
    if prefix is not None:
        metadata["instance_prefix"] = prefix
    return TerrainState(
        schema_version=state.schema_version,
        members=members,
        portals=portals,
        connectivity=dict(state.connectivity),
        metadata=metadata,
    )


def merge_terrain_states(states: list[TerrainState], metadata: dict[str, Any] | None = None) -> TerrainState:
    if len(states) == 0:
        return TerrainState(metadata={} if metadata is None else dict(metadata))
    schema_version = max(state.schema_version for state in states)
    members: list[TerrainMember] = []
    portals: list[TerrainPortal] = []
    merged_metadata = {"source_state_count": len(states)}
    if metadata is not None:
        merged_metadata.update(metadata)
    return TerrainState(
        schema_version=schema_version,
        members=[member for state in states for member in state.members],
        portals=[portal for state in states for portal in state.portals],
        connectivity={},
        metadata=merged_metadata,
    )


def _box_center_extents(bounds: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center = tuple(np.mean(bounds, axis=0).astype(np.float32).tolist())
    extents = tuple((bounds[1] - bounds[0]).astype(np.float32).tolist())
    return center, extents


def make_box_member(
    member_id: str,
    kind: str,
    role: MemberRole,
    bounds: np.ndarray,
    *,
    yaw_deg: float = 0.0,
    traversable: bool = False,
    boundary_contacts: tuple[str, ...] = (),
    params: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TerrainMember:
    center, extents = _box_center_extents(np.asarray(bounds, dtype=np.float32))
    return TerrainMember(
        member_id=member_id,
        kind=kind,
        role=role,
        center=center,
        extents=extents,
        yaw_deg=float(yaw_deg),
        traversable=traversable,
        boundary_contacts=tuple(boundary_contacts),
        params={} if params is None else dict(params),
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "TerrainMember",
    "TerrainPortal",
    "TerrainState",
    "ensure_terrain_state",
    "merge_terrain_states",
    "make_box_member",
    "transform_terrain_state",
    "terrain_state_to_serializable_dict",
]
