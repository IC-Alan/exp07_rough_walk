from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ..mesh_utils import merge_meshes
from ..terrain_state import TerrainPortal, TerrainState, make_box_member
from .base import ProceduralTerrainResult, centered_boundary_edge_constraints, ground_box


def _wall_segment(
    length: float,
    thickness: float,
    height: float,
    angle: float,
    center_xy: np.ndarray,
    radius: float,
) -> trimesh.Trimesh:
    segment = trimesh.creation.box(extents=[thickness, length, height])
    base = np.eye(4)
    base[:3, 3] = [0.0, 0.0, height / 2.0]
    segment.apply_transform(base)
    segment.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1.0]))
    px = center_xy[0] + radius * np.cos(angle)
    py = center_xy[1] + radius * np.sin(angle)
    segment.apply_transform(trimesh.transformations.translation_matrix([px, py, 0.0]))
    return segment


@dataclass
class DoorTerrainCfg:
    size: tuple[float, float] = (10.0, 10.0)
    ground_thickness: float = 0.1
    margin: float = 0.8
    wall_thickness: float = 0.10
    wall_height: float = 1.8
    door_width: float = 1.2
    wall_length: float = 1.8
    ring_spacing: float = 1.0
    min_radius: float = 1.4
    max_radius: float | None = None
    target_panel_arc: float = 0.40
    jamb_tangential_thickness: float = 0.10


def generate_door_terrain(cfg: DoorTerrainCfg) -> ProceduralTerrainResult:
    size_x, size_y = cfg.size
    center_xy = np.array([size_x * 0.5, size_y * 0.5], dtype=np.float32)
    origin = np.array([center_xy[0], center_xy[1], 0.3], dtype=np.float32)
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

    max_radius_allowed = min(size_x, size_y) * 0.5 - cfg.margin - cfg.wall_thickness
    max_radius = max_radius_allowed if cfg.max_radius is None else min(cfg.max_radius, max_radius_allowed)
    ring_count = 0
    door_count = 0

    radius = cfg.min_radius
    while radius <= max_radius + 1e-6:
        circumference = 2.0 * np.pi * radius
        unit_length = cfg.wall_length + cfg.door_width
        slots = max(1, int(np.floor(circumference / max(unit_length, 1e-6))))
        sep_ang = 2.0 * np.pi * cfg.wall_length / (slots * unit_length)
        door_ang = 2.0 * np.pi * cfg.door_width / (slots * unit_length)

        theta = -np.pi
        for _ in range(slots):
            th0 = theta
            th1 = theta + sep_ang
            arc_length = th1 - th0
            step = float(np.clip(cfg.target_panel_arc / max(radius, 1e-6), 0.05, 0.30))
            n_seg = max(1, int(np.ceil(arc_length / step)))
            panel_step = arc_length / n_seg

            for seg_idx in range(n_seg):
                theta_mid = th0 + (seg_idx + 0.5) * panel_step
                chord_len = float(max(0.12, 2.0 * radius * np.sin(0.5 * panel_step)))
                wall = _wall_segment(chord_len, cfg.wall_thickness, cfg.wall_height, theta_mid, center_xy, radius)
                meshes.append(wall)
                members.append(
                    make_box_member(
                        f"wall_{ring_count}_{door_count}_{seg_idx}",
                        kind="door_ring_wall",
                        role="wall",
                        bounds=wall.bounds,
                        yaw_deg=float(np.degrees(theta_mid)),
                        traversable=False,
                        params={"radius": radius, "chord_len": chord_len},
                    )
                )

            left_edge = th1
            right_edge = th1 + door_ang
            left_jamb = _wall_segment(
                cfg.jamb_tangential_thickness,
                cfg.wall_thickness,
                cfg.wall_height,
                left_edge,
                center_xy,
                radius,
            )
            right_jamb = _wall_segment(
                cfg.jamb_tangential_thickness,
                cfg.wall_thickness,
                cfg.wall_height,
                right_edge,
                center_xy,
                radius,
            )
            meshes.extend([left_jamb, right_jamb])
            members.extend(
                [
                    make_box_member(
                        f"jamb_left_{ring_count}_{door_count}",
                        kind="door_jamb",
                        role="wall",
                        bounds=left_jamb.bounds,
                        yaw_deg=float(np.degrees(left_edge)),
                        traversable=False,
                        params={"radius": radius},
                    ),
                    make_box_member(
                        f"jamb_right_{ring_count}_{door_count}",
                        kind="door_jamb",
                        role="wall",
                        bounds=right_jamb.bounds,
                        yaw_deg=float(np.degrees(right_edge)),
                        traversable=False,
                        params={"radius": radius},
                    ),
                ]
            )
            theta = right_edge
            door_count += 1

        ring_count += 1
        radius += cfg.ring_spacing

    mesh = merge_meshes(meshes, minimal_triangles=False)
    route_half_span = min(
        max(cfg.door_width * 0.65, cfg.wall_length * 0.4),
        min(size_x, size_y) * 0.22,
    )
    return ProceduralTerrainResult(
        mesh=mesh,
        terrain_mesh=mesh,
        origin=origin,
        terrain_state=TerrainState(
            members=members,
            portals=[
                TerrainPortal(
                    name="center_clearance",
                    boundary="interior",
                    center=(float(center_xy[0]), float(center_xy[1]), 0.0),
                    span=(2.0 * cfg.min_radius, 2.0 * cfg.min_radius, cfg.wall_height),
                    normal=(0.0, 0.0, 1.0),
                )
            ],
            connectivity={"ring_count": ring_count, "door_count": door_count},
            metadata={"generator": "door"},
        ),
        metadata={
            "ring_count": ring_count,
            "door_count": door_count,
            "planning": {"route_constraints": centered_boundary_edge_constraints(cfg.size, route_half_span)},
        },
    )


__all__ = ["DoorTerrainCfg", "generate_door_terrain"]