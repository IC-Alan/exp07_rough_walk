from __future__ import annotations

from typing import Any

import trimesh

from ..terrain_state import MemberRole, TerrainMember, make_box_member

Bounds6 = tuple[float, float, float, float, float, float]


def box_mesh(bounds: Bounds6) -> trimesh.Trimesh:
    x_min, x_max, y_min, y_max, z_min, z_max = (float(value) for value in bounds)
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(f"Invalid box bounds: {bounds}")
    return trimesh.creation.box(
        extents=[x_max - x_min, y_max - y_min, z_max - z_min],
        transform=trimesh.transformations.translation_matrix(
            [(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5]
        ),
    )


def add_box(
    meshes: list[trimesh.Trimesh],
    members: list[TerrainMember],
    member_id: str,
    *,
    kind: str,
    role: MemberRole,
    bounds: Bounds6,
    traversable: bool,
    boundary_contacts: tuple[str, ...] = (),
    params: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> trimesh.Trimesh:
    mesh = box_mesh(bounds)
    meshes.append(mesh)
    members.append(
        make_box_member(
            member_id,
            kind=kind,
            role=role,
            bounds=mesh.bounds,
            traversable=traversable,
            boundary_contacts=boundary_contacts,
            params=params,
            metadata=metadata,
        )
    )
    return mesh


def ring_bounds(
    center_x: float,
    center_y: float,
    outer_half: float,
    inner_half: float,
    z_min: float,
    z_max: float,
) -> list[Bounds6]:
    outer_half = float(outer_half)
    inner_half = max(0.0, min(float(inner_half), outer_half))
    x0, x1 = center_x - outer_half, center_x + outer_half
    y0, y1 = center_y - outer_half, center_y + outer_half
    ix0, ix1 = center_x - inner_half, center_x + inner_half
    iy0, iy1 = center_y - inner_half, center_y + inner_half
    bounds: list[Bounds6] = []
    if iy0 > y0:
        bounds.append((x0, x1, y0, iy0, z_min, z_max))
    if y1 > iy1:
        bounds.append((x0, x1, iy1, y1, z_min, z_max))
    if ix0 > x0 and iy1 > iy0:
        bounds.append((x0, ix0, iy0, iy1, z_min, z_max))
    if x1 > ix1 and iy1 > iy0:
        bounds.append((ix1, x1, iy0, iy1, z_min, z_max))
    return bounds


def rect_ring_bounds(
    center_x: float,
    center_y: float,
    outer_half_x: float,
    outer_half_y: float,
    inner_half_x: float,
    inner_half_y: float,
    z_min: float,
    z_max: float,
) -> list[Bounds6]:
    outer_half_x = float(outer_half_x)
    outer_half_y = float(outer_half_y)
    inner_half_x = max(0.0, min(float(inner_half_x), outer_half_x))
    inner_half_y = max(0.0, min(float(inner_half_y), outer_half_y))
    x0, x1 = center_x - outer_half_x, center_x + outer_half_x
    y0, y1 = center_y - outer_half_y, center_y + outer_half_y
    ix0, ix1 = center_x - inner_half_x, center_x + inner_half_x
    iy0, iy1 = center_y - inner_half_y, center_y + inner_half_y
    bounds: list[Bounds6] = []
    if iy0 > y0:
        bounds.append((x0, x1, y0, iy0, z_min, z_max))
    if y1 > iy1:
        bounds.append((x0, x1, iy1, y1, z_min, z_max))
    if ix0 > x0 and iy1 > iy0:
        bounds.append((x0, ix0, iy0, iy1, z_min, z_max))
    if x1 > ix1 and iy1 > iy0:
        bounds.append((ix1, x1, iy0, iy1, z_min, z_max))
    return bounds


__all__ = ["Bounds6", "add_box", "box_mesh", "rect_ring_bounds", "ring_bounds"]
