from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import trimesh


ENGINE = "blender"


def get_flip_transform(direction: Literal["x", "y"]) -> np.ndarray:
    if direction == "x":
        return trimesh.transformations.scale_matrix(-1, [0, 0, 0], [1, 0, 0])
    if direction == "y":
        return trimesh.transformations.scale_matrix(-1, [0, 0, 0], [0, 1, 0])
    raise ValueError(f"Unsupported flip direction: {direction}")


def get_yaw_rotation_transform(deg: Literal[90, 180, 270]) -> np.ndarray:
    if deg == 90:
        return trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    if deg == 180:
        return trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    if deg == 270:
        return trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
    raise ValueError(f"Unsupported rotation degree: {deg}")


def merge_meshes(meshes: list[trimesh.Trimesh], minimal_triangles: bool = False, engine: str = "blender") -> trimesh.Trimesh:
    if not meshes:
        return trimesh.Trimesh()
    if minimal_triangles:
        try:
            merged = trimesh.boolean.union(meshes, engine=engine)
            if merged is not None:
                return merged
        except Exception:
            pass
    return trimesh.util.concatenate(meshes)


def export_mesh_obj(mesh: trimesh.Trimesh, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return output_path


def canonicalize_mesh_to_dim(mesh: trimesh.Trimesh, dim: tuple[float, float, float]) -> trimesh.Trimesh:
    canonical = mesh.copy()
    bounds = canonical.bounding_box.bounds
    extents = bounds[1] - bounds[0]
    if extents[0] > dim[0] + 1e-6 or extents[1] > dim[1] + 1e-6 or extents[2] > dim[2] + 1e-6:
        raise ValueError("Mesh extents exceed target tile dimensions")

    xy_center = (bounds[0, :2] + bounds[1, :2]) * 0.5
    z_center = (bounds[0, 2] + bounds[1, 2]) * 0.5
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [-xy_center[0], -xy_center[1], -z_center]
    canonical.apply_transform(transform)
    return canonical


def flip_mesh(mesh: trimesh.Trimesh, direction: Literal["x", "y"]) -> trimesh.Trimesh:
    new_mesh = mesh.copy()
    transform = get_flip_transform(direction)
    new_mesh.apply_transform(transform)
    return new_mesh


def yaw_rotate_mesh(mesh: trimesh.Trimesh, deg: Literal[90, 180, 270]) -> trimesh.Trimesh:
    new_mesh = mesh.copy()
    transform = get_yaw_rotation_transform(deg)
    new_mesh.apply_transform(transform)
    return new_mesh


def rotate_mesh(mesh: trimesh.Trimesh, deg: float = 0.0, axis: list[float] = [0.0, 0.0, 1.0]) -> trimesh.Trimesh:
    new_mesh = mesh.copy()
    transform = trimesh.transformations.rotation_matrix(np.pi * deg / 180.0, axis)
    new_mesh.apply_transform(transform)
    return new_mesh


def get_heights_from_mesh(mesh: trimesh.Trimesh, origins: np.ndarray) -> np.ndarray:
    if origins.shape[1] == 2:
        origins = np.concatenate([origins, np.ones((origins.shape[0], 1), dtype=origins.dtype) * 100.0], axis=1)

    vectors = np.stack(
        [np.zeros_like(origins[:, 0]), np.zeros_like(origins[:, 1]), -np.ones_like(origins[:, 2])],
        axis=-1,
    )
    points, index_ray, _ = mesh.ray.intersects_location(origins, vectors, multiple_hits=False)
    heights = np.zeros((origins.shape[0],), dtype=np.float32)
    if len(points) > 0:
        heights[index_ray] = points[:, 2]
    return heights


def get_height_array_of_mesh(
    mesh: trimesh.Trimesh,
    dim: tuple[float, float, float],
    num_points: int = 100,
    offset: float = 0.01,
) -> np.ndarray:
    array = np.zeros((num_points * num_points,), dtype=np.float32)
    if mesh.is_empty:
        return array.reshape((num_points, num_points))

    x = np.linspace(-dim[0] / 2.0 + offset, dim[0] / 2.0 - offset, num_points)
    y = np.linspace(dim[1] / 2.0 - offset, -dim[1] / 2.0 + offset, num_points)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    origins = np.stack([xv, yv, np.ones_like(xv) * dim[2] * 2], axis=-1)
    vectors = np.stack([np.zeros_like(xv), np.zeros_like(yv), -np.ones_like(xv)], axis=-1)

    points, index_ray, _ = mesh.ray.intersects_location(origins, vectors, multiple_hits=False)
    if len(points) > 0:
        array[index_ray] = points[:, 2] + dim[2] / 2.0
        array = np.round(array, 1)
    return array.reshape(num_points, num_points)


def convert_heightfield_to_trimesh(
    height_field_raw: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: float | None = None,
) -> trimesh.Trimesh:
    height_field = height_field_raw
    num_rows, num_cols = height_field.shape

    y_min = -num_cols * horizontal_scale / 2.0
    y_max = num_cols * horizontal_scale / 2.0
    x_min = -num_rows * horizontal_scale / 2.0
    x_max = num_rows * horizontal_scale / 2.0

    y = np.linspace(y_min, y_max, num_cols)
    x = np.linspace(x_min, x_max, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        threshold = slope_threshold * horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += height_field[1:num_rows, :] - height_field[: num_rows - 1, :] > threshold
        move_x[1:num_rows, :] -= height_field[: num_rows - 1, :] - height_field[1:num_rows, :] > threshold
        move_y[:, : num_cols - 1] += height_field[:, 1:num_cols] - height_field[:, : num_cols - 1] > threshold
        move_y[:, 1:num_cols] -= height_field[:, : num_cols - 1] - height_field[:, 1:num_cols] > threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            height_field[1:num_rows, 1:num_cols] - height_field[: num_rows - 1, : num_cols - 1] > threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            height_field[: num_rows - 1, : num_cols - 1] - height_field[1:num_rows, 1:num_cols] > threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = height_field.flatten() * vertical_scale

    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for row in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + row * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * row * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return trimesh.Trimesh(vertices=vertices, faces=triangles)


def merge_two_height_meshes(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> trimesh.Trimesh:
    merged_mesh = trimesh.util.concatenate(mesh1, mesh2)
    vertices = merged_mesh.vertices
    faces = merged_mesh.faces
    n_vertices = vertices.shape[0] // 2
    n = int(np.sqrt(n_vertices))
    indices = np.arange(n_vertices).reshape(n, n)

    new_faces = []
    edges = [indices[0, :], indices[-1, :], indices[:, 0], indices[:, -1]]
    for edge in edges:
        ind0 = edge[:-1]
        ind1 = edge[1:]
        ind2 = ind0 + n_vertices
        ind3 = ind1 + n_vertices

        face_a = np.zeros((n - 1, 3), dtype=np.uint32)
        face_a[:, 0] = ind0
        face_a[:, 1] = ind1
        face_a[:, 2] = ind2
        new_faces.append(face_a)

        face_b = np.zeros((n - 1, 3), dtype=np.uint32)
        face_b[:, 0] = ind2
        face_b[:, 1] = ind3
        face_b[:, 2] = ind1
        new_faces.append(face_b)

    return trimesh.Trimesh(vertices=vertices, faces=np.vstack([faces, np.vstack(new_faces)]))


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    new_mesh = mesh.copy()
    internal_faces = trimesh.intersections.mesh_plane(new_mesh, plane_normal=[1, 0, 0], plane_origin=[0.5, 0, 0])
    if len(internal_faces) > 0:
        new_mesh.remove_faces(internal_faces)
        new_mesh.remove_unreferenced_vertices()
    return new_mesh


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required only for SDF computation; install the `sdf` extra to enable it") from exc
    return o3d


def compute_sdf(mesh: trimesh.Trimesh, dim: tuple[float, float, float] = (2.0, 2.0, 2.0), resolution: float = 0.1) -> np.ndarray:
    o3d = _require_open3d()
    mesh = mesh.copy()
    rng = np.random.default_rng(0)
    mesh.vertices += rng.uniform(-1e-4, 1e-4, size=mesh.vertices.shape)

    mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh.as_open3d)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_o3d)

    dim_array = np.asarray(dim, dtype=np.float32)
    num_elements = np.ceil(dim_array / resolution).astype(int)
    xyz_range = [np.linspace(-dim_array[i] / 2, dim_array[i] / 2, num=num_elements[i]) for i in range(len(dim_array))]
    query_points = np.stack(np.meshgrid(*xyz_range), axis=-1).astype(np.float32)

    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points["points"].numpy(), axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    is_inside = scene.count_intersections(rays).numpy() % 2 == 1
    distance[is_inside] *= -1

    sdf = distance.reshape(*num_elements)
    return sdf.transpose(1, 0, 2)
