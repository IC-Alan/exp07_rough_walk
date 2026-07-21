from __future__ import annotations

import dataclasses

import numpy as np
import trimesh

from ..mesh_utils import (
    ENGINE,
    canonicalize_mesh_to_dim,
    convert_heightfield_to_trimesh,
    merge_meshes,
    merge_two_height_meshes,
)
from .mesh_parts_cfg import (
    BoxMeshPartsCfg,
    CapsuleMeshPartsCfg,
    HeightMapMeshPartsCfg,
    PlatformMeshPartsCfg,
    ProceduralMeshPartsCfg,
    WallPartsCfg,
)


def create_floor(cfg, **_kwargs):
    dims = [cfg.dim[0], cfg.dim[1], cfg.floor_thickness]
    pose = np.eye(4)
    pose[:3, -1] = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness / 2.0 + cfg.height_offset]
    return trimesh.creation.box(dims, pose)


def create_standard_wall(cfg: WallPartsCfg, edge: str = "bottom", **_kwargs):
    if edge == "bottom":
        dim = [cfg.dim[0], cfg.wall_thickness, cfg.wall_height]
        pos = [0, -cfg.dim[1] / 2.0 + cfg.wall_thickness / 2.0, -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0]
    elif edge == "up":
        dim = [cfg.dim[0], cfg.wall_thickness, cfg.wall_height]
        pos = [0, cfg.dim[1] / 2.0 - cfg.wall_thickness / 2.0, -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0]
    elif edge == "left":
        dim = [cfg.wall_thickness, cfg.dim[1], cfg.wall_height]
        pos = [-cfg.dim[0] / 2.0 + cfg.wall_thickness / 2.0, 0, -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0]
    elif edge == "right":
        dim = [cfg.wall_thickness, cfg.dim[1], cfg.wall_height]
        pos = [cfg.dim[0] / 2.0 - cfg.wall_thickness / 2.0, 0, -cfg.dim[2] / 2.0 + cfg.wall_height / 2.0]
    else:
        raise ValueError(f"Edge {edge} is not defined.")

    pose = np.eye(4)
    pose[:3, -1] = pos
    return trimesh.creation.box(dim, pose)


def create_door(cfg: WallPartsCfg, door_direction: str = "up", **_kwargs):
    if door_direction in {"bottom", "up"}:
        dim = [cfg.door_width, 2.0, cfg.door_height]
        pos = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0]
    elif door_direction in {"left", "right"}:
        dim = [2.0, cfg.door_width, cfg.door_height]
        pos = [0, 0, -cfg.dim[2] / 2.0 + cfg.floor_thickness + cfg.door_height / 2.0]
    else:
        return trimesh.Trimesh()

    pose = np.eye(4)
    pose[:3, -1] = pos
    return trimesh.creation.box(dim, pose)


def create_wall_mesh(cfg: WallPartsCfg, **kwargs):
    meshes = [create_floor(cfg)]
    for wall_edge in cfg.wall_edges:
        meshes.append(create_standard_wall(cfg, wall_edge, **kwargs))
    mesh = merge_meshes(meshes, cfg.minimal_triangles)
    if cfg.create_door:
        door = create_door(cfg, cfg.door_direction)
        try:
            mesh = trimesh.boolean.difference([mesh, door], engine=ENGINE)
        except Exception:
            pass
    return mesh


def create_platform_mesh(cfg: PlatformMeshPartsCfg, **kwargs):
    meshes = []
    min_h = 0.0
    if cfg.add_floor:
        meshes.append(create_floor(cfg))
        min_h = cfg.floor_thickness

    arrays = [cfg.array]
    z_dim_arrays = [cfg.z_dim_array]
    if cfg.arrays is not None:
        arrays += list(cfg.arrays)
    if cfg.z_dim_arrays is not None:
        z_dim_arrays += list(cfg.z_dim_arrays)

    for array, z_dim_array in zip(arrays, z_dim_arrays):
        dim_xy = [cfg.dim[0] / array.shape[0], cfg.dim[1] / array.shape[1]]
        for y in range(array.shape[1]):
            for x in range(array.shape[0]):
                if array[y, x] > min_h:
                    h = array[y, x]
                    dim = [dim_xy[0], dim_xy[1], h]
                    if cfg.use_z_dim_array:
                        z = z_dim_array[y, x]
                        if 0.0 < z < h:
                            dim = np.array([dim_xy[0], dim_xy[1], z_dim_array[y, x]])
                    pos = np.array([
                        x * dim[0] - cfg.dim[0] / 2.0 + dim[0] / 2.0,
                        -y * dim[1] + cfg.dim[1] / 2.0 - dim[1] / 2.0,
                        h - dim[2] / 2.0 - cfg.dim[2] / 2.0,
                    ])
                    meshes.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

    if cfg.wall is not None:
        meshes.append(create_wall_mesh(cfg.wall, **kwargs))
    mesh = merge_meshes(meshes, cfg.minimal_triangles)
    mesh.fill_holes()
    return mesh


def create_from_height_map(cfg: HeightMapMeshPartsCfg, **_kwargs):
    mesh = trimesh.Trimesh()
    height_map = cfg.height_map
    if cfg.fill_borders:
        mesh = create_floor(cfg)

    height_map = height_map.copy() - cfg.dim[2] / 2.0
    height_map_mesh = convert_heightfield_to_trimesh(
        height_map,
        cfg.horizontal_scale,
        cfg.vertical_scale,
        cfg.slope_threshold,
    )
    bottom_height_map = height_map * 0.0 + cfg.floor_thickness - cfg.dim[2] / 2.0
    bottom_mesh = convert_heightfield_to_trimesh(
        bottom_height_map,
        cfg.horizontal_scale,
        cfg.vertical_scale,
        cfg.slope_threshold,
    )
    height_map_mesh = merge_two_height_meshes(height_map_mesh, bottom_mesh)
    mesh = merge_meshes([mesh, height_map_mesh], False)
    if cfg.simplify:
        mesh = mesh.simplify_quadratic_decimation(cfg.target_num_faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


def create_box_mesh(cfg: BoxMeshPartsCfg, **_kwargs):
    meshes = [create_floor(cfg)] if cfg.add_floor else []
    for idx in range(len(cfg.box_dims)):
        transform = cfg.transformations[idx].copy()
        transform[2, 3] -= cfg.dim[2] / 2.0
        box = trimesh.creation.box(cfg.box_dims[idx], transform)
        meshes.append(box)
    return merge_meshes(meshes, cfg.minimal_triangles)


def create_capsule_mesh(cfg: CapsuleMeshPartsCfg, mesh: trimesh.Trimesh | None = None, **_kwargs):
    meshes = []
    positions = []
    for idx in range(len(cfg.radii)):
        capsule = trimesh.creation.capsule(radius=cfg.radii[idx], height=cfg.heights[idx])
        transform = cfg.transformations[idx].copy()
        transform[2, 3] -= cfg.dim[2] / 2.0
        positions.append(transform[0:3, 3])
        capsule.apply_transform(transform)
        meshes.append(capsule)

    if mesh is not None and len(meshes) > 0:
        positions = np.array(positions)
        x = positions[:, 0]
        y = positions[:, 1]
        origins = np.stack([x, y, np.ones_like(x) * cfg.dim[2] * 2], axis=-1)
        vectors = np.stack([np.zeros_like(x), np.zeros_like(y), -np.ones_like(x)], axis=-1)
        points, index_ray, _ = mesh.ray.intersects_location(origins, vectors, multiple_hits=True)
        translations = []
        for idx in index_ray:
            translations.append(cfg.dim[2] / 2.0 + points[idx, 2])
        for idx, capsule_mesh in enumerate(meshes):
            capsule_mesh.apply_translation([0, 0, translations[idx]])

    return merge_meshes(meshes, cfg.minimal_triangles)


def create_procedural_mesh(cfg: ProceduralMeshPartsCfg, **_kwargs):
    if cfg.terrain_generator is None:
        raise ValueError("terrain_generator must be provided for ProceduralMeshPartsCfg")
    if cfg.terrain_cfg is None:
        raise ValueError("terrain_cfg must be provided for ProceduralMeshPartsCfg")

    terrain_cfg = cfg.terrain_cfg
    if hasattr(terrain_cfg, "size"):
        terrain_cfg = dataclasses.replace(terrain_cfg, size=(cfg.dim[0], cfg.dim[1]))
    result = cfg.terrain_generator(terrain_cfg)
    return canonicalize_mesh_to_dim(result.mesh, cfg.dim)
