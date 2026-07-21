from __future__ import annotations

from typing import Callable

import numpy as np
import trimesh

from ..mesh_utils import ENGINE, get_height_array_of_mesh, get_heights_from_mesh, merge_meshes, rotate_mesh, yaw_rotate_mesh
from ..utils import euler_angles_to_rotation_matrix
from .mesh_parts_cfg import (
    BoxMeshPartsCfg,
    FloatingBoxesPartsCfg,
    OverhangingBoxesPartsCfg,
    OverhangingMeshPartsCfg,
    PlatformMeshPartsCfg,
    WallMeshPartsCfg,
)


_DEFAULT_RNG_SEED = 0


def create_wall(width, height, depth):
    return trimesh.creation.box([width, height, depth])


def generate_wall_from_array(cfg: WallMeshPartsCfg) -> trimesh.Trimesh:
    assert cfg.connection_array.shape == (3, 3)
    if cfg.connection_array.sum() == 0:
        return trimesh.Trimesh()

    grid_size = cfg.dim[0] / cfg.connection_array.shape[0]
    meshes = []
    for y in range(cfg.connection_array.shape[1]):
        for x in range(cfg.connection_array.shape[0]):
            if cfg.connection_array[x, y] <= 0:
                continue
            pos = np.array([x * grid_size, y * grid_size, 0], dtype=np.float32)
            pos[:2] += grid_size / 2.0 - cfg.dim[0] / 2.0
            pos[2] += cfg.wall_height / 2.0 - cfg.dim[2] / 2.0
            if np.abs(pos[0]) > 1.0e-4 and np.abs(pos[1]) < 1.0e-4:
                mesh = create_wall(grid_size, cfg.wall_thickness, cfg.wall_height)
                mesh.apply_translation(pos)
                meshes.append(mesh)
            elif np.abs(pos[0]) < 1.0e-4 and np.abs(pos[1]) > 1.0e-4:
                mesh = create_wall(cfg.wall_thickness, grid_size, cfg.wall_height)
                mesh.apply_translation(pos)
                meshes.append(mesh)
            elif np.abs(pos[0]) < 1.0e-4 and np.abs(pos[1]) < 1.0e-4:
                neighbors = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    if 0 <= x + dx < cfg.connection_array.shape[0] and 0 <= y + dy < cfg.connection_array.shape[1]:
                        if cfg.connection_array[x + dx, y + dy] > 0:
                            neighbors.append((dx, dy))
                for dx, dy in neighbors:
                    width = grid_size / 2.0
                    height = grid_size / 2.0
                    depth = cfg.wall_height
                    p = pos.copy()
                    if dx == 1:
                        width += cfg.wall_thickness / 2.0
                        height = cfg.wall_thickness
                        p[0] += -cfg.wall_thickness / 4.0 + grid_size / 4.0
                    elif dx == -1:
                        width += cfg.wall_thickness / 2.0
                        height = cfg.wall_thickness
                        p[0] -= -cfg.wall_thickness / 4.0 + grid_size / 4.0
                    elif dy == 1:
                        height += cfg.wall_thickness / 2.0
                        width = cfg.wall_thickness
                        p[1] += -cfg.wall_thickness / 4.0 + grid_size / 4.0
                    elif dy == -1:
                        height += cfg.wall_thickness / 2.0
                        width = cfg.wall_thickness
                        p[1] -= -cfg.wall_thickness / 4.0 + grid_size / 4.0
                    mesh = create_wall(width, height, depth)
                    mesh.apply_translation(p)
                    meshes.append(mesh)
            else:
                mesh = create_wall(grid_size, grid_size, cfg.wall_height)
                mesh.apply_translation(pos)
                meshes.append(mesh)
    mesh = merge_meshes(meshes, minimal_triangles=cfg.minimal_triangles, engine=ENGINE)
    return yaw_rotate_mesh(mesh, 270)


def create_overhanging_boxes(cfg: OverhangingBoxesPartsCfg, **_kwargs):
    rng = np.random.default_rng(_DEFAULT_RNG_SEED)
    if cfg.mesh is not None:
        height_array = get_height_array_of_mesh(cfg.mesh, cfg.dim, cfg.box_grid_n)
    elif cfg.height_array is not None:
        height_array = cfg.height_array
    else:
        height_array = np.zeros((cfg.box_grid_n, cfg.box_grid_n))
    array = rng.normal(cfg.gap_mean, cfg.gap_std, size=height_array.shape)
    array += height_array
    z_dim_array = np.ones_like(array) * cfg.box_height
    floating_array = array + z_dim_array
    return PlatformMeshPartsCfg(
        name="floating_boxes",
        dim=cfg.dim,
        array=floating_array,
        z_dim_array=z_dim_array,
        rotations=(90, 180, 270),
        flips=("x", "y"),
        weight=0.1,
        minimal_triangles=False,
        add_floor=False,
        use_z_dim_array=True,
    )


def create_floating_boxes(cfg: FloatingBoxesPartsCfg, **_kwargs):
    if cfg.mesh is None:
        raise ValueError("mesh must be provided")
    rng = np.random.default_rng(_DEFAULT_RNG_SEED)
    x = rng.uniform(-cfg.dim[0] / 2.0, cfg.dim[0] / 2.0, size=(cfg.n_boxes,))
    y = rng.uniform(-cfg.dim[1] / 2.0, cfg.dim[1] / 2.0, size=(cfg.n_boxes,))
    z = rng.uniform(cfg.min_height, cfg.max_height, size=(cfg.n_boxes,))
    terrain_heights = get_heights_from_mesh(cfg.mesh, np.stack([x, y], axis=1))
    z += terrain_heights
    z += cfg.dim[2] / 2.0
    positions = np.stack([x, y, z], axis=1)

    roll = rng.uniform(cfg.roll_pitch_range[0], cfg.roll_pitch_range[1], size=(cfg.n_boxes,))
    pitch = rng.uniform(cfg.roll_pitch_range[0], cfg.roll_pitch_range[1], size=(cfg.n_boxes,))
    yaw = rng.uniform(cfg.yaw_range[0], cfg.yaw_range[1], size=(cfg.n_boxes,))
    rotation_matrices = euler_angles_to_rotation_matrix(roll, pitch, yaw)
    transformations = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(cfg.n_boxes, axis=0)
    transformations[:, :3, :3] = rotation_matrices
    transformations[:, :3, 3] = positions

    box_x = rng.uniform(cfg.box_dim_min[0], cfg.box_dim_max[0], size=(cfg.n_boxes,))
    box_y = rng.uniform(cfg.box_dim_min[1], cfg.box_dim_max[1], size=(cfg.n_boxes,))
    box_z = rng.uniform(cfg.box_dim_min[2], cfg.box_dim_max[2], size=(cfg.n_boxes,))
    box_dims = np.stack([box_x, box_y, box_z], axis=1)
    return BoxMeshPartsCfg(box_dims=tuple(box_dims), transformations=tuple(transformations))


def get_cfg_gen(cfg: OverhangingMeshPartsCfg) -> Callable:
    if isinstance(cfg, OverhangingBoxesPartsCfg):
        return create_overhanging_boxes
    if isinstance(cfg, FloatingBoxesPartsCfg):
        return create_floating_boxes
    raise NotImplementedError


__all__ = ["create_floating_boxes", "create_overhanging_boxes", "generate_wall_from_array", "get_cfg_gen"]
