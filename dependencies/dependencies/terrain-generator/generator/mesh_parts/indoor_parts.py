from __future__ import annotations

import numpy as np
import trimesh

from ..mesh_utils import merge_meshes, yaw_rotate_mesh
from .basic_parts import create_floor, create_standard_wall
from .mesh_parts_cfg import StairMeshPartsCfg


def create_standard_stairs(cfg: StairMeshPartsCfg.Stair):
    n_steps = cfg.n_steps
    step_height = cfg.total_height / n_steps
    step_depth = cfg.step_depth
    residual_depth = cfg.dim[1] - (n_steps + 1) * step_depth
    mesh = trimesh.Trimesh()
    stair_start_pos = np.array([0.0, -cfg.dim[1] / 2.0, -cfg.dim[2] / 2.0])
    current_pos = stair_start_pos

    if cfg.add_residual_side_up is False:
        dims = np.array([cfg.step_width, residual_depth, cfg.height_offset])
        current_pos += np.array([0.0, dims[1], 0.0])
        if cfg.height_offset != 0.0:
            pos = current_pos + np.array([0.0, dims[1] / 2.0, dims[2] / 2.0])
            step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
            mesh = merge_meshes([mesh, step], cfg.minimal_triangles)

    for step_idx in range(n_steps + 1):
        if step_idx == 0:
            dims = [cfg.step_width, cfg.step_depth, cfg.height_offset if cfg.height_offset > 0 else cfg.floor_thickness]
        else:
            dims = [cfg.step_width, cfg.step_depth, step_idx * step_height + cfg.height_offset]
        pos = current_pos + np.array([0.0, dims[1] / 2.0, dims[2] / 2.0])
        step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
        current_pos += np.array([0.0, dims[1], 0.0])
        mesh = merge_meshes([mesh, step], cfg.minimal_triangles)

    if cfg.add_residual_side_up is True:
        dims = np.array([cfg.step_width, residual_depth, n_steps * step_height + cfg.height_offset])
        pos = current_pos + np.array([0.0, dims[1] / 2.0, dims[2] / 2.0])
        step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
        mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    return mesh


def create_stairs(cfg: StairMeshPartsCfg.Stair):
    if cfg.stair_type != "standard":
        raise NotImplementedError(f"Unsupported stair type: {cfg.stair_type}")

    mesh = create_standard_stairs(cfg)
    dim = np.array([cfg.step_width, cfg.dim[1], cfg.total_height])
    if cfg.direction == "front":
        pass
    elif cfg.direction == "left":
        mesh = yaw_rotate_mesh(mesh, 90)
        dim = dim[np.array([1, 0, 2])]
    elif cfg.direction == "back":
        mesh = yaw_rotate_mesh(mesh, 180)
    elif cfg.direction == "right":
        mesh = yaw_rotate_mesh(mesh, 270)
        dim = dim[np.array([1, 0, 2])]

    if "left" in cfg.attach_side:
        mesh.apply_translation([-cfg.dim[0] / 2.0 + dim[0] / 2.0, 0, 0])
    if "right" in cfg.attach_side:
        mesh.apply_translation([cfg.dim[0] / 2.0 - dim[0] / 2.0, 0, 0])
    if "front" in cfg.attach_side:
        mesh.apply_translation([0, cfg.dim[1] / 2.0 - dim[1] / 2.0, 0])
    if "back" in cfg.attach_side:
        mesh.apply_translation([0, -cfg.dim[1] / 2.0 + dim[1] / 2.0, 0])
    return mesh


def create_stairs_mesh(cfg: StairMeshPartsCfg):
    mesh = create_floor(cfg)
    for stair in cfg.stairs:
        mesh = merge_meshes([mesh, create_stairs(stair)], cfg.minimal_triangles)
    if cfg.wall is not None:
        for wall_edge in cfg.wall.wall_edges:
            mesh = merge_meshes([mesh, create_standard_wall(cfg.wall, wall_edge)], cfg.minimal_triangles)
    return mesh


__all__ = ["create_stairs", "create_stairs_mesh", "create_standard_stairs"]