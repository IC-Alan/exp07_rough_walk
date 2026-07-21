from __future__ import annotations

from typing import Tuple

import trimesh

from ..mesh_utils import convert_heightfield_to_trimesh
from .noise import generate_fractal_noise_2d, generate_perlin_noise_2d
from .tree import add_trees_on_terrain


def generate_perlin_terrain(
    base_shape: Tuple[int, int] = (256, 256),
    base_res: Tuple[int, int] = (2, 2),
    base_octaves: int = 2,
    base_fractal_weight: float = 0.2,
    noise_res: Tuple[int, int] = (4, 4),
    noise_octaves: int = 5,
    base_scale: float = 2.0,
    noise_scale: float = 1.0,
    horizontal_scale: float = 1.0,
    vertical_scale: float = 10.0,
    seed: int | None = None,
):
    base = generate_perlin_noise_2d(base_shape, base_res, tileable=(True, True), seed=seed)
    base += generate_fractal_noise_2d(
        base_shape,
        base_res,
        base_octaves,
        tileable=(True, True),
        seed=None if seed is None else seed + 17,
    ) * base_fractal_weight
    noise = generate_fractal_noise_2d(
        base_shape,
        noise_res,
        noise_octaves,
        tileable=(True, True),
        seed=None if seed is None else seed + 31,
    )
    terrain_height = base * base_scale + noise * noise_scale
    terrain_mesh = convert_heightfield_to_trimesh(terrain_height, horizontal_scale, vertical_scale)
    terrain_mesh.vertices[:, 2] = trimesh.smoothing.filter_humphrey(terrain_mesh).vertices[:, 2]
    return terrain_mesh


__all__ = ["add_trees_on_terrain", "generate_perlin_terrain"]