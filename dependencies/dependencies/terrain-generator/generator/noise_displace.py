"""Multi-scale Perlin noise displacement for terrain meshes.

Applies layered fractal noise as Z-axis vertex displacement.  Designed to add
gentle hills (~0.5 m) and fine surface roughness on top of any procedural
terrain while preserving tile-boundary continuity (tileable noise).

Usage::

    from generator.noise_displace import displace_mesh, NoiseDisplaceConfig

    cfg = NoiseDisplaceConfig(hill_amplitude=0.5, roughness_amplitude=0.05)
    noisy_mesh = displace_mesh(mesh, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from .mesh_parts.noise import generate_fractal_noise_2d, generate_perlin_noise_2d
from .utils import random_seed


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NoiseDisplaceConfig:
    """Multi-layer Perlin noise displacement settings.

    Two layers are composed additively:

    - **Hill layer**: low-frequency, ~0.5 m amplitude — gentle mounds.
    - **Roughness layer**: high-frequency, ~0.05 m amplitude — fine texture.
    """

    # ── hill (coarse) layer ──
    hill_enabled: bool = True
    hill_amplitude: float = 0.5          # peak-to-peak height (meters)
    hill_frequency: float = 0.5          # base frequency (features / meter)
    hill_octaves: int = 2                # fractal octaves for organic shape
    hill_persistence: float = 0.45       # how much each octave contributes
    hill_lacunarity: float = 2.3         # frequency multiplier per octave

    # ── roughness (fine) layer ──
    roughness_enabled: bool = True
    roughness_amplitude: float = 0.05    # peak-to-peak height (meters)
    roughness_frequency: float = 8.0     # base frequency (features / meter)
    roughness_octaves: int = 4           # more octaves → sharper detail
    roughness_persistence: float = 0.55
    roughness_lacunarity: float = 2.1

    # ── sampling resolution (pixels along longest axis) ──
    noise_resolution: int = 512

    # ── seed ──
    seed: int = field(default_factory=random_seed)


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_noise_shape_and_scale(
    mesh: trimesh.Trimesh,
    noise_resolution: int,
) -> tuple[tuple[int, int], float]:
    """Return (rows, cols) noise shape and the per-pixel world-unit scale."""
    bounds = mesh.bounding_box.bounds
    extent_x = float(bounds[1, 0] - bounds[0, 0])
    extent_y = float(bounds[1, 1] - bounds[0, 1])
    max_extent = max(extent_x, extent_y, 1e-3)
    pixel_scale = max_extent / noise_resolution
    cols = max(1, int(extent_x / pixel_scale))
    rows = max(1, int(extent_y / pixel_scale))
    return (rows, cols), pixel_scale


def _sample_displacement_field(
    shape: tuple[int, int],
    pixel_scale: float,
    cfg: NoiseDisplaceConfig,
) -> np.ndarray:
    """Build a 2D displacement field (rows, cols) from layered noise."""
    field = np.zeros(shape, dtype=np.float32)

    if cfg.hill_enabled:
        hill_res = max(1, int(shape[0] * pixel_scale * cfg.hill_frequency)), max(
            1, int(shape[1] * pixel_scale * cfg.hill_frequency)
        )
        hills = generate_fractal_noise_2d(
            shape,
            hill_res,
            octaves=cfg.hill_octaves,
            persistence=cfg.hill_persistence,
            lacunarity=cfg.hill_lacunarity,
            tileable=(True, True),
            seed=cfg.seed,
        )
        # Map from [-1,1] to [-hill_amplitude/2, +hill_amplitude/2]
        field += hills * (cfg.hill_amplitude / 2.0)

    if cfg.roughness_enabled:
        rough_res = max(1, int(shape[0] * pixel_scale * cfg.roughness_frequency)), max(
            1, int(shape[1] * pixel_scale * cfg.roughness_frequency)
        )
        roughness = generate_fractal_noise_2d(
            shape,
            rough_res,
            octaves=cfg.roughness_octaves,
            persistence=cfg.roughness_persistence,
            lacunarity=cfg.roughness_lacunarity,
            tileable=(True, True),
            seed=None if cfg.seed == 0 else cfg.seed + 1000,
        )
        field += roughness * (cfg.roughness_amplitude / 2.0)

    return field


def displace_mesh(
    mesh: trimesh.Trimesh,
    cfg: NoiseDisplaceConfig | None = None,
) -> trimesh.Trimesh:
    """Return a copy of *mesh* with multi-scale Perlin noise displacing Z.

    Parameters
    ----------
    mesh:
        Input terrain mesh.  Vertices are displaced along the world Z axis
        based on their (x, y) position.
    cfg:
        Noise configuration.  Uses defaults (gentle hills + fine roughness)
        when ``None``.

    Returns
    -------
    trimesh.Trimesh
        New mesh with displaced vertices (input is unchanged).
    """
    if cfg is None:
        cfg = NoiseDisplaceConfig()

    bounds = mesh.bounding_box.bounds
    x_min, y_min = float(bounds[0, 0]), float(bounds[0, 1])

    # ── build displacement field ──
    noise_shape, pixel_scale = _compute_noise_shape_and_scale(mesh, cfg.noise_resolution)
    field = _sample_displacement_field(noise_shape, pixel_scale, cfg)

    # ── sample field at each vertex ──
    verts = mesh.vertices.copy()
    vx = verts[:, 0]
    vy = verts[:, 1]

    col_idx = np.clip(((vx - x_min) / pixel_scale).astype(int), 0, noise_shape[1] - 1)
    row_idx = np.clip(((vy - y_min) / pixel_scale).astype(int), 0, noise_shape[0] - 1)
    dz = field[row_idx, col_idx]

    # Displace along Z (world up)
    verts[:, 2] += dz

    displaced = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
    # Copy visual attributes if present
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        displaced.visual.vertex_colors = mesh.visual.vertex_colors.copy()
    return displaced


__all__ = ["NoiseDisplaceConfig", "displace_mesh"]
