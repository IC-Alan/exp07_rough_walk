from __future__ import annotations

import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def generate_perlin_noise_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    tileable: tuple[bool, bool] = (False, False),
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gradients_shape = (res[0] + 1, res[1] + 1)
    angles = rng.uniform(0, 2 * np.pi, size=gradients_shape)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]

    grid_y, grid_x = np.meshgrid(
        np.arange(shape[0]) * res[0] / shape[0],
        np.arange(shape[1]) * res[1] / shape[1],
        indexing="ij",
    )
    x0 = np.floor(grid_x).astype(int)
    y0 = np.floor(grid_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    sx = grid_x - x0
    sy = grid_y - y0

    g00 = gradients[y0, x0]
    g10 = gradients[y0, x1]
    g01 = gradients[y1, x0]
    g11 = gradients[y1, x1]

    d00 = np.stack((sx, sy), axis=-1)
    d10 = np.stack((sx - 1, sy), axis=-1)
    d01 = np.stack((sx, sy - 1), axis=-1)
    d11 = np.stack((sx - 1, sy - 1), axis=-1)

    n00 = np.sum(g00 * d00, axis=-1)
    n10 = np.sum(g10 * d10, axis=-1)
    n01 = np.sum(g01 * d01, axis=-1)
    n11 = np.sum(g11 * d11, axis=-1)

    u = _fade(sx)
    v = _fade(sy)
    nx0 = _lerp(n00, n10, u)
    nx1 = _lerp(n01, n11, u)
    return _lerp(nx0, nx1, v)


def generate_fractal_noise_2d(
    shape: tuple[int, int],
    res: tuple[int, int],
    octaves: int,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    tileable: tuple[bool, bool] = (False, False),
    seed: int | None = None,
) -> np.ndarray:
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0
    for octave in range(octaves):
        octave_res = (max(1, int(res[0] * frequency)), max(1, int(res[1] * frequency)))
        octave_seed = None if seed is None else seed + octave
        noise += amplitude * generate_perlin_noise_2d(shape, octave_res, tileable=tileable, seed=octave_seed)
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return noise / max(max_amplitude, 1e-6)


__all__ = ["generate_fractal_noise_2d", "generate_perlin_noise_2d"]