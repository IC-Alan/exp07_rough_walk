#!/usr/bin/env python3
"""
Route visualization with path-following camera (Open3D).

Generates terrain -> plans navigation route -> renders a "walk along the route" video.

Features:
- Red route tube (Open3D TriangleMesh with real lighting)
- Green start sphere + red goal sphere markers with text overlays
- Route resampling (0.05 m spacing) + XY moving average for ultra-smooth camera
- Fixed-height camera (only XY and yaw adjust; no terrain-following, no pitch change)
- Route tube radius 4 cm (clearly visible without obscuring terrain)
- Default camera height 2.5 m
- Improved planner: asymmetric height cost + morphological dilation + obstacle margin

Usage:
    # Requires Python <= 3.12 (Open3D compatible)
    /path/to/py312/bin/python examples/visualize_route.py

    # Basic usage
    python examples/visualize_route.py                           # forest, 8x8m
    python examples/visualize_route.py --mode pile --seed 123    # pile terrain
    python examples/visualize_route.py --mode wfc --wfc-shape 4 4  # WFC composite
    python examples/visualize_route.py --size 12 12              # 12x12m large map

    # Custom start/goal + camera params
    python examples/visualize_route.py --start 1 1 --goal 7 7
    python examples/visualize_route.py --speed 0.8 --camera-height 1.8
    python examples/visualize_route.py --output my_route.mp4 --fps 24 --noise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generator import (
    MeshTerrain,
    MeshTerrainCfg,
    NoiseDisplaceConfig,
    displace_mesh,
    list_registered_terrain_names,
)
from generator.presets import procedural_terrain_result, procedural_wfc_specs
from generator.utils import random_seed
from generator.visualization import (
    _mesh_to_open3d,
    _ground_plane_open3d,
    _segment_transform,
    _O3D_BG_COLOR as BG_COLOR,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Open3D import / version check
# ═══════════════════════════════════════════════════════════════════════════════

def _require_open3d():
    """Import open3d; raise a clear error if Python version is incompatible."""
    try:
        import open3d as o3d
        return o3d
    except ImportError as e:
        ver = sys.version_info
        if ver.major == 3 and ver.minor >= 13:
            raise RuntimeError(
                f"Open3D is not supported on Python {ver.major}.{ver.minor}.\n"
                f"Please use a Python <= 3.12 environment, e.g.:\n"
                f"  /home/PJLAB/wangzirui/miniconda3/envs/py312/bin/python {sys.argv[0]} ...\n"
                f"  conda run -n py312 python {sys.argv[0]} ..."
            ) from e
        raise


# ═══════════════════════════════════════════════════════════════════════════════
#  Rendering constants
# ═══════════════════════════════════════════════════════════════════════════════

# material parameters
_MAT_ROUGHNESS: float = 0.72
_MAT_REFLECTANCE: float = 0.28
_LINE_WIDTH: float = 2.5

# colors (RGB 0-1)
_START_COLOR = np.array([0.18, 0.72, 0.28])   # green
_GOAL_COLOR = np.array([0.86, 0.12, 0.12])    # red
_ROUTE_COLOR = np.array([0.85, 0.14, 0.14])   # red route

# geometry sizes
_MARKER_RADIUS: float = 0.18           # sphere marker radius
_ROUTE_TUBE_RADIUS: float = 0.07       # route tube radius (7 cm, visible from distance)
_ROUTE_TUBE_SECTIONS: int = 10         # tube cross-section segments

# camera
_FOV_DEG: float = 75.0                 # field of view
_DEFAULT_CAMERA_HEIGHT: float = 2.5    # default camera height (meters)
_ROUTE_RESAMPLE_SPACING: float = 0.05  # route resample spacing (meters, smaller = smoother)
_ROUTE_RENDER_Z_OFFSET: float = 0.5    # visual route overlay height

# video
_DEFAULT_RESOLUTION = (1600, 896)
_MIN_ENDPOINT_DISTANCE_RATIO: float = 0.35


# ═══════════════════════════════════════════════════════════════════════════════
#  Open3D geometry builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_route_tube_o3d(route: np.ndarray, radius: float = _ROUTE_TUBE_RADIUS,
                          sections: int = _ROUTE_TUBE_SECTIONS, o3d=None):
    """Build a thick route tube as an Open3D TriangleMesh from cylinder segments."""
    if o3d is None:
        o3d = _require_open3d()
    import trimesh as _tm

    if len(route) < 2:
        return None

    cylinders = []
    for start, end in zip(route[:-1], route[1:]):
        length = float(np.linalg.norm(end - start))
        if length < 1e-8:
            continue
        cyl = _tm.creation.cylinder(radius=radius, height=length, sections=sections)
        cyl.apply_transform(_segment_transform(start, end))
        cylinders.append(cyl)

    if not cylinders:
        return None

    merged = _tm.util.concatenate(cylinders)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(merged.vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(merged.faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(_ROUTE_COLOR.astype(np.float64))
    return mesh


def _build_sphere_marker_o3d(center: np.ndarray, radius: float,
                             color: np.ndarray, o3d=None):
    """Create an Open3D sphere marker."""
    if o3d is None:
        o3d = _require_open3d()
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color.astype(np.float64))
    sphere.translate(np.asarray(center, dtype=np.float64))
    return sphere


def _project_route_to_navigation_surface(
    route: np.ndarray,
    height_array: np.ndarray,
    mesh_bounds: np.ndarray,
    *,
    resolution: float,
    z_offset: float,
) -> np.ndarray:
    """Build a visual-only route that follows the navigable surface height."""
    projected = np.asarray(route, dtype=np.float32).copy()
    bounds_min = np.asarray(mesh_bounds[0], dtype=np.float32)
    for point in projected:
        col = int(np.round((float(point[0]) - float(bounds_min[0])) / resolution))
        row = int(np.round((float(point[1]) - float(bounds_min[1])) / resolution))
        row = int(np.clip(row, 0, height_array.shape[0] - 1))
        col = int(np.clip(col, 0, height_array.shape[1] - 1))
        height = float(height_array[row, col])
        if np.isfinite(height):
            point[2] = height + float(z_offset)
        else:
            point[2] = float(point[2]) + float(z_offset)
    return projected


def _planning_height_limits(terrain_state) -> tuple[float | None, float | None]:
    planning = getattr(terrain_state, "metadata", {}).get("planning", {}) if terrain_state is not None else {}
    if not isinstance(planning, dict):
        return None, None
    limits = planning.get("height_limits", {})
    if not isinstance(limits, dict):
        return None, None
    min_height = limits.get("min")
    max_height = limits.get("max")
    return (
        None if min_height is None else float(min_height),
        None if max_height is None else float(max_height),
    )


def _random_endpoint_pair(candidates: np.ndarray, rng: np.random.Generator, min_distance: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(candidates, dtype=np.float32)
    if points.shape[0] < 2:
        raise ValueError("Need at least two candidate endpoints")
    for _ in range(128):
        start_idx = int(rng.integers(0, points.shape[0]))
        distances = np.linalg.norm(points[:, :2] - points[start_idx, :2], axis=1)
        eligible = np.flatnonzero(distances >= min_distance)
        if eligible.size:
            goal_idx = int(rng.choice(eligible))
            return points[start_idx].copy(), points[goal_idx].copy()
    distances = np.linalg.norm(points[:, None, :2] - points[None, :, :2], axis=2)
    start_idx, goal_idx = np.unravel_index(np.argmax(distances), distances.shape)
    return points[int(start_idx)].copy(), points[int(goal_idx)].copy()


def _random_navigation_endpoints(
    height_array: np.ndarray,
    mesh_bounds: np.ndarray,
    *,
    resolution: float,
    rng: np.random.Generator,
    terrain_state=None,
) -> tuple[np.ndarray, np.ndarray]:
    heights = np.asarray(height_array, dtype=np.float32)
    valid = np.isfinite(heights)
    min_height, max_height = _planning_height_limits(terrain_state)
    if min_height is not None:
        valid &= heights >= min_height
    if max_height is not None:
        valid &= heights <= max_height
    rows, cols = np.nonzero(valid)
    if rows.size < 2:
        raise ValueError("No enough navigable points for random route endpoints")
    bounds_min = np.asarray(mesh_bounds[0], dtype=np.float32)
    points = np.stack(
        [
            bounds_min[0] + cols.astype(np.float32) * float(resolution),
            bounds_min[1] + rows.astype(np.float32) * float(resolution),
            heights[rows, cols],
        ],
        axis=1,
    ).astype(np.float32)
    extent_xy = np.asarray(mesh_bounds[1], dtype=np.float32)[:2] - bounds_min[:2]
    min_distance = float(np.linalg.norm(extent_xy) * _MIN_ENDPOINT_DISTANCE_RATIO)
    return _random_endpoint_pair(points, rng, min_distance)


# ═══════════════════════════════════════════════════════════════════════════════
#  Text overlay (PIL)
# ═══════════════════════════════════════════════════════════════════════════════

def _overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Overlay text lines on a rendered frame; returns RGBA uint8 array."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # try to load a system font
    font = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    ]:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, 22)
                break
            except Exception:
                continue

    x, y = 20, 24
    for line in lines:
        draw.text((x + 1, y + 1), line, fill=(0, 0, 0), font=font)  # shadow
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += 28

    return np.array(img)


# ═══════════════════════════════════════════════════════════════════════════════
#  Route resampling -> smooth camera motion
# ═══════════════════════════════════════════════════════════════════════════════

def _resample_route(route: np.ndarray, spacing: float = _ROUTE_RESAMPLE_SPACING):
    """Resample waypoints by XY arc length into fine-grained waypoints.

    Camera speed is horizontal speed. Z is interpolated only to keep metadata
    coherent; camera height itself is fixed later.
    """
    if len(route) < 2:
        return route

    diffs = np.diff(route, axis=0)
    seg_lengths = np.linalg.norm(diffs[:, :2], axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cum_lengths[-1])
    if total <= 1e-8:
        return route[:1].astype(np.float32)

    n_samples = max(int(total / spacing), 2)
    sample_dists = np.linspace(0, total, n_samples)

    indices = np.searchsorted(cum_lengths[1:], sample_dists, side="right")
    indices = np.clip(indices, 0, len(route) - 2)

    seg_t = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        idx = indices[i]
        denom = seg_lengths[idx]
        seg_t[i] = (sample_dists[i] - cum_lengths[idx]) / max(denom, 1e-8)
    seg_t = np.clip(seg_t, 0.0, 1.0)

    resampled = route[indices] + seg_t[:, None] * diffs[indices]
    return resampled.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera trajectory sampling
# ═══════════════════════════════════════════════════════════════════════════════

def sample_camera_along_route(
    route: np.ndarray,
    camera_height: float,
    look_ahead_distance: float,
    speed_ms: float,
    fps: int,
    resample_spacing: float = _ROUTE_RESAMPLE_SPACING,
) -> tuple[list[np.ndarray], list[np.ndarray], float, int]:
    """Sample camera positions and look-at targets at constant XY speed.

    The camera keeps fixed Z; speed_ms and look_ahead_distance are measured in
    the XY plane, not along terrain height changes.

    Returns:
        cam_positions:  (N,) list of (3,) float32
        cam_targets:    (N,) list of (3,) float32
        total_length:   route total length (meters)
        n_frames:       total frame count
    """
    # route resampling -> smooth basis
    fine_route = _resample_route(route, spacing=resample_spacing)

    if len(fine_route) < 2:
        raise ValueError("Route must have at least 2 waypoints")

    diffs = np.diff(fine_route, axis=0)
    seg_lengths = np.linalg.norm(diffs[:, :2], axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cum_lengths[-1])
    if total_length <= 1e-8:
        raise ValueError("Route must move in XY to sample a moving camera")

    duration = total_length / speed_ms
    n_frames = max(int(np.ceil(duration * fps)) + 1, 2)

    # Fixed camera Z: only XY and yaw adjust; Z and pitch stay constant.
    ground_z = max(0.0, float(np.max(np.asarray(route, dtype=np.float32)[:, 2])))
    fixed_cam_z = ground_z + camera_height
    fixed_look_z = max(0.2, ground_z + camera_height * 0.35)

    cam_positions = []
    cam_targets = []

    for i in range(n_frames):
        dist = min(float(i) * speed_ms / float(fps), total_length)

        idx = int(np.searchsorted(cum_lengths[1:], dist, side="right"))
        idx = min(idx, len(fine_route) - 2)

        if idx == 0 and cum_lengths[1] < 1e-6:
            pos = fine_route[0].copy()
        else:
            seg_t = float(np.clip(
                (dist - cum_lengths[idx]) / max(seg_lengths[idx], 1e-8), 0.0, 1.0))
            pos = fine_route[idx] + seg_t * diffs[idx]

        cam_pos = np.array([pos[0], pos[1], fixed_cam_z], dtype=np.float32)

        target_dist = min(dist + look_ahead_distance, total_length)
        target_idx = int(np.searchsorted(cum_lengths, target_dist))
        target_idx = min(target_idx, len(fine_route) - 1)
        look_xy = fine_route[target_idx][:2]
        look_target = np.array([look_xy[0], look_xy[1], fixed_look_z], dtype=np.float32)

        cam_positions.append(cam_pos)
        cam_targets.append(look_target)

    # apply slight moving average to camera XY to eliminate residual jitter
    if len(cam_positions) > 4:
        positions_arr = np.array(cam_positions)
        targets_arr = np.array(cam_targets)
        window = max(3, int(fps * 0.08))  # ~80 ms window
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window) / window
        half = window // 2
        for dim in [0, 1]:  # only smooth XY
            smoothed = np.convolve(positions_arr[:, dim], kernel, mode='same')
            smoothed[:half] = positions_arr[:half, dim]
            smoothed[-half:] = positions_arr[-half:, dim]
            positions_arr[:, dim] = smoothed
            t_smoothed = np.convolve(targets_arr[:, dim], kernel, mode='same')
            t_smoothed[:half] = targets_arr[:half, dim]
            t_smoothed[-half:] = targets_arr[-half:, dim]
            targets_arr[:, dim] = t_smoothed
        cam_positions = [p.astype(np.float32) for p in positions_arr]
        cam_targets = [t.astype(np.float32) for t in targets_arr]

    return cam_positions, cam_targets, total_length, n_frames


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    terrain_choices = list_registered_terrain_names()
    all_modes = [*terrain_choices, "wfc"]
    p = argparse.ArgumentParser(
        description="Route visualization (Open3D) — follow-camera along navigation route, output MP4")
    p.add_argument("--mode", choices=all_modes, default="forest",
                   help=f"terrain type: {', '.join(terrain_choices)}, or wfc (WFC composite)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed; omit to use a fresh random seed")
    p.add_argument("--size", type=float, nargs=2, default=[8.0, 8.0],
                   help="single terrain size or WFC tile size (width height) meters")
    p.add_argument("--wfc-shape", type=int, nargs=2, default=[4, 4],
                   help="WFC grid (rows cols), e.g. --wfc-shape 6 6")
    p.add_argument("--wfc-terrains", type=str, default=None,
                   help="comma-separated WFC terrain families, default auto-selects all")
    p.add_argument("--output", type=str, default="route_visualization.mp4")
    p.add_argument("--fps", type=int, default=30, help="frame rate")
    p.add_argument("--speed", type=float, default=1.0,
                   help="camera speed (m/s)")
    p.add_argument("--camera-height", type=float, default=_DEFAULT_CAMERA_HEIGHT,
                   help=f"camera height above ground (m), default {_DEFAULT_CAMERA_HEIGHT} (eye-level)")
    p.add_argument("--look-ahead", type=float, default=5.0,
                   help="camera look-ahead distance (m)")
    p.add_argument("--start", type=float, nargs=2, default=None,
                   help="start XY (e.g. --start 1.5 2.0)")
    p.add_argument("--goal", type=float, nargs=2, default=None,
                   help="goal XY (e.g. --goal 6.5 6.0)")
    p.add_argument("--strict", action="store_true",
                   help="use strict traversable route")
    p.add_argument("--noise", action="store_true",
                   help="apply Perlin noise displacement")
    p.add_argument("--resolution", type=int, nargs=2,
                   default=list(_DEFAULT_RESOLUTION),
                   help=f"output resolution (width height), default {_DEFAULT_RESOLUTION}")
    p.add_argument("--no-text", action="store_true",
                   help="do not overlay text")
    return p


def main():
    args = build_parser().parse_args()
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive")
    if args.seed is None:
        args.seed = random_seed()
        print(f"Using random seed: {args.seed}")
    o3d = _require_open3d()
    tile_size = tuple(args.size)

    # ── 1. Generate terrain ─────────────────────────────────────────────
    if args.mode == "wfc":
        _generate_wfc(args, o3d, tile_size)
    else:
        _generate_single(args, o3d, tile_size)


def _generate_single(args, o3d, tile_size):
    """Single terrain mode from the registry."""
    print(f"Generating terrain: mode={args.mode}, size={tile_size}, seed={args.seed}")
    result, _cfg = procedural_terrain_result(args.mode, size=tile_size, seed=args.seed)
    mesh = result.mesh

    if args.noise:
        noise_cfg = NoiseDisplaceConfig(
            hill_amplitude=0.4, roughness_amplitude=0.04, seed=args.seed)
        mesh = displace_mesh(mesh, noise_cfg)
        print("  + Perlin noise applied")

    _run_pipeline(args, o3d, mesh, f"Terrain: {args.mode}", terrain_state=result.terrain_state)


def _generate_wfc(args, o3d, tile_size):
    """WFC composite terrain mode."""
    from generator.wfc import run_wfc_scene_with_procedural_specs
    from generator.terrain_registry import list_wfc_terrain_names

    shape = tuple(args.wfc_shape)

    if args.wfc_terrains is None:
        terrain_list = list_wfc_terrain_names(default_only=True)
    else:
        terrain_list = [t.strip() for t in args.wfc_terrains.split(",") if t.strip()]

    print(f"Generating WFC terrain: shape={shape}, tiles={tile_size}, "
          f"terrains={terrain_list}, seed={args.seed}")

    specs = procedural_wfc_specs(terrain_list, tile_size=tile_size, seed=args.seed)
    scene, _solver = run_wfc_scene_with_procedural_specs(
        shape, specs, tile_size=tile_size, seed=args.seed)
    mesh = scene.mesh

    _setup_wfc_route_endpoints(args, scene)

    if args.noise:
        noise_cfg = NoiseDisplaceConfig(
            hill_amplitude=0.4, roughness_amplitude=0.04, seed=args.seed)
        mesh = displace_mesh(mesh, noise_cfg)
        print("  + Perlin noise applied")

    total_size = (shape[1] * tile_size[0], shape[0] * tile_size[1])
    label = f"WFC {shape[0]}x{shape[1]} ({','.join(terrain_list)})"
    print(f"  WFC total size: {total_size[0]:.0f}x{total_size[1]:.0f} m,  "
          f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    _run_pipeline(args, o3d, mesh, label, scene=scene)


def _setup_wfc_route_endpoints(args, scene):
    """Pick random WFC start/goal tile centers when not provided.

    Hierarchical routing will decide the intermediate tile sequence itself.
    """
    placements = scene.placements  # (N, 3) all tile centers
    if len(placements) < 2:
        return

    xy = placements[:, :2]
    rng = np.random.default_rng(args.seed + 17)
    extent_xy = np.ptp(xy, axis=0)
    min_distance = float(np.linalg.norm(extent_xy) * _MIN_ENDPOINT_DISTANCE_RATIO)

    if args.start is None or args.goal is None:
        if args.start is None and args.goal is None:
            start, goal = _random_endpoint_pair(placements, rng, min_distance)
            args.start = start[:2]
            args.goal = goal[:2]
        elif args.start is None:
            goal_xy = np.asarray(args.goal, dtype=np.float32)
            distances = np.linalg.norm(xy - goal_xy[None, :], axis=1)
            eligible = np.flatnonzero(distances >= min_distance)
            args.start = xy[int(rng.choice(eligible if eligible.size else np.arange(len(xy))))]
        elif args.goal is None:
            start_xy = np.asarray(args.start, dtype=np.float32)
            distances = np.linalg.norm(xy - start_xy[None, :], axis=1)
            eligible = np.flatnonzero(distances >= min_distance)
            args.goal = xy[int(rng.choice(eligible if eligible.size else np.arange(len(xy))))]
    wave = scene.wave
    tile_names = scene.tile_names
    cols = wave.shape[1]
    start_idx = _closest_point_idx(xy, args.start)
    goal_idx = _closest_point_idx(xy, args.goal)

    def _flat_to_grid(idx):
        return idx // cols, idx % cols

    syi, sxi = _flat_to_grid(start_idx)
    gyi, gxi = _flat_to_grid(goal_idx)
    ns = tile_names[int(wave[syi, sxi])]
    ng = tile_names[int(wave[gyi, gxi])]
    print(f"  WFC start: tile[{syi},{sxi}]={ns}  ({args.start[0]:.1f}, {args.start[1]:.1f})")
    print(f"  WFC goal:  tile[{gyi},{gxi}]={ng}  ({args.goal[0]:.1f}, {args.goal[1]:.1f})")


def _rasterize_tile_line(start_rc, end_rc, shape):
    """Bresenham-like grid line rasterization.

    Returns list of (row, col) tuples for all tiles on the line
    from start_rc to end_rc (inclusive).
    """
    r0, c0 = start_rc
    r1, c1 = end_rc
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1

    points = []
    if dc > dr:
        # shallow slope: step through columns
        err = dc / 2.0
        r = r0
        for c in range(c0, c1 + sc, sc):
            # clamp to grid bounds
            points.append((max(0, min(shape[0] - 1, r)), max(0, min(shape[1] - 1, c))))
            err -= dr
            if err < 0:
                r += sr
                err += dc
    else:
        # steep slope: step through rows
        err = dr / 2.0
        c = c0
        for r in range(r0, r1 + sr, sr):
            points.append((max(0, min(shape[0] - 1, r)), max(0, min(shape[1] - 1, c))))
            err -= dc
            if err < 0:
                c += sc
                err += dr
    return points


def _closest_point_idx(points: np.ndarray, target: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(points - np.array(target), axis=1)))


def _run_pipeline(args, o3d, mesh, terrain_label: str, scene=None, terrain_state=None):
    """Common pipeline: MeshTerrain -> start/goal -> route planning -> camera -> Open3D render."""

    # ── 2. MeshTerrain ───────────────────────────────────────────────────
    terrain = MeshTerrain(MeshTerrainCfg(
        mesh=mesh,
        mesh_dim=tuple(mesh.bounding_box.extents.tolist()),
        terrain_state=terrain_state,
        auto_compute_sdf=False,
        auto_compute_distance=True,
    ))

    # Use planning heights for endpoints so void bottoms do not leak into route endpoints.
    # visual top surfaces into the route query.
    from generator.nav_utils import get_navigation_height_array_of_mesh_with_resolution
    ha, _ = get_navigation_height_array_of_mesh_with_resolution(
        mesh,
        resolution=0.1,
        terrain_state=terrain_state if scene is None else scene.terrain_state,
    )
    mbounds = mesh.bounding_box.bounds

    # ── 3. Start / goal ──────────────────────────────────────────────────
    seed = int(getattr(args, "seed", random_seed()))
    rng = np.random.default_rng(seed + 31)
    if scene is None:
        try:
            start_pos, goal_pos = _random_navigation_endpoints(
                ha,
                mbounds,
                resolution=0.1,
                rng=rng,
                terrain_state=terrain_state,
            )
            start_xy = start_pos[:2]
            goal_xy = goal_pos[:2]
        except ValueError:
            bounds = mesh.bounding_box.bounds
            start_xy = np.array([bounds[0, 0] + 1.0, bounds[0, 1] + 1.0], dtype=np.float32)
            goal_xy = np.array([bounds[1, 0] - 1.0, bounds[1, 1] - 1.0], dtype=np.float32)
    elif args.start is not None and args.goal is not None:
        start_xy = np.asarray(args.start, dtype=np.float32)
        goal_xy = np.asarray(args.goal, dtype=np.float32)
    else:
        start_pos, goal_pos = _random_navigation_endpoints(
            ha,
            mbounds,
            resolution=0.1,
            rng=rng,
            terrain_state=scene.terrain_state,
        )
        start_xy = start_pos[:2]
        goal_xy = goal_pos[:2]

    if args.start is not None:
        start_xy = np.array(args.start, dtype=np.float32)
    if args.goal is not None:
        goal_xy = np.array(args.goal, dtype=np.float32)

    def _xy_to_z(xy: np.ndarray) -> float:
        col = int(np.round((xy[0] - mbounds[0, 0]) / 0.1))
        row = int(np.round((xy[1] - mbounds[0, 1]) / 0.1))
        row = int(np.clip(row, 0, ha.shape[0] - 1))
        col = int(np.clip(col, 0, ha.shape[1] - 1))
        return float(ha[row, col])

    start_pos = np.array([start_xy[0], start_xy[1], _xy_to_z(start_xy)], dtype=np.float32)
    goal_pos = np.array([goal_xy[0], goal_xy[1], _xy_to_z(goal_xy)], dtype=np.float32)

    print(f"  Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})")
    print(f"  Goal:  ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})")

    # ── 4. Route planning ───────────────────────────────────────────────
    if scene is not None:
        route = scene.plan_route(start_pos, goal_pos, height_map_resolution=0.1)
        print(f"  Route: {len(route)} waypoints  (hierarchical WFC planner)")
    else:
        route = terrain.get_route(start_pos, goal_pos, use_diagonal=True)
        print(f"  Route: {len(route)} waypoints")

    # ── 5. Camera trajectory ─────────────────────────────────────────────
    cam_positions, cam_targets, route_length, n_frames = sample_camera_along_route(
        route,
        camera_height=args.camera_height,
        look_ahead_distance=args.look_ahead,
        speed_ms=args.speed,
        fps=args.fps,
    )
    print(f"  Route length: {route_length:.2f} m")
    print(f"  Duration:    {route_length / args.speed:.1f} s  "
          f"({n_frames} frames @ {args.fps} fps)")

    # ── 6. Build Open3D scene ────────────────────────────────────────────
    rendering = o3d.visualization.rendering
    width, height = args.resolution

    renderer = rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background(np.array(BG_COLOR, dtype=np.float32))
    scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
        np.array([0.577, -0.577, -0.577], dtype=np.float32),
    )

    # ── Materials ──
    lit = rendering.MaterialRecord()
    lit.shader = "defaultLit"
    lit.base_roughness = _MAT_ROUGHNESS
    lit.base_reflectance = _MAT_REFLECTANCE

    unlit = rendering.MaterialRecord()
    unlit.shader = "defaultUnlit"

    route_mat = rendering.MaterialRecord()
    route_mat.shader = "defaultUnlit"
    route_mat.base_color = np.array([_ROUTE_COLOR[0], _ROUTE_COLOR[1], _ROUTE_COLOR[2], 1.0], dtype=np.float32)
    route_mat.emissive_color = np.array([_ROUTE_COLOR[0], _ROUTE_COLOR[1], _ROUTE_COLOR[2], 1.0], dtype=np.float32)

    # ── Add geometry ──
    # ground plane
    ground = _ground_plane_open3d(mesh.bounding_box.bounds)
    scene.add_geometry("ground", ground, lit)

    # terrain mesh
    terrain_o3d = _mesh_to_open3d(mesh)
    scene.add_geometry("terrain", terrain_o3d, lit)

    route_visible = _project_route_to_navigation_surface(
        route,
        ha,
        mbounds,
        resolution=0.1,
        z_offset=_ROUTE_RENDER_Z_OFFSET,
    )
    route_tube = _build_route_tube_o3d(route_visible, o3d=o3d)
    if route_tube is not None:
        scene.add_geometry("route", route_tube, route_mat)

    # start / goal markers
    start_marker = _build_sphere_marker_o3d(start_pos, _MARKER_RADIUS, _START_COLOR, o3d=o3d)
    goal_marker = _build_sphere_marker_o3d(goal_pos, _MARKER_RADIUS, _GOAL_COLOR, o3d=o3d)
    scene.add_geometry("start_marker", start_marker, unlit)
    scene.add_geometry("goal_marker", goal_marker, unlit)

    # ── 7. Render frame sequence ─────────────────────────────────────────
    import imageio.v2 as imageio
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_path), fps=args.fps, codec="libx264",
        format="FFMPEG", macro_block_size=1,
    )
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    print(f"  Rendering {n_frames} frames -> {args.output}  "
          f"({width}x{height}) [Open3D]")

    try:
        for i, (cam_pos, cam_target) in enumerate(zip(cam_positions, cam_targets)):
            renderer.setup_camera(_FOV_DEG, cam_target, cam_pos, up)
            frame = np.asarray(renderer.render_to_image())

            if not args.no_text:
                dist_traveled = (i / max(n_frames - 1, 1)) * route_length
                text_lines = [
                    f"Terrain: {terrain_label}  |  seed={args.seed}",
                    f"Route: {route_length:.1f} m  |  traveled: {dist_traveled:.1f} m",
                    f"Speed: {args.speed:.1f} m/s  |  frame: {i + 1}/{n_frames}",
                    f"* START  ------------------  * GOAL",
                ]
                frame = _overlay_text(frame, text_lines)

            writer.append_data(frame)

            if (i + 1) % max(n_frames // 10, 1) == 0 or i == n_frames - 1:
                pct = (i + 1) / n_frames * 100
                print(f"  [{pct:3.0f}%]  frame {i + 1}/{n_frames}")

    finally:
        writer.close()
        renderer.scene.clear_geometry()

    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
