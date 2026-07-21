#!/usr/bin/env python3
"""
Terrain visualization tool with true spatial proportions.

Usage:
    # Headless MP4 sweep (Open3D preferred)
    python examples/visualize_terrain.py --headless --output sweep.mp4

    # WFC scene sweep
    python examples/visualize_terrain.py --headless --mode wfc --shape 4 4 --output wfc_sweep.mp4

    # WFC scene with custom terrain families
    python examples/visualize_terrain.py --headless --mode wfc --shape 4 4 \
        --terrains pyramid_stairs,platform_gap,stakes,pile --output terrain_sweep.mp4

    # Interactive single-terrain view
    python examples/visualize_terrain.py --mode pile

Tweak rendering quality / lighting / camera by editing the CONSTANTS block below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── matplotlib setup ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # safe default; switched to interactive later if needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ── project imports ─────────────────────────────────────────────────────────
from generator.presets import procedural_terrain_result, procedural_wfc_specs
from generator.utils import random_seed
from generator import (
    NoiseDisplaceConfig,
    displace_mesh,
    list_registered_terrain_names,
    list_wfc_terrain_names,
    run_wfc_scene_with_procedural_specs,
)
from generator.visualization import (
    build_scene,
    camera_pose_from_bounds,
    draw_open3d,
    render_open3d_frame_sequence,
    render_scene_offscreen,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION CONSTANTS — tweak rendering quality / lighting here
# ═══════════════════════════════════════════════════════════════════════════════

# ── output quality ──
RESOLUTION_W: int = 1600
RESOLUTION_H: int = 896  # divisible by 16 for mp4 compatibility
RENDER_DPI: int = 120     # matplotlib-only
FIG_SIZE: tuple[float, float] = (16, 10)  # matplotlib figure (inches)
MAX_MESH_VERTS: int = 45000  # downsample target for matplotlib

# ── tile / grid defaults ──
TILE_SIZE: tuple[float, float] = (8.0, 8.0)  # meters (width, height)
GRID_SPACING: float = 0.5  # fine grid line interval (meters)
SHOW_GRID: bool = True
SHOW_AXES: bool = True
SHOW_MARKERS: bool = True
SHOW_GROUND: bool = True

# ── noise displacement ──
NOISE_ENABLED: bool = False  # set True for multi-scale Perlin hills+roughness
NOISE_HILL_AMPLITUDE: float = 0.5   # gentle mound height (meters)
NOISE_ROUGH_AMPLITUDE: float = 0.05  # fine surface texture (meters)

# ── camera ──
TRAJECTORY: str = "orbit"  # "orbit" | "flyover"
DEFAULT_ELEV: float = 30.0
DEFAULT_AZIM: float = -45.0

# ── renderer fallback chain ("auto" → try each in order) ──
RENDERER_AUTO_CHAIN: tuple[str, ...] = ("open3d", "matplotlib")


# ═══════════════════════════════════════════════════════════════════════════════
#  helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _downsample_mesh(vertices: np.ndarray, faces: np.ndarray):
    """Reduce mesh complexity for plotting speed while preserving shape."""
    max_verts = MAX_MESH_VERTS
    if len(vertices) <= max_verts:
        return vertices, faces
    rng = np.random.default_rng(42)
    idx = rng.choice(len(vertices), max_verts, replace=False)
    idx_set = set(idx)
    mask = np.array([all(v in idx_set for v in f) for f in faces])
    faces_sub = faces[mask]
    old_to_new = {old: new for new, old in enumerate(idx)}
    faces_sub = np.vectorize(old_to_new.get)(faces_sub)
    return vertices[idx], faces_sub


def _compute_grid_meta(placements: np.ndarray, tile_dim: tuple[float, float, float], shape: tuple[int, int]):
    """Return grid-line positions and bounds from placements + tile_dim."""
    half_x, half_y = tile_dim[0] / 2, tile_dim[1] / 2
    xs = placements[:, 0]
    ys = placements[:, 1]
    x_min, x_max = xs.min() - half_x, xs.max() + half_x
    y_min, y_max = ys.min() - half_y, ys.max() + half_y

    # tile boundary grid lines
    rows, cols = int(shape[0]), int(shape[1])
    x_lines = np.linspace(x_min + half_x, x_max - half_x, cols + 1)
    y_lines = np.linspace(y_min + half_y, y_max - half_y, rows + 1)

    # fine grid lines at GRID_SPACING intervals (0.5 m)
    fine_x_lines = np.arange(
        np.ceil(x_min / GRID_SPACING) * GRID_SPACING,
        x_max + GRID_SPACING / 2,
        GRID_SPACING,
    )
    fine_y_lines = np.arange(
        np.ceil(y_min / GRID_SPACING) * GRID_SPACING,
        y_max + GRID_SPACING / 2,
        GRID_SPACING,
    )

    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "x_lines": x_lines,
        "y_lines": y_lines,
        "fine_x_lines": fine_x_lines,
        "fine_y_lines": fine_y_lines,
        "half_x": half_x, "half_y": half_y,
        "rows": rows, "cols": cols,
    }


def _build_ground_plane(x_range: tuple[float, float], y_range: tuple[float, float], res: int = 30):
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], res),
                         np.linspace(y_range[0], y_range[1], res))
    return xx, yy, np.zeros_like(xx)


def _sphere_mesh(center: np.ndarray, radius: float = 0.15, res: int = 8):
    u = np.linspace(0, 2 * np.pi, res * 2)
    v = np.linspace(0, np.pi, res)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _terrain_facecolors(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = verts[faces]
    centers = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normal_norm = np.where(normal_norm > 1e-6, normal_norm, 1.0)
    normals = normals / normal_norm

    z = centers[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    denom = max(z_max - z_min, 1e-6)
    base = plt.get_cmap("terrain")((z - z_min) / denom)[:, :3]

    light_dir = np.array([0.55, -0.35, 0.76], dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)
    diffuse = np.clip(normals @ light_dir, 0.0, 1.0)
    ambient = 0.38
    rim = np.clip(1.0 - normals[:, 2], 0.0, 1.0) * 0.10
    intensity = np.clip(ambient + 0.72 * diffuse + rim, 0.0, 1.15)
    shaded = np.clip(base * intensity[:, None], 0.0, 1.0)
    alpha = np.full((shaded.shape[0], 1), 0.96, dtype=np.float32)
    return np.concatenate([shaded, alpha], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  core drawing (reused per frame)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_frame(
    ax: plt.Axes,
    *,
    verts: np.ndarray,
    faces: np.ndarray,
    grid_meta: dict,
    placements: np.ndarray,
    tile_names: list[str] | None,
    wave: np.ndarray | None,
    elev: float,
    azim: float,
):
    """Draw one frame onto *ax* (cleared beforehand)."""
    ax.clear()

    # ── terrain mesh ──
    facecolors = _terrain_facecolors(verts, faces)
    mesh_collection = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        shade=False,
        linewidth=0.08,
        antialiased=True,
        edgecolor=(0.04, 0.04, 0.05, 0.08),
    )
    mesh_collection.set_facecolors(facecolors)

    # ── ground plane ──
    if SHOW_GROUND:
        xx, yy, zz = _build_ground_plane((grid_meta["x_min"], grid_meta["x_max"]),
                                         (grid_meta["y_min"], grid_meta["y_max"]))
        ax.plot_surface(xx, yy, zz, color="#aab3b8", alpha=0.14, linewidth=0)

    # ── grid lines (tile boundaries) ──
    if SHOW_GRID:
        for xv in grid_meta["x_lines"]:
            ax.plot([xv, xv], [grid_meta["y_min"], grid_meta["y_max"]], [0, 0],
                    color="white", linewidth=0.6, alpha=0.40, linestyle="--")
        for yv in grid_meta["y_lines"]:
            ax.plot([grid_meta["x_min"], grid_meta["x_max"]], [yv, yv], [0, 0],
                    color="white", linewidth=0.6, alpha=0.40, linestyle="-")
        # fine grid (0.5 m)
        for xv in grid_meta["fine_x_lines"]:
            ax.plot([xv, xv], [grid_meta["y_min"], grid_meta["y_max"]], [0, 0],
                    color="white", linewidth=0.3, alpha=0.18, linestyle=":")
        for yv in grid_meta["fine_y_lines"]:
            ax.plot([grid_meta["x_min"], grid_meta["x_max"]], [yv, yv], [0, 0],
                    color="white", linewidth=0.3, alpha=0.18, linestyle=":")

    # ── world axes ──
    if SHOW_AXES:
        axis_len = max(grid_meta["x_max"] - grid_meta["x_min"],
                       grid_meta["y_max"] - grid_meta["y_min"]) * 0.35
        origin = np.array([grid_meta["x_min"], grid_meta["y_min"], 0.0])
        ax.quiver(*origin, axis_len, 0, 0, color="red", linewidth=2,
                  arrow_length_ratio=0.08)
        ax.quiver(*origin, 0, axis_len, 0, color="green", linewidth=2,
                  arrow_length_ratio=0.08)
        ax.quiver(*origin, 0, 0, axis_len * 0.6, color="blue", linewidth=2,
                  arrow_length_ratio=0.08)
        off = axis_len * 0.08
        ax.text(origin[0] + axis_len + off, origin[1], origin[2], "X",
                color="red", fontsize=11, fontweight="bold")
        ax.text(origin[0], origin[1] + axis_len + off, origin[2], "Y",
                color="green", fontsize=11, fontweight="bold")
        ax.text(origin[0], origin[1], origin[2] + axis_len * 0.6 + off, "Z",
                color="blue", fontsize=11, fontweight="bold")

    # ── origin markers ──
    if SHOW_MARKERS and len(placements) > 0 and tile_names is not None and wave is not None:
        shape_cols = grid_meta["cols"]
        for i, pos in enumerate(placements):
            gy, gx = i // shape_cols, i % shape_cols
            name = tile_names[int(wave[gy, gx])]
            sx, sy, sz = _sphere_mesh(pos, radius=0.15)
            ax.plot_surface(sx, sy, sz, color="gold", alpha=0.9, linewidth=0)
            ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2], pos[2] + 0.5],
                    color="orange", linewidth=1.2, alpha=0.7)
            ax.text(pos[0], pos[1], pos[2] + 0.7, name[:4],
                    color="gold", fontsize=6, ha="center", va="bottom",
                    fontweight="bold")

    # ── true spatial proportions ──
    x_range = grid_meta["x_max"] - grid_meta["x_min"]
    y_range = grid_meta["y_max"] - grid_meta["y_min"]
    z_data = verts[:, 2]
    z_range = max(z_data.max() - z_data.min(), x_range * 0.15)
    ax.set_box_aspect((x_range, y_range, z_range))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=elev, azim=azim)

    return ax


# ═══════════════════════════════════════════════════════════════════════════════
#  camera trajectory
# ═══════════════════════════════════════════════════════════════════════════════

def _orbit_trajectory(n_frames: int, start_azim: float = 0, total_rotation: float = 360):
    """Camera orbit: full horizontal sweep with elevation dip in the middle."""
    azim = np.linspace(start_azim, start_azim + total_rotation, n_frames)
    # Elevation: start high, dip low in middle, rise again
    t = np.linspace(0, 2 * np.pi, n_frames)
    elev = 25 + 20 * np.sin(t)  # oscillates between 5° and 45°
    return elev, azim


def _flyover_trajectory(n_frames: int):
    """Fly-over: start from one side, fly across, turn around, fly back."""
    half = n_frames // 2
    azim_p1 = np.linspace(-60, -60, half)       # hold direction
    azim_p2 = np.linspace(-60, 120, n_frames - half)  # turn
    azim = np.concatenate([azim_p1, azim_p2])

    elev_p1 = np.linspace(60, 15, half)          # descend
    elev_p2 = np.linspace(15, 50, n_frames - half)  # rise
    elev = np.concatenate([elev_p1, elev_p2])
    return elev, azim


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    terrain_choices = list_registered_terrain_names()
    default_wfc_terrains = ",".join(list_wfc_terrain_names(default_only=True))

    p = argparse.ArgumentParser(
        description="Terrain visualization — interactive or headless MP4 sweep")
    p.add_argument("--mode", choices=[*terrain_choices, "wfc"],
                   default="wfc", help="Terrain generation mode")
    p.add_argument("--shape", type=int, nargs=2, default=[4, 4],
                   help="WFC grid shape (rows cols)")
    p.add_argument("--terrains", type=str, default=default_wfc_terrains,
                   help="Comma-separated terrain family list for WFC mode")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed; omit to use a fresh random seed")
    p.add_argument("--output", type=str, default="terrain_sweep.mp4",
                   help="Output MP4 path (headless mode)")
    p.add_argument("--headless", action="store_true",
                   help="Generate MP4 instead of interactive window")
    p.add_argument("--duration", type=float, default=10.0,
                   help="Video duration in seconds")
    p.add_argument("--fps", type=int, default=30, help="Frames per second")
    p.add_argument("--renderer", choices=["auto", "matplotlib", "open3d", "pyrender"], default="auto",
                   help="Rendering backend; auto tries open3d → matplotlib")
    p.add_argument("--noise", action="store_true", default=None,
                   help="Enable multi-scale Perlin noise displacement (hills + roughness)")
    p.add_argument("--no-noise", action="store_false", dest="noise",
                   help="Disable noise displacement")
    return p


def main():
    args = build_parser().parse_args()
    if args.seed is None:
        args.seed = random_seed()
        print(f"Using random seed: {args.seed}")

    shape = tuple(args.shape)
    tile_size = TILE_SIZE

    # ── generate terrain ──
    if args.mode == "wfc":
        terrain_list = [name.strip() for name in args.terrains.split(",") if name.strip()]
        specs = procedural_wfc_specs(terrain_list, tile_size=tile_size, seed=args.seed)
        scene, _solver = run_wfc_scene_with_procedural_specs(
            shape, specs, tile_size=tile_size, seed=args.seed)
        mesh = scene.mesh
        placements = scene.placements
        tile_names = scene.tile_names
        wave = scene.wave
        tile_dim = scene.tile_dim
        title_mode = f"WFC {shape[0]}×{shape[1]} ({','.join(terrain_list)})"
    else:
        result, _cfg = procedural_terrain_result(args.mode, size=tile_size, seed=args.seed)
        mesh = result.mesh
        bbox = mesh.bounding_box.bounds
        center = np.mean(bbox, axis=0)
        center[2] = 0.0
        mesh_copy = mesh.copy()
        mesh_copy.apply_translation(-center)
        mesh = mesh_copy
        placements = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        tile_names = [args.mode]
        wave = np.array([[0]])
        tile_dim = (tile_size[0], tile_size[1], float(bbox[1, 2] - bbox[0, 2]))
        shape = (1, 1)
        title_mode = args.mode.capitalize()

    # ── apply noise displacement ──
    use_noise = args.noise if args.noise is not None else NOISE_ENABLED
    if use_noise:
        noise_cfg = NoiseDisplaceConfig(
            hill_amplitude=NOISE_HILL_AMPLITUDE,
            roughness_amplitude=NOISE_ROUGH_AMPLITUDE,
            seed=args.seed,
        )
        mesh = displace_mesh(mesh, noise_cfg)
        print(f"Noise displacement applied (hills={NOISE_HILL_AMPLITUDE}m, roughness={NOISE_ROUGH_AMPLITUDE}m)")

    # ── downsample for matplotlib fallback ──
    verts_full = mesh.vertices.copy()
    faces_full = mesh.faces.copy()
    verts, faces = _downsample_mesh(verts_full, faces_full)
    print(f"Mesh: {len(verts)} verts, {len(faces)} faces  "
          f"(from {len(verts_full)} / {len(faces_full)})")

    # ── grid metadata ──
    grid_meta = _compute_grid_meta(placements, tile_dim, shape)
    print(f"Grid: {grid_meta['cols']}×{grid_meta['rows']}  "
          f"bounds X=[{grid_meta['x_min']:.1f}, {grid_meta['x_max']:.1f}]  "
          f"Y=[{grid_meta['y_min']:.1f}, {grid_meta['y_max']:.1f}]")

    n_frames = int(args.duration * args.fps)

    if TRAJECTORY == "orbit":
        elev_seq, azim_seq = _orbit_trajectory(n_frames, total_rotation=360)
    else:
        elev_seq, azim_seq = _flyover_trajectory(n_frames)

    # ══════════════════════════════════════════════════════════════════════════
    #  renderer implementations
    # ══════════════════════════════════════════════════════════════════════════

    def _render_headless_matplotlib():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Rendering {n_frames} frames → {args.output}  "
              f"({args.duration}s @ {args.fps}fps, dpi={RENDER_DPI}) [matplotlib]")

        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot(111, projection="3d")
        draw_frame(ax, verts=verts, faces=faces, grid_meta=grid_meta,
                   placements=placements, tile_names=tile_names, wave=wave,
                   elev=elev_seq[0], azim=azim_seq[0])
        ax.set_title(f"{title_mode}  |  frame 0/{n_frames}", fontsize=13)

        def _update(frame_idx: int):
            ax.clear()
            draw_frame(ax, verts=verts, faces=faces, grid_meta=grid_meta,
                       placements=placements, tile_names=tile_names, wave=wave,
                       elev=elev_seq[frame_idx], azim=azim_seq[frame_idx])
            ax.set_title(f"{title_mode}  |  frame {frame_idx + 1}/{n_frames}", fontsize=13)
            return [ax]

        ani = animation.FuncAnimation(fig, _update, frames=n_frames,
                                      interval=1000 / args.fps, blit=False)
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=-1,
                                        codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p",
                                                    "-preset", "medium"])
        ani.save(str(output_path), writer=writer, dpi=RENDER_DPI)
        plt.close(fig)
        print(f"✅ Saved {output_path}")

    def _render_headless_open3d():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Rendering {n_frames} frames → {args.output}  "
              f"({args.duration}s @ {args.fps}fps, {RESOLUTION_W}x{RESOLUTION_H}) [open3d]")
        render_open3d_frame_sequence(
            mesh,
            output_path=str(output_path),
            width=RESOLUTION_W,
            height=RESOLUTION_H,
            fps=args.fps,
            elev_seq=elev_seq,
            azim_seq=azim_seq,
            points=placements,
            show_ground=SHOW_GROUND,
            show_axes=SHOW_AXES,
            grid_meta=grid_meta,
        )
        print(f"✅ Saved {output_path}")

    def _render_headless_pyrender():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Rendering {n_frames} frames → {args.output}  "
              f"({args.duration}s @ {args.fps}fps, {RESOLUTION_W}x{RESOLUTION_H}) [pyrender]")

        initial_pose = camera_pose_from_bounds(mesh.bounds, elev_seq[0], azim_seq[0])
        scene, nodes = build_scene(
            mesh,
            points=placements,
            camera_pose=initial_pose,
            return_nodes=True,
        )

        import pyrender
        renderer = pyrender.OffscreenRenderer(viewport_width=RESOLUTION_W, viewport_height=RESOLUTION_H)
        writer = imageio.get_writer(str(output_path), fps=args.fps, codec="libx264", format="FFMPEG")
        try:
            camera_node = nodes["camera"]
            for frame_idx in range(n_frames):
                pose = camera_pose_from_bounds(mesh.bounds, elev_seq[frame_idx], azim_seq[frame_idx])
                scene.set_pose(camera_node, pose)
                frame = render_scene_offscreen(scene, RESOLUTION_W, RESOLUTION_H, renderer=renderer, shadows=True)
                writer.append_data(frame)
        finally:
            writer.close()
            renderer.delete()
        print(f"✅ Saved {output_path}")

    def _render_interactive_matplotlib():
        matplotlib.use("TkAgg")
        plt.ion()
        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot(111, projection="3d")
        draw_frame(ax, verts=verts, faces=faces, grid_meta=grid_meta,
                   placements=placements, tile_names=tile_names, wave=wave,
                   elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
        ax.set_title(f"{title_mode}  (drag to rotate)", fontsize=13)
        plt.show(block=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  dispatch
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    #  dispatch helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _run_headless(renderer_choice: str):
        if renderer_choice == "auto":
            renderer_choice = RENDERER_AUTO_CHAIN[0]
        if renderer_choice == "open3d":
            try:
                return _render_headless_open3d()
            except Exception as exc:
                if args.renderer == "open3d":
                    raise
                print(f"open3d headless unavailable ({exc}); falling back")
                renderer_choice = RENDERER_AUTO_CHAIN[1] if len(RENDERER_AUTO_CHAIN) > 1 else "matplotlib"
        if renderer_choice == "pyrender":
            try:
                return _render_headless_pyrender()
            except Exception as exc:
                if args.renderer == "pyrender":
                    raise
                print(f"pyrender headless unavailable ({exc}); falling back to matplotlib")
        _render_headless_matplotlib()

    def _run_interactive(renderer_choice: str):
        if renderer_choice == "auto":
            renderer_choice = RENDERER_AUTO_CHAIN[0]
        if renderer_choice == "open3d":
            try:
                draw_open3d(
                    mesh,
                    points=placements,
                    show_ground=SHOW_GROUND,
                    show_axes=SHOW_AXES,
                    grid_meta=grid_meta,
                )
                return
            except Exception as exc:
                if args.renderer == "open3d":
                    raise
                print(f"open3d interactive unavailable ({exc}); falling back to matplotlib")
        _render_interactive_matplotlib()

    if args.headless:
        _run_headless(args.renderer)
    else:
        _run_interactive(args.renderer)


if __name__ == "__main__":
    main()
