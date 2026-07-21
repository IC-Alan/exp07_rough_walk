from __future__ import annotations

from typing import Iterable

import numpy as np
import trimesh


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDERING CONSTANTS — tweak lighting / material / camera here
# ═══════════════════════════════════════════════════════════════════════════════

# ── camera defaults ──
_CAM_ELEV_DEG: float = 30.0
_CAM_AZIM_DEG: float = -45.0
_CAM_DISTANCE_SCALE: float = 2.4

# ── Open3D material ──
_O3D_BASE_ROUGHNESS: float = 0.72
_O3D_BASE_REFLECTANCE: float = 0.28
_O3D_LINE_WIDTH: float = 2.5
_O3D_BG_COLOR: tuple[float, float, float, float] = (0.93, 0.95, 0.98, 1.0)

# ── pyrender studio lights: (direction, intensity) ──
_PYRENDER_LIGHT_SPECS: tuple[tuple[tuple[float, ...], float], ...] = (
    ((1.2, -1.0, 1.8), 4.2),
    ((-1.0, 1.1, 1.3), 1.8),
    ((-1.6, -1.2, 1.6), 1.0),
)
_PYRENDER_LIGHT_RADIUS_SCALE: float = 2.8

# ── marker / ground / polyline sizes ──
_MARKER_SPHERE_RADIUS: float = 0.12
_MARKER_COLOR: tuple[float, float, float] = (0.86, 0.12, 0.12)
_POLYLINE_RADIUS: float = 0.03
_ROUTE_RADIUS: float = 0.025
_ROUTE_COLOR: tuple[float, float, float] = (0.08, 0.55, 0.94)
_GROUND_COLOR: tuple[float, float, float] = (0.80, 0.84, 0.86)


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for Open3D visualization") from exc
    return o3d


def _require_pyrender():
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError("pyrender is required for interactive visualization") from exc
    return pyrender


def _require_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for scalar color mapping") from exc
    return plt


def _scalar_to_rgb(values: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if values.ndim == 2 and values.shape[1] == 3:
        return values.astype(np.float32)
    plt = _require_matplotlib_pyplot()
    cmap = plt.get_cmap(cmap_name)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    return cmap(norm(values.reshape(-1)))[:, :3].astype(np.float32)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32) if up is None else np.asarray(up, dtype=np.float32)
    forward = np.asarray(target, dtype=np.float32) - np.asarray(eye, dtype=np.float32)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def camera_pose_from_bounds(
    bounds: np.ndarray,
    elev_deg: float = _CAM_ELEV_DEG,
    azim_deg: float = _CAM_AZIM_DEG,
    distance_scale: float = _CAM_DISTANCE_SCALE,
) -> np.ndarray:
    eye, target, up = camera_vectors_from_bounds(bounds, elev_deg=elev_deg, azim_deg=azim_deg, distance_scale=distance_scale)
    return _look_at(eye, target, up)


def camera_vectors_from_bounds(
    bounds: np.ndarray,
    elev_deg: float = _CAM_ELEV_DEG,
    azim_deg: float = _CAM_AZIM_DEG,
    distance_scale: float = _CAM_DISTANCE_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = np.asarray(bounds, dtype=np.float32)
    target = np.mean(bounds, axis=0).astype(np.float32)
    target[2] = float(bounds[0, 2] + 0.35 * (bounds[1, 2] - bounds[0, 2]))

    extent = float(np.max(bounds[1] - bounds[0]))
    radius = max(extent * distance_scale, 2.0)
    elev = np.deg2rad(float(elev_deg))
    azim = np.deg2rad(float(azim_deg))
    eye = target + radius * np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=np.float32,
    )
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return eye, target, up


def _colorize_mesh_by_height(mesh: trimesh.Trimesh, cmap_name: str = "terrain") -> trimesh.Trimesh:
    colored = mesh.copy()
    if len(colored.vertices) == 0:
        return colored
    rgb = _scalar_to_rgb(colored.vertices[:, 2], cmap_name=cmap_name)
    rgba = np.concatenate([rgb, np.ones((len(rgb), 1), dtype=np.float32)], axis=1)
    colored.visual.vertex_colors = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    return colored


def _add_studio_lights(scene: object, center: np.ndarray, extent: float):
    pyrender = _require_pyrender()

    radius = max(extent * _PYRENDER_LIGHT_RADIUS_SCALE, 3.0)
    for direction, intensity in _PYRENDER_LIGHT_SPECS:
        eye = center + radius * np.array(direction, dtype=np.float32)
        pose = _look_at(eye, center)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=float(intensity)), pose=pose)


def _segment_transform(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    length = np.linalg.norm(vector)
    transform = np.eye(4, dtype=np.float32)
    if np.isclose(length, 0.0):
        transform[:3, 3] = start
        return transform

    direction = vector / length
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if np.allclose(direction, z_axis):
        rotation = np.eye(4)
    elif np.allclose(direction, -z_axis):
        rotation = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        rotation = trimesh.transformations.rotation_matrix(angle, axis)

    transform = rotation.astype(np.float32)
    transform[:3, 3] = (start + end) * 0.5
    return transform


def _polyline_mesh(points: np.ndarray, radius: float = _POLYLINE_RADIUS) -> trimesh.Trimesh | None:
    if len(points) < 2:
        return None
    meshes: list[trimesh.Trimesh] = []
    for start, end in zip(points[:-1], points[1:]):
        length = np.linalg.norm(end - start)
        if np.isclose(length, 0.0):
            continue
        cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        cylinder.apply_transform(_segment_transform(start, end))
        meshes.append(cylinder)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _mesh_to_open3d(mesh: trimesh.Trimesh, cmap_name: str = "terrain"):
    o3d = _require_open3d()
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    if len(mesh.vertices) > 0:
        colors = _scalar_to_rgb(np.asarray(mesh.vertices)[:, 2], cmap_name=cmap_name)
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    return o3d_mesh


def _points_to_open3d_cloud(points: np.ndarray, colors: np.ndarray | None = None):
    o3d = _require_open3d()
    if points is None or len(points) == 0:
        return None
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def _polyline_to_open3d_lineset(points: np.ndarray, color: tuple[float, float, float]):
    o3d = _require_open3d()
    if points is None or len(points) < 2:
        return None
    points_np = np.asarray(points, dtype=np.float64)
    lines = np.array([[index, index + 1] for index in range(len(points_np) - 1)], dtype=np.int32)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_np)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1)))
    return line_set


def _ground_plane_open3d(bounds: np.ndarray):
    o3d = _require_open3d()
    bounds = np.asarray(bounds, dtype=np.float32)
    extent = bounds[1] - bounds[0]
    plane = o3d.geometry.TriangleMesh.create_box(
        width=float(extent[0]),
        height=float(extent[1]),
        depth=max(0.01, 0.01 * float(np.max(extent))),
    )
    plane.compute_vertex_normals()
    plane.paint_uniform_color(_GROUND_COLOR)
    plane.translate((float(bounds[0, 0]), float(bounds[0, 1]), float(bounds[0, 2]) - 0.02))
    return plane


def _goal_marker_open3d(goal_pos: np.ndarray):
    o3d = _require_open3d()
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=_MARKER_SPHERE_RADIUS)
    marker.compute_vertex_normals()
    marker.paint_uniform_color(_MARKER_COLOR)
    marker.translate(np.asarray(goal_pos, dtype=np.float64))
    return marker


def _axis_frame_open3d(bounds: np.ndarray):
    o3d = _require_open3d()
    bounds = np.asarray(bounds, dtype=np.float32)
    extent = float(np.max(bounds[1] - bounds[0]))
    origin = bounds[0].copy()
    origin[2] = bounds[0, 2]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(extent * 0.18, 0.6))
    frame.translate(origin.astype(np.float64))
    return frame


def _wfc_grid_lines_open3d(
    x_lines: np.ndarray,
    y_lines: np.ndarray,
    y_min: float,
    y_max: float,
    x_min: float,
    x_max: float,
    *,
    fine_x_lines: np.ndarray | None = None,
    fine_y_lines: np.ndarray | None = None,
):
    """Build Open3D LineSet for WFC tile grid at z=0 (incl. optional 0.5m fine grid)."""
    o3d = _require_open3d()
    points: list[list[float]] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []

    def _add_point(x: float, y: float):
        idx = len(points)
        points.append([x, y, 0.0])
        return idx

    # tile boundary lines
    for x_val in x_lines:
        a = _add_point(x_val, y_min)
        b = _add_point(x_val, y_max)
        lines.append([a, b])
        colors.append([0.82, 0.82, 0.84])
    for y_val in y_lines:
        a = _add_point(x_min, y_val)
        b = _add_point(x_max, y_val)
        lines.append([a, b])
        colors.append([0.82, 0.82, 0.84])

    # fine grid lines (0.5 m)
    if fine_x_lines is not None:
        for x_val in fine_x_lines:
            a = _add_point(x_val, y_min)
            b = _add_point(x_val, y_max)
            lines.append([a, b])
            colors.append([0.92, 0.92, 0.94])
    if fine_y_lines is not None:
        for y_val in fine_y_lines:
            a = _add_point(x_min, y_val)
            b = _add_point(x_max, y_val)
            lines.append([a, b])
            colors.append([0.92, 0.92, 0.94])

    if not points:
        return None

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return line_set


def build_open3d_geometry(
    mesh: trimesh.Trimesh,
    points: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
    goal_pos: np.ndarray | None = None,
    route_points: np.ndarray | None = None,
    *,
    show_ground: bool = True,
    show_axes: bool = True,
    grid_meta: dict | None = None,
) -> list[object]:
    bounds = mesh.bounding_box.bounds
    geometries: list[object] = []
    if show_ground:
        geometries.append(_ground_plane_open3d(bounds))
    geometries.append(_mesh_to_open3d(mesh))
    if points is not None and len(points) > 0:
        cloud = _points_to_open3d_cloud(points, point_colors)
        if cloud is not None:
            geometries.append(cloud)
    if goal_pos is not None:
        geometries.append(_goal_marker_open3d(goal_pos))
    if route_points is not None and len(route_points) > 1:
        route = _polyline_to_open3d_lineset(np.asarray(route_points, dtype=np.float32), _ROUTE_COLOR)
        if route is not None:
            geometries.append(route)
    if show_axes:
        geometries.append(_axis_frame_open3d(bounds))
    if grid_meta is not None:
        grid_lines = _wfc_grid_lines_open3d(
            np.asarray(grid_meta["x_lines"]),
            np.asarray(grid_meta["y_lines"]),
            grid_meta["y_min"],
            grid_meta["y_max"],
            grid_meta["x_min"],
            grid_meta["x_max"],
            fine_x_lines=np.asarray(grid_meta.get("fine_x_lines", [])),
            fine_y_lines=np.asarray(grid_meta.get("fine_y_lines", [])),
        )
        if grid_lines is not None:
            geometries.append(grid_lines)
    return geometries


def build_open3d_scene_items(
    mesh: trimesh.Trimesh,
    points: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
    goal_pos: np.ndarray | None = None,
    route_points: np.ndarray | None = None,
    *,
    show_ground: bool = True,
    show_axes: bool = True,
    grid_meta: dict | None = None,
) -> list[tuple[str, object, object]]:
    o3d = _require_open3d()
    rendering = o3d.visualization.rendering
    items: list[tuple[str, object, object]] = []

    lit = rendering.MaterialRecord()
    lit.shader = "defaultLit"
    lit.base_roughness = _O3D_BASE_ROUGHNESS
    lit.base_reflectance = _O3D_BASE_REFLECTANCE

    unlit = rendering.MaterialRecord()
    unlit.shader = "defaultUnlit"

    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = _O3D_LINE_WIDTH

    bounds = mesh.bounding_box.bounds
    if show_ground:
        items.append(("ground", _ground_plane_open3d(bounds), lit))
    items.append(("terrain", _mesh_to_open3d(mesh), lit))
    if points is not None and len(points) > 0:
        cloud = _points_to_open3d_cloud(points, point_colors)
        if cloud is not None:
            point_mat = rendering.MaterialRecord()
            point_mat.shader = "defaultUnlit"
            point_mat.point_size = 6.0
            items.append(("points", cloud, point_mat))
    if goal_pos is not None:
        items.append(("goal", _goal_marker_open3d(goal_pos), unlit))
    if route_points is not None and len(route_points) > 1:
        route = _polyline_to_open3d_lineset(np.asarray(route_points, dtype=np.float32), _ROUTE_COLOR)
        if route is not None:
            items.append(("route", route, line_mat))
    if show_axes:
        items.append(("axes", _axis_frame_open3d(bounds), unlit))
    if grid_meta is not None:
        grid_lines = _wfc_grid_lines_open3d(
            np.asarray(grid_meta["x_lines"]),
            np.asarray(grid_meta["y_lines"]),
            grid_meta["y_min"],
            grid_meta["y_max"],
            grid_meta["x_min"],
            grid_meta["x_max"],
            fine_x_lines=np.asarray(grid_meta.get("fine_x_lines", [])),
            fine_y_lines=np.asarray(grid_meta.get("fine_y_lines", [])),
        )
        if grid_lines is not None:
            items.append(("grid_lines", grid_lines, line_mat))
    return items


def draw_open3d(
    mesh: trimesh.Trimesh,
    points: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
    goal_pos: np.ndarray | None = None,
    route_points: np.ndarray | None = None,
    *,
    show_ground: bool = True,
    show_axes: bool = True,
    grid_meta: dict | None = None,
):
    o3d = _require_open3d()
    items = build_open3d_scene_items(
        mesh,
        points=points,
        point_colors=point_colors,
        goal_pos=goal_pos,
        route_points=route_points,
        show_ground=show_ground,
        show_axes=show_axes,
        grid_meta=grid_meta,
    )
    eye, target, up = camera_vectors_from_bounds(mesh.bounding_box.bounds)
    draw_items = [
        {"name": name, "geometry": geometry, "material": material}
        for name, geometry, material in items
    ]
    return o3d.visualization.draw(
        draw_items,
        bg_color=_O3D_BG_COLOR,
        show_ui=True,
        lookat=target,
        eye=eye,
        up=up,
        field_of_view=60.0,
    )


def render_open3d_frame_sequence(
    mesh: trimesh.Trimesh,
    output_path: str,
    width: int,
    height: int,
    fps: int,
    elev_seq: np.ndarray,
    azim_seq: np.ndarray,
    *,
    points: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
    goal_pos: np.ndarray | None = None,
    route_points: np.ndarray | None = None,
    show_ground: bool = True,
    show_axes: bool = True,
    grid_meta: dict | None = None,
):
    o3d = _require_open3d()
    renderer = o3d.visualization.rendering.OffscreenRenderer(int(width), int(height))
    scene = renderer.scene
    scene.set_background(np.array(_O3D_BG_COLOR, dtype=np.float32))
    scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, np.array([0.577, -0.577, -0.577], dtype=np.float32))
    scene.show_axes(show_axes)

    for name, geometry, material in build_open3d_scene_items(
        mesh,
        points=points,
        point_colors=point_colors,
        goal_pos=goal_pos,
        route_points=route_points,
        show_ground=show_ground,
        show_axes=show_axes,
        grid_meta=grid_meta,
    ):
        scene.add_geometry(name, geometry, material)

    import imageio.v2 as imageio

    writer = imageio.get_writer(str(output_path), fps=max(1, int(fps)))
    try:
        for elev, azim in zip(elev_seq, azim_seq):
            eye, target, up = camera_vectors_from_bounds(mesh.bounding_box.bounds, float(elev), float(azim))
            renderer.setup_camera(60.0, target, eye, up)
            image = np.asarray(renderer.render_to_image())
            writer.append_data(image)
    finally:
        writer.close()
        renderer.scene.clear_geometry()
        renderer = None


def _points_mesh(points: np.ndarray, colors: np.ndarray | None = None, radius: float = 0.05):
    pyrender = _require_pyrender()
    if points.size == 0:
        return None
    try:
        return pyrender.Mesh.from_points(points, colors=colors)
    except Exception:
        spheres = []
        for idx, point in enumerate(points):
            sphere = trimesh.creation.uv_sphere(radius=radius)
            sphere.apply_translation(point)
            if colors is not None:
                color = np.clip(colors[idx] * 255.0, 0, 255).astype(np.uint8)
                sphere.visual.vertex_colors = np.tile(np.append(color, 255), (len(sphere.vertices), 1))
            spheres.append(sphere)
        return pyrender.Mesh.from_trimesh(trimesh.util.concatenate(spheres), smooth=False)


def build_scene(
    mesh: trimesh.Trimesh,
    points: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
    goal_pos: np.ndarray | None = None,
    route_points: np.ndarray | None = None,
    camera_pose: np.ndarray | None = None,
    return_nodes: bool = False,
) -> object:
    pyrender = _require_pyrender()

    scene = pyrender.Scene(bg_color=np.array(_O3D_BG_COLOR), ambient_light=np.array([0.10, 0.10, 0.12]))
    render_mesh = _colorize_mesh_by_height(mesh)
    scene.add(pyrender.Mesh.from_trimesh(render_mesh, smooth=True))

    if points is not None and len(points) > 0:
        point_mesh = _points_mesh(points, point_colors)
        if point_mesh is not None:
            scene.add(point_mesh)

    if goal_pos is not None:
        marker = trimesh.creation.uv_sphere(radius=_MARKER_SPHERE_RADIUS)
        marker.apply_translation(np.asarray(goal_pos, dtype=np.float32))
        marker.visual.vertex_colors = np.tile(np.array([220, 30, 30, 255], dtype=np.uint8), (len(marker.vertices), 1))
        scene.add(pyrender.Mesh.from_trimesh(marker, smooth=False))

        stem = _polyline_mesh(np.stack([goal_pos, np.asarray(goal_pos) + np.array([0.0, 0.0, 1.0])]), radius=_POLYLINE_RADIUS * 0.67)
        if stem is not None:
            stem.visual.vertex_colors = np.tile(np.array([220, 30, 30, 255], dtype=np.uint8), (len(stem.vertices), 1))
            scene.add(pyrender.Mesh.from_trimesh(stem, smooth=False))

    if route_points is not None and len(route_points) > 1:
        route_mesh = _polyline_mesh(np.asarray(route_points, dtype=np.float32), radius=_ROUTE_RADIUS)
        if route_mesh is not None:
            route_mesh.visual.vertex_colors = np.tile(np.array([20, 140, 240, 255], dtype=np.uint8), (len(route_mesh.vertices), 1))
            scene.add(pyrender.Mesh.from_trimesh(route_mesh, smooth=False))

    bounds = mesh.bounding_box.bounds
    center = bounds.mean(axis=0)
    extent = np.max(bounds[1] - bounds[0])
    if camera_pose is None:
        camera_pose = camera_pose_from_bounds(bounds)

    _add_studio_lights(scene, center, float(extent))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_node = scene.add(camera, pose=camera_pose)
    if return_nodes:
        return scene, {"camera": camera_node}
    return scene


def render_scene_offscreen(
    scene: object,
    width: int,
    height: int,
    *,
    renderer: object | None = None,
    shadows: bool = True,
) -> np.ndarray:
    pyrender = _require_pyrender()
    own_renderer = False
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        own_renderer = True

    flags = pyrender.RenderFlags.RGBA
    if shadows:
        flags |= pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    color, _depth = renderer.render(scene, flags=flags)
    if own_renderer:
        renderer.delete()
    return color


def visualize_mesh(
    mesh: trimesh.Trimesh,
    points: np.ndarray | None = None,
    color_values: np.ndarray | None = None,
    goal_pos: np.ndarray | None = None,
    route_points: np.ndarray | None = None,
    show: bool = True,
) -> object:
    point_colors = None if color_values is None else _scalar_to_rgb(np.asarray(color_values))
    if show:
        try:
            return draw_open3d(
                mesh,
                points=points,
                point_colors=point_colors,
                goal_pos=goal_pos,
                route_points=route_points,
            )
        except ImportError:
            pass
    pyrender = _require_pyrender()
    scene = build_scene(mesh, points=points, point_colors=point_colors, goal_pos=goal_pos, route_points=route_points)
    if show:
        pyrender.Viewer(scene, use_raymond_lighting=False)
    return scene
