from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.visualize_route import _ROUTE_RENDER_Z_OFFSET, _random_navigation_endpoints, _run_pipeline, sample_camera_along_route
from generator import MeshTerrain, MeshTerrainCfg, list_registered_terrain_names, list_wfc_terrain_names
from generator.route_graph import route_graph_nodes_by_id, route_graph_shortest_path
from generator.nav_utils import _close_xy_mask, compute_traversability_map, get_navigation_height_array_of_mesh_with_resolution
from generator.presets import procedural_terrain_result, procedural_wfc_specs
from generator.terrains import (
    PileTerrainCfg,
    PlatformGapTerrainCfg,
    PyramidStairsTerrainCfg,
    generate_platform_gap_terrain,
    generate_pyramid_stairs_terrain,
)
from generator.wfc import compose_wave_mesh, create_procedural_wfc_tiles
from generator.wfc import run_wfc_scene_with_procedural_specs


def _xy_step_lengths(route: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(np.asarray(route)[:, :2], axis=0), axis=1)


def _non_platform_step_lengths(
    route: np.ndarray,
    *,
    platform_center: tuple[float, float] = (4.0, 4.0),
    platform_half: float = 1.0,
) -> np.ndarray:
    points = np.asarray(route)
    center = np.asarray(platform_center, dtype=np.float32)
    inside = (np.abs(points[:, 0] - center[0]) <= platform_half) & (np.abs(points[:, 1] - center[1]) <= platform_half)
    steps = _xy_step_lengths(points)
    keep = ~(inside[:-1] & inside[1:])
    return steps[keep]


def _pile_terrain(size: tuple[float, float] = (8.0, 8.0)) -> tuple[object, MeshTerrain]:
    result, _ = procedural_terrain_result("pile", size=size, seed=42)
    terrain = MeshTerrain(
        MeshTerrainCfg(
            mesh=result.mesh,
            terrain_state=result.terrain_state,
            auto_compute_sdf=False,
            auto_compute_distance=False,
            distance_matrix=np.zeros((1, 1), dtype=np.float32),
            distance_shape=(1, 1),
        )
    )
    return result, terrain


def _pile_default_max_step_distance() -> float:
    cfg = PileTerrainCfg(size=(8.0, 8.0))
    spacing = max(cfg.pillar_spacing, 2.0 * cfg.pillar_radius + cfg.pillar_clearance_between)
    return float(spacing * 1.35)


def _pile_support_xy(result) -> np.ndarray:
    centers = [
        np.asarray(member.center, dtype=np.float32)[:2]
        for member in result.terrain_state.members
        if member.kind == "pillar"
    ]
    return np.asarray(centers, dtype=np.float32)


def _assert_non_platform_points_on_support(route: np.ndarray, support_xy: np.ndarray, platform_center: tuple[float, float], platform_half: float = 1.0):
    points = np.asarray(route, dtype=np.float32)
    center = np.asarray(platform_center, dtype=np.float32)
    inside_platform = (np.abs(points[:, 0] - center[0]) <= platform_half + 1e-4) & (
        np.abs(points[:, 1] - center[1]) <= platform_half + 1e-4
    )
    non_platform = points[~inside_platform]
    if non_platform.size == 0:
        return
    distances = np.linalg.norm(non_platform[:, None, :2] - support_xy[None, :, :], axis=2)
    assert np.all(np.min(distances, axis=1) <= 1e-3)


def _route_contains_xy(route: np.ndarray, xy: tuple[float, float] | np.ndarray, atol: float = 1e-4) -> bool:
    points = np.asarray(route, dtype=np.float32)
    target = np.asarray(xy, dtype=np.float32)
    return bool(np.any(np.linalg.norm(points[:, :2] - target[None, :2], axis=1) <= atol))


def _graph_path_between_boundary_sides(route_graph: dict[str, object], start_side: str, goal_side: str) -> list[str]:
    start_id = route_graph["boundaries"][start_side][0]
    goal_id = route_graph["boundaries"][goal_side][0]
    return route_graph_shortest_path(route_graph, start_id, goal_id)


def _assert_boundary_paths_include(route_graph: dict[str, object], node_id: str, side_pairs: tuple[tuple[str, str], ...]):
    for start_side, goal_side in side_pairs:
        assert node_id in _graph_path_between_boundary_sides(route_graph, start_side, goal_side)


def _assert_non_platform_steps_are_axis_aligned(
    route: np.ndarray,
    *,
    platform_centers: tuple[tuple[float, float], ...],
    platform_half: float = 1.0,
):
    points = np.asarray(route, dtype=np.float32)
    centers = [np.asarray(center, dtype=np.float32) for center in platform_centers]

    def inside_platform(point: np.ndarray) -> bool:
        return any(np.all(np.abs(point[:2] - center) <= platform_half + 1e-4) for center in centers)

    for start, end in zip(points[:-1], points[1:]):
        if not np.isclose(start[2], end[2], atol=1e-4):
            continue
        if inside_platform(start) or inside_platform(end):
            continue
        delta = np.abs(start[:2] - end[:2])
        assert delta[0] <= 1e-5 or delta[1] <= 1e-5


def test_registry_replaces_legacy_ceiling_and_stairs_with_new_terrains():
    names = set(list_registered_terrain_names())

    assert "ceiling" not in names
    assert "stairs" not in names
    assert {"platform_gap", "pyramid_stairs", "stakes"}.issubset(names)
    assert "ceiling" not in set(list_wfc_terrain_names(default_only=True))
    assert "stairs" not in set(list_wfc_terrain_names(default_only=True))


def test_single_pile_route_uses_metadata_footholds_not_heightmap_edge_walk():
    result, terrain = _pile_terrain()
    start = np.array([1.0, 4.0, 0.0], dtype=np.float32)
    goal = np.array([7.0, 4.0, 0.0], dtype=np.float32)

    route = terrain.get_route(start, goal, use_diagonal=True)

    assert route.shape[0] < 20
    assert np.max(_non_platform_step_lengths(route)) <= _pile_default_max_step_distance()
    assert np.min(route[:, 0]) > 0.5
    assert _route_contains_xy(route, (4.0, 4.0))
    _assert_non_platform_points_on_support(route, _pile_support_xy(result), platform_center=(4.0, 4.0))


def test_pile_cross_route_graph_forces_boundary_paths_through_center_platform():
    result, _terrain = _pile_terrain()
    route_graph = result.terrain_state.metadata["planning"]["route_graph"]
    nodes_by_id = route_graph_nodes_by_id(route_graph)

    assert "platform_center" in nodes_by_id
    _assert_boundary_paths_include(
        route_graph,
        "platform_center",
        (("left", "right"), ("left", "up"), ("down", "right")),
    )


def test_pile_default_pillar_radius_is_0_3m():
    cfg = PileTerrainCfg(size=(8.0, 8.0))
    result, _terrain = _pile_terrain()
    pillar_members = [member for member in result.terrain_state.members if member.kind == "pillar"]

    assert cfg.pillar_radius == pytest.approx(0.3)
    assert pillar_members
    assert all(member.params["radius"] == pytest.approx(0.3) for member in pillar_members)


def test_pile_cross_route_graph_has_no_diagonal_support_shortcuts():
    result, _terrain = _pile_terrain()
    route_graph = result.terrain_state.metadata["planning"]["route_graph"]
    nodes_by_id = route_graph_nodes_by_id(route_graph)

    for edge in route_graph["edges"]:
        node_a = nodes_by_id[edge["a"]]
        node_b = nodes_by_id[edge["b"]]
        if node_a["kind"] != "pillar" or node_b["kind"] != "pillar":
            continue
        delta = np.abs(node_a["point"][:2] - node_b["point"][:2])
        assert delta[0] <= 1e-5 or delta[1] <= 1e-5


def test_single_pile_visual_entry_has_no_edge_skirt_route():
    result, _terrain = _pile_terrain()
    visual_terrain = MeshTerrain(
        MeshTerrainCfg(
            mesh=result.mesh,
            terrain_state=result.terrain_state,
            auto_compute_sdf=False,
            auto_compute_distance=True,
        )
    )

    route = visual_terrain.get_route(
        np.array([1.0, 4.0, 0.0], dtype=np.float32),
        np.array([7.0, 4.0, 0.0], dtype=np.float32),
        use_diagonal=True,
    )

    assert np.min(route[:, 1]) > 3.0
    assert np.max(_non_platform_step_lengths(route)) <= _pile_default_max_step_distance()
    assert _route_contains_xy(route, (4.0, 4.0))
    _assert_non_platform_points_on_support(route, _pile_support_xy(result), platform_center=(4.0, 4.0))


def test_wfc_grid_mapping_uses_tile_centers_after_centering():
    specs = procedural_wfc_specs(["pile"], tile_size=(8.0, 8.0), seed=42)
    scene, _ = run_wfc_scene_with_procedural_specs((2, 2), specs, tile_size=(8.0, 8.0), seed=42)

    for row in range(2):
        for col in range(2):
            assert scene._grid_from_world_xy(scene._tile_offset(row, col)[:2]) == (row, col)


def test_wfc_pile_route_stays_on_true_3d_footholds():
    specs = procedural_wfc_specs(["pile"], tile_size=(8.0, 8.0), seed=42)
    scene, _ = run_wfc_scene_with_procedural_specs((1, 1), specs, tile_size=(8.0, 8.0), seed=42)

    route = scene.plan_route(
        np.array([-3.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        height_map_resolution=0.1,
    )

    assert np.allclose(route[-1, :2], [0.0, 0.0])
    assert np.allclose(route[1:-1, 2], scene.mesh.bounds[1, 2])
    assert _route_contains_xy(route, (0.0, 0.0))
    _assert_non_platform_steps_are_axis_aligned(route, platform_centers=((0.0, 0.0),), platform_half=1.0)


def test_wfc_pile_bridge_has_only_boundary_footholds_not_interpolated_void_points():
    specs = procedural_wfc_specs(["pile"], tile_size=(8.0, 8.0), seed=42)
    scene, _ = run_wfc_scene_with_procedural_specs((1, 2), specs, tile_size=(8.0, 8.0), seed=42)

    route = scene.plan_route(
        scene._tile_offset(0, 0) + np.array([1.0, 0.0, 0.0], dtype=np.float32),
        scene._tile_offset(0, 1) + np.array([0.0, 0.0, 0.0], dtype=np.float32),
        height_map_resolution=0.1,
    )
    x_between_tiles = route[(route[:, 0] > -0.5) & (route[:, 0] < 0.5)]

    assert x_between_tiles.shape[0] <= 2
    assert np.allclose(route[1:-1, 2], scene.mesh.bounds[1, 2])


def test_wfc_pile_cross_tile_route_includes_each_tile_center_platform():
    specs = procedural_wfc_specs(["pile"], tile_size=(8.0, 8.0), seed=42)
    scene, _ = run_wfc_scene_with_procedural_specs((1, 2), specs, tile_size=(8.0, 8.0), seed=42)

    left_center = scene._tile_offset(0, 0)[:2]
    right_center = scene._tile_offset(0, 1)[:2]
    route = scene.plan_route(
        scene._tile_offset(0, 0) + np.array([-3.0, 0.0, 0.0], dtype=np.float32),
        scene._tile_offset(0, 1) + np.array([3.0, 0.0, 0.0], dtype=np.float32),
        height_map_resolution=0.1,
    )

    assert _route_contains_xy(route, left_center)
    assert _route_contains_xy(route, right_center)


def test_visual_pipeline_passes_single_terrain_state_to_route_planner(monkeypatch):
    result, _ = procedural_terrain_result("pile", size=(8.0, 8.0), seed=42)
    captured = {}

    class FakeTerrain:
        def __init__(self, cfg):
            captured["terrain_state"] = cfg.terrain_state
            self.cfg = SimpleNamespace(
                spawnable_locations=np.array(
                    [[1.0, 4.0, 0.0], [7.0, 4.0, 0.0]],
                    dtype=np.float32,
                )
            )

        def get_route(self, start_pos, goal_pos, use_diagonal=True):
            return np.stack([start_pos, goal_pos]).astype(np.float32)

    monkeypatch.setattr("examples.visualize_route.MeshTerrain", FakeTerrain)
    monkeypatch.setattr(
        "examples.visualize_route.sample_camera_along_route",
        lambda route, camera_height, look_ahead_distance, speed_ms, fps: (
            [np.array([0.0, 0.0, 1.0], dtype=np.float32)],
            [np.array([1.0, 0.0, 1.0], dtype=np.float32)],
            1.0,
            1,
        ),
    )

    class StopAfterPlanning(RuntimeError):
        pass

    class FakeRenderer:
        def __init__(self, *_args, **_kwargs):
            raise StopAfterPlanning()

    fake_o3d = SimpleNamespace(
        visualization=SimpleNamespace(rendering=SimpleNamespace(OffscreenRenderer=FakeRenderer))
    )
    args = SimpleNamespace(
        start=[1.0, 4.0],
        goal=[7.0, 4.0],
        camera_height=2.5,
        look_ahead=5.0,
        speed=1.0,
            fps=30,
            seed=42,
            resolution=[64, 64],
        output="/tmp/unused.mp4",
        no_text=True,
    )

    with pytest.raises(StopAfterPlanning):
        _run_pipeline(args, fake_o3d, result.mesh, "Terrain: pile", terrain_state=result.terrain_state)

    assert captured["terrain_state"] is result.terrain_state
    assert _ROUTE_RENDER_Z_OFFSET == pytest.approx(0.5)


def test_down_pyramid_stairs_are_traversable_below_default_min_height():
    result = generate_pyramid_stairs_terrain(
        PyramidStairsTerrainCfg(size=(8.0, 8.0), direction="down", step_count_range=(5, 5), step_height_range=(0.12, 0.12), seed=4)
    )
    terrain = MeshTerrain(
        MeshTerrainCfg(
            mesh=result.mesh,
            terrain_state=result.terrain_state,
            auto_compute_sdf=False,
            auto_compute_distance=False,
            distance_matrix=np.zeros((1, 1), dtype=np.float32),
            distance_shape=(1, 1),
        )
    )

    route = terrain.get_route(np.array([0.8, 4.0, 0.0], dtype=np.float32), np.array([7.2, 4.0, 0.0], dtype=np.float32), use_diagonal=True)

    assert result.metadata["direction"] == "down"
    assert route.shape[0] >= 3
    assert np.min(route[:, 2]) < 0.0
    assert np.max(np.abs(np.diff(route[:, 2]))) <= 0.4


def test_pyramid_stairs_route_graph_connects_center_to_all_boundaries_with_true_step_heights():
    result = generate_pyramid_stairs_terrain(
        PyramidStairsTerrainCfg(
            size=(8.0, 8.0),
            direction="up",
            step_count_range=(4, 4),
            step_width_range=(0.5, 0.5),
            step_height_range=(0.1, 0.1),
            platform_half_range=(0.8, 0.8),
            seed=0,
        )
    )
    route_graph = result.terrain_state.metadata["planning"]["route_graph"]
    nodes_by_id = route_graph_nodes_by_id(route_graph)

    assert set(route_graph["boundaries"]) == {"left", "right", "up", "down"}
    assert np.allclose(nodes_by_id["center_4"]["point"], np.array([4.0, 4.0, 0.4], dtype=np.float32))
    for side in ("left", "right", "up", "down"):
        boundary_node_id = route_graph["boundaries"][side][0]
        assert np.isclose(nodes_by_id[boundary_node_id]["point"][2], 0.0)


def test_platform_gap_default_depth_matches_pile_depth():
    cfg = PlatformGapTerrainCfg(size=(8.0, 8.0), seed=3)
    pile_cfg = PileTerrainCfg(size=(8.0, 8.0))
    result = generate_platform_gap_terrain(cfg)

    assert cfg.gap_depth == pytest.approx(pile_cfg.ground_depth)
    assert result.metadata["gap_depth"] == pytest.approx(pile_cfg.ground_depth)
    assert result.mesh.bounds[0, 2] == pytest.approx(-pile_cfg.ground_depth)


def test_platform_gap_default_gap_width_is_narrow():
    result, _ = procedural_terrain_result("platform_gap", size=(8.0, 8.0), seed=3)

    assert result.metadata["gap_width"] == pytest.approx(0.25)
    assert result.metadata["gap_half"] - result.metadata["platform_half"] == pytest.approx(0.25)


def test_platform_gap_default_seed_is_randomized_but_explicit_seed_is_deterministic():
    seeds = {PlatformGapTerrainCfg().seed for _ in range(8)}
    preset_seeds = {procedural_terrain_result("platform_gap", size=(8.0, 8.0))[1].seed for _ in range(8)}
    first = generate_platform_gap_terrain(PlatformGapTerrainCfg(size=(8.0, 8.0), seed=3))
    second = generate_platform_gap_terrain(PlatformGapTerrainCfg(size=(8.0, 8.0), seed=3))

    assert len(seeds) > 1
    assert len(preset_seeds) > 1
    assert first.metadata["gap_half"] == pytest.approx(second.metadata["gap_half"])
    assert first.metadata["platform_half"] == pytest.approx(second.metadata["platform_half"])


def test_visual_route_random_endpoints_are_seeded_and_avoid_platform_gap_floor():
    result, _ = procedural_terrain_result("platform_gap", size=(8.0, 8.0), seed=3)
    height_array, _ = get_navigation_height_array_of_mesh_with_resolution(
        result.mesh,
        resolution=0.1,
        terrain_state=result.terrain_state,
    )
    first = _random_navigation_endpoints(
        height_array,
        result.mesh.bounds,
        resolution=0.1,
        rng=np.random.default_rng(34),
        terrain_state=result.terrain_state,
    )
    second = _random_navigation_endpoints(
        height_array,
        result.mesh.bounds,
        resolution=0.1,
        rng=np.random.default_rng(34),
        terrain_state=result.terrain_state,
    )
    third = _random_navigation_endpoints(
        height_array,
        result.mesh.bounds,
        resolution=0.1,
        rng=np.random.default_rng(35),
        terrain_state=result.terrain_state,
    )

    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])
    assert not (np.allclose(first[0], third[0]) and np.allclose(first[1], third[1]))
    assert first[0][2] >= -0.2
    assert first[1][2] >= -0.2


def test_platform_gap_route_graph_uses_center_star_not_gap_floor():
    result, _ = procedural_terrain_result("platform_gap", size=(8.0, 8.0), seed=3)
    route_graph = result.terrain_state.metadata["planning"]["route_graph"]
    nodes_by_id = route_graph_nodes_by_id(route_graph)
    points = np.stack([np.asarray(node["point"], dtype=np.float32) for node in route_graph["nodes"]], axis=0)
    gap_half = float(result.metadata["gap_half"])
    platform_half = float(result.metadata["platform_half"])
    center = np.array([4.0, 4.0], dtype=np.float32)
    distances = np.max(np.abs(points[:, :2] - center), axis=1)

    assert "center_platform" in nodes_by_id
    assert np.all((distances >= gap_half - 1e-4) | (distances <= platform_half + 1e-4))
    assert np.all(points[:, 2] >= -1e-5)
    _assert_boundary_paths_include(
        route_graph,
        "center_platform",
        (("left", "right"), ("left", "up"), ("down", "right")),
    )


def test_platform_gap_route_graph_has_no_ring_or_diagonal_edges():
    result, _ = procedural_terrain_result("platform_gap", size=(8.0, 8.0), seed=3)
    route_graph = result.terrain_state.metadata["planning"]["route_graph"]
    nodes_by_id = route_graph_nodes_by_id(route_graph)
    side_ids = {"left", "right", "up", "down"}

    for edge in route_graph["edges"]:
        edge_ids = {edge["a"], edge["b"]}
        assert len(edge_ids & side_ids) < 2

        point_a = nodes_by_id[edge["a"]]["point"]
        point_b = nodes_by_id[edge["b"]]["point"]
        delta = np.abs(point_a[:2] - point_b[:2])
        assert delta[0] <= 1e-5 or delta[1] <= 1e-5


def test_single_platform_gap_route_passes_through_center_platform():
    result, _ = procedural_terrain_result("platform_gap", size=(8.0, 8.0), seed=3)
    terrain = MeshTerrain(
        MeshTerrainCfg(
            mesh=result.mesh,
            terrain_state=result.terrain_state,
            auto_compute_sdf=False,
            auto_compute_distance=False,
            distance_matrix=np.zeros((1, 1), dtype=np.float32),
            distance_shape=(1, 1),
        )
    )

    route = terrain.get_route(np.array([0.5, 4.0, 0.0], dtype=np.float32), np.array([7.5, 4.0, 0.0], dtype=np.float32), use_diagonal=True)

    assert _route_contains_xy(route, (4.0, 4.0))


def test_wfc_platform_gap_route_passes_through_tile_center_platform():
    specs = procedural_wfc_specs(["platform_gap"], tile_size=(8.0, 8.0), seed=3)
    scene, _ = run_wfc_scene_with_procedural_specs((1, 1), specs, tile_size=(8.0, 8.0), seed=3)

    route = scene.plan_route(
        scene._tile_offset(0, 0) + np.array([-3.5, 0.0, 0.0], dtype=np.float32),
        scene._tile_offset(0, 0) + np.array([3.5, 0.0, 0.0], dtype=np.float32),
        height_map_resolution=0.1,
    )

    assert _route_contains_xy(route, scene._tile_offset(0, 0)[:2])


def test_platform_gap_marks_gap_floor_as_void_not_walkable():
    result, _ = procedural_terrain_result("platform_gap", size=(8.0, 8.0), seed=3)

    traversable = compute_traversability_map(
        result.mesh,
        height_map_resolution=0.1,
        terrain_state=result.terrain_state,
        graph_ratio=1,
    )
    bounds = result.mesh.bounding_box.bounds
    gap_half = float(result.metadata["gap_half"])
    platform_half = float(result.metadata["platform_half"])
    center = np.array([4.0, 4.0], dtype=np.float32)
    gap_point = center + np.array([0.5 * (gap_half + platform_half), 0.0], dtype=np.float32)
    row = int(np.clip(round((float(gap_point[1]) - bounds[0, 1]) / 0.1), 0, traversable.shape[0] - 1))
    col = int(np.clip(round((float(gap_point[0]) - bounds[0, 0]) / 0.1), 0, traversable.shape[1] - 1))

    assert not traversable[row, col]
    assert any(member.role == "void" for member in result.terrain_state.members)


def test_stakes_route_uses_foothold_graph_not_void_heightmap():
    result, _ = procedural_terrain_result("stakes", size=(8.0, 8.0), seed=7)
    terrain = MeshTerrain(
        MeshTerrainCfg(
            mesh=result.mesh,
            terrain_state=result.terrain_state,
            auto_compute_sdf=False,
            auto_compute_distance=False,
            distance_matrix=np.zeros((1, 1), dtype=np.float32),
            distance_shape=(1, 1),
        )
    )

    route = terrain.get_route(np.array([1.0, 4.0, 0.0], dtype=np.float32), np.array([7.0, 4.0, 0.0], dtype=np.float32), use_diagonal=True)
    support_xy = np.asarray(
        [np.asarray(member.center, dtype=np.float32)[:2] for member in result.terrain_state.members if member.kind == "stake"],
        dtype=np.float32,
    )

    assert result.terrain_state.metadata["planning"]["foothold_graph"]["enabled"] is True
    assert route.shape[0] >= 3
    _assert_non_platform_points_on_support(route, support_xy, platform_center=(4.0, 4.0), platform_half=0.9)


def test_support_route_graph_nodes_stay_on_supports_for_pile_and_stakes():
    pile_result, _ = procedural_terrain_result("pile", size=(8.0, 8.0), seed=42)
    pile_nodes = pile_result.terrain_state.metadata["planning"]["route_graph"]["nodes"]
    pile_points = np.asarray([node["point"] for node in pile_nodes if node["kind"] == "pillar"], dtype=np.float32)
    _assert_non_platform_points_on_support(pile_points, _pile_support_xy(pile_result), platform_center=(4.0, 4.0))

    stakes_result, _ = procedural_terrain_result("stakes", size=(8.0, 8.0), seed=7)
    stake_nodes = stakes_result.terrain_state.metadata["planning"]["route_graph"]["nodes"]
    support_xy = np.asarray(
        [np.asarray(member.center, dtype=np.float32)[:2] for member in stakes_result.terrain_state.members if member.kind == "stake"],
        dtype=np.float32,
    )
    stake_points = np.asarray([node["point"] for node in stake_nodes if node["kind"] == "stake"], dtype=np.float32)
    _assert_non_platform_points_on_support(stake_points, support_xy, platform_center=(4.0, 4.0), platform_half=0.9)


def test_wfc_stakes_bridge_uses_neighbor_boundary_footholds():
    specs = procedural_wfc_specs(["stakes"], tile_size=(8.0, 8.0), seed=7)
    scene, _ = run_wfc_scene_with_procedural_specs((1, 2), specs, tile_size=(8.0, 8.0), seed=7)

    route = scene.plan_route(
        scene._tile_offset(0, 0) + np.array([1.0, 0.0, 0.0], dtype=np.float32),
        scene._tile_offset(0, 1) + np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        height_map_resolution=0.1,
    )

    assert np.min(route[:, 0]) < -1.0
    assert np.max(route[:, 0]) > 1.0
    assert np.allclose(route[1:-1, 2], scene.mesh.bounds[1, 2])


def test_wfc_transforms_planning_height_limits_with_tile_mesh():
    specs = procedural_wfc_specs(["stakes"], tile_size=(8.0, 8.0), seed=7)
    scene, _ = run_wfc_scene_with_procedural_specs((1, 1), specs, tile_size=(8.0, 8.0), seed=7)
    tile_name = scene.tile_names[int(scene.wave[0, 0])]
    planning = scene.tile_terrain_states[tile_name].metadata["planning"]
    support_top = scene.tile_meshes[tile_name].bounds[1, 2]

    assert planning["height_limits"]["min"] == pytest.approx(support_top - 0.2)
    assert planning["height_limits"]["max"] == pytest.approx(support_top + 0.3)


def test_wfc_transforms_route_graph_boundaries_with_tile_rotation():
    specs = procedural_wfc_specs(["pyramid_stairs"], tile_size=(8.0, 8.0), seed=0)
    base_tile = create_procedural_wfc_tiles(specs, tile_size=(8.0, 8.0))[0]
    rotated = base_tile.get_rotated_tile(90)
    planning = rotated.terrain_state.metadata["planning"]
    assert planning["route_graph"]["boundaries"]["up"] == ["right_0"]
    assert planning["route_graph"]["boundaries"]["left"] == ["up_0"]


def test_wfc_transition_from_rotated_pyramid_stairs_to_flipped_door_plans_successfully():
    specs = procedural_wfc_specs(["pyramid_stairs", "door"], tile_size=(8.0, 8.0), seed=0)
    stairs_tile, door_tile = create_procedural_wfc_tiles(specs, tile_size=(8.0, 8.0))
    rotated_stairs = stairs_tile.get_rotated_tile(90)
    flipped_door = door_tile.get_flipped_tile("y").get_rotated_tile(180)
    scene = compose_wave_mesh(np.array([[0, 1]], dtype=np.int64), [rotated_stairs.name, flipped_door.name], [rotated_stairs, flipped_door])

    route = scene.plan_route(
        scene._tile_offset(0, 0) + np.array([0.0, 0.0, 0.0], dtype=np.float32),
        scene._tile_offset(0, 1) + np.array([0.0, 0.0, 0.0], dtype=np.float32),
        height_map_resolution=0.1,
    )

    assert route.shape[0] >= 4
    assert np.max(route[:, 0]) > 0.5


def test_low_void_morphology_fills_gap_then_adds_safety_margin():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, :] = True
    mask[3, 3] = False

    closed = _close_xy_mask(mask, gap_fill_iterations=1, safe_margin_iterations=1)

    assert closed[3, 3]
    assert closed[2, 3]
    assert closed[4, 3]


def test_camera_uses_constant_z_from_full_route_and_xy_speed():
    route = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 5.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    positions, _targets, route_length, frame_count = sample_camera_along_route(
        route,
        camera_height=2.5,
        look_ahead_distance=1.0,
        speed_ms=1.0,
        fps=10,
        resample_spacing=0.1,
    )

    positions_array = np.asarray(positions)
    xy_steps = np.linalg.norm(np.diff(positions_array[:, :2], axis=0), axis=1)
    assert route_length == pytest.approx(3.0)
    assert frame_count == 31
    assert np.allclose(positions_array[:, 2], 7.5)
    assert np.all(xy_steps[:-1] <= 0.1001)


def test_wfc_seed_42_scene_routes_corner_to_corner_with_noise_planning_surface():
    specs = procedural_wfc_specs(list_wfc_terrain_names(default_only=True), tile_size=(8.0, 8.0), seed=42)
    scene, _ = run_wfc_scene_with_procedural_specs((6, 6), specs, tile_size=(8.0, 8.0), seed=42)
    start = scene._tile_offset(0, 0)
    goal = scene._tile_offset(5, 5)

    route = scene.plan_route(start.astype(np.float32), goal.astype(np.float32), height_map_resolution=0.1)

    assert route.shape[0] >= 2
    assert np.allclose(route[0][:2], start[:2], atol=1e-4)
    assert np.allclose(route[-1][:2], goal[:2], atol=1e-4)
