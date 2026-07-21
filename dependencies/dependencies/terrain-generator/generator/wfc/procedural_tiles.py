from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import trimesh

from ..nav_utils import compute_boundary_anchors
from ..terrain_state import transform_terrain_state
from ..terrains.base import ProceduralTerrainResult
from .tiles import MeshTile, Tile
from .wfc import WFCSolver


TerrainGenerator = Callable[[Any], ProceduralTerrainResult]


@dataclass
class ProceduralWFCTileSpec:
    name: str
    terrain_cfg: Any
    generator: TerrainGenerator
    weight: float = 1.0
    rotations: tuple[int, ...] = ()
    flips: tuple[str, ...] = ()
    array_sample_size: int = 5


def _clone_cfg_with_size(cfg: Any, size_xy: tuple[float, float]) -> Any:
    if hasattr(cfg, "size"):
        return replace(cfg, size=tuple(float(v) for v in size_xy))
    return cfg


def _resolve_tile_size(specs: Sequence[ProceduralWFCTileSpec], tile_size: tuple[float, float] | None) -> tuple[float, float]:
    if tile_size is not None:
        return tuple(float(v) for v in tile_size)
    max_x = 0.0
    max_y = 0.0
    for spec in specs:
        size = getattr(spec.terrain_cfg, "size", None)
        if size is None:
            raise ValueError("tile_size must be provided when a terrain config has no `size` field")
        max_x = max(max_x, float(size[0]))
        max_y = max(max_y, float(size[1]))
    return max_x, max_y


def _generate_results_for_tile_size(
    specs: Sequence[ProceduralWFCTileSpec],
    tile_size: tuple[float, float],
) -> list[tuple[ProceduralWFCTileSpec, Any, ProceduralTerrainResult]]:
    generated = []
    for spec in specs:
        cfg = _clone_cfg_with_size(spec.terrain_cfg, tile_size)
        result = spec.generator(cfg)
        generated.append((spec, cfg, result))
    return generated


def _resolve_vertical_interval(results: Sequence[tuple[ProceduralWFCTileSpec, Any, ProceduralTerrainResult]], tile_height: float | None):
    z_min = min(float(result.mesh.bounds[0, 2]) for _, _, result in results)
    z_max = max(float(result.mesh.bounds[1, 2]) for _, _, result in results)
    actual_height = z_max - z_min
    common_height = float(tile_height) if tile_height is not None else float(actual_height)
    if common_height < actual_height - 1e-6:
        raise ValueError("tile_height must be at least as large as the max terrain vertical extent")
    z_shift = -z_min
    return common_height, z_shift


def _canonicalize_procedural_mesh(
    mesh: trimesh.Trimesh,
    tile_size: tuple[float, float],
    z_shift: float,
) -> trimesh.Trimesh:
    canonical = mesh.copy()
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [-tile_size[0] / 2.0, -tile_size[1] / 2.0, z_shift]
    canonical.apply_transform(transform)
    return canonical


def create_procedural_wfc_tiles(
    specs: Sequence[ProceduralWFCTileSpec],
    tile_size: tuple[float, float] | None = None,
    tile_height: float | None = None,
) -> list[MeshTile]:
    tile_size = _resolve_tile_size(specs, tile_size)
    generated = _generate_results_for_tile_size(specs, tile_size)
    common_height, z_shift = _resolve_vertical_interval(generated, tile_height)
    mesh_dim = (tile_size[0], tile_size[1], common_height)

    tiles: list[MeshTile] = []
    for spec, _cfg, result in generated:
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [-tile_size[0] / 2.0, -tile_size[1] / 2.0, z_shift]
        canonical_mesh = _canonicalize_procedural_mesh(result.mesh, tile_size, z_shift)
        tile = MeshTile(
            name=spec.name,
            mesh=canonical_mesh,
            mesh_dim=mesh_dim,
            array_sample_size=spec.array_sample_size,
            weight=spec.weight,
            terrain_state=transform_terrain_state(result.terrain_state, transform),
        )
        tile.rotations = spec.rotations
        tile.flips = spec.flips
        tile.metadata = dict(result.metadata)
        planning_metadata = dict(tile.terrain_state.metadata.get("planning", {}))
        if "boundary_anchors" not in planning_metadata:
            planning_metadata["boundary_anchors"] = compute_boundary_anchors(
                canonical_mesh,
                height_map_resolution=0.1,
                min_traversable_height=-0.2,
                pit_inset_radius=0,
                obstacle_inflation_radius=1,
                terrain_state=tile.terrain_state,
            )
            tile.terrain_state.metadata["planning"] = planning_metadata
        tile.metadata["planning"] = planning_metadata
        tiles.append(tile)
    return tiles


def build_wfc_solver_from_tiles(
    shape: Sequence[int],
    tiles: Sequence[Tile],
    dimensions: int = 2,
    seed: int | None = None,
    observation_mode: str = "weighted",
) -> WFCSolver:
    solver = WFCSolver(tuple(shape), dimensions=dimensions, seed=seed, observation_mode=observation_mode)
    for tile in tiles:
        solver.register_tile(tile.name, tile.edges, weight=tile.weight)
    return solver


def expand_tiles_for_wfc(tiles: Sequence[MeshTile]) -> list[MeshTile]:
    expanded: list[MeshTile] = []
    for tile in tiles:
        rotations = getattr(tile, "rotations", ())
        flips = getattr(tile, "flips", ())
        expanded.extend(tile.get_all_tiles(rotations=rotations, flips=flips))
    return expanded


def run_wfc_with_tiles(
    shape: Sequence[int],
    tiles: Sequence[MeshTile],
    dimensions: int = 2,
    seed: int | None = None,
    observation_mode: str = "weighted",
    init_tiles: Sequence[tuple[str, tuple[int, ...]]] = (),
    max_steps: int = 1000,
) -> tuple[np.ndarray, WFCSolver]:
    expanded_tiles = expand_tiles_for_wfc(tiles)
    solver = build_wfc_solver_from_tiles(shape, expanded_tiles, dimensions=dimensions, seed=seed, observation_mode=observation_mode)
    wave = solver.run(list(init_tiles), max_steps=max_steps)
    return wave, solver


__all__ = [
    "ProceduralWFCTileSpec",
    "build_wfc_solver_from_tiles",
    "create_procedural_wfc_tiles",
    "expand_tiles_for_wfc",
    "run_wfc_with_tiles",
]
