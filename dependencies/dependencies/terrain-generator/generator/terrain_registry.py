from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .terrains import (
    DoorTerrainCfg,
    ForestTerrainCfg,
    PileTerrainCfg,
    PlatformGapTerrainCfg,
    PyramidStairsTerrainCfg,
    StakesTerrainCfg,
    generate_door_terrain,
    generate_forest_terrain,
    generate_pile_terrain,
    generate_platform_gap_terrain,
    generate_pyramid_stairs_terrain,
    generate_stakes_terrain,
)
from .utils import random_seed
from .wfc import ProceduralWFCTileSpec


TerrainConfigFactory = Callable[[tuple[float, float], int | None], Any]
TerrainGenerator = Callable[[Any], Any]


@dataclass(frozen=True)
class TerrainRegistryEntry:
    name: str
    config_factory: TerrainConfigFactory
    generator: TerrainGenerator
    wfc_rotations: tuple[int, ...] = ()
    wfc_flips: tuple[str, ...] = ()
    wfc_enabled: bool = True
    default_wfc: bool = False
    weight: float = 1.0


_TERRAIN_REGISTRY: dict[str, TerrainRegistryEntry] = {
    "door": TerrainRegistryEntry(
        name="door",
        config_factory=lambda size, seed: DoorTerrainCfg(size=size),
        generator=generate_door_terrain,
        wfc_rotations=(90, 180, 270),
        wfc_flips=("x", "y"),
        default_wfc=True,
    ),
    "forest": TerrainRegistryEntry(
        name="forest",
        config_factory=lambda size, seed: ForestTerrainCfg(size=size, seed=seed),
        generator=generate_forest_terrain,
        wfc_rotations=(90, 180, 270),
        wfc_flips=("x", "y"),
        default_wfc=True,
    ),
    "platform_gap": TerrainRegistryEntry(
        name="platform_gap",
        config_factory=lambda size, seed: PlatformGapTerrainCfg(size=size, seed=seed),
        generator=generate_platform_gap_terrain,
        wfc_rotations=(90, 180, 270),
        wfc_flips=("x", "y"),
        default_wfc=True,
    ),
    "pile": TerrainRegistryEntry(
        name="pile",
        config_factory=lambda size, seed: PileTerrainCfg(size=size, route_mode="cross_route", route_line_count=3),
        generator=generate_pile_terrain,
        wfc_rotations=(90, 180, 270),
        wfc_flips=("x", "y"),
        default_wfc=True,
    ),
    "pyramid_stairs": TerrainRegistryEntry(
        name="pyramid_stairs",
        config_factory=lambda size, seed: PyramidStairsTerrainCfg(size=size, seed=seed),
        generator=generate_pyramid_stairs_terrain,
        wfc_rotations=(90, 180, 270),
        wfc_flips=("x", "y"),
        default_wfc=True,
    ),
    "stakes": TerrainRegistryEntry(
        name="stakes",
        config_factory=lambda size, seed: StakesTerrainCfg(size=size, seed=seed),
        generator=generate_stakes_terrain,
        wfc_rotations=(90, 180, 270),
        wfc_flips=("x", "y"),
        default_wfc=True,
    ),
}


def get_terrain_registry() -> dict[str, TerrainRegistryEntry]:
    return dict(_TERRAIN_REGISTRY)


def get_terrain_entry(name: str) -> TerrainRegistryEntry:
    key = name.lower()
    if key not in _TERRAIN_REGISTRY:
        supported = ", ".join(sorted(_TERRAIN_REGISTRY))
        raise ValueError(f"Unsupported terrain name: {name}. Supported terrains: {supported}")
    return _TERRAIN_REGISTRY[key]


def list_registered_terrain_names() -> tuple[str, ...]:
    return tuple(_TERRAIN_REGISTRY.keys())


def list_wfc_terrain_names(default_only: bool = False) -> tuple[str, ...]:
    return tuple(
        entry.name
        for entry in _TERRAIN_REGISTRY.values()
        if entry.wfc_enabled and (entry.default_wfc or not default_only)
    )


def _resolve_seed(seed: int | None) -> int:
    return random_seed() if seed is None else int(seed)


def create_terrain_result(terrain_name: str, size: tuple[float, float], seed: int | None = None):
    entry = get_terrain_entry(terrain_name)
    cfg = entry.config_factory(tuple(float(value) for value in size), _resolve_seed(seed))
    return entry.generator(cfg), cfg


def create_wfc_specs(
    terrain_names: Sequence[str],
    tile_size: tuple[float, float],
    seed: int | None = None,
) -> list[ProceduralWFCTileSpec]:
    specs: list[ProceduralWFCTileSpec] = []
    normalized_size = tuple(float(value) for value in tile_size)
    base_seed = _resolve_seed(seed)
    for index, terrain_name in enumerate(terrain_names):
        entry = get_terrain_entry(terrain_name)
        if not entry.wfc_enabled:
            raise ValueError(f"Terrain `{entry.name}` is not enabled for WFC presets")
        cfg = entry.config_factory(normalized_size, base_seed + index)
        specs.append(
            ProceduralWFCTileSpec(
                name=entry.name,
                terrain_cfg=cfg,
                generator=entry.generator,
                weight=entry.weight,
                rotations=entry.wfc_rotations,
                flips=entry.wfc_flips,
            )
        )
    return specs


__all__ = [
    "TerrainRegistryEntry",
    "create_terrain_result",
    "create_wfc_specs",
    "get_terrain_entry",
    "get_terrain_registry",
    "list_registered_terrain_names",
    "list_wfc_terrain_names",
]
