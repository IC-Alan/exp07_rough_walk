from __future__ import annotations

from typing import Sequence

from .terrain_registry import create_terrain_result, create_wfc_specs, list_registered_terrain_names, list_wfc_terrain_names
from .wfc import ProceduralWFCTileSpec


def procedural_terrain_result(terrain_name: str, size: tuple[float, float], seed: int | None = None):
    return create_terrain_result(terrain_name, size=size, seed=seed)


def procedural_wfc_specs(
    terrain_names: Sequence[str],
    tile_size: tuple[float, float],
    seed: int | None = None,
) -> list[ProceduralWFCTileSpec]:
    return create_wfc_specs(terrain_names, tile_size=tile_size, seed=seed)


__all__ = [
    "list_registered_terrain_names",
    "list_wfc_terrain_names",
    "procedural_terrain_result",
    "procedural_wfc_specs",
]
