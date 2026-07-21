from __future__ import annotations

import trimesh

from ..mesh_utils import merge_meshes
from ..utils import get_cached_mesh_gen
from ..wfc.tiles import MeshTile
from .basic_parts import (
    create_box_mesh,
    create_capsule_mesh,
    create_floor,
    create_from_height_map,
    create_platform_mesh,
    create_procedural_mesh,
    create_wall_mesh,
)
from .indoor_parts import create_stairs_mesh
from .overhanging_parts import create_floating_boxes, create_overhanging_boxes, generate_wall_from_array
from .mesh_parts_cfg import (
    BoxMeshPartsCfg,
    CapsuleMeshPartsCfg,
    CombinedMeshPartsCfg,
    FloatingBoxesPartsCfg,
    HeightMapMeshPartsCfg,
    MeshPartsCfg,
    MeshPattern,
    OverhangingBoxesPartsCfg,
    PlatformMeshPartsCfg,
    ProceduralMeshPartsCfg,
    StairMeshPartsCfg,
    WallMeshPartsCfg,
    WallPartsCfg,
)


def get_mesh_gen(cfg: MeshPartsCfg):
    if isinstance(cfg, WallPartsCfg):
        return create_wall_mesh
    if isinstance(cfg, StairMeshPartsCfg):
        return create_stairs_mesh
    if isinstance(cfg, PlatformMeshPartsCfg):
        return create_platform_mesh
    if isinstance(cfg, HeightMapMeshPartsCfg):
        return create_from_height_map
    if isinstance(cfg, CapsuleMeshPartsCfg):
        return create_capsule_mesh
    if isinstance(cfg, BoxMeshPartsCfg):
        return create_box_mesh
    if isinstance(cfg, WallMeshPartsCfg):
        return generate_wall_from_array
    if isinstance(cfg, OverhangingBoxesPartsCfg):
        return create_overhanging_boxes
    if isinstance(cfg, FloatingBoxesPartsCfg):
        return create_floating_boxes
    if isinstance(cfg, ProceduralMeshPartsCfg):
        return create_procedural_mesh
    if isinstance(cfg, CombinedMeshPartsCfg):
        mesh_gens = [get_mesh_gen(sub_cfg) for sub_cfg in cfg.cfgs]

        def mesh_gen(current_cfg: CombinedMeshPartsCfg):
            mesh = trimesh.Trimesh()
            for idx, gen in enumerate(mesh_gens):
                new_mesh = gen(current_cfg.cfgs[idx], mesh=mesh)
                mesh = merge_meshes([mesh, new_mesh], False)
            return mesh

        return mesh_gen
    raise NotImplementedError(f"Mesh generator for {cfg} not implemented")


def create_mesh_tile(cfg: MeshPartsCfg) -> MeshTile:
    mesh_gen = get_mesh_gen(cfg)
    cached_mesh_gen = get_cached_mesh_gen(mesh_gen, cfg, verbose=False, use_cache=cfg.load_from_cache)
    mesh = cached_mesh_gen()
    if cfg.edge_array is None:
        from ..mesh_utils import get_height_array_of_mesh

        cfg.edge_array = get_height_array_of_mesh(mesh, cfg.dim, 5)

    if cfg.use_generator:
        tile = MeshTile(cfg.name, cached_mesh_gen, array=cfg.edge_array, mesh_dim=cfg.dim, weight=cfg.weight)
    else:
        tile = MeshTile(cfg.name, mesh, array=cfg.edge_array, mesh_dim=cfg.dim, weight=cfg.weight)
    tile.rotations = cfg.rotations
    tile.flips = cfg.flips
    return tile


def create_mesh_pattern(cfg: MeshPattern) -> dict[str, MeshTile]:
    tiles = []
    for mesh_cfg in cfg.mesh_parts:
        tile = create_mesh_tile(mesh_cfg)
        tiles.extend(tile.get_all_tiles(rotations=mesh_cfg.rotations, flips=mesh_cfg.flips))
    return {tile.name: tile for tile in tiles}