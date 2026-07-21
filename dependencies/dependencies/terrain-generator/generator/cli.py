from __future__ import annotations

import argparse
from pathlib import Path

from .presets import list_registered_terrain_names, list_wfc_terrain_names, procedural_terrain_result, procedural_wfc_specs
from .utils import random_seed
from .wfc import run_wfc_scene_with_procedural_specs


def _save_result(result, output_dir: Path, bundle_mode: str, show: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle_mode == "mesh":
        result.save_mesh(output_dir / "mesh.obj") if hasattr(result, "save_mesh") else result.mesh.export(output_dir / "mesh.obj")
    elif bundle_mode == "light":
        result.save(output_dir, auto_compute_sdf=False)
    elif bundle_mode == "full":
        result.save(output_dir, auto_compute_sdf=True)
    else:
        raise ValueError(f"Unsupported bundle mode: {bundle_mode}")

    if show:
        result.visualize(show=True)


def _ensure_seed(args: argparse.Namespace) -> int:
    if args.seed is None:
        args.seed = random_seed()
        print(f"Using random seed: {args.seed}")
    return int(args.seed)


def _cmd_procedural(args: argparse.Namespace) -> int:
    _ensure_seed(args)
    result, _cfg = procedural_terrain_result(args.terrain, tuple(args.size), seed=args.seed)
    _save_result(result, Path(args.output), args.bundle_mode, args.show)
    return 0


def _cmd_wfc_scene(args: argparse.Namespace) -> int:
    _ensure_seed(args)
    terrain_names = [name.strip() for name in args.terrains.split(",") if name.strip()]
    specs = procedural_wfc_specs(terrain_names, tuple(args.tile_size), seed=args.seed)
    scene, _solver = run_wfc_scene_with_procedural_specs(
        shape=tuple(args.shape),
        specs=specs,
        tile_size=tuple(args.tile_size),
        seed=args.seed,
        max_steps=args.max_steps,
    )
    _save_result(scene, Path(args.output), args.bundle_mode, args.show)
    return 0


def build_parser() -> argparse.ArgumentParser:
    terrain_choices = list_registered_terrain_names()
    default_wfc_terrains = ",".join(list_wfc_terrain_names(default_only=True))

    parser = argparse.ArgumentParser(description="terrain-generator command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    procedural = subparsers.add_parser("procedural", help="Generate one procedural terrain")
    procedural.add_argument("terrain", choices=terrain_choices)
    procedural.add_argument("--output", type=str, required=True)
    procedural.add_argument("--size", type=float, nargs=2, default=(8.0, 8.0))
    procedural.add_argument("--seed", type=int, default=None, help="Random seed; omit to use a fresh random seed")
    procedural.add_argument("--bundle-mode", choices=["mesh", "light", "full"], default="light")
    procedural.add_argument("--show", action="store_true")
    procedural.set_defaults(func=_cmd_procedural)

    wfc_scene = subparsers.add_parser("wfc-scene", help="Generate a WFC-composed scene from procedural terrains")
    wfc_scene.add_argument("--output", type=str, required=True)
    wfc_scene.add_argument("--shape", type=int, nargs=2, default=(4, 4))
    wfc_scene.add_argument("--tile-size", type=float, nargs=2, default=(8.0, 8.0))
    wfc_scene.add_argument("--terrains", type=str, default=default_wfc_terrains)
    wfc_scene.add_argument("--seed", type=int, default=None, help="Random seed; omit to use a fresh random seed")
    wfc_scene.add_argument("--max-steps", type=int, default=1000)
    wfc_scene.add_argument("--bundle-mode", choices=["mesh", "light", "full"], default="light")
    wfc_scene.add_argument("--show", action="store_true")
    wfc_scene.set_defaults(func=_cmd_wfc_scene)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = ["build_parser", "main"]
