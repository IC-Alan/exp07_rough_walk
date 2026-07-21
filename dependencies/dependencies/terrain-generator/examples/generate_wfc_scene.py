from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generator import list_wfc_terrain_names, run_wfc_scene_with_procedural_specs
from generator.presets import procedural_wfc_specs
from generator.utils import random_seed


def main():
    output_dir = ROOT / "example_outputs" / "wfc_scene"
    seed = random_seed()
    print(f"Using random seed: {seed}")
    specs = procedural_wfc_specs(list_wfc_terrain_names(default_only=True), tile_size=(8.0, 8.0), seed=seed)
    scene, _solver = run_wfc_scene_with_procedural_specs((4, 4), specs, tile_size=(8.0, 8.0), seed=seed)
    scene.save(output_dir, auto_compute_sdf=False)
    print(f"Saved WFC scene example to {output_dir}")


if __name__ == "__main__":
    main()
