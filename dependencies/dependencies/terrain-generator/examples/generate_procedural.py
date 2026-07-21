from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generator import PileTerrainCfg, generate_pile_terrain


def main():
    output_dir = ROOT / "example_outputs" / "procedural_pile"
    result = generate_pile_terrain(PileTerrainCfg(size=(8.0, 8.0), route_mode="cross_route", route_line_count=3))
    result.save(output_dir, auto_compute_sdf=False)
    print(f"Saved procedural example to {output_dir}")


if __name__ == "__main__":
    main()