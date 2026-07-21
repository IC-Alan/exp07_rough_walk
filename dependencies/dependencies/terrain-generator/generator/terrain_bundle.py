from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .terrain_state import ensure_terrain_state, terrain_state_to_serializable_dict
from .utils import NpEncoder


def _cfg_to_serializable_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        cfg_dict = asdict(cfg)
    elif isinstance(cfg, dict):
        cfg_dict = dict(cfg)
    else:
        raise ValueError("cfg must be a dataclass or dict")

    for key in ("mesh", "sdf", "distance_matrix", "spawnable_locations"):
        cfg_dict[key] = None
    cfg_dict["terrain_state"] = terrain_state_to_serializable_dict(cfg_dict.get("terrain_state"))
    return cfg_dict


def save_terrain_bundle(
    output_dir: str | Path,
    metadata_name: str,
    cfg: Any,
    sdf_array: np.ndarray,
    sdf_center: np.ndarray,
    sdf_resolution: float,
    sdf_max_value: float,
    distance_matrix: np.ndarray,
    distance_shape: tuple[int, int],
    distance_center: np.ndarray,
    distance_resolution: float,
    distance_max_value: float,
    spawnable_locations: np.ndarray,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_json = json.dumps(_cfg_to_serializable_dict(cfg), cls=NpEncoder)
    metadata_path = output_dir / metadata_name
    np.savez_compressed(
        metadata_path,
        bundle_schema_version=np.array(getattr(cfg, "bundle_schema_version", 1), dtype=np.int32),
        cfg_json=np.array(cfg_json),
        mesh_path=np.array(cfg.mesh_path),
        origin=np.asarray(cfg.origin, dtype=np.float32),
        mesh_dim=np.asarray(cfg.mesh_dim, dtype=np.float32),
        sdf_array=sdf_array,
        sdf_center=sdf_center,
        sdf_resolution=np.array(sdf_resolution, dtype=np.float32),
        sdf_max_value=np.array(sdf_max_value, dtype=np.float32),
        distance_matrix=distance_matrix,
        distance_shape=np.asarray(distance_shape, dtype=np.int32),
        distance_center=distance_center,
        distance_resolution=np.array(distance_resolution, dtype=np.float32),
        distance_max_value=np.array(distance_max_value, dtype=np.float32),
        spawnable_locations=spawnable_locations,
    )
    return metadata_path


def load_terrain_bundle_cfg(bundle_path: str | Path) -> dict[str, Any]:
    data = np.load(bundle_path, allow_pickle=True)
    cfg_json = str(data["cfg_json"].item()) if "cfg_json" in data else "{}"
    cfg_dict = json.loads(cfg_json)

    cfg_dict["bundle_schema_version"] = int(data["bundle_schema_version"].item()) if "bundle_schema_version" in data else 1
    cfg_dict["mesh_path"] = str(data["mesh_path"].item()) if "mesh_path" in data else None
    cfg_dict["mesh"] = None
    cfg_dict["sdf"] = data["sdf_array"] if "sdf_array" in data and data["sdf_array"].size > 0 else None
    cfg_dict["distance_matrix"] = (
        data["distance_matrix"] if "distance_matrix" in data and data["distance_matrix"].size > 0 else None
    )
    cfg_dict["distance_shape"] = (
        tuple(data["distance_shape"].tolist())
        if "distance_shape" in data and data["distance_shape"].size > 0
        else None
    )
    cfg_dict["spawnable_locations"] = (
        data["spawnable_locations"]
        if "spawnable_locations" in data and data["spawnable_locations"].size > 0
        else None
    )
    if "origin" in data:
        cfg_dict["origin"] = tuple(data["origin"].tolist())
    if "mesh_dim" in data:
        cfg_dict["mesh_dim"] = tuple(data["mesh_dim"].tolist())
    if "sdf_center" in data:
        cfg_dict["sdf_center"] = tuple(data["sdf_center"].tolist())
    if "sdf_resolution" in data:
        cfg_dict["sdf_resolution"] = float(data["sdf_resolution"].item())
    if "sdf_max_value" in data:
        cfg_dict["sdf_max_value"] = float(data["sdf_max_value"].item())
    if "distance_center" in data:
        cfg_dict["distance_center"] = tuple(data["distance_center"].tolist())
    cfg_dict["terrain_state"] = ensure_terrain_state(cfg_dict.get("terrain_state"))
    return cfg_dict


__all__ = ["load_terrain_bundle_cfg", "save_terrain_bundle"]