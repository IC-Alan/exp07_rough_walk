from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation


CACHE_DIR = Path(__file__).resolve().parent / "__cache__"


def random_seed() -> int:
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, trimesh.Trimesh):
            return None
        if callable(obj):
            return f"{obj.__module__}:{getattr(obj, '__qualname__', obj.__name__)}"
        return super().default(obj)


def cfg_to_hash(cfg: Any, exclude_keys: tuple[str, ...] = ("weight", "load_from_cache")) -> str:
    def tuple_to_str(data: dict[str, Any]) -> dict[str, Any]:
        new_data: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                value = tuple_to_str(value)
            new_key = str(key) if isinstance(key, tuple) else key
            new_data[new_key] = value
        return new_data

    if isinstance(cfg, dict):
        cfg_dict = copy.deepcopy(cfg)
    elif is_dataclass(cfg):
        cfg_dict = asdict(cfg)
    else:
        raise ValueError("cfg must be a dict or dataclass")

    for key in exclude_keys:
        cfg_dict.pop(key, None)

    encoded = json.dumps(tuple_to_str(cfg_dict), sort_keys=True, cls=NpEncoder).encode()
    digest = hashlib.md5()
    digest.update(encoded)
    return digest.hexdigest()


def _points_in_bounds(indices: torch.Tensor, grid_shape: tuple[int, ...]) -> torch.Tensor:
    valid = torch.ones(indices.shape[:-1], dtype=torch.bool, device=indices.device)
    for dim, limit in enumerate(grid_shape):
        valid = torch.logical_and(valid, indices[..., dim] >= 0)
        valid = torch.logical_and(valid, indices[..., dim] <= limit - 1)
    return valid


def sample_interpolated(
    grid: np.ndarray | torch.Tensor,
    indices: np.ndarray | torch.Tensor,
    padding_mode: str = "zeros",
    invalid_value: float = 0.0,
    no_grad: bool = True,
) -> np.ndarray | torch.Tensor:
    use_torch = isinstance(grid, torch.Tensor)

    if isinstance(grid, np.ndarray):
        grid = torch.from_numpy(grid)
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)

    grid = grid.float()
    indices = indices.float()

    grid_shape = grid.shape
    old_indices = indices.clone()

    if len(grid_shape) == 4:
        indices[..., 0] = old_indices[..., 1] / (grid_shape[-1] - 1) * 2 - 1
        indices[..., 1] = old_indices[..., 0] / (grid_shape[-2] - 1) * 2 - 1
        spatial_shape = (grid_shape[-2], grid_shape[-1])
    elif len(grid_shape) == 5:
        indices[..., 0] = old_indices[..., 2] / (grid_shape[-1] - 1) * 2 - 1
        indices[..., 1] = old_indices[..., 1] / (grid_shape[-2] - 1) * 2 - 1
        indices[..., 2] = old_indices[..., 0] / (grid_shape[-3] - 1) * 2 - 1
        spatial_shape = (grid_shape[-3], grid_shape[-2], grid_shape[-1])
    else:
        raise ValueError(f"Unsupported grid rank: {len(grid_shape)}")

    valid = _points_in_bounds(old_indices, spatial_shape)

    if no_grad:
        with torch.no_grad():
            values = F.grid_sample(grid, indices, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    else:
        values = F.grid_sample(grid, indices, mode="bilinear", padding_mode=padding_mode, align_corners=True)

    values = torch.where(valid.unsqueeze(1), values, torch.full_like(values, invalid_value))

    if not use_torch:
        return values.cpu().numpy()
    return values


def get_cached_mesh_gen(
    mesh_gen_fn: Callable[[Any], trimesh.Trimesh],
    cfg: Any,
    verbose: bool = False,
    use_cache: bool = True,
) -> Callable[[], trimesh.Trimesh]:
    code = cfg_to_hash(cfg)
    mesh_cache_dir = CACHE_DIR / "mesh_cache"
    mesh_cache_dir.mkdir(parents=True, exist_ok=True)
    name = getattr(cfg, "name", "mesh")
    mesh_name = f"{name}_{code}.obj"

    def mesh_gen() -> trimesh.Trimesh:
        mesh_path = mesh_cache_dir / mesh_name
        if mesh_path.exists() and use_cache:
            if verbose:
                print(f"Loading mesh {name} from cache {mesh_name} ...")
            return trimesh.load_mesh(mesh_path)

        if use_cache and verbose:
            print(f"{name} does not exist in cache, creating {mesh_name} ...")
        mesh = mesh_gen_fn(cfg)
        mesh.export(mesh_path)
        return mesh

    return mesh_gen


def euler_angles_to_rotation_matrix(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    roll = np.expand_dims(roll, axis=-1)
    pitch = np.expand_dims(pitch, axis=-1)
    yaw = np.expand_dims(yaw, axis=-1)
    rotation = Rotation.from_euler("xyz", np.concatenate([roll, pitch, yaw], axis=-1), degrees=False)
    return rotation.as_matrix()
