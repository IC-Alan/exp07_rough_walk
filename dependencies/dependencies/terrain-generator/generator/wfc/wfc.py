from __future__ import annotations

import copy
import os
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..utils import cfg_to_hash

try:
    from alive_progress import alive_bar
except ImportError:
    @contextmanager
    def alive_bar(*_args, **_kwargs):
        def noop(*__args, **__kwargs):
            return None

        yield noop


CACHE_DIR = Path(__file__).resolve().parents[1] / "__cache__"


class Wave:
    def __init__(self, n_tiles: int, shape: list | tuple, dimensions: int = 2):
        self.n_tiles = n_tiles
        self.shape = tuple(shape)
        self.dimensions = dimensions
        self.wave = np.zeros(self.shape, dtype=np.int32)
        self.valid = np.ones((self.n_tiles, *self.shape), dtype=bool)
        self.is_collapsed = np.zeros(self.shape, dtype=bool)
        self.wave_order = np.zeros_like(self.wave)

    def substitute(self, obj: "Wave"):
        self.wave = copy.deepcopy(obj.wave)
        self.valid = copy.deepcopy(obj.valid)
        self.is_collapsed = copy.deepcopy(obj.is_collapsed)
        self.wave_order = copy.deepcopy(obj.wave_order)

    def copy(self):
        new_wave = Wave(self.n_tiles, self.shape, self.dimensions)
        new_wave.wave = copy.deepcopy(self.wave)
        new_wave.valid = copy.deepcopy(self.valid)
        new_wave.is_collapsed = copy.deepcopy(self.is_collapsed)
        new_wave.wave_order = copy.deepcopy(self.wave_order)
        return new_wave


@dataclass
class Direction2D:
    up: tuple = (-1, 0)
    left: tuple = (0, -1)
    down: tuple = (1, 0)
    right: tuple = (0, 1)
    base_directions: tuple = ("up", "left", "down", "right")

    def __post_init__(self):
        self.directions: Dict[int, tuple] = {
            0: self.base_directions,
            90: tuple(np.roll(np.array(self.base_directions), -1)),
            180: tuple(np.roll(np.array(self.base_directions), -2)),
            270: tuple(np.roll(np.array(self.base_directions), -3)),
        }
        self.flipped_directions: Dict[str, tuple] = {
            "x": tuple(np.array(self.base_directions)[[0, 3, 2, 1]]),
            "y": tuple(np.array(self.base_directions)[[2, 1, 0, 3]]),
        }
        self.is_edge_flipped: Dict[str, tuple] = {
            "x": ("up", "down", "left", "right"),
            "y": ("left", "right", "up", "down"),
        }


@dataclass
class Direction3D:
    up: tuple = (0, 0, 1)
    down: tuple = (0, 0, -1)
    front: tuple = (-1, 0, 0)
    left: tuple = (0, -1, 0)
    back: tuple = (1, 0, 0)
    right: tuple = (0, 1, 0)
    base_directions: tuple = ("up", "down", "front", "left", "back", "right")

    def __post_init__(self):
        self.directions: dict = {
            0: self.base_directions,
            90: self.base_directions[:2] + tuple(np.roll(np.array(self.base_directions[2:]), -1)),
            180: self.base_directions[:2] + tuple(np.roll(np.array(self.base_directions[2:]), -2)),
            270: self.base_directions[:2] + tuple(np.roll(np.array(self.base_directions[2:]), -3)),
        }
        self.flipped_directions: dict = {
            "x": tuple(np.array(self.base_directions)[[0, 1, 2, 5, 4, 3]]),
            "y": tuple(np.array(self.base_directions)[[0, 1, 4, 3, 2, 5]]),
            "z": tuple(np.array(self.base_directions)[[1, 0, 2, 3, 4, 5]]),
        }
        self.is_edge_flipped: dict = {
            "x": ("front", "back", "up", "down"),
            "y": ("left", "right", "up", "down"),
            "z": ("left", "right", "front", "back"),
        }


class WFCCore:
    def __init__(
        self,
        n_tiles: int,
        connections: dict,
        shape: list | tuple,
        tile_weights: list | tuple | np.ndarray = (),
        dimensions: int = 2,
        observation_mode: str = "random",
        max_backtracking: int = 10000,
        rng: np.random.Generator | None = None,
    ):
        self.n_tiles = n_tiles
        self.connections = connections
        self.shape = tuple(shape)
        self.dimensions = dimensions
        self.new_idx = None
        self.wave = Wave(n_tiles, self.shape, dimensions)
        self.history = []
        self.tile_weights = np.array(tile_weights) if len(tile_weights) > 0 else np.ones(n_tiles)
        self.back_track_cnt = 0
        self.prev_remaining_grid_num = np.sum(self.wave.is_collapsed == False)
        self.total_back_track_cnt = 0
        self.observation_mode = observation_mode
        self.max_backtracking = max_backtracking
        self.rng = np.random.default_rng() if rng is None else rng

    def _get_neighbours(self, idx):
        neighbours = np.tile(idx, (2 * self.dimensions, 1))
        delta = np.vstack([np.eye(self.dimensions, dtype=int), -np.eye(self.dimensions, dtype=int)])
        neighbours += delta
        neighbours = neighbours[np.all(neighbours >= 0, axis=1) & np.all(neighbours < self.shape, axis=1)]
        directions = neighbours - idx
        return neighbours, directions

    def _get_possible_tiles(self, tile_id, directions):
        return [self.connections[tile_id][tuple(direction)] for direction in directions]

    def _update_wave(self, idx: np.ndarray, tile_id: int):
        self.wave.wave[tuple(idx)] = tile_id
        self.wave.wave_order[tuple(idx)] = self.wave.is_collapsed.sum()
        self.wave.is_collapsed[tuple(idx)] = True
        self.wave.valid[(slice(None),) + tuple(idx)] = False
        self.wave.valid[(tile_id,) + tuple(idx)] = True
        self._update_validity(idx, tile_id)

    def _update_validity(self, new_idx: np.ndarray, tile_id: int):
        neighbours, directions = self._get_neighbours(new_idx)
        possible_tiles = self._get_possible_tiles(tile_id, directions)
        non_possible_tiles = [np.setdiff1d(np.arange(self.n_tiles), allowed) for allowed in possible_tiles]
        for neighbour, tiles in zip(neighbours, non_possible_tiles):
            self.wave.valid[(tiles,) + tuple(neighbour)] = False

    def _random_collapse(self, entropy):
        indices = np.argwhere(entropy == np.min(entropy))
        return indices[int(self.rng.integers(len(indices)))]

    def collapse(self, entropy):
        return self._random_collapse(entropy)

    def init_randomly(self):
        idx = self.rng.integers(0, self.shape, self.dimensions)
        tile_id = int(self.rng.integers(0, self.n_tiles))
        self._update_wave(idx, tile_id)
        self.new_idx = idx

    def init(self, idx=None, tile_id=None):
        if idx is None or tile_id is None:
            self.init_randomly()
        else:
            self._update_wave(idx, tile_id)
            self.new_idx = idx

    def random_observe(self, idx):
        return int(self.rng.choice(np.arange(self.n_tiles)[self.wave.valid[(slice(None),) + tuple(idx)]]))

    def weighted_random_observe(self, idx):
        valid_tiles = np.arange(self.n_tiles)[self.wave.valid[(slice(None),) + tuple(idx)]].astype(int)
        valid_tile_weights = self.tile_weights[valid_tiles]
        return int(self.rng.choice(valid_tiles, p=valid_tile_weights / np.sum(valid_tile_weights)))

    def observe(self, idx):
        if self.observation_mode == "random":
            tile_id = self.random_observe(idx)
        elif self.observation_mode == "weighted":
            tile_id = self.weighted_random_observe(idx)
        else:
            raise NotImplementedError
        self._update_wave(idx, tile_id)

    def solve(self):
        with alive_bar(manual=True) as bar:
            while True:
                entropy = np.sum(self.wave.valid, axis=0)
                entropy[self.wave.is_collapsed] = self.n_tiles + 1
                idx = self.collapse(entropy)
                if entropy[tuple(idx)] == self.n_tiles + 1:
                    break
                if entropy[tuple(idx)] == 0:
                    self._back_track()
                    continue

                remaining = np.sum(self.wave.is_collapsed == False)
                if remaining < self.prev_remaining_grid_num or remaining <= 1:
                    self.prev_remaining_grid_num = remaining
                    self.back_track_cnt = 0
                    self.update_history()
                self.observe(idx)

                if self.dimensions == 2:
                    total = self.wave.shape[0] * self.wave.shape[1]
                else:
                    total = np.prod(self.wave.shape)
                bar(np.sum(self.wave.is_collapsed) / total)
        return self.wave.wave

    def update_history(self):
        self.history.append(self.wave.copy())

    def _back_track(self):
        self.back_track_cnt += 1
        self.total_back_track_cnt += 1
        look_back = max(min(self.back_track_cnt // 10, len(self.history) - 2), 0)
        if self.total_back_track_cnt > self.max_backtracking:
            raise ValueError("Too many total backtracks.", self.total_back_track_cnt)
        if ((look_back + 1) > len(self.history)) or (len(self.history) <= 1):
            self.wave = self.history[0].copy()
            self.history = [self.history[0]]
            self.prev_remaining_grid_num = np.sum(self.wave.is_collapsed == False)
            self.back_track_cnt = 0
        else:
            self.wave = self.history[-1 - look_back].copy()
            self.history = self.history[: -1 - look_back]


class Edge:
    def __init__(self, dimension=2, edge_types: Dict[str, tuple] | None = None):
        self.dimension = dimension
        if dimension == 2:
            self.directions = Direction2D()
        elif dimension == 3:
            self.directions = Direction3D()
        else:
            raise ValueError("Dimension must be 2 or 3.")
        if edge_types is not None and len(edge_types) > 0:
            self.register_edge_types(edge_types)

    def register_edge_types(self, edge_types: Dict[str, tuple]):
        self._check_edge_types(edge_types)
        self.edge_types = edge_types

    def _direction_to_tuple(self, direction):
        return getattr(self.directions, direction)

    def _check_edge_types(self, edge_types):
        for key in self.directions.base_directions:
            if key not in edge_types:
                raise ValueError(f"Edge type {key} is not defined.")

    def get_all_directions_in_tuple(self):
        for direction in self.directions.base_directions:
            yield self._direction_to_tuple(direction)

    def get_tuple_edge_types(self):
        return {self._direction_to_tuple(key): value for key, value in self.edge_types.items()}


class ConnectionManager:
    def __init__(self, dimension=2, load_from_cache=True):
        self.connections = {}
        self.names = []
        self.edge_def = Edge(dimension=dimension)
        self.edges = {}
        self.flipped_edges = {}
        for edge_type in self.edge_def.get_all_directions_in_tuple():
            self.edges[edge_type] = []
            self.flipped_edges[edge_type] = []
        self.edge_types_of_tiles = {}
        self.all_tiles_of_edge_type = {}
        self.dimension = dimension
        self.cache_dir = os.path.join(CACHE_DIR, "connection_cache")
        self.load_from_cache = load_from_cache

    def register_tile(self, name: str, edge_types: dict):
        edges = Edge(edge_types=edge_types, dimension=self.dimension)
        self._register_tile(name, edges.get_tuple_edge_types())

    def _register_tile(self, name: str, edge_types: Dict[tuple, tuple]):
        tile_id = len(self.names)
        self.names.append(name)
        for direction, edge_type in edge_types.items():
            self.edges[direction].append(edge_type)
            self.flipped_edges[direction].append(edge_type[::-1])

        self.edge_types_of_tiles[tile_id] = edge_types
        for direction, edge in edge_types.items():
            if edge not in self.all_tiles_of_edge_type:
                self.all_tiles_of_edge_type[edge] = []
            self.all_tiles_of_edge_type[edge].append((direction, tile_id))

    def get_connection_dict(self):
        return self._load_from_cache()

    def _compute_connection_dict(self):
        for edge_type, edges in self.edges.items():
            self.edges[edge_type] = np.array(edges)
        for edge_type, edges in self.flipped_edges.items():
            self.flipped_edges[edge_type] = np.array(edges)

        connectivities = {}
        for edge_dir, edges in self.edges.items():
            opposite_edge_dir = tuple(-np.array(edge_dir))
            opposite = np.array(self.flipped_edges[opposite_edge_dir])
            connectivity = np.array(np.all((edges[:, None, :] == opposite[None, :, :]), axis=-1))
            connectivities[edge_dir] = connectivity

        connections = {}
        n_tiles = len(self.names)
        for tile_id in range(n_tiles):
            connections[tile_id] = {}
            for edge_dir, connectivity in connectivities.items():
                indices = connectivity[tile_id].nonzero()[0].tolist()
                connections[tile_id][edge_dir] = tuple(sorted(indices))
        return connections

    def _load_from_cache(self):
        code = cfg_to_hash(self.edge_types_of_tiles)
        os.makedirs(self.cache_dir, exist_ok=True)
        filename = os.path.join(self.cache_dir, code + ".pkl")
        if os.path.exists(filename) and self.load_from_cache:
            with open(filename, "rb") as handle:
                connections = pickle.load(handle)
        else:
            connections = self._compute_connection_dict()
            with open(filename, "wb") as handle:
                pickle.dump(connections, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return connections


class WFCSolver:
    def __init__(self, shape, dimensions, seed=None, observation_mode="weighted"):
        self.rng = np.random.default_rng(seed)
        self.cm = ConnectionManager(dimension=dimensions)
        self.shape = shape
        self.dimensions = dimensions
        self.observation_mode = observation_mode
        self.tile_weights = {}

    def register_tile(self, name, edge_types, weight=1):
        self.cm.register_tile(name, edge_types)
        self.tile_weights[name] = weight

    def run(self, init_tiles: List[Tuple[str, Tuple[int, ...]]] = (), max_steps=1000):
        connections = self.cm.get_connection_dict()
        tile_weights = [self.tile_weights[name] for name in self.cm.names]
        self.wfc = WFCCore(
            len(self.cm.names),
            connections,
            self.shape,
            tile_weights=tile_weights,
            dimensions=self.dimensions,
            observation_mode=self.observation_mode,
            max_backtracking=max_steps,
            rng=self.rng,
        )
        if len(init_tiles) > 0:
            for name, index in init_tiles:
                tile_id = self.cm.names.index(name)
                self.wfc.init(index, tile_id)
        else:
            self.wfc.init_randomly()
        return self.wfc.solve()

    @property
    def names(self):
        return self.cm.names

    def get_history(self):
        return self.wfc.history


__all__ = [
    "ConnectionManager",
    "Direction2D",
    "Direction3D",
    "Edge",
    "WFCCore",
    "WFCSolver",
    "Wave",
]
