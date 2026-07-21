from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import trimesh

from ..mesh_utils import flip_mesh, get_flip_transform, get_height_array_of_mesh, get_yaw_rotation_transform, yaw_rotate_mesh
from ..terrain_state import TerrainState, ensure_terrain_state, transform_terrain_state
from .wfc import Direction2D, Direction3D


class Tile:
    def __init__(
        self,
        name: str,
        edges: Dict[str, tuple | str],
        dimension: int = 2,
        weight: float = 1.0,
        terrain_state: TerrainState | dict | None = None,
    ):
        self.name = name
        self.dimension = dimension
        self.edges = edges
        self.weight = weight
        self.terrain_state = ensure_terrain_state(terrain_state)
        self.directions = Direction2D() if dimension == 2 else Direction3D()

    def get_dict_tile(self):
        return self.name, self.edges, self.weight

    def get_flipped_tile(self, direction):
        if direction not in ["x", "y", "z"]:
            raise ValueError(f"Direction {direction} is not defined.")
        new_name = f"{self.name}_{direction}"
        new_edges = {}
        for key, value in self.edges.items():
            new_key = self.directions.flipped_directions[direction][self.directions.base_directions.index(key)]
            if key in self.directions.is_edge_flipped[direction]:
                new_edges[new_key] = value[::-1]
            else:
                new_edges[new_key] = value
        return self._copy_metadata_to(Tile(
            name=new_name,
            edges=new_edges,
            dimension=self.dimension,
            weight=self.weight,
            terrain_state=transform_terrain_state(self.terrain_state, get_flip_transform(direction)),
        ))

    def get_rotated_tile(self, deg):
        if deg not in self.directions.directions:
            raise ValueError(f"Rotation degree {deg} is not defined.")
        new_name = f"{self.name}_{deg}"
        basic_directions = self.directions.directions[0]
        new_directions = self.directions.directions[deg]
        new_edges = {new_key: self.edges[key] for new_key, key in zip(new_directions, basic_directions)}
        return self._copy_metadata_to(Tile(
            name=new_name,
            edges=new_edges,
            dimension=self.dimension,
            weight=self.weight,
            terrain_state=transform_terrain_state(self.terrain_state, get_yaw_rotation_transform(deg)),
        ))

    def get_all_tiles(self, rotations=(), flips=()):
        tiles = [self]
        for rotation in rotations:
            tiles.append(self.get_rotated_tile(rotation))
        for flip_direction in flips:
            tiles.append(self.get_flipped_tile(flip_direction))
            for rotation in rotations:
                tiles.append(self.get_flipped_tile(flip_direction).get_rotated_tile(rotation))
        return tiles

    def _copy_metadata_to(self, other: "Tile"):
        metadata = dict(getattr(self, "metadata", {}))
        planning = other.terrain_state.metadata.get("planning")
        if isinstance(planning, dict):
            metadata["planning"] = dict(planning)
        other.metadata = metadata
        return other

    def __str__(self):
        return f"Tile {self.name} with edges {self.edges}, weight {self.weight}"


class ArrayTile(Tile):
    def __init__(
        self,
        name: str,
        array: np.ndarray,
        edges: Optional[Dict[str, tuple | str]] = None,
        dimension: int = 2,
        weight: float = 1.0,
        terrain_state: TerrainState | dict | None = None,
    ):
        self.array = array
        self.directions = Direction2D()
        if edges is None:
            edges = self.create_edges_from_array(array)
        super().__init__(name, edges, dimension, weight, terrain_state=terrain_state)

    def get_array(self, name=None):
        if name is None:
            return self.array
        for deg in (90, 180, 270):
            if name == f"{self.name}_{deg}":
                return np.rot90(self.array, deg // 90)
        return self.array

    def get_flipped_tile(self, direction):
        if direction == "x":
            array = np.flip(self.array, 1)
        elif direction == "y":
            array = np.flip(self.array, 0)
        else:
            raise ValueError(f"Direction {direction} is not defined.")
        tile = super().get_flipped_tile(direction)
        return self._copy_metadata_to(ArrayTile(
            name=tile.name,
            array=array,
            edges=tile.edges,
            dimension=self.dimension,
            weight=tile.weight,
            terrain_state=tile.terrain_state,
        ))

    def get_rotated_tile(self, deg):
        if deg not in self.directions.directions:
            raise ValueError(f"Rotation degree {deg} is not defined.")
        array = np.rot90(self.array, deg // 90)
        tile = super().get_rotated_tile(deg)
        return self._copy_metadata_to(ArrayTile(
            name=tile.name,
            array=array,
            edges=tile.edges,
            dimension=self.dimension,
            weight=tile.weight,
            terrain_state=tile.terrain_state,
        ))

    def create_edges_from_array(self, array):
        edges = {}
        for direction in self.directions.base_directions:
            if direction == "up":
                edges[direction] = tuple(np.round(array[0, :], 1))
            elif direction == "down":
                edges[direction] = tuple(np.round(array[-1, :][::-1], 1))
            elif direction == "left":
                edges[direction] = tuple(np.round(array[:, 0][::-1], 1))
            elif direction == "right":
                edges[direction] = tuple(np.round(array[:, -1], 1))
            else:
                raise ValueError(f"Direction {direction} is not defined.")
        return edges

    def __str__(self):
        return super().__str__() + f"\n {self.array}"


class MeshTile(ArrayTile):
    def __init__(
        self,
        name: str,
        mesh: Union[trimesh.Trimesh, Callable[[], trimesh.Trimesh]],
        array: Optional[np.ndarray] = None,
        edges: Optional[Dict[str, tuple | str]] = None,
        mesh_dim: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        array_sample_size: int = 5,
        dimension: int = 2,
        weight: float = 1.0,
        terrain_state: TerrainState | dict | None = None,
    ):
        self.mesh_gen = lambda: mesh() if callable(mesh) else mesh
        self.mesh_dim = mesh_dim
        self.array_sample_size = array_sample_size
        if array is None:
            array = get_height_array_of_mesh(self.mesh_gen(), mesh_dim, array_sample_size)
        super().__init__(name, array, edges, dimension, weight=weight, terrain_state=terrain_state)

    def get_flipped_tile(self, direction):
        if direction == "x":
            mesh_gen = lambda: flip_mesh(self.mesh_gen(), "x")
        elif direction == "y":
            mesh_gen = lambda: flip_mesh(self.mesh_gen(), "y")
        else:
            raise ValueError(f"Direction {direction} is not defined.")
        tile = super().get_flipped_tile(direction)
        return self._copy_metadata_to(MeshTile(
            name=tile.name,
            array=tile.array,
            mesh=mesh_gen,
            edges=tile.edges,
            mesh_dim=self.mesh_dim,
            array_sample_size=self.array_sample_size,
            dimension=self.dimension,
            weight=self.weight,
            terrain_state=tile.terrain_state,
        ))

    def get_rotated_tile(self, deg):
        if deg not in self.directions.directions:
            raise ValueError(f"Rotation degree {deg} is not defined.")
        mesh_gen = lambda: yaw_rotate_mesh(self.mesh_gen(), deg)
        tile = super().get_rotated_tile(deg)
        return self._copy_metadata_to(MeshTile(
            name=tile.name,
            array=tile.array,
            mesh=mesh_gen,
            edges=tile.edges,
            mesh_dim=self.mesh_dim,
            array_sample_size=self.array_sample_size,
            dimension=self.dimension,
            weight=self.weight,
            terrain_state=tile.terrain_state,
        ))

    def get_mesh(self):
        return self.mesh_gen()

    def __str__(self):
        return "MeshTile: " + super().__str__()


__all__ = ["ArrayTile", "MeshTile", "Tile"]
