from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import trimesh

from ..mesh_utils import get_heights_from_mesh

try:
    from alive_progress import alive_bar
except ImportError:
    @contextmanager
    def alive_bar(*_args, **_kwargs):
        def noop(*__args, **__kwargs):
            return None

        yield noop


@contextmanager
def _preserve_numpy_random_state():
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


class LSystem:
    def __init__(self, axiom, rules, angle_adjustment=22.5, num_branches=2, rng: random.Random | None = None):
        self.axiom = axiom
        self.rules = rules
        self.angle_adjustment = angle_adjustment
        self.num_branches = num_branches
        self.rng = random.Random() if rng is None else rng

    def generate(self, iterations):
        state = self.axiom
        for _ in range(iterations):
            new_state = ""
            for char in state:
                if char in self.rules:
                    rule = self.rules[char]
                    if "{" in rule:
                        directions = self.rng.sample(["+", "-", "^", "&", "\\", "/"], self.num_branches + 2)
                        rule = rule.format(*([directions[0], directions[1]] + [f"X{{{i}}}" for i in range(2, self.num_branches + 2)]))
                    else:
                        rule = rule.format(self.angle_adjustment)
                    new_state += rule
                else:
                    new_state += char
            state = new_state
        return state


def generate_tree_mesh(
    num_branches: int = 2,
    iterations: int = 3,
    angle_adjustment: float = 22.5,
    cylinder_sections: int = 8,
    rng: random.Random | None = None,
):
    rules = {
        "F": "FF",
        "X": "F[+X]F[-X]{0}" + "".join([f"[{{0}}+X{{{i}}}][{{0}}-X{{{i}}}]" for i in range(2, 2 + num_branches - 2)]),
        "+": "+{0}",
        "-": "-{0}",
    }
    lsys = LSystem("X", rules, angle_adjustment=angle_adjustment, num_branches=num_branches, rng=rng)
    state = lsys.generate(iterations)

    position = np.array([0, 0, 0], dtype=np.float32)
    direction = np.array([0, 0, 1], dtype=np.float32)
    rot_mat = np.eye(4)
    stack = []
    step_size = 0.1
    angle = 20.7 * (np.pi / 180.0)
    cylinders = []

    for char in state:
        if char == "F":
            endpoint = position + step_size * direction
            distance = np.linalg.norm(endpoint)
            radius = max(0.005, min(0.04 - 0.02 * distance, 0.03))
            cylinder = trimesh.creation.cylinder(radius=radius, height=step_size, sections=cylinder_sections)
            center = (position + endpoint) / 2.0
            cylinder.apply_transform(rot_mat)
            cylinder.apply_translation(center)
            cylinders.append(cylinder)
            position = endpoint
        elif char in ["+", "-", "&", "^", "\\", "/"]:
            axis_map = {
                "+": ([1, 0, 0], angle),
                "-": ([1, 0, 0], -angle),
                "&": ([0, 1, 0], angle),
                "^": ([0, 1, 0], -angle),
                "\\": ([0, 0, 1], angle),
                "/": ([0, 0, 1], -angle),
            }
            axis, angle_value = axis_map[char]
            rot_matrix = trimesh.transformations.rotation_matrix(angle_value, axis)
            direction = np.dot(rot_matrix[:3, :3], direction)
            direction /= np.linalg.norm(direction)
            rot_mat = np.dot(rot_matrix, rot_mat)
        elif char == "[":
            stack.append((position.copy(), direction.copy(), rot_mat.copy()))
        elif char == "]":
            position, direction, rot_mat = stack.pop()

    with _preserve_numpy_random_state():
        tree = trimesh.util.concatenate(cylinders) if cylinders else trimesh.Trimesh()
        if len(tree.vertices) > 0:
            tree = trimesh.smoothing.filter_humphrey(tree)
        tree.apply_scale(10.0)
        return tree


def add_trees_on_terrain(
    terrain_mesh: trimesh.Trimesh,
    num_trees: int = 10,
    tree_scale_range: Tuple[float, float] = (0.5, 1.5),
    tree_deg_range: Tuple[float, float] = (-30.0, 30.0),
    tree_cylinder_sections: int = 6,
    seed: int | None = None,
):
    with _preserve_numpy_random_state():
        np_rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        bbox = terrain_mesh.bounding_box.bounds
        tree_meshes = []
        positions = np.zeros((num_trees, 3), dtype=np.float32)
        positions[:, 0] = np_rng.uniform(bbox[0][0], bbox[1][0], size=(num_trees,))
        positions[:, 1] = np_rng.uniform(bbox[0][1], bbox[1][1], size=(num_trees,))
        positions[:, 2] = get_heights_from_mesh(terrain_mesh, positions[:, :2])

        tree_rad_range = (tree_deg_range[0] * np.pi / 180.0, tree_deg_range[1] * np.pi / 180.0)
        with alive_bar(num_trees, dual_line=True, title="tree generation") as bar:
            for idx in range(num_trees):
                num_branches = int(np_rng.integers(2, 4))
                tree_mesh = generate_tree_mesh(num_branches=num_branches, cylinder_sections=tree_cylinder_sections, rng=py_rng)
                tree_mesh.apply_scale(np_rng.uniform(*tree_scale_range))
                pose = np.eye(4)
                pose[:3, 3] = positions[idx]
                q = trimesh.transformations.quaternion_from_euler(
                    np_rng.uniform(*tree_rad_range),
                    np_rng.uniform(*tree_rad_range),
                    np_rng.uniform(0, 2 * np.pi),
                )
                pose[:3, :3] = trimesh.transformations.quaternion_matrix(q)[:3, :3]
                tree_mesh.apply_transform(pose)
                tree_meshes.append(tree_mesh)
                bar()

        return trimesh.util.concatenate(tree_meshes) if tree_meshes else trimesh.Trimesh()


__all__ = ["LSystem", "add_trees_on_terrain", "generate_tree_mesh"]
