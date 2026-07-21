from __future__ import annotations

import numpy as np
import torch

from .utils import sample_interpolated


def _as_float_tensor(value: np.ndarray | torch.Tensor, device: str | torch.device) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    return value.to(device).float()


class SDFArray:
    def __init__(
        self,
        array: np.ndarray | torch.Tensor,
        center: np.ndarray | torch.Tensor,
        resolution: float = 0.1,
        max_value: float = 1000.0,
        device: str | torch.device = "cpu",
    ):
        self.array = _as_float_tensor(array, device)
        self.center = _as_float_tensor(center, device)
        self.resolution = float(resolution)
        self.max_value = float(max_value)
        self.device = torch.device(device)

    def to(self, device: str | torch.device):
        device = torch.device(device)
        self.array = self.array.to(device)
        self.center = self.center.to(device)
        self.device = device
        return self

    def transform(self, transformation: np.ndarray | torch.Tensor):
        transformation = _as_float_tensor(transformation, self.device)
        homogeneous_center = torch.cat([self.center, torch.tensor([1.0], device=self.device)], 0)
        self.center = transformation.matmul(homogeneous_center)[:3]

    def get_sdf(self, points: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        use_torch = isinstance(points, torch.Tensor)
        points = _as_float_tensor(points, self.device)
        point_count = points.shape[0]
        indices = (points - self.center) / self.resolution
        indices += torch.tensor(self.array.shape, device=self.device) // 2

        grid = self.array.reshape(1, 1, *self.array.shape)
        sample_points = indices.reshape(1, point_count, 1, 1, 3)
        sdf = sample_interpolated(grid, sample_points, invalid_value=self.max_value)
        sdf = sdf.reshape(point_count)
        if not use_torch:
            return sdf.cpu().numpy()
        return sdf


class NavDistance:
    def __init__(
        self,
        matrix: np.ndarray | torch.Tensor,
        shape: tuple[int, int],
        center: np.ndarray | torch.Tensor,
        resolution: float = 0.1,
        max_value: float = 1000.0,
        device: str | torch.device = "cpu",
    ):
        self.matrix = _as_float_tensor(matrix, device)
        self.center = _as_float_tensor(center, device)[:2]
        self.shape = tuple(shape)
        self.resolution = float(resolution)
        self.max_value = float(max_value)
        self.device = torch.device(device)

    def to(self, device: str | torch.device):
        device = torch.device(device)
        self.matrix = self.matrix.to(device)
        self.center = self.center.to(device)
        self.device = device
        return self

    def transform(self, transformation: np.ndarray | torch.Tensor):
        transformation = _as_float_tensor(transformation, self.device)
        homogeneous_center = torch.cat([self.center, torch.tensor([0.0, 1.0], device=self.device)], 0)
        self.center = transformation.matmul(homogeneous_center)[:2]

    def get_distance(
        self,
        point: np.ndarray | torch.Tensor,
        goal_pos: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        use_torch = isinstance(point, torch.Tensor)
        point = _as_float_tensor(point, self.device)
        goal_pos = _as_float_tensor(goal_pos, self.device)

        goal_pos = (goal_pos - self.center) / self.resolution
        goal_pos += torch.tensor(self.shape, device=self.device) // 2
        goal_idx = (torch.round(goal_pos[:, 1]) * self.shape[0] + torch.round(goal_pos[:, 0])).long()
        goal_idx = torch.clip(goal_idx, 0, self.shape[0] * self.shape[1] - 1)
        distance_map = self.matrix[goal_idx, :].reshape(-1, self.shape[0], self.shape[1]).transpose(1, 2)

        point = (point - self.center) / self.resolution
        point += torch.tensor(self.shape, device=self.device) // 2

        if point.shape[0] != goal_pos.shape[0]:
            goal_indices = torch.arange(goal_pos.shape[0], device=self.device).unsqueeze(1).repeat(1, point.shape[0])
            point = point.repeat(goal_pos.shape[0], 1)
        else:
            goal_indices = torch.arange(goal_pos.shape[0], device=self.device).unsqueeze(1).repeat(1, point.shape[1])
            point = point.reshape(-1, 2)
        point = torch.cat([goal_indices.reshape(-1, 1), point], dim=-1)

        distance_map = distance_map.reshape(1, 1, *distance_map.shape)
        sample_points = point.reshape(1, -1, 1, 1, 3)
        distances = sample_interpolated(distance_map, sample_points, invalid_value=self.max_value)
        distances = distances.reshape(goal_pos.shape[0], -1)
        if not use_torch:
            return distances.cpu().numpy()
        return distances


__all__ = ["NavDistance", "SDFArray"]