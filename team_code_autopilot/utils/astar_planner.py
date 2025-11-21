import numpy as np
import heapq
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple


class AStarPlanner:
    def __init__(
        self,
        points: np.ndarray,
        neighbor_radius: float,
        heuristic_weight: float = 1.0,
    ):
        assert points.ndim == 2 and points.shape[1] == 2, "points must be (N,2)"
        self.points = points.astype(np.float32)
        self.N = self.points.shape[0]
        self.neighbor_radius = float(neighbor_radius)
        self.neighbor_radius_sq = self.neighbor_radius ** 2
        self.heuristic_weight = float(heuristic_weight)
        self.adj_list: List[List[int]] = self._build_adjacency()

    def _build_adjacency(self) -> List[List[int]]:
        adj = [[] for _ in range(self.N)]
        for i in range(self.N):
            pi = self.points[i]
            diff = self.points - pi
            dist_sq = np.sum(diff * diff, axis=1)
            mask = (dist_sq > 0.0) & (dist_sq <= self.neighbor_radius_sq)
            neighbors = np.nonzero(mask)[0].tolist()
            adj[i] = neighbors
        return adj

    def _neighbors(self, idx: int) -> List[int]:
        return self.adj_list[idx]

    def _dist(self, i: int, j: int) -> float:
        pi = self.points[i]
        pj = self.points[j]
        return float(np.linalg.norm(pi - pj))

    def _heuristic(self, idx: int, goal_idx: int) -> float:
        return self.heuristic_weight * self._dist(idx, goal_idx)

    def find_closest_node(self, pos: np.ndarray) -> int:
        pos = np.asarray(pos, dtype=np.float32).reshape(1, 2)
        diff = self.points - pos
        dist_sq = np.sum(diff * diff, axis=1)
        return int(np.argmin(dist_sq))

    def plan_indices(
        self,
        start_idx: int,
        goal_idx: int,
    ) -> Optional[List[int]]:
        if start_idx == goal_idx:
            return [start_idx]

        N = self.N
        g_cost = np.full(N, np.inf, dtype=np.float32)
        g_cost[start_idx] = 0.0
        parent = np.full(N, -1, dtype=np.int32)
        open_heap: List[Tuple[float, int]] = []
        import math
        heapq.heappush(open_heap, (self._heuristic(start_idx, goal_idx), start_idx))
        closed = np.zeros(N, dtype=bool)

        while open_heap:
            f_curr, curr = heapq.heappop(open_heap)
            if closed[curr]:
                continue
            if curr == goal_idx:
                return self._reconstruct_path(parent, start_idx, goal_idx)
            closed[curr] = True
            for nb in self._neighbors(curr):
                if closed[nb]:
                    continue
                tentative_g = g_cost[curr] + self._dist(curr, nb)
                if tentative_g < g_cost[nb]:
                    g_cost[nb] = tentative_g
                    parent[nb] = curr
                    f_nb = tentative_g + self._heuristic(nb, goal_idx)
                    heapq.heappush(open_heap, (f_nb, nb))
        return None

    def _reconstruct_path(
        self, parent: np.ndarray, start_idx: int, goal_idx: int
    ) -> List[int]:
        path = [goal_idx]
        curr = goal_idx
        while curr != start_idx:
            curr = int(parent[curr])
            if curr < 0:
                raise RuntimeError("Path reconstruction failed: broken parent chain")
            path.append(curr)
        path.reverse()
        return path

    def plan(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        snap_to_closest: bool = True,
    ) -> Optional[np.ndarray]:
        if snap_to_closest:
            start_idx = self.find_closest_node(start_xy)
            goal_idx = self.find_closest_node(goal_xy)
        else:
            raise NotImplementedError(
                "Non-snapping mode not implemented; provide node indices instead."
            )

        idx_path = self.plan_indices(start_idx, goal_idx)
        if idx_path is None:
            return None

        return self.points[idx_path, :]