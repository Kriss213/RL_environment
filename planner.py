from typing import Tuple, List
import heapq

class AStarPlanner:
    def __init__(self, grid_map: List[List[int]]):
        self.map = grid_map

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Plans a path from start to goal using A*.

        Returns:
            List[Tuple[int, int]]: Waypoints from start to goal.
        """
        # Simple dummy path
        return [start, goal]
