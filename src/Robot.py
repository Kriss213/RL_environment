"""
Virtual Robot class
"""

from typing import Tuple, List, Union, Any

from scipy.spatial.transform import Rotation as R
import numpy as np

from src.Classes import Position, Navigator
from src.Map import Map

from nav_msgs.msg import Path

np.random.seed(42)

class Robot:
    MAX_LIN_VEL = 0.26 # m/s
    MAX_ANG_VEL = 0.35 # rad/s
    def __init__(self,
                 robot_id: str,
                 start_pos: Position,
                 map: Map,
                 footprint: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 dist_tolerance:float,
                 heading_tolerance:float,
                 logging:bool=False,
                ):
        """
        Initializes a robot.

        Args:
            robot_id (str): Unique identifier.
            start_pos (Tuple[float, float]): Starting position.
            footprint (Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]): Robot's footprint.
            map (Map): The map of the environment.
            
        """
        self.init_pos = Position(*start_pos())
        self.id:str = robot_id
        self.position:Position = start_pos
        self.map:Map = map
        self.logging=logging

        
        self.footprint = np.array(footprint)

        self.path:List[Tuple[float, float]] = []
        self._goal = None

        self.planning_map = self.map.planning_map

        # goal tolerance
        self.dist_tolerance:float = dist_tolerance
        self.head_tolerance:float = heading_tolerance

        self.navigator = Navigator(f"navigator_{self.id}", ns=self.id)

    @property
    def goal(self) -> Position:
        """Returns the robot's goal."""
        return self._goal
    
    @goal.setter
    def goal(self, value: Position):
        """Sets the robot's goal."""
        if isinstance(value, Position):
            self._goal = value # set goal only if valid path found
            self._plan_path()                
            if self.path:
                if self.logging:
                    print(f"Path planned for robot {self.id}. Length: {len(self.path)} points")
                # simplify path will need to be tested since
                # ROS2 plans path with a lot of points

                # self.path = self.__simplify_path()
                # print(f"Simplified path for robot {self.id}. Length: {len(self.path)} points")
                # self.path = self.__smooth_path(self.path, num_points=len(self.path))
                # print(f"Smoothed path for robot {self.id}. Length: {len(self.path)} points")
            else:
                self._goal = None
        else:
            raise ValueError("Goal must be an instance of Position class.")

    def get_bbox(self) -> np.ndarray:
        """
        Returns the bounding box of the robot.
        """
        x, y, theta = self.position.x, self.position.y, self.position.theta

        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated = (rot @ self.footprint.T).T + np.array([x, y])
        
        return rotated


    def clear_goal(self):
        self._goal = None
        self.path.clear()

    def move(self, lin_vel:float, ang_vel:float, dt:float):
        """
        Moves the robot using differential drive kinematics.

        Args:
            lin_vel (float): Linear velocity.
            ang_vel (float): Angular velocity.
            dt (float): Time step in seconds.
        """
               
        # Update position
        theta = self.position.theta
        dx = lin_vel * np.cos(theta) * dt
        dy = lin_vel * np.sin(theta) * dt
        self.position.x, self.position.y = (self.position.x + dx, self.position.y + dy)

        # Update orientation (theta adjusts automatically in Position)
        self.position.theta += ang_vel * dt

    def _plan_path(self):
        """
        Plans a path for the robot to the goal using A* and stores it in self.path (world coordinates).
        """
        if self.logging:
            print(f"Planning path for robot {self.id} from {self.position} to {self.goal}")

        # Call ROS2 service to get the path
        start_pose = self.navigator.create_pose(self.position.x, self.position.y, self.position.theta)
        goal_pose = self.navigator.create_pose(self.goal.x, self.goal.y, self.goal.theta)
        path:Path = self.navigator.compute_path(start_pose, goal_pose)

        self.path = []
        if not path:
            return
        
        if self.logging:
            print(f"Path found for robot {self.id}. Length: {len(path.poses)} poses")
        
        for pose in path.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            #theta = pose.pose.orientation.z
            orientation = R.from_quat([pose.pose.orientation.x,
                                      pose.pose.orientation.y,
                                      pose.pose.orientation.z,
                                      pose.pose.orientation.w])
            theta = orientation.as_euler('xyz')[2]  # yaw
            # TODO add theta
            self.path.append((x, y, theta))


        # grid = self.planning_map
        # height, width = grid.shape

        # start = self.position
        # goal = self.goal

        # sx, sy = self.map.world_to_map(start.x, start.y)
        # gx, gy = self.map.world_to_map(goal.x, goal.y)

        # # scale map points
        # scale_factor = self.map.downsample_factor
        # sx //= scale_factor
        # sy //= scale_factor
        # gx //= scale_factor
        # gy //= scale_factor


        # if not (0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height):
        #     self.path = []
        #     if self.logging:
        #         print(f"Path planning failed for robot {self.id}. Start or goal out of bounds.")
        #     return []
        # if grid[gy, gx] != self.map.FREE:
        #     self.path = []
        #     if self.logging:
        #         print(f"Path planning failed for robot {self.id}. Goal blocked.")
        #     return []

        # def h(p1, p2):
        #     #return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])
        #     return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

        # open_set = []
        # heapq.heappush(open_set, (h((sx, sy), (gx, gy)), 0, (sx, sy)))

        # came_from = {}
        # g_score = np.full_like(grid, np.inf, dtype=np.float32)
        # g_score[sy, sx] = 0

        # visited = np.zeros_like(grid, dtype=bool)
        # neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]  # 8-connected

        # while open_set:
        #     _, _, current = heapq.heappop(open_set)
        #     cx, cy = current

        #     if visited[cy, cx]:
        #         continue
        #     visited[cy, cx] = True

        #     if current == (gx, gy):
        #         # Reconstruct path
        #         path = [current]
        #         while current in came_from:
        #             current = came_from[current]
        #             path.append(current)
        #         path.reverse()

        #         # scale map back up and return 
        #         self.path = [self.map.map_to_world(col*scale_factor, row*scale_factor) for col, row in path]
        #         self.orig_path = path.copy()
        #         return  # Found, exit early

        #     for dx, dy in neighbors:
        #         nx, ny = cx + dx, cy + dy
        #         if not (0 <= nx < width and 0 <= ny < height):
        #             continue
        #         if grid[ny, nx] != self.map.FREE:
        #             continue

        #         tentative_g = g_score[cy, cx] + 1
        #         if tentative_g < g_score[ny, nx]:
        #             g_score[ny, nx] = tentative_g
        #             f = tentative_g + h((nx, ny), (gx, gy))
        #             heapq.heappush(open_set, (f, tentative_g, (nx, ny)))
        #             came_from[(nx, ny)] = (cx, cy)
        # if self.logging:
        #     print(f"Path planning failed for robot {self.id}. No valid path found.")
        # self.path = []  # No path found

    def follow_path(self, dt: float, lin_gain: float = 1.0):
        """
        Follows the path step-by-step using proportional control.
        Respects heading and distance tolerances.
        """
        if self.path:
            self.position.x = self.path[0][0]
            self.position.y = self.path[0][1]
            self.position.theta = self.path[0][2]
            self.path.pop(0)
            return
    
    def reached_target(self, target:Position) -> bool:       
        heading_error = (target - self.position).theta
        distance_to_goal = np.hypot(*self.position()[:2]-target()[:2])
        
        #print(f"[{self.id}] Heading error: {heading_error},  Distance to goal: {distance_to_goal}")

        return heading_error < self.head_tolerance and distance_to_goal < self.dist_tolerance