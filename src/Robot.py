"""
Virtual Robot class
"""

from typing import Tuple, List, Union, Any

from scipy.spatial.transform import Rotation as R
import numpy as np

from src.Classes import Position
from src.Map import Map
from src.dubins import Dubins

import heapq
from skimage.draw import line, polygon2mask
import matplotlib.pyplot as plt

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
                 turn_radius:float,
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
        self.__orig_planning_map = self.planning_map.copy()

        # goal tolerance
        self.dist_tolerance:float = dist_tolerance
        self.head_tolerance:float = heading_tolerance

        # failed path plan attempts
        self.failed_path_plan_attempts = 0
        
        self.turn_radius:float = turn_radius
        self.planner_dubins = Dubins(self.turn_radius, point_separation=0.2)
                
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
                self.failed_path_plan_attempts = 0
                if self.logging:
                    print(f"Path planned for robot {self.id}. Length: {len(self.path)} points")

            else:
                self.failed_path_plan_attempts += 1
                self._goal = None
        else:
            raise ValueError("Goal must be an instance of Position class.")

    def get_bbox(self, _footrprint:np.ndarray=None) -> np.ndarray:
        """
        Returns the bounding box of the robot.
        """
        x, y, theta = self.position.x, self.position.y, self.position.theta

        if _footrprint is None:
            _footrprint = self.footprint
        
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated = (rot @ _footrprint.T).T + np.array([x, y])
        
        return rotated
    
    def set_obstacle(self, bounding_box:np.ndarray):
        # Convert bounding box to world coords
        bbbox_in_map = []
        scale_factor = self.map.downsample_factor
        for x, y in bounding_box:
            mx, my = self.map.world_to_map(x, y)
            bbbox_in_map.append((my//scale_factor, mx//scale_factor))

        poly = np.array(bbbox_in_map)
        
        shape = self.planning_map.shape
        mask = polygon2mask(shape, poly)
        self.planning_map[mask] = self.map.OCCUPIED
        return mask

    def reset_planning_map(self):
        """
        Remove all non_static obstacles from planning map.
        """
        # set values without copying
        self.planning_map[:,:] = self.__orig_planning_map[:,:]

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

    def _plan_path(self, goal:Position=None):
        """
        Plan path with the following steps:
        1) Plan A* path
        2) Perfomr LOS optimization
        3) Add DUBINS curves (using: ```https://github.com/FelicienC/RRT-Dubins/blob/master/code/dubins.py```)
        4) Calculate yaw for each coordinate pair
        """
        
        if not self.goal:
            if self.logging: print(f"[{self.id}] Planning path failed, since there is no goal")
            return  
        
        start = self.position()
        goal = self.goal()
            
        if self.logging:
            print(f"[{self.id}] Planning path from {self.position} to {self.goal}")
        
        # 1st step - plan a star path to goal
        a_star_path = self._astar()
        if not a_star_path:
            self.path = []
            return
    
        # 2nd step - apply LOS optimization
        LOS_path = self.__simplify_path(a_star_path)
        if not LOS_path:
            self.path = []
            return
            
        # scale back to map coords
        scale_factor = self.map.downsample_factor
        LOS_path_1_scale = [self.map.map_to_world(col*scale_factor, row*scale_factor) for col, row in LOS_path]
        
        # 3rd step - apply dubins
        DUBINS_path = []
        for i, cur_p in enumerate(LOS_path_1_scale):
            if i==0: continue
            prev_p = LOS_path_1_scale[i-1]
            dx = cur_p[0] - prev_p[0]
            dy = cur_p[1] - prev_p[1]
            
            if np.linalg.norm(np.array(cur_p) - np.array(prev_p)) < self.planner_dubins.radius * 1.5:
                continue
            
            if i == 1: start_yaw = start[2]
            else: start_yaw = np.arctan2(dy, dx)
                
            if i < len(LOS_path_1_scale)-1:
                next_p = LOS_path_1_scale[i+1]
                dx = next_p[0] - cur_p[0]
                dy = next_p[1] - cur_p[1]
                end_yaw = np.arctan2(dy, dx)
            else:
                end_yaw = goal[2]
            
            dub_start = (*prev_p, start_yaw)
            dub_end = (*cur_p, end_yaw)
            tmp_dubins = self.planner_dubins.dubins_path(start=dub_start, end=dub_end)
            
            # 4th step - calculate yaw for DUBINS path
            for j, dub_p in enumerate(tmp_dubins):
                if j==0:
                    continue
                if j == len(tmp_dubins)-1:
                    yaw = end_yaw
                else:
                    prev_dub_point = tmp_dubins[j-1]
                    dx = dub_p[0] - prev_dub_point[0]
                    dy = dub_p[1] - prev_dub_point[1]
                    yaw = np.arctan2(dy, dx)
                
                point = (*dub_p, yaw)
                 
                DUBINS_path.append(point)
        
        if DUBINS_path:
            # add end point
            end_p = (self.goal.x, self.goal.y, self.goal.theta)
            DUBINS_path.append(end_p)
        
            self.path = DUBINS_path

    def follow_path(self) -> bool:
        """
        Follows path assuming it has x, y, theta (yaw).
        
        Args:
        :param courier_map: map to check for other couriers to see if following path would cause collision
        
        Return:
        :return moved: True if has path and move is possible
        """
        if self.path:
            self.position.x = self.path[0][0]
            self.position.y = self.path[0][1]
            self.position.theta = self.path[0][2]
            self.path.pop(0)
            return True
        else:
            # goal is reached
            self.clear_goal()
            return False

    def reached_target(self, target:Position) -> bool:       
        heading_error = (target - self.position).theta
        distance_to_goal = np.hypot(*self.position()[:2]-target()[:2])

        return heading_error < self.head_tolerance and distance_to_goal < self.dist_tolerance
    
    def _astar(self) -> List[Tuple[int, int]]:
        """
        Plans a path for the robot to the goal using A*.
        
        :return path: A list of x, y coordinates in map coordinates.
        """
            
        grid = self.planning_map
        height, width = grid.shape

        start = self.position
        goal = self.goal

        sx, sy = self.map.world_to_map(start.x, start.y)
        gx, gy = self.map.world_to_map(goal.x, goal.y)

        # scale map points
        scale_factor = self.map.downsample_factor
        sx //= scale_factor
        sy //= scale_factor
        gx //= scale_factor
        gy //= scale_factor


        if not (0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height):
            if self.logging:
                print(f"Path planning failed for robot {self.id}. Start or goal out of bounds.")
            return []
        if grid[gy, gx] != self.map.FREE:
            if self.logging:
                print(f"Path planning failed for robot {self.id}. Goal blocked.")
            return []

        def h(p1, p2):
            #return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])
            return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

        open_set = []
        heapq.heappush(open_set, (h((sx, sy), (gx, gy)), 0, (sx, sy)))

        came_from = {}
        g_score = np.full_like(grid, np.inf, dtype=np.float32)
        g_score[sy, sx] = 0

        visited = np.zeros_like(grid, dtype=bool)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]  # 8-connected

        while open_set:
            _, _, current = heapq.heappop(open_set)
            cx, cy = current

            if visited[cy, cx]:
                continue
            visited[cy, cx] = True

            if current == (gx, gy):
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                return path
                
                # scale map back up and return found path
                #return [(col*scale_factor, row*scale_factor) for col, row in path]

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if grid[ny, nx] != self.map.FREE:
                    continue

                tentative_g = g_score[cy, cx] + 1
                if tentative_g < g_score[ny, nx]:
                    g_score[ny, nx] = tentative_g
                    f = tentative_g + h((nx, ny), (gx, gy))
                    heapq.heappush(open_set, (f, tentative_g, (nx, ny)))
                    came_from[(nx, ny)] = (cx, cy)
        if self.logging:
            print(f"Path planning failed for robot {self.id}. No valid path found.")
        return []
    
    def __simplify_path(self, path) -> List[Tuple[int, int]]:
        """
        Prunes intermediate points with line-of-sight (LOS) optimization.
        :param path: A list of (x,y) in planning map scale coordinates
        """
        grid = self.planning_map

        if not path:
            return []

        simplified = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                # in grid coords
                p_i = (path[i][0], path[i][1])
                p_j = (path[j][0], path[j][1])
                if self.__is_visible(p_i, p_j, grid):
                    break

                j -= 1
            simplified.append(path[j])
            i = j

        return simplified
    
    def __is_visible(self, p1, p2, grid) -> bool:
        """
        Helper function.
        Checks if line between p1 and p2 is obstacle-free (LOS).
        """
        rr, cc = line(p1[1], p1[0], p2[1], p2[0])
        return np.all(grid[rr, cc] == self.map.FREE)