"""
Virtual Robot class
"""

from typing import Tuple, List, Union, Any

from scipy.spatial.transform import Rotation as R
import numpy as np

from src.Classes import Position, Navigator
from src.Map import Map

from nav_msgs.msg import Path

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

        # failed path plan attempts
        self.failed_path_plan_attempts = 0

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

    def _replan_local_path(self, n:int=50):
        """
        Simulate local path replanning by replacing next n points with new path.
        """
        n = min(n, len(self.path))

        new_goal = self.path[n]

        start_pose = self.navigator.create_pose(self.position.x, self.position.y, self.position.theta)
        goal_pose = self.navigator.create_pose(new_goal[0], new_goal[1], new_goal[2])

        new_local_path:Path = self.navigator.compute_path(start_pose, goal_pose)

        if self.logging:
            print(f"Replanning LOCAL path for robot {self.id} from {self.position} to {new_goal}")

        if not new_local_path:
            return False
        
        new_local_path_xyyaw = []
        for pose in new_local_path.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            #theta = pose.pose.orientation.z
            orientation = R.from_quat([pose.pose.orientation.x,
                                      pose.pose.orientation.y,
                                      pose.pose.orientation.z,
                                      pose.pose.orientation.w])
            theta = orientation.as_euler('xyz')[2]  # yaw
            new_local_path_xyyaw.append((x, y, theta))

        self.path = new_local_path_xyyaw + self.path[n:]

        return True

    def _plan_path(self, goal:Position=None):
        """
        Plans a path for the robot to the goal using A* and stores it in self.path (world coordinates).
        """
        if self.logging:
            print(f"Planning path for robot {self.id} from {self.position} to {self.goal}")
        
        if not goal:
            goal = self.goal

        # Call ROS2 service to get the path
        start_pose = self.navigator.create_pose(self.position.x, self.position.y, self.position.theta)
        goal_pose = self.navigator.create_pose(goal.x, goal.y, goal.theta)
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
           
            self.path.append((x, y, theta))

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
        else:
            # goal is reached
            self.clear_goal()

    def reached_target(self, target:Position) -> bool:       
        heading_error = (target - self.position).theta
        distance_to_goal = np.hypot(*self.position()[:2]-target()[:2])
        
        #print(f"[{self.id}] Heading error: {heading_error},  Distance to goal: {distance_to_goal}")

        return heading_error < self.head_tolerance and distance_to_goal < self.dist_tolerance