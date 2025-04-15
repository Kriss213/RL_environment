from typing import Tuple, List, Union, Any

from classes import Position
from map import Map
import numpy as np


class Robot:
    MAX_LIN_VEL = 0.26 # m/s
    MAX_ANG_VEL = 0.35 # rad/s
    def __init__(self,
                 robot_id: str,
                 start_pos: Position,
                 #map: Map,
                 footprint: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                 task_allocation: Union[Any] = None, #TODO
                 velocity: float = 1.0, # m/s # TODO
                ):
        """
        Initializes a robot.

        Args:
            robot_id (str): Unique identifier.
            start_pos (Tuple[float, float]): Starting position.
            footprint (Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]): Robot's footprint.
            map (Map): The map of the environment.
            task_allocation (Union[Any], optional): Task allocation. Defaults to None. TODO
            
        """
        self.id:str = robot_id
        self.position:Position = start_pos

        self.velocity:float = velocity
        self.footprint = np.array(footprint)

        
        #self.goal = None
        #self.path: List[Tuple[float, float]] = []

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


    def reset(self):
        """Resets robot's path and goal."""
        self.goal = None
        self.path = []

    def move(self, lin_vel:float, ang_vel:float, dt:float):
        """
        Moves the robot using differential drive kinematics.

        Args:
            lin_vel (float): Linear velocity.
            ang_vel (float): Angular velocity.
            dt (float): Time step in seconds.
        """
        # if lin_dir not in [-1, 1] or rot_dir not in [-1, 1]:
        #     raise ValueError("lin_dir and rot_dir must be either -1 or 1.")
        
        # Update position
        theta = self.position.theta
        dx = lin_vel * np.cos(theta) * dt
        dy = lin_vel * np.sin(theta) * dt
        self.position.x, self.position.y = (self.position.x + dx, self.position.y + dy)

        # Update orientation
        self.position.theta += ang_vel * dt
        # keep between [-pi, pi]
        self.position.theta = (self.position.theta + np.pi) % (2 * np.pi) - np.pi

    def get_next_path_points(self, n: int) -> List[Tuple[float, float]]:
        """Returns next N points from path."""
        return self.path[:n]

    def reached_goal(self) -> bool:
        """Returns True if robot has reached its goal."""
        return self.goal is not None and self.position == self.goal
    


# test code
if __name__ == "__main__":
    robot = Robot("robot1", Position(10, 13, np.deg2rad(37)), [[0.81, 0.46], [-0.31, 0.46], [-0.31, -0.46], [0.81, -0.46]])

    print(f"Robot ID: {robot.id}")
    x, y, theta = robot.position.x, robot.position.y, robot.position.theta
    print(f"Initial Position: {x}, {y}, {theta}")
    bbox = robot.get_bbox()
    print(f"Footprint: {bbox}")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    poly = Polygon(bbox, closed=True, color='blue', alpha=0.5)
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.add_patch(poly)
    ax.set_aspect('equal') #prevent skewed view
    ax.plot(x, y, 'ro')  # robot position point
    plt.title("Robot Bounding Box")
    plt.show()