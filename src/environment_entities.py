import heapq
from typing import Optional, Tuple, List, Union, Any

from scipy.interpolate import CubicSpline
import numpy as np
from skimage.draw import line

from src.classes import Position
from src.map import Map


class Robot:
    MAX_LIN_VEL = 0.26 # m/s
    MAX_ANG_VEL = 0.35 # rad/s
    def __init__(self,
                 robot_id: str,
                 start_pos: Position,
                 map: Map,
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
        self.map:Map = map
        
        self.velocity:float = velocity
        self.footprint = np.array(footprint)

        self.path:List[Tuple[float, float]] = []
        self._goal = None

        self.map.inflate_map(self.footprint)

    @property
    def goal(self) -> Position:
        """Returns the robot's goal."""
        return self._goal
    
    @goal.setter
    def goal(self, value: Position):
        """Sets the robot's goal."""
        if isinstance(value, Position):
            self._goal = value
            self.__plan_path()
            if self.path:
                print(f"Path planned for robot {self.id}. Length: {len(self.path)} points")
                # simplify path will need to be tested since
                # ROS2 plans path with a lot of points

                # self.path = self.__simplify_path()
                # print(f"Simplified path for robot {self.id}. Length: {len(self.path)} points")
                # self.path = self.__smooth_path(self.path, num_points=len(self.path))
                # print(f"Smoothed path for robot {self.id}. Length: {len(self.path)} points")
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
               
        # Update position
        theta = self.position.theta
        dx = lin_vel * np.cos(theta) * dt
        dy = lin_vel * np.sin(theta) * dt
        self.position.x, self.position.y = (self.position.x + dx, self.position.y + dy)

        # Update orientation
        self.position.theta += ang_vel * dt
        # keep between [-pi, pi]
        self.position.theta = (self.position.theta + np.pi) % (2 * np.pi) - np.pi

    def __plan_path(self):
        """
        Plans a path for the robot to the goal using A* and stores it in self.path (world coordinates).
        """
        print(f"Planning path for robot {self.id} from {self.position} to {self.goal}")
        grid = self.map.inflated_map
        height, width = self.map.height, self.map.width

        start = self.position
        goal = self.goal

        sx, sy = self.map.world_to_map(start.x, start.y)
        gx, gy = self.map.world_to_map(goal.x, goal.y)

        if not (0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height):
            self.path = None
            print(f"Path planning failed for robot {self.id}. Start or goal out of bounds.")
            return
        if grid[gy, gx] != self.map.FREE:
            #grid[sy, sx] != self.map.FREE or
            self.path = None
            print(f"Path planning failed for robot {self.id}. Goal blocked.")
            return

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

                self.path = [self.map.map_to_world(col, row) for col, row in path]
                return  # Found, exit early

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

        print(f"Path planning failed for robot {self.id}. No valid path found.")
        self.path = None  # No path found

    def __is_visible(self, p1, p2, grid) -> bool:
            """
            Helper function.
            Checks if line between p1 and p2 is obstacle-free (LOS).
            """
            rr, cc = line(p1[1], p1[0], p2[1], p2[0])
            return np.all(grid[rr, cc] == self.map.FREE)
    
    def __simplify_path(self, max_cut_per_jump:int = 50) -> List[Tuple[int, int]]:
        """
        Prunes intermediate points with line-of-sight optimization.
        """
       
        path = self.path
        grid = self.map.inflated_map

        if not path:
            return []

        simplified = [path[0]]
        i = 0
        while i < len(path) - 1:
            # max_j = min(i + max_cut_per_jump + 1, len(path) - 1)
            # j = max_j
            j = len(path) - 1
            while j > i + 1:
                # in grid coords
                p_i = self.map.world_to_map(path[i][0], path[i][1])
                p_j = self.map.world_to_map(path[j][0], path[j][1])
                if self.__is_visible(p_i, p_j, grid):
                    break
               
                j -= 1
            simplified.append(path[j])
            i = j

        return simplified

    def __smooth_path(self, path: List[Tuple[float, float]], num_points=100):
        """
        Smooths a path using cubic spline interpolation.

        Returns:
            List of (x, y) points.
        """
        if len(path) < 3:
            return path

        path = np.array(path)
        dist = np.cumsum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])))
        dist = np.insert(dist, 0, 0)

        cs_x = CubicSpline(dist, path[:, 0])
        cs_y = CubicSpline(dist, path[:, 1])

        distances = np.linspace(0, dist[-1], num_points)
        smoothed = list(zip(cs_x(distances), cs_y(distances)))
        return smoothed

    def follow_path(self, dt: float, lin_gain: float = 1.0, ang_gain: float = 3.0, goal_tolerance: float = 0.1, heading_tolerance: float = 0.05):
        """
        Follows a given path using proportional control.

        Args:
            path (List[Tuple[float, float]]): List of (x, y) waypoints in world coordinates.
            dt (float): Time step in seconds.
            lin_gain (float): Linear velocity gain.
            ang_gain (float): Angular velocity gain.
            goal_tolerance (float): Distance threshold to consider goal reached.
            heading_tolerance (float): Angle threshold to consider heading aligned.
        """
        path = self.path
           
        if self.goal is None:
            return
        
        heading_error = self.goal.theta - self.position.theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # after reaching goal point rotate to correct orientation
        if abs(heading_error) > heading_tolerance+1e-2 and not path:
            # rotate closer to goal pos
            lin_vel = 0.0
            ang_vel = ang_gain* heading_error
            self.move(lin_vel, ang_vel, dt)
            # Update heading error
            heading_error = self.goal.theta - self.position.theta
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        if not path:
            return

        # Track next point
        target_x, target_y = path[0]
        dx = target_x - self.position.x
        dy = target_y - self.position.y
        distance = np.hypot(dx, dy)

        # Check if waypoint reached
        if distance < goal_tolerance:
            heading_error = self.goal.theta - self.position.theta
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

            path.pop(0)
            return

        # Compute angle to target and heading error
        target_theta = np.arctan2(dy, dx)
        heading_error = target_theta - self.position.theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

        # Control law
        lin_vel = lin_gain * distance
        ang_vel = ang_gain * heading_error

        # Limit speeds if needed
        lin_vel = np.clip(lin_vel, -self.MAX_LIN_VEL, self.MAX_LIN_VEL)

        # Move robot
        self.move(lin_vel, ang_vel, dt)

    
    # def get_next_path_points(self, n: int) -> List[Tuple[float, float]]:
    #     """Returns next N points from path."""
    #     return self.path[:n]

    # def reached_goal(self) -> bool:
    #     """Returns True if robot has reached its goal."""
    #     return self.goal is not None and self.position == self.goal
    
class Loader:
    """
    Spawns tasks and 
    """
    pass

class Unloader:
    pass

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