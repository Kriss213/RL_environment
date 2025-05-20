"""
Contains all agent definitions.
"""
from typing import List
from src.Robot import Robot
from src.Classes import Position, Task, PointCloudPublisher
import numpy as np
from matplotlib.path import Path


class Courier(Robot):
    """
    Courier agent. Inherits robot.
    """
    def __init__(self,
                 robot_id: str,
                 start_pos: Position,
                 map,
                 footprint,
                 dist_tolerance:float,
                 heading_tolerance:float,
                 logging:bool=False):
        super().__init__(
            robot_id=robot_id,
            start_pos=start_pos,
            map=map,
            footprint=footprint,
            dist_tolerance=dist_tolerance,
            heading_tolerance=heading_tolerance,
            logging=logging
        )

        # Agent specific attributes
        self.active_task:Task = None
        self.other_couriers:List[Courier] = []

        self.PC_publisher = PointCloudPublisher(self.navigator.node,
                                                topic=f"/{robot_id}/position")
        
        # action metrics/counters
        self._failed_validations_in_row:int = 0
        self._failed_plans_in_row: int = 0
        self._idle_time:float = 0.0
        self._last_action:int = 0
        self._path_plan_cooldown_counter:int = 0
        
        # reward metrics/counters
        self._awarded_reached_goal = False

    def reset(self):
        """Resets Courier agent."""
        self._goal = None
        self.path.clear()
        self.position = Position(*self.init_pos()) # copy
        self.active_task:Task = None
        
        self._failed_validations_in_row:int = 0
        self._failed_plans_in_row: int = 0
        self._idle_time:float = 0.0
        self._last_action:int = 0
        self._path_plan_cooldown_counter:int = 0
    
    def publish_position(self):
        """
        Converts current position to a PointCloud2 message and publishes it.
        """
        # if too close to task goal, do not publish to speed up simulation
        # otherwise it prevents others from planning path to the same goal
        if self.active_task is not None:
            dist = np.linalg.norm(self.position()[:2] - self.active_task.unloader.position()[:2])
            if dist < 1.5:
                return

        # Convert position to PointCloud2 message
        corners = self.get_bbox()
        resolution = 0.1
        z_value = 0.0

        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)

        x_vals = np.arange(min_x, max_x + resolution, resolution)
        y_vals = np.arange(min_y, max_y + resolution, resolution)
        xv, yv = np.meshgrid(x_vals, y_vals)
        xy = np.vstack((xv.flatten(), yv.flatten())).T  # shape (N, 2)

        # Filter: inside quadrilateral
        path = Path(corners)
        mask = path.contains_points(xy)
        inside = xy[mask]

        # Add z
        z = np.full((inside.shape[0], 1), z_value)
        points = np.hstack((inside, z)).astype(np.float32)

        # createa PC2 message and publish
        self.PC_publisher.publish_points(points)

    def validate_path(self, n:int=None):
        """
        Validates the path of the robot.
        Args:
            n (int): Number of points to validate.
        Returns:
            bool: True if path is valid, False otherwise.
        """
        # Path is only invalid if other robot's bounding box
        # intersects path >=3 m away from the goal
        if not self.path:
            self.failed_validations_in_row = 0
            return True
        
        own_dist_to_goal = np.linalg.norm(self.position()[:2] - self.path[-1][:2])
        if own_dist_to_goal < 3.0:
            self.failed_validations_in_row = 0
            return True
        
        n = len(self.path) if n is None else min(n, len(self.path))
        next_path_points = np.array(self.path[:n])[:, :2]

        # if within the next path points there is another courier
        # whose distance to their goal is greater than 3.0
        # and whose bounding box intersects the path, then
        # the path is invalid
        for c in self.other_couriers:
            pos = c.position()[:2]
            # calculate pos distance to all path points
            dist = np.linalg.norm(next_path_points - pos, axis=1)
            # check if any distance is less than tolerance
            if np.any(dist < 1.0):
                # get other courier dist to their goal
                if c.path:
                    dist_to_their_goal = np.linalg.norm(c.position()[:2] - c.path[-1][:2])
                    # if they are close to their goal, do not validate path
                    if dist_to_their_goal < 3.0:
                        continue
                self.failed_validations_in_row += 1
                return False
            
        self.failed_validations_in_row = 0
        return True

    def perform(self, action: int, previous_obs:np.ndarray, dt:float):
        """
        Perform an action.
        """
        assert action in (0, 1), f"[{self.id}] Invalid action: {action}!"
        
        self._last_action = action
        
        # check distance to loader/unloader
        if self.active_task:
            dx_l = self.position.x - self.active_task.loader.position.x
            dy_l = self.position.y - self.active_task.loader.position.y
            dx_ul = self.position.x - self.active_task.unloader.position.x
            dy_ul = self.position.y - self.active_task.unloader.position.y
            dist_to_loader = np.hypot(dx_l, dy_l)
            dist_to_unloader = np.hypot(dx_ul, dy_ul)
        else:
            dist_to_loader = 0.0
            dist_to_unloader = 0.0
        
        # if not close to goal (loader or unloader)
        if dist_to_unloader > 1.0 and dist_to_loader > 1.0:
            # publish own position
            self.publish_position()
        
        # Plan path if necessary (max every 10 steps)
        if self._should_replan_path():
            self._path_plan_cooldown_counter = 0
            # update goal to active task (trigger path plan)
            self.goal = self.active_task.active_goal
            if self.path:
                self._failed_plans_in_row = 0
                
                # when there is new path, clear goal reached reward flag
                self._awarded_reached_goal = False
            else:
                self._failed_plans_in_row += 1
                    
        self._path_plan_cooldown_counter += 1           
        if action == 0:
            self._idle_time += dt
        elif action == 1:
            self._idle_time = 0.0 if self.follow_path() else self._idle_time+dt
                   
    def _should_replan_path(self) -> bool:
        # should replan if no path, has task goal and is not at loader/unloader, and cooldown expired
        is_blocked = not self.validate_path()
        
        if is_blocked:
            # wait for replan
            self.clear_goal()
        if not self.active_task:
            return False
        
        cond_no_path = self.active_task.active_goal and not self.path
        cond_not_loading = not (self.active_task.status in (Task.AT_PICKUP, Task.AT_DROPOFF))
        cond_cooldown_pass = self._path_plan_cooldown_counter >= 10
        return (cond_no_path or is_blocked) and cond_not_loading and cond_cooldown_pass

    
class Loader:
    """
    Loaders spawn tasks.
    """
    def __init__(self, loader_id:str, pos:Position, max_tasks:int, load_time:float, logging:bool):
        """
        Initializes a loader.
        Args:
            id (str): Unique identifier.
            pos (Position): Position of the loader.
        """
        self.__orig_pos = Position(*pos())
        self.id = loader_id
        self.tasks:List[Task] = []
        self.position:Position = pos
        self.task_allocation = None #TODO
        self.task_spawn_chance = 0.01 # each time step
        self.max_tasks = max_tasks
        self.load_time = load_time

    def reset(self):
        """
        Reset loader agent to original state.
        """
        self.tasks.clear()
        self.position = Position(*self.__orig_pos())

    def generate_task(self, id:str, unloader:"Unloader", start_time:float) -> Task:
        """
        Generates a task.
        Args:
            id (str): Unique identifier.
            unloader (Position): Goal position.
        """

        if len(self.tasks) >= self.max_tasks:
            return None
        
        loader = self
        task = Task(id, loader, unloader, start_time=start_time)
        self.tasks.append(task)
        return task
    
class Unloader:
    """
    Unloaders accept task drop offs.
    """
    def __init__(self, unloader_id:str, pos:Position, unload_time:float, logging:bool):
        """
        Initializes an unloader.
        Args:
            id (str): Unique identifier.
            pos (Position): Position of the unloader.
        """
        self.id = unloader_id
        self.__orig_pos = Position(*pos())
        #self.tasks:List[Task] = []
        self.position:Position = pos
        self.unload_time = unload_time

    def reset(self):
        """
        Reset loader agent to original state.
        """
        self.position = Position(*self.__orig_pos())