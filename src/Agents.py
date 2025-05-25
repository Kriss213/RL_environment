"""
Contains all agent definitions.
"""
from typing import List
from src.Robot import Robot
from src.Classes import Position, Task
import numpy as np
from typing import Any
from collections import deque
from shapely import Polygon

class Courier(Robot):
    POSITION_NOISE = 0.05 # +- m
    HEADING_NOISE = np.deg2rad(5.0) # +- 5 degrees
    REPLAN_CHANCE = lambda x: np.random.rand() < 0.7 # 70 % chance of path replan
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
                 turn_radius:float,
                 logging:bool=False):
        super().__init__(
            robot_id=robot_id,
            start_pos=start_pos,
            map=map,
            footprint=footprint,
            dist_tolerance=dist_tolerance,
            heading_tolerance=heading_tolerance,
            turn_radius=turn_radius,
            logging=logging
        )

        # Agent specific attributes
        self.active_task:Task = None
        self.other_couriers:List[Courier] = []

        # action metrics/counters
        self._failed_plans_in_row: int = 0
        self._idle_time:float = 0.0
        self._last_actions:deque= deque([0], maxlen=20) # 1 init to avoid div by zero when using len() in init stages
        self._path_plan_cooldown_counter:int = 0
        
        # reward metrics/counters
        self._awarded_reached_goal = False
        self.replanned_this_step = False
        self.was_blocked_last_step = False
        self.packages_delivered: int = 0
        
        # sign doesnt matter
        footprint_dx, footprint_dy = np.min(self.footprint), np.max(self.footprint)
        radius = np.hypot(footprint_dx, footprint_dy)
        footprint_inflation_d = np.cos(np.deg2rad(45.0)) * radius
        self._inflated_footprint = self.footprint + np.where(self.footprint>0, +footprint_inflation_d, -footprint_inflation_d)
                
        self._other_courier_map:np.ndarray = np.ones_like(self.planning_map) * self.map.FREE # A scaled binary map containint only other couriers
        self.d_lim_path_blocked:float = 2.0 # distance to check for blocked path
        

    def reset(self, pos:Position|Any=None):
        """Resets Courier agent."""
        self._goal = None
        self.path.clear()
        self.position = Position(*self.init_pos()) if pos is None else pos # copy
        self.active_task:Task = None
        
        self._failed_plans_in_row: int = 0
        self._idle_time:float = 0.0
        self._last_actions:deque= deque([0], maxlen=20)# 1 init to avoid div by zero when using len() in init stages
        self._path_plan_cooldown_counter:int = 0
        self.reset_planning_map()
        
    def _is_path_blocked(self):# -> Tuple[bool, str]:
        """
        Check if in given distance another courier blocks the path.
        """
        if not self.path:
            return False
        
        # Get path points in the given distance
        path = np.array(self.path)
        path_xy = path[:, :2]
        path_deltas = np.diff(path_xy, axis=0)
        segment_lengths = np.linalg.norm(path_deltas, axis=1)
        total_len = np.sum(segment_lengths)
        if total_len < self.d_lim_path_blocked:
            n = len(path)
        else:
            cum_dist = np.cumsum(segment_lengths)
            n = np.searchsorted(cum_dist, self.d_lim_path_blocked, side='right')
            
        path_to_check = path[:n]
        
        #construct own polygon list
        poly_list = [Polygon(self.get_bbox(xytheta=(x,y,theta))) for (x,y,theta) in path_to_check]
        
        for target_courier in self.other_couriers:
            other_courier_poly = Polygon(target_courier.get_bbox())
            # if any future configuration intersects with other robot
            # return true
            for p in poly_list:
                if other_courier_poly.intersects(p):
                    return True
        return False

    def perform(self, action: int, dt:float):
        """
        Perform an action.
        """
        assert action in (0, 1, 2), f"[{self.id}] Invalid action: {action}!"
        
        self._last_actions.append(action)
        
        # later set to true if replanned
        self.replanned_this_step = False
        
        # add random noise to agents
        self._add_noise()
        
        # Make sure that planning map is updated
        self.update_planning_map()
        
        # set flag if agent was blocked before performing action
        self.was_blocked_last_step = self._is_path_blocked()
        
        if action == 0:
            self._idle_time += dt
        elif action == 1:
            self._idle_time = 0.0 if self.follow_path() else self._idle_time+dt
        elif action == 2:
            # Plan path if necessary (max every 10 follow path actions)
            #self._path_plan_cooldown_counter += 1
            #if self._should_replan_path() and self.REPLAN_CHANCE():
            #self._path_plan_cooldown_counter = 0    
                            
            # update goal to active task (trigger path plan)
            if self.active_task:
                self.goal = self.active_task.active_goal
                self.replanned_this_step = True
                        
            if self.path:
                self._failed_plans_in_row = 0    
                # when there is new path, clear goal reached reward flag
                self._awarded_reached_goal = False
            else:
                self._failed_plans_in_row += 1
                
                   
    # def _should_replan_path(self) -> bool:
    #     is_blocked = self.was_blocked_last_step
    #     if is_blocked:
    #         # wait for replan
    #         self.clear_goal()
    #     if not self.active_task:
    #         return False
        
    #     cond_no_path = self.active_task.active_goal and not self.path
    #     cond_not_loading = not (self.active_task.status in (Task.AT_PICKUP, Task.AT_DROPOFF))
    #     cond_cooldown_pass = self._path_plan_cooldown_counter >= 10 or self._path_plan_cooldown_counter == 0
    #     return (cond_no_path or is_blocked) and cond_not_loading and cond_cooldown_pass

    def set_obstacle(self, bounding_box):
        mask = super().set_obstacle(bounding_box)
        # add obstacle to only courier map
        self._other_courier_map[mask] = self.map.OCCUPIED
    
    def reset_planning_map(self):
        self._other_courier_map[:,:] = self.map.FREE
        
        return super().reset_planning_map()
    
    def update_planning_map(self):
        self.reset_planning_map()
        # add couriers as obstacles but only if closer than 3 meters
        for c in self.other_couriers:
            _diff = c.position - self.position
            dx, dy = _diff.x, _diff.y
            heading_err = _diff.theta
            dist_to_other = np.hypot(dx, dy)
            # check distance to self
            #dist = np.hypot(*(self.position()[:2] - c.position()[:2]))
            if heading_err > 0 or dist_to_other < 5.0:
                bbox = c.get_bbox(_footrprint=self._inflated_footprint)
                self.set_obstacle(bbox)
    
    def _add_noise(self):
        """
        Add noise to robot's properties
        """
        self.position.x +=  (2*np.random.rand()-1.0) * self.POSITION_NOISE
        self.position.y += (2*np.random.rand()-1.0) * self.POSITION_NOISE
        self.position.theta +=  (2*np.random.rand()-1.0) * self.HEADING_NOISE
        
    
    
    
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