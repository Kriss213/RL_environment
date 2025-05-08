"""
Contains misc classes.
"""

import numpy as np

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.Agents import Courier, Unloader, Loader



class Position:
    def __init__(self, x: float, y: float, theta: float):
        self.x = x # meters
        self.y = y # meters
        self._theta = None # radians
        self.theta = theta # radians


    @property
    def theta(self) -> float:
        return self._theta
    
    @theta.setter
    def theta(self, value: float):
        # clip theta to [-pi, pi]
        self._theta = (value+np.pi) % (2 * np.pi) - np.pi

    def __sub__(self, other):
        if isinstance(other, Position):
            return Position(
                self.x - other.x,
                self.y - other.y,
                self.theta - other.theta
            )
        return NotImplemented


    def __call__(self, *args, **kwds) -> np.ndarray:
        return np.array([
            self.x,
            self.y,
            self.theta
        ])
    
    def __repr__(self):
        return f"Position(x={self.x:.2f}, y={self.y:.2f}, theta={self.theta:.2f})"
    
class Task:
    PENDING = 0
    ASSIGNED = 1
    AWAITING_PICKUP = 2
    AT_PICKUP = 3
    EN_ROUTE = 4
    AT_DROPOFF = 5
    DELIVERED = 6

    def __init__(self,
                 task_id:str,
                 loader:'Loader',
                 unloader:'Unloader',
                 start_time:float):
        self.id = task_id
        self.loader:'Loader' = loader
        self.unloader:'Unloader' = unloader
        self.courier:'Courier' = None

        self.active_goal: Position = None
        
        self.status:int = self.PENDING
        self.start_time:float = start_time
        self.end_time:float = None

        # track how long task has not been completed
        self.elapsed_time:float = 0.0

        # track load/unload time
        self.elapsed_load_time:float = 0.0
        self.elapsed_unload_time:float = 0.0

    def update_task_time(self, dt:float):
        """
        Update task time
        """
        self.elapsed_time += dt
        
class CBBA:
    """
    A centralized task allocator.
    Manages all loaders, couriers and unloaders.
    """
    def __init__(self, couriers:List['Courier'], loaders:List['Loader'], unloaders:List['Unloader'], dt:float, seed:int, logging:bool=False):
        #super().__init__()
        
        self.loaders = loaders
        self.unloaders = unloaders
        self.couriers = couriers
        
        self.runtime:float = 0.0
        self.dt = dt
        
        self.logging = logging
        
        self.seed = seed
        
        self.delivered_tasks:int = 0
        
        
        if logging:
            print('CBBA task allocator initialized')
    
    def __assign_task(self, courier):
        """
        Temporary simple logic to asign task to robot
        """
        if not courier.active_task is None:
            return
        # shuffle loaders
        np.random.shuffle(self.loaders)
        for l in self.loaders:
            for t in l.tasks:
                if t.status == Task.PENDING:
                    t.status = Task.ASSIGNED
                    t.start_time = self.runtime
                    courier.active_task = t
                    return
                
    def __update_task_status(self): 
        for courier in self.couriers:
            active_task = courier.active_task

            if active_task is None:
                if self.logging:
                    print(f"[{courier.id}] No active task assigned.")
                continue

            task_status = active_task.status
            if self.logging:
                print(f"[{courier.id}] Task {active_task.id} status: {task_status}")

            # ==== TASK EXECUTION LOGIC ====
            if task_status == active_task.ASSIGNED:
                #courier.goal = active_task.loader.position # NOTE DEBUG
                active_task.active_goal = active_task.loader.position
                active_task.status = active_task.AWAITING_PICKUP
                if self.logging:
                    print(f"[{courier.id}] Task {active_task.id} → AWAITING_PICKUP.")# Goal set to loader at {robot.goal}.")

            elif task_status == active_task.AWAITING_PICKUP:
                task_pick_up_pos = active_task.loader.position
                if courier.reached_target(task_pick_up_pos):
                    active_task.status = active_task.AT_PICKUP
                    if self.logging:
                        print(f"[{courier.id}] Arrived at loader for Task {active_task.id}. Status → AT_PICKUP.")
                else:
                    if self.logging:
                        print(f"[{courier.id}] En route to loader... Current position: {courier.position} Target: {task_pick_up_pos}")

            elif task_status == active_task.AT_PICKUP:
                elapsed_load_time = active_task.elapsed_load_time
                load_time = active_task.loader.load_time
                if self.logging:
                    print(f"[{courier.id}] Loading Task {active_task.id}... {elapsed_load_time:.2f}/{load_time:.2f}s")
                if elapsed_load_time >= load_time:
                    active_task.status = active_task.EN_ROUTE
                    unloader_pos = active_task.unloader.position
                    active_task.active_goal = unloader_pos
                    if self.logging:
                        print(f"[{courier.id}] Finished loading Task {active_task.id}. Status → EN_ROUTE.")
                else:
                    active_task.elapsed_load_time += self.dt

            elif task_status == active_task.EN_ROUTE:
                unloader_pos = active_task.unloader.position
                # NOTE DEBUG:
                # if courier.goal != unloader_pos:
                #     courier.goal = unloader_pos
                #     if self.logging:
                #         print(f"[{courier.id}] Set goal to unloader at {unloader_pos} for Task {active_task.id}")

                if courier.reached_target(unloader_pos):
                    active_task.status = active_task.AT_DROPOFF
                    if self.logging:
                        print(f"[{courier.id}] Arrived at unloader. Status → AT_DROPOFF.")

            elif task_status == active_task.AT_DROPOFF:
                elapsed_unload_time = active_task.elapsed_unload_time
                unload_time = active_task.unloader.unload_time
                if self.logging:
                    print(f"[{courier.id}] Unloading Task {active_task.id}... {elapsed_unload_time:.2f}/{unload_time:.2f}s")
                if elapsed_unload_time >= unload_time:
                    active_task.status = active_task.DELIVERED
                    if self.logging:
                        print(f"[{courier.id}] Finished unloading Task {active_task.id}. Status → DELIVERED.")
                else:
                    active_task.elapsed_unload_time += self.dt

            elif task_status == active_task.DELIVERED:
                active_task.end_time = self.runtime
                courier.active_task = None
                active_task.loader.tasks.remove(active_task) # remove task from loader
                self.delivered_tasks += 1
                if self.logging:
                    print(f"[{courier.id}] Task {active_task.id} completed at t={self.runtime:.2f}s.")

            else:
                raise Exception(f"[{courier.id}] Unknown task status: {task_status}")

            # update task execution time
            active_task.elapsed_time += self.dt
            # =========================================================

    def run(self):
        """
        Perform a single timestep for CBBA.
        """

        # generate tasks on loaders
        for loader in self.loaders:
            random_unloader = np.random.choice(self.unloaders)
            if loader.generate_task(
                id=f"T{self.runtime}",
                unloader=random_unloader,
                start_time=self.runtime):
                # ensure only one task can be generated in each time step
                break
        
        # assign task to courier
        for courier in self.couriers:
            self.__assign_task(courier)
        
        # update task status' for all couriers
        self.__update_task_status()
        
        self.runtime += self.dt
        
    def reset(self):
        self.delivered_tasks = 0
        self.runtime = 0.0
        np.random.seed(self.seed) 
        