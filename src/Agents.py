"""
Contains all agent definitions.
"""
from typing import List
from src.Robot import Robot
from src.Classes import Position, Task

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
        # TODO CBBA attributes



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