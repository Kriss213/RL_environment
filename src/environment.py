"""
A simulated warehouse enviornment
"""

import heapq
from typing import List, Tuple, Dict, Optional, Any, Union
import gym
import numpy as np
from shapely.geometry import Polygon, Point

import yaml
from PIL import Image
from pathlib import Path


from src.environment_entities import Robot, Loader, Unloader, CBBA
from src.map import Map
from src.classes import Position

import pygame

np.random.seed(42)


class Environment(gym.Env):

    def __init__(self, map:Map,
                 robots:List[Robot],
                 loaders:List[Loader],
                 unloaders:List[Unloader],
                 visualize:bool=False,
                 logging:bool=False):
        """
        Initializes the simulation environment.

        """
        super().__init__()

        self.logging:bool = logging

        self.map:Map = map
        self.robots:List[Robot] = robots
        self.loaders:List[Loader] = loaders
        self.unloaders:List[Unloader] = unloaders

        
        scr_bg = self.__init_pygame() if visualize else None
        self.__screen = scr_bg[0] if visualize else None
        self.__bg_img = scr_bg[1] if visualize else None
        self.__visualize:bool = visualize
        self.runtime:float = 0.0

        self.TA:CBBA = CBBA()

    def _check_robot_collisions(self) -> List[Tuple[int, int]]:
        """
        Checks collisions between robots.

        Returns:
            List of pairs of robot indices that are colliding.
        """
        collisions = []
        id_and_polygons = [(robot.id, Polygon(robot.get_bbox()) ) for robot in self.robots]

        for i, (rob_id_1, poly1) in enumerate(id_and_polygons):
            for j, (rob_id_2, poly2) in enumerate(id_and_polygons):
                if i >= j:
                    continue
                if poly1.intersects(poly2):
                    collisions.append((rob_id_1, rob_id_2))
        return collisions 
    
    def visualize(self):
        """
        Visualizes the environment and the robots.
        """
        if not self.__visualize:
            return
        
        # draw background
        self.__screen.blit(self.__bg_img, (0, 0))
        
        # add robot to screen
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)

        # font init
        font = pygame.font.SysFont('Comic Sans MS', 24)

        for loader in self.loaders:
            # visualize loaders
            loader_id_surface = font.render(loader.id, True, RED)
            loader_pos_screen = self.map.world_to_map(*loader.position()[:2])
            self.__screen.blit(loader_id_surface, loader_pos_screen)

        for unloader in self.unloaders:
            # visualize unloaders
            unloader_id_surface = font.render(unloader.id, True, GREEN)
            unloader_pos_screen = self.map.world_to_map(*unloader.position()[:2])
            self.__screen.blit(unloader_id_surface, unloader_pos_screen)

        for robot in self.robots:
            # =====VISUALIZE ROBOT=====
            # get robot bounding box
            bbox = robot.get_bbox() # this is in meters
            # convert to pixel coords
            bbox_map = []
            for point in bbox:
                # convert to map coordinates
                mx, my = self.map.world_to_map(*point[:2])
                bbox_map.append((mx, my))
            # robot pos in pixel coords
            robot_pos = self.map.world_to_map(*robot.position()[:2])
            # draw the robot on the screen
            pygame.draw.polygon(self.__screen, BLUE, bbox_map)
            pygame.draw.circle(self.__screen, RED, robot_pos[:2], 5)
            # add robot name
            text_surface = font.render(robot.id, True, RED)
            # place label in middle of robot box
            self.__screen.blit(text_surface, np.sum(bbox_map, axis=0) // 4)

            # =====VISUALIZE PATH=====
            if robot.path:# is not None and len(robot.path) > 0:
                path_map = [self.map.world_to_map(*robot.position()[:2])]
                for point in robot.path:
                    # convert to map coordinates
                    mx, my = self.map.world_to_map(*point[:2])
                    path_map.append((mx, my))
                # draw the path on the screen
                pygame.draw.lines(self.__screen, GREEN, False, path_map, 2)

        # update display loop
        for event in pygame.event.get():
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
                pixel_pos = pygame.mouse.get_pos()
                world_pos = self.map.map_to_world(*pixel_pos)

                pos = Position(*world_pos, theta=0.0)
                # DEBUG
                # send robot 1 to pos
                self.robots[0].goal = pos

                print(f"Clicked pixel: {pixel_pos}")
                print(f"Clicked world: {world_pos}")
                print(f"Pixel value at clicked: {self.map()[pixel_pos[1], pixel_pos[0]]}")
        pygame.display.update()
  
    def __init_pygame(self) -> pygame.Surface:
        """
        Initializes the Pygame screen and returns it.
        """
        pygame.init()
        h, w = self.map.height, self.map.width
        #res = self.map.resolution
        
        # screen size must match map resolution (h,w)
        screen = pygame.display.set_mode((w,h))
        pygame.display.set_caption("Environment")

        # Load the map image
        map_image = Image.fromarray(self.map()).convert('RGB')

        # set the screen to the map image
        img = pygame.image.fromstring(map_image.tobytes(), map_image.size, map_image.mode)
        screen.blit(img, (0, 0))


        return screen, img#, surface
    

    # def reset(self) -> List[Dict[str, Any]]:
    #     """
    #     Resets the environment to its initial state.

    #     Returns:
    #         List[Dict[str, Any]]: Initial observations for each robot.
    #     """
    #     for robot in self.robots:
    #         robot.reset()
    #     return self._get_obs()

    def step(self, dt:float=0.1):#, actions: List[Tuple[float, float]]) -> Tuple[List[Dict[str, Any]], List[float], bool, Dict[str, Any]]:
        """
        Advances the simulation by one time step using the given actions.

        Args:
            action - goal position

        Returns:
        """

        # generate tasks if needed
        for loader in self.loaders:
            random_unloader = np.random.choice(self.unloaders)
            if loader.generate_task(
                id=f"T{self.runtime}",
                unloader=random_unloader,
                start_time=self.runtime):
                # ensure only one task can be generated in each time step
                break


        for robot in self.robots:
            # assign task if possible
            self.TA.assign_task(robot, self.loaders, self.runtime)

            active_task = robot.active_task

            if active_task is None:
                if self.logging:
                    print(f"[{robot.id}] No active task assigned.")
                continue

            task_status = active_task.status
            if self.logging:
                print(f"[{robot.id}] Task {active_task.id} status: {task_status}")

            # ==== TASK EXECUTION LOGIC ====
            if task_status == active_task.ASSIGNED:
                robot.goal = active_task.loader.position
                active_task.status = active_task.AWAITING_PICKUP
                if self.logging:
                    print(f"[{robot.id}] Task {active_task.id} → AWAITING_PICKUP. Goal set to loader at {robot.goal}.")

            elif task_status == active_task.AWAITING_PICKUP:
                task_pick_up_pos = active_task.loader.position
                if robot.reached_target(task_pick_up_pos):
                    active_task.status = active_task.AT_PICKUP
                    if self.logging:
                        print(f"[{robot.id}] Arrived at loader for Task {active_task.id}. Status → AT_PICKUP.")
                else:
                    if self.logging:
                        print(f"[{robot.id}] En route to loader... Current position: {robot.position} Target: {task_pick_up_pos}")

            elif task_status == active_task.AT_PICKUP:
                elapsed_load_time = active_task.elapsed_load_time
                load_time = active_task.loader.load_time
                if self.logging:
                    print(f"[{robot.id}] Loading Task {active_task.id}... {elapsed_load_time:.2f}/{load_time:.2f}s")
                if elapsed_load_time >= load_time:
                    active_task.status = active_task.EN_ROUTE
                    if self.logging:
                        print(f"[{robot.id}] Finished loading Task {active_task.id}. Status → EN_ROUTE.")
                else:
                    active_task.elapsed_load_time += dt

            elif task_status == active_task.EN_ROUTE:
                unloader_pos = active_task.unloader.position
                if robot.goal != unloader_pos:
                    robot.goal = unloader_pos
                    if self.logging:
                        print(f"[{robot.id}] Set goal to unloader at {unloader_pos} for Task {active_task.id}")

                if robot.reached_target(unloader_pos):
                    active_task.status = active_task.AT_DROPOFF
                    if self.logging:
                        print(f"[{robot.id}] Arrived at unloader. Status → AT_DROPOFF.")

            elif task_status == active_task.AT_DROPOFF:
                elapsed_unload_time = active_task.elapsed_unload_time
                unload_time = active_task.unloader.unload_time
                if self.logging:
                    print(f"[{robot.id}] Unloading Task {active_task.id}... {elapsed_unload_time:.2f}/{unload_time:.2f}s")
                if elapsed_unload_time >= unload_time:
                    active_task.status = active_task.DELIVERED
                    if self.logging:
                        print(f"[{robot.id}] Finished unloading Task {active_task.id}. Status → DELIVERED.")
                else:
                    active_task.elapsed_unload_time += dt

            elif task_status == active_task.DELIVERED:
                active_task.end_time = self.runtime
                robot.active_task = None
                active_task.loader.tasks.remove(active_task) # remove task from loader
                if self.logging:
                    print(f"[{robot.id}] Task {active_task.id} completed at t={self.runtime:.2f}s.")

            else:
                raise Exception(f"[{robot.id}] Unknown task status: {task_status}")

            # update task execution time
            active_task.elapsed_time += dt
            # =========================================================

            # follow path
            robot.follow_path(dt)


        

        # check robot collisions
        collisions = self._check_robot_collisions()
        # rewards = self._calculate_rewards(collisions)
        # done = self._check_done()
        # obs = self._get_obs()
        # info = {"collisions": collisions}


        self.runtime += dt
        # return obs, rewards, done, info

    def _get_obs(self) -> List[Dict[str, Any]]:
        """
        Gathers observations for all robots in the environment.
        """
    
    def _calculate_rewards(self, collisions: List[Tuple[int, int]]) -> List[float]:
        """
        Calculates reward values for each robot based on goal achievement and collisions.
        """



# test code
if __name__ == "__main__":
    map_yaml_path = Path("maps/warehouse_map.yaml")
    map = Map(map_yaml_path)
    #robot = Robot((0, 0), (5, 5))
    env = Environment(map, [], visualize=True)

    # Test the visualization
    while True:
        try:
            env.visualize()
        except KeyboardInterrupt:
            print("Exiting...")
            pygame.quit()
            break
    