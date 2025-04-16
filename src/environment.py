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


from src.environment_entities import Robot
from src.map import Map
from src.classes import Position

import pygame


class Environment(gym.Env):

    def __init__(self, map:Map, robots:List[Robot], visualize:bool=False):
        """
        Initializes the simulation environment.

        """
        super().__init__()

        self.map:Map = map
        self.robots:List[Robot] = robots

        
        scr_bg = self.__init_pygame() if visualize else None
        self.__screen = scr_bg[0] if visualize else None
        self.__bg_img = scr_bg[1] if visualize else None
        self.__visualize:bool = visualize

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
        BLUE = (0,0,255)
        RED = (255, 0, 0)

        # font init
        font = pygame.font.SysFont('Comic Sans MS', 24)

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
                pygame.draw.lines(self.__screen, (0, 255, 0), False, path_map, 2)

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
            actions (List[Tuple[float, float]]): List of velocity vectors for each robot.

        Returns:
            Tuple:
                - observations (List[Dict[str, Any]]): Updated observations for each robot.
                - rewards (List[float]): Reward values for each robot.
                - done (bool): Flag indicating if all tasks are completed.
                - info (Dict[str, Any]): Additional information, such as collisions.
        """
        # check for collisions
        collisions = self._check_robot_collisions()
        for robot in self.robots:
            self.map.check_map_collisions(robot=robot)

        for robot in self.robots:
            robot.follow_path(dt)

        #if collisions:
        #print(f"Collisions detected: {collisions}")
        # if map_collisions:
        #     print(f"Map collisions detected: {map_collisions}")
        
        # for robot, action in zip(self.robots, actions):
        #     robot.move(action, self.time_step)
        
        # collisions = check_collisions(self.robots)
        # rewards = self._calculate_rewards(collisions)
        # done = self._check_done()
        # obs = self._get_obs()
        # info = {"collisions": collisions}
        # return obs, rewards, done, info

    # def _get_obs(self) -> List[Dict[str, Any]]:
    #     """
    #     Gathers observations for all robots in the environment.

    #     Returns:
    #         List[Dict[str, Any]]: Observation dictionary for each robot.
    #     """
    #     observations = []
    #     for i, robot in enumerate(self.robots):
    #         nearby_agents = []
    #         for j, other in enumerate(self.robots):
    #             if i != j:
    #                 dx = other.position[0] - robot.position[0]
    #                 dy = other.position[1] - robot.position[1]
    #                 nearby_agents.append({
    #                     "rel_pos": (dx, dy),
    #                     "goal": other.goal
    #                 })
    #         observations.append({
    #             "position": robot.position,
    #             "goal": robot.goal,
    #             "path": robot.get_next_path_points(5),
    #             "nearby_agents": nearby_agents
    #         })
    #     return observations

    # def _calculate_rewards(self, collisions: List[Tuple[int, int]]) -> List[float]:
    #     """
    #     Calculates reward values for each robot based on goal achievement and collisions.

    #     Args:
    #         collisions (List[Tuple[int, int]]): Pairs of robot indices that collided.

    #     Returns:
    #         List[float]: Reward value for each robot.
    #     """
    #     rewards = []
    #     for robot in self.robots:
    #         if robot.reached_goal():
    #             rewards.append(1.0)
    #         else:
    #             rewards.append(-0.1)
    #     for i, j in collisions:
    #         rewards[i] -= 1.0
    #         rewards[j] -= 1.0
    #     return rewards

    # def _check_done(self) -> bool:
    #     """
    #     Checks whether all robots have reached their goals.

    #     Returns:
    #         bool: True if all robots reached their goals, False otherwise.
    #     """
    #     return all(robot.reached_goal() for robot in self.robots)


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
    