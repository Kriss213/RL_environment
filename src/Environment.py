"""
Defines MARL environment
"""

import configparser
import ast

from typing import List, Dict, Tuple

from shapely.geometry import Polygon
from PIL import Image
from pathlib import Path
from copy import deepcopy

#from src.environment_entities import Robot, Loader, Unloader, CBBA
from src.Agents import Courier, Loader, Unloader
from src.Map import Map
from src.Classes import Position, CBBA, Task

import pygame
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym

from ray.rllib.env import EnvContext

class WarehouseEnv(MultiAgentEnv):

    @classmethod
    def parse_config(cls, config:configparser.ConfigParser) -> Dict:
        """
        Parse config.ini and convert it to python dict with proper types.
        """
        conf_env = config['ENVIRONMENT']
        conf_TA = config['TASK_ALLOCATION']
        conf_robot = config['ROBOT']
        conf_courier = config['COURIER']
        conf_loader = config['LOADER']
        conf_unloader = config['UNLOADER']

        config_dict = {
            'ENVIRONMENT': {
                'map_yaml': conf_env['map_yaml'],
                'dt': conf_env.getfloat('dt'),
                'downsample_factor': conf_env.getint('downsample_factor'),
                'extra_padding': conf_env.getfloat('extra_padding'),
                'logging': conf_env.getboolean('logging'),
                'episode_length': conf_env.getint('episode_length'),
                'visualize': conf_env.getboolean('visualize'),
                'map_tuple' : Map.load_map(conf_env['map_yaml'])
            },
            'TASK_ALLOCATION': {
                'logging': conf_TA.getboolean('logging'),
                'seed': conf_TA.getint('seed')
            },
            'ROBOT': {
                'footprint': np.array(ast.literal_eval(conf_robot['footprint'])),
                'init_poses': np.array(ast.literal_eval(conf_robot['init_poses'])),
                'distance_tolerance': conf_robot.getfloat('distance_tolerance'),
                'heading_tolerance': conf_robot.getfloat('heading_tolerance'),
                'logging': conf_robot.getboolean('logging'),
            },
            'COURIER': {
                'logging': conf_courier.getboolean('logging'),
            },
            'LOADER': {
                'init_poses': np.array(ast.literal_eval(conf_loader['init_poses'])),
                'max_tasks': conf_loader.getint('max_tasks'),
                'loading_delay': conf_loader.getfloat('loading_delay'),
                'logging': conf_loader.getboolean('logging')

            },
            'UNLOADER': {
                'init_poses': np.array(ast.literal_eval(conf_unloader['init_poses'])),
                'unloading_delay': conf_unloader.getfloat('unloading_delay'),
                'logging': conf_unloader.getboolean('logging')
            }
        }

        return config_dict
    

    def __init__(self, config:EnvContext):
        super().__init__()
        """
        Initialize MARL environment.
        """        

        # ============ Parse config.ini ============
        #self.__config = configparser.ConfigParser()
        self.__config = config
        #self.__config.read('config.ini')

        # Environment config
        self.config_env = self.__config['ENVIRONMENT']
        map_yaml_path = self.config_env['map_yaml']
        self.dt = self.config_env['dt']
        downsample_factor = self.config_env['downsample_factor']
        extra_padding = self.config_env['extra_padding']
        self.logging = self.config_env['logging']
        self.episode_length = self.config_env['episode_length']

        # Task allocation config
        self.config_task_alloc = self.__config['TASK_ALLOCATION']
        TA_logging = self.config_task_alloc['logging']
        TA_seed = self.config_task_alloc['seed']
        
        # Robot config
        self.config_robot = self.__config['ROBOT']
        robot_footprint = self.config_robot['footprint']

        # Loader config
        self.config_loader = self.__config['LOADER']
        
        # Unloader config
        self.config_unloader = self.__config['UNLOADER']
        # ============ ================ ============

        # ============ Initialize environment map ============
        self.map = Map(
            map_yaml=map_yaml_path,
            map_tuple=self.config_env['map_tuple'],
            footprint=robot_footprint+ np.where(robot_footprint>0, extra_padding, -extra_padding),
            downsample_factor=downsample_factor)
        # ============ ================ ============
        # ============ Initialize agents ============
        self.__init_agents()
        self.possible_agents:List[str] = [c.id for c in self.couriers]
        self.agents:List[str] = [c.id for c in self.couriers]
        self.agent_count = len(self.possible_agents)

        # Observations:
        # all agent positions (x,y,theta)
        #    (sorted by distance to each agent):
        # all agent navigation goals (x, y, theta)
        # all agent TASK goals (x,y,theta)
        obs_len = self.agent_count * 3 * 3
        obs_len = 9 * self.agent_count
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_len,),
            dtype=np.float32
        )

        self.observation_spaces = {
            c.id: deepcopy(self.single_observation_space)
            for c in self.couriers
        }
        
        # Action space
        self.single_action_space = gym.spaces.Discrete(2) # 0 - idle, 1 - follow path

        self.action_spaces = {
            c.id: deepcopy(self.single_action_space)
            for c in self.couriers
        }
        
        
        # ============ ================ ============
        
        # ============ Initialize task allocator ============
        self.TA:CBBA = CBBA(
            couriers=self.couriers,
            unloaders=self.unloaders,
            loaders=self.loaders,
            dt=self.dt,
            seed=TA_seed,
            logging=TA_logging
        )
        # ============ ================ ============ 
        
        
        # ============ Initialize Visualizer ============
        self.visualizer = \
            Visualize(map=self.map, couriers=self.couriers, loaders=self.loaders, unloaders=self.unloaders ) \
            if self.config_env['visualize'] else None
        # ============ ================ ============ 
        
        # reset the environment
        self.reset()
        
        # Run task allocator
        self.TA.run()

    def __del__(self):
        """
        Destructor for WarehouseEnv.
        """
        if self.visualizer:
            pygame.quit()
            del self.visualizer
            self.visualizer = None
        print("Environment destroyed.")
    
    def close(self):
        self.__del__()

    def __init_agents(self):
        """
        Initialize agents.
        """
        # Loader config
        init_loader_poses = self.config_loader['init_poses']
        load_delay = self.config_loader['loading_delay']
        max_tasks = self.config_loader['max_tasks']
        loader_logging = self.config_loader['logging']
        
        # Unloader config
        init_unloader_poses = self.config_unloader['init_poses']
        unload_delay = self.config_unloader['unloading_delay']
        unloader_logging = self.config_unloader['logging']

        # Courier (robot) config
        robot_footprint = self.config_robot['footprint']
        init_robot_poses = self.config_robot['init_poses']
        dist_tolerance = self.config_robot['distance_tolerance']
        head_tolerance = self.config_robot['heading_tolerance']
        courier_logging = self.config_robot['logging']

        # ============ Initialize agents ============
        self.loaders = [
                Loader(loader_id=f"L{i+1}",
                    pos=Position(init_loader_poses[i][0], init_loader_poses[i][1], init_loader_poses[i][2]),
                    load_time=load_delay,
                    max_tasks=max_tasks,
                    logging=loader_logging)
                    for i in range(len(init_loader_poses))
            ]
        self.unloaders = [
                Unloader(unloader_id=f"U{i+1}",
                        pos=Position(init_unloader_poses[i][0], init_unloader_poses[i][1], init_unloader_poses[i][2]),
                        unload_time=unload_delay,
                        logging=unloader_logging)
                        for i in range(len(init_unloader_poses))
            ]
        self.couriers = [
                Courier(robot_id=f"R{i}",
                    start_pos=Position(init_robot_poses[i-1][0], init_robot_poses[i-1][1], init_robot_poses[i-1][2]),
                    footprint=robot_footprint,
                    dist_tolerance=dist_tolerance,
                    heading_tolerance=head_tolerance,
                    map=self.map, 
                    logging=courier_logging)
                    for i in range(1, len(init_robot_poses) + 1)
            ]
        # ============ ================ ============

    def _get_obs(self, agent_id):
        """
        Get observation for a single agent:
        - Own position
        - Own navigation goal
        - Own task goal
        - Other agents' positions, nav goals, and task goals (sorted by distance to self)
        """
        # Get reference agent
        self_agent = next(c for c in self.couriers if c.id == agent_id)

        # Own data
        own_pos = [self_agent.position.x, self_agent.position.y, self_agent.position.theta]

        nav_goal = self_agent.goal or Position(*own_pos)
        own_nav = [nav_goal.x, nav_goal.y, nav_goal.theta]

        task_goal = self_agent.active_task.active_goal if self_agent.active_task else Position(*own_pos)
        own_task = [task_goal.x, task_goal.y, task_goal.theta]

        # Prepare other agents' data
        others = []
        for c in self.couriers:
            if c.id == agent_id:
                continue

            pos = np.array([c.position.x, c.position.y])
            dist = np.linalg.norm(pos - np.array([self_agent.position.x, self_agent.position.y]))

            nav = c.goal or c.position
            task = c.active_task.active_goal if c.active_task else c.position

            other_data = {
                "distance": dist,
                "position": [c.position.x, c.position.y, c.position.theta],
                "nav_goal": [nav.x, nav.y, nav.theta],
                "task_goal": [task.x, task.y, task.theta],
            }
            others.append(other_data)

        # Sort other agents by distance
        others.sort(key=lambda d: d["distance"])

        # Flatten other agents' data
        others_flat = []
        for o in others:
            others_flat.extend(o["position"])
            others_flat.extend(o["nav_goal"])
            others_flat.extend(o["task_goal"])

        # Final observation
        obs = np.array(own_pos + own_nav + own_task + others_flat, dtype=np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to original state.
        """
        
        if self.visualizer:
            self.visualizer.render()

        # reset agents
        for courier in self.couriers:
            courier.reset()
        for loader in self.loaders:
            loader.reset()
        for unloader in self.unloaders:
            unloader.reset()
            
        self.TA.reset()

        # return observation dict and infos dict.
        
        # Observation is flat list of all [x, y, theta] positions, nav_goals and task_goals
        # all agents have full info
        observations = {}
        infos = {}
        for courier in self.couriers:
            observations[courier.id] = self._get_obs(courier.id)
            infos[courier.id] = {}
        
        # obs = self._get_obs()
        # print(obs, obs.shape )
        return observations, infos
        
    def step(self, action_dict):
        """
        Perform a step in environment/
        
        :param action_dict: Dictionary of agent actions like:
        
        courier.id : (mode, x, y, theta) 
        """
        #print(f"Recieved action_dict: {action_dict}")
        # Empty return dicts
        obs, rewards, terminateds, infos = {}, {}, {}, {}
        
        # perform task allocation step
        self.TA.run()
        
        # get robot collisions
        collisions = self._check_robot_collisions()
        
        
        for courier in self.couriers:

            # set new goal if active task active goal differs
            if courier.active_task and not courier.goal:
                # update goal to active 
                courier.goal = courier.active_task.active_goal
                if courier.logging:
                    print(f"Robot {c_id} set new goal to task target: {courier.goal}")

            c_id = courier.id
            action = action_dict[c_id]

            if courier.logging:
                print(f"Courier {c_id} action: {action}")

            if action == 0:
                # stay idle
                if courier.logging:
                    print(f"Robot {c_id} is idle.")
            elif action == 1:
                # follow path
                courier.follow_path(dt=self.dt)
                if courier.logging:
                    print(f"Robot {c_id} is following path.")
            else:
                raise ValueError(f"Invalid action {action} for agent {c_id}.")
            
             # Collect agent data
            obs[c_id] = self._get_obs(c_id)
            rewards[c_id] = self.get_reward(courier, collisions, action_dict)
            terminateds[c_id] = self.TA.delivered_tasks >= self.episode_length
            infos[c_id] = {}

        # RLlib requires "__all__" key in done dict
        terminateds["__all__"] = all(terminateds[agent.id] for agent in self.couriers)
        #terminateds["__all__"] = all(terminateds.get(agent.id, True) for agent in self.couriers)
        truncateds = deepcopy(terminateds)

        if self.visualizer:
            self.render()
        
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        """
        Override default render behaviour.
        """
        if self.visualizer:
            self.visualizer.render()
    
    def get_reward(self, courier:Courier, collisions:List[Tuple[int, int]], action:int) -> float:
        """
        Calculate reward for courier.
        """
        # TODO - adjust reward function
        penalty = 0 
        
        # penalize being far from goal
        if courier.active_task:
            dist_to_task_goal = np.hypot(*(courier.active_task.active_goal - courier.position)()[:2])
            penalty += dist_to_task_goal
        else:
            penalty += 50 # no goal
        
        # penalize going away from loading and unloading position
        if courier.active_task:
            if courier.active_task.status in (Task.AT_PICKUP, Task.AT_DROPOFF) \
                and action != 0:
                    penalty += 500
        
        # penalzie collisions
        for col_ids in collisions:
            if courier.id in col_ids:
                penalty += 1000
                break
        
        # reward following_path
        if action == 1:
            penalty -= 10
                
        # negative!
        return -penalty 
        
    def _check_robot_collisions(self) -> List[Tuple[int, int]]:
        """
        Checks collisions between robots.

        Returns:
            List of pairs of robot indices that are colliding.
        """
        collisions = []
        id_and_polygons = [(courier.id, Polygon(courier.get_bbox()) ) for courier in self.couriers]

        for i, (rob_id_1, poly1) in enumerate(id_and_polygons):
            for j, (rob_id_2, poly2) in enumerate(id_and_polygons):
                if i >= j:
                    continue
                if poly1.intersects(poly2):
                    collisions.append((rob_id_1, rob_id_2))
        return collisions
    
        
        
    
class Visualize:
    """
    A class for environment visualization
    """
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    def __init__(self,
                 map:Map,
                 couriers:List[Courier],
                 loaders:List[Loader],
                 unloaders:List[Unloader],
                 ):
        """
        Initializes the Pygame screen and returns it.
        """
        pygame.init()
        pygame.display.init()
        self.env_map = map
        self.couriers = couriers
        self.loaders = loaders
        self.unloaders=unloaders
        

        self.font = pygame.font.SysFont('Comic Sans MS', 24)

        h, w = map.height, map.width
        
        # screen size must match map resolution (h,w)
        self.screen = pygame.display.set_mode((w,h))
        pygame.display.set_caption("Environment")

        # Load the map image
        map_image = Image.fromarray(self.env_map()).convert('RGB')

        # set the screen to the map image
        self.bg_img = pygame.image.fromstring(map_image.tobytes(), map_image.size, map_image.mode)
        self.screen.blit(self.bg_img, (0, 0))

    def render(self):
        """
        Update pygame window. Render the environment with pygame.
        """
        # draw background
        self.screen.blit(self.bg_img, (0, 0))
        env_map = self.env_map

        # Visualize loaders
        for loader in self.loaders:
            # visualize loaders
            loader_id_surface = self.font.render(loader.id, True, self.RED)
            loader_pos_screen = env_map.world_to_map(*loader.position()[:2])
            self.screen.blit(loader_id_surface, loader_pos_screen)

        # visualize unloaders
        for unloader in self.unloaders:
            # visualize unloaders
            unloader_id_surface = self.font.render(unloader.id, True, self.GREEN)
            unloader_pos_screen = env_map.world_to_map(*unloader.position()[:2])
            self.screen.blit(unloader_id_surface, unloader_pos_screen)

        # visualize couriers
        for courier in self.couriers:
            # =====VISUALIZE ROBOT=====
            # get robot bounding box
            bbox = courier.get_bbox() # this is in meters
            # convert to pixel coords
            bbox_map = []
            for point in bbox:
                # convert to map coordinates
                mx, my = env_map.world_to_map(*point[:2])
                bbox_map.append((mx, my))
            assert len(bbox_map) == 4, f"Invalid bbox_map: {bbox_map}"
            # robot pos in pixel coords
            robot_pos = env_map.world_to_map(*courier.position()[:2])
            # draw the robot on the screen
            pygame.draw.polygon(self.screen, self.BLUE, bbox_map)
            pygame.draw.circle(self.screen, self.RED, robot_pos[:2], 5)
            # add robot name
            text_surface = self.font.render(courier.id, True, self.RED)
            # place label in middle of robot box
            self.screen.blit(text_surface, np.sum(bbox_map, axis=0) // 4)

            # =====VISUALIZE PATH=====
            if courier.path:# is not None and len(robot.path) > 0:
                path_map = [env_map.world_to_map(*courier.position()[:2])]
                for point in courier.path:
                    # convert to map coordinates
                    mx, my = env_map.world_to_map(*point[:2])
                    path_map.append((mx, my))
                # draw the path on the screen
                pygame.draw.lines(self.screen, self.GREEN, False, path_map, 2)

        # # DEBUG
        # for event in pygame.event.get():
            
        #     if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
        #         pixel_pos = pygame.mouse.get_pos()
        #         world_pos = self.env_map.map_to_world(*pixel_pos)

        #         pos = Position(*world_pos, theta=0.0)
        #         # DEBUG
        #         # send robot 1 to pos
        #         self.couriers[0].goal = pos

        #         print(f"Clicked pixel: {pixel_pos}")
        #         print(f"Clicked world: {world_pos}")
        #         print(f"Pixel value at clicked: {self.env_map()[pixel_pos[1], pixel_pos[0]]}")
        #         break
        
        # update display loop
        pygame.event.pump()
        pygame.display.update()