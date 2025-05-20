"""
Defines MARL environment
"""

import configparser
import ast

from typing import List, Dict, Tuple

from shapely.geometry import Polygon
from PIL import Image
from copy import deepcopy

#from src.environment_entities import Robot, Loader, Unloader, CBBA
from src.Agents import Courier, Loader, Unloader
from src.Map import Map
from src.Classes import Position, TaskAllocator, Task
import rclpy

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
        if not rclpy.ok():
            rclpy.init()

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

        # 9 - own observations
        # 2x5 - other agent observations
        obs_len = 9 + 2*5
        self.single_observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_len,),
            dtype=np.float32
        )

        self.observation_spaces = {
            c.id: deepcopy(self.single_observation_space)
            for c in self.couriers
        }
        
        self.IDX = dict(
            dx_to_goal= 0,
            dy_to_goal= 1,
            heading_sin= 2,
            heading_cos= 3,
            rem_path_len= 4,
            idle_timer= 5,
            front_busy= 6,
            blocking_other= 7,
            closest_front_dist= 8,
            # indices 9-18 are the two-neighbour bundle
        )
        
        # Default weights – override in call if needed
        self.DEFAULT_W = dict(
            progress            = +10.0,   # reward per delta progress to goal
            goal_arrival        = +200.0,  # one-time bonus when dx & dy both ~0
            collision           = -500.0,
            idle_penalty        = -1.0,    # per step when WAIT without good reason
            front_busy_penalty  = -5.0,    # waiting while nothing blocks front
            blocking_penalty    = -8.0,    # is blocking somebody else
            follow_dist_penalty = -3.0,    # tail-gating, dist < safe
            plan_fail_penalty   = -3.0,    # FOLLOW_PATH but path length == 0
        )
        
        
        # OBSERVATION CONSTANTS
        self.D_MAX = np.hypot(*self.map().shape) * self.map.resolution
        self.T_IDLE_MAX:float = 30.0 #seconds
        self.HEADING_ERR_MAX:float = np.deg2rad(35.0)
        self.HEADING_FOLLOWING_MAX:float = np.deg2rad(10.0)
        self.DIST_FOLLOWING_MAX:float = 3.0 # m
        
        # Threshold that counts as “at goal” (normalised units)
        self.GOAL_THRESH = 0.1 / self.D_MAX
        self.SAFE_FOLLOW_DIST = 2.0 / self.DIST_FOLLOWING_MAX    # normalised [0,1]
        self.IDLE_PATIENCE    = 10.0 / self.T_IDLE_MAX     # normalised idle_timer tolerated
        
        self._previous_observations:dict = {c.id: [0]*obs_len for c in self.couriers}
        
        # Action space
        self.single_action_space = gym.spaces.Discrete(2) # 0 - idle, 1 - follow path

        self.action_spaces = {
            c.id: deepcopy(self.single_action_space)
            for c in self.couriers
        }
        
        
        # ============ ================ ============
        
        # ============ Initialize task allocator ============
        self.TA:TaskAllocator = TaskAllocator(
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
        for c1 in self.couriers:
            # set other courier positions
            c1.other_couriers = [c2 for c2 in self.couriers if c1.id != c2.id]
        # ============ ================ ============

    def _get_obs(self, agent:Courier):
        """
        Get observation for a single agent:
        - dx_to_goal [-1; 1]
        - dy_to_goal [-1; 1]
        - heading_sin [-1; 1]
        - heading_cos [-1; 1]
        - remaining_path_len [0; 1]
        - idle_timer [0; 1]
        - front_busy {0, 1}
        - no_blocked_agents {0, 1}
        - min_front_dist [0; 1]
        - For 2 closest agents (if not enough agents, pad with 0):
            - dx_rel [-1; 1]
            - dy_rel [-1; 1]
            - heading_sin [-1; 1]
            - heading_cos [-1; 1]
            - is_waiting {0, 1}
            
        :return observations:
        """      
        
        # Get reference agent
        goal = agent.goal if agent.goal else agent.position
        pos = agent.position
        
        
        # distance_to_goal
        dx_to_goal = (goal.x - pos.x) / self.D_MAX
        dy_to_goal = (goal.y - pos.y) / self.D_MAX
        
        # heading
        heading_sin = np.sin(pos.theta)
        heading_cos = np.cos(pos.theta)
        
        # remaining_path_len (m)
        path = np.array(agent.path)[:,:2] if agent.path else []
        rem_path_len = min(1.0, np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)) / self.D_MAX) if len(path) else 0.0
        
        # time spent idle
        idle_time = min(1.0, agent._idle_time / self.T_IDLE_MAX)
        
        # is front of robot busy
        is_front_occupied, _ = self._is_front_occupied(agent)
        is_front_occupied = float(is_front_occupied)
        
        # is robot blocking other
        # and
        # distance to closest agent that is IN FRONT and is heading roughly the same way
        
        is_blocking_other = 0.0
        _min_following_dist = self.DIST_FOLLOWING_MAX
        other_ag_info = []
        couriers_sorted_by_dist = sorted(self.couriers, key=lambda c: np.hypot(c.position.x - agent.position.x, c.position.y - agent.position.y))
        for i, target_courier in enumerate(couriers_sorted_by_dist):
            if target_courier.id == agent.id:
                continue
            
            # get distance to closest agent that this agent is following
            _diff = target_courier.position - agent.position
            dist_to_target_courier = np.hypot(_diff.x, _diff.y)
            _heading_err = _diff.theta # to other courier's (x,y)
            _is_target_courier_in_front = _heading_err < self.HEADING_ERR_MAX
            _agent_heading_diff = abs(target_courier.position.theta - agent.position.theta)
            if _is_target_courier_in_front \
                and _agent_heading_diff < self.HEADING_FOLLOWING_MAX \
                and dist_to_target_courier < _min_following_dist:
                    _min_following_dist = dist_to_target_courier
            
            # collect info about closest 2 agents
            if i < 2:
                _dx_rel = _diff.x / self.D_MAX
                _dy_rel = _diff.y / self.D_MAX
                
                _is_waiting = float(target_courier._last_action == 0)
                
                _other_heading_sin = np.sin(target_courier.position.theta)
                _other_heading_cos = np.cos(target_courier.position.theta)
                
                other_ag_info += [_dx_rel, _dy_rel, _other_heading_sin, _other_heading_cos, _is_waiting]
            
            
            # is agent blocking someone else?
            _other_blocked, _blocker_id = self._is_front_occupied(target_courier)
            if _other_blocked and _blocker_id == agent.id:
                is_blocking_other = 1.0
            
        closest_following_dist = min(1.0, _min_following_dist / self.HEADING_FOLLOWING_MAX)
        
        while len(other_ag_info) < 10:
            other_ag_info.append(0.0)
            
        assert len(other_ag_info) == 10, f"Invalid other agent info list length ({len(other_ag_info)}). It must be 2x5=10."
        
        obs = np.array([dx_to_goal, dy_to_goal,
                        heading_sin, heading_cos,
                        rem_path_len,
                        idle_time,
                        is_front_occupied,
                        is_blocking_other,
                        closest_following_dist,
                        ] + other_ag_info, dtype=np.float32)

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
        observations = {}
        infos = {}
        for courier in self.couriers:
            observations[courier.id] = self._get_obs(courier)
            infos[courier.id] = {}

        self.episode_steps = 0

        return observations, infos
        
    def step(self, action_dict):
        """
        Perform a step in environment/
        
        :param action_dict: Dictionary of agent actions like:
        
        courier.id : 0 or 1
        0 - wait
        1 - follow path 
        """

        # Empty return dicts
        obs, rewards, terminateds, infos = {}, {}, {}, {}
        self.episode_steps += 1
        
        # perform task allocation step
        self.TA.run()
        
        collided_at_least_once = False
        for courier in self.couriers:
            c_id = courier.id
            action = action_dict.get(c_id, None)
            if action == None:
                action = 0
                print(f"[WARN] No action in action dict for agent [{c_id}]. Defaulting to aciton 0")
            
            # Perform action
            courier.perform(action=action, previous_obs=self._previous_observations[c_id], dt=self.dt)
            
            # get environment feedback
            new_obs = self._get_obs(courier)
            
            collision, collided_id = self._check_single_robot_collision(courier)
            if collision:
                collided_at_least_once = True
            # TODO check deadlocks
            hard_limits_met = self._check_hard_limits()
            
            # collect episode data 
            obs[c_id] = new_obs
            rewards[c_id] = self.get_reward(
                courier=courier,
                prev_obs=self._previous_observations[c_id],
                new_obs=new_obs,
                collided=collision,
                action=action)
            
            
            # save as previous obs
            self._previous_observations[c_id] = new_obs
        
        # RLlib requires "__all__" key in done dict
        terminateds["__all__"] = hard_limits_met or collided_at_least_once
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
    
    def get_reward(
        self,
        courier: Courier,
        prev_obs: np.ndarray,
        new_obs : np.ndarray,
        collided: bool,
        action  : int,
    ) -> float:
        """
        Reward = ∑ w_i * term_i
        All shaping terms use NORMALISED observation values.
        """

        w = self.DEFAULT_W
        IDX = self.IDX
        GOAL_THRESH = self.GOAL_THRESH
        IDLE_PATIENCE = self.IDLE_PATIENCE
        SAFE_FOLLOW_DIST = self.SAFE_FOLLOW_DIST
        r = 0.0

        # ------------------------------------------------------------------ #
        # 1) Task-level progress (euclidean distance shrinkage to goal)
        prev_dist = np.hypot(prev_obs[IDX["dx_to_goal"]], prev_obs[IDX["dy_to_goal"]])
        new_dist  = np.hypot(new_obs [IDX["dx_to_goal"]], new_obs [IDX["dy_to_goal"]])
        d_progress = prev_dist - new_dist # >0 if closer
        r += w["progress"] * d_progress

        # a large terminal bonus once agent is at goal
        if new_dist < GOAL_THRESH and not courier._awarded_reached_goal:
            r += w["goal_arrival"]
            courier._awarded_reached_goal = True

        # ------------------------------------------------------------------ #
        # 2) Collisions (episode-ending)
        if collided:
            r += w["collision"]

        # ------------------------------------------------------------------ #
        # 3) Action-specific shaping
        if action == 0: # WAIT
            # How long has the agent idled
            idle_now = new_obs[IDX["idle_timer"]]
            # if front IS free and agent waits -> penalise
            front_busy = new_obs[IDX["front_busy"]]
            if idle_now > IDLE_PATIENCE and front_busy < 0.5:
                r += w["idle_penalty"]

        elif action == 1: # FOLLOW_PATH
            # discourage attempting to move when there is no path
            if new_obs[IDX["rem_path_len"]] < 1e-3:
                r += w["plan_fail_penalty"]

        # ------------------------------------------------------------------ #
        # 4) Social-safety shaping
        if new_obs[IDX["blocking_other"]] > 0.5:
            r += w["blocking_penalty"]

        # being too close to other agent that is being followed
        if new_obs[IDX["closest_front_dist"]] < SAFE_FOLLOW_DIST:
            r += w["follow_dist_penalty"]

        return float(r)

    def _check_deadlocks(self, courier:Courier, n=5) -> bool:
        """
        Check if courier is in deadlock (cannot plan path for extended period of time)
        for whatever reason (usually facing different robot in corridor etc.).

        Args:
            courier (Courier): The courier to check.
            n (int): Number of failed path plan attempts to check (Default 5).
        """
        return courier.failed_path_plan_attempts > n
        
    def _check_single_robot_collision(self, courier:Courier) -> Tuple[bool, str]:
        
        courier_poly = Polygon(courier.get_bbox())
        id_and_polygons = [(c.id, Polygon(c.get_bbox()) ) for c in self.couriers if c.id != courier.id]
        
        for rob_id_other, poly_other in id_and_polygons:
            if courier_poly.intersects(poly_other):
                return True, rob_id_other
        
        return False, None
    
    def _check_robot_collisions(self) -> List[Tuple[str, str]]:
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
    
    def _is_front_occupied(self, courier:Courier, d_lim:float=1.0) -> Tuple[bool, str]:
        """
        Check if robot (likely) prevented collision by waiting (waiting action is assumed)
        
        :Return bool | (bool, str):
        """
              
        for target_courier in self.couriers:
            if target_courier.id == courier.id:
                continue
            # get distance and heading error to other courier from courier
            diff = target_courier.position - courier.position
            
            dist = np.hypot(diff.x, diff.y)
            heading_error = diff.theta
            
            
            if dist < d_lim and (-self.HEADING_ERR_MAX < heading_error < self.HEADING_ERR_MAX):
                #print(f"{courier.id} is blocked by {target_courier.id}")
                return True, target_courier.id
          
        return False, None

    def _check_hard_limits(self) -> bool:
        cond_agent_path_plan = any([c._failed_plans_in_row >= 3 for c in self.couriers])
        cond_ep_steps = self.episode_steps > 2000
        cond_delivered_tasks = self.TA.delivered_tasks >= self.episode_length
        # TODO time in deadlock (2 agents not moving)
        
        return cond_agent_path_plan or cond_ep_steps

    def _likely_blocking_path(self, courier:Courier) -> bool:
        """
        Check if robot (likely) is blocking someone's path by moving
        """

        if not courier.path:
            return False

        # assumes action is 1 (following path)
        for c in self.couriers:
            if c.id == courier.id:
                continue
            # get distance to other courier from courier
            dist = np.linalg.norm(np.array(c.position()[:2]) - np.array(courier.position()[:2]))
            if dist > 3.0 or not c.path:
                continue
            # check if courier is NEAR other courier's path points
            n_points = min(50, len(courier.path))
            path_points = c.path[:n_points]
            for point in path_points:
                dist = np.linalg.norm(np.array(point[:2]) - np.array(c.position()[:2]))
                if dist < 0.5:
                    return True
        return False

        
    
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