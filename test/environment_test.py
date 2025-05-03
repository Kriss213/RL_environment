"""
Test environment.

This program initializes an environment and performs random actions.

"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import configparser
from src.Environment import WarehouseEnv

from ray.rllib.env import EnvContext

import numpy as np

if __name__ == "__main__":
    
    # read config
    config = configparser.ConfigParser()
    config.read('config.ini')
    config_dict = WarehouseEnv.parse_config(config)
    env_config = EnvContext(config_dict, worker_index=0)


    env = WarehouseEnv(config=env_config)
    np.random.seed(env.TA.seed)
    
    map_min, map_max = env.map.map_min_world, env.map.man_max_world
    
    agents = env.courier_ids
    actions = {}
    for agent in agents:
        mode = np.random.choice([0, 1], p=[0.8, 0.2])
        x = np.random.uniform(map_min[0], map_max[0])
        y = np.random.uniform(map_min[1], map_max[1])
        theta = np.random.uniform(-np.pi, np.pi)
        actions[agent] = (mode, x, y, theta)
    
    
   # env.couriers[0].goal = Position(4.23, -5.38, 0.0)
    while True:
        try:
            actions = {}
            for agent in agents:
                mode = np.random.choice([0, 1], p=[0.99, 0.01])
                x = np.random.uniform(map_min[0], map_max[0])
                y = np.random.uniform(map_min[1], map_max[1])
                theta = np.random.uniform(-np.pi, np.pi)
                actions[agent] = (mode, x, y, theta)
            obs, rewards, dones, infos, _ = env.step(action_dict=actions)
            #print(rewards)
            env.render()
        except KeyboardInterrupt:
            break