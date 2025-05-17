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
    
    agents = env.agents
    action_space = env.single_action_space
    
    while True:
        try:
            actions = {}
            for agent in agents:
                action = action_space.sample()#
                actions[agent] = action
            obs, rewards, dones, infos, _ = env.step(action_dict=actions)

        except KeyboardInterrupt:
            break