"""
Test environment.

This program initializes an environment and performs random actions or a checkpoint can be loaded with --checkpoint.

"""
import sys
import os
from pathlib import Path
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import configparser
from src.Environment import WarehouseEnv

from ray.rllib.env import EnvContext
from ray.tune.registry import register_env
from ray.rllib.core.rl_module import RLModule

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Ray Tune checkpoint directory")
    args = parser.parse_args()
    
    # read config
    config = configparser.ConfigParser()
    config.read('config.ini')
    config_dict = WarehouseEnv.parse_config(config)
    config_dict['ENVIRONMENT']['visualize'] = True
    env_config = EnvContext(config_dict, worker_index=0)


    env = WarehouseEnv(config=env_config)
    register_env("warehouse_env", lambda env_config: WarehouseEnv(env_config))
    
    map_min, map_max = env.map.map_min_world, env.map.man_max_world
    
    agents = env.agents
    action_space = env.single_action_space
    
    # Load checkpoint
    rl_module = None
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        # create the RL module
        rl_module = RLModule.from_checkpoint(
            Path(args.checkpoint)
            / "learner_group"
            / "learner"
            / "rl_module"
            / "shared_policy"
        )
        

    # https://docs.ray.io/en/latest/rllib/getting-started.html?_gl=1*f6wmei*_up*MQ..*_ga*MTA4ODQwNDI4NS4xNzQ3ODUyMTM2*_ga_0LCWHW1N3S*czE3NDc4NTIxMzUkbzEkZzAkdDE3NDc4NTIxMzUkajAkbDAkaDAkZFIyRE03b3VVbHBhUklnMEs3OEV1Y0Y5ZXJvczI0REpCZHc.
    
    obs, info = env.reset()
    episode_reward = {c: 0 for c in agents}
    episode_len = 0
    while True:
        try:
            actions = {}
            for agent_id in agents:
                if rl_module:
                    obs_batch = torch.from_numpy(obs[agent_id]).unsqueeze(0)  # add batch B=1 dimension
                    model_outputs = rl_module.forward_inference({"obs": obs_batch})
                    
                    # Extract the action distribution parameters from the output and dissolve batch dim.
                    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
                    # For discrete actions, you should take the argmax over the logits:
                    greedy_action = np.argmax(action_dist_params)
                    
                    actions[agent_id] = greedy_action
                    
                else:
                    actions[agent_id] = action_space.sample()
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            episode_len += 1
            for c, r in rewards.items():
                episode_reward[c] += r
                
            if terminated["__all__"]:
                print(f"Episode length: {episode_len} steps")
                for c, r in episode_reward.items():
                    print(f"[{c}] Mean reward: {(r/episode_len):.4f}")
                
                episode_len = 0
                episode_reward = {c: 0 for c in agents}
                obs, info = env.reset()

        except KeyboardInterrupt:
            
            break