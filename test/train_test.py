import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

import configparser

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Environment import WarehouseEnv
from ray.rllib.env import EnvContext

from copy import deepcopy

if __name__ == "__main__":
    ray.init()

    config = configparser.ConfigParser()
    config.read('config.ini')
    config_dict = WarehouseEnv.parse_config(config)
    env_config = EnvContext(config_dict, worker_index=0)


    env = WarehouseEnv(config=env_config)
    obs_space = deepcopy(env.single_observation_space)
    act_space = deepcopy(env.single_action_space)
    agent_ids = deepcopy(env.agents)
   
    register_env("warehouse_env", lambda env_config: WarehouseEnv(env_config))

    config = (
        PPOConfig()
        .environment(env="warehouse_env",env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=0)#.rollouts(num_rollout_workers=0)
        .resources(num_cpus_for_main_process=0)
        .learners(num_gpus_per_learner=1)
        .rl_module(model_config={
            'lr': 5e-4,
        })
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        .evaluation(
            evaluation_interval=100,
            evaluation_duration=3,
            evaluation_config={
                "explore": True,
            }
        )
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=tune.RunConfig(
            name="warehouse_marl_train_300",
            stop={"training_iteration": 300},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            )
        ),
        param_space=config.to_dict()
    )

    tuner.fit()
