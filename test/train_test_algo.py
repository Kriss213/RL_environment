import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

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

    # sample = obs_space.sample()
    # print(sample)
    # print(obs_space.contains(sample))
    # print(sample.shape)
    # sample2 = env._get_obs()
    # print(obs_space.contains(sample2))
    # env.close()
    # exit()

    algo_ppo = (
        PPOConfig()
        .environment(env="warehouse_env",env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=0)#.rollouts(num_rollout_workers=0)
        #.resources(num_cpus_for_main_process=0)
        .learners(num_gpus_per_learner=1)
        .rl_module(model_config={
            'train_batch_size': 4000,
            'minibatch_size': 128,
            'lr': 5e-4,
            'gamma': 0.99,
            'vf_clip_param': 10.0,
        })
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        )
        # .evaluation(
        #     evaluation_interval=100,
        #     evaluation_duration=3,
        #     evaluation_config={
        #         "explore": True,
        #     }
        # )
        .build()
    )

    for i in range(100):
        result = algo_ppo.train()
        print(pretty_print(result))
        if i % 5 == 0:
            checkpoint_dir = algo_ppo.save()
            print(f"checkpoint saved in directory {checkpoint_dir}")


    # tuner = tune.Tuner(
    #     "PPO",
    #     run_config=tune.RunConfig(
    #         name="warehouse_marl_train",
    #         stop={"training_iteration": 100},
    #         checkpoint_config=tune.CheckpointConfig(
    #             checkpoint_frequency=1,
    #             checkpoint_at_end=True,
    #         )
    #     ),
    #     param_space=config.to_dict()
    # )

    # tuner.fit()
