import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

import configparser
import os
import sys
import argparse
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.Environment import WarehouseEnv
from ray.rllib.env import EnvContext

# https://github.com/ray-project/ray/issues/51560#issuecomment-2831921054
# Fix for optimizer betas as tensors after checkpoint restore
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.algorithm import Algorithm
from torch import Tensor
from typing import Optional
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

class AlgorithmFix(RLlibCallback):
    def __init__(self, **kwargs):
        super().__init__()

    def on_algorithm_init(self,* , algorithm: "Algorithm", metrics_logger: Optional[MetricsLogger] = None, **kwargs,) -> None:
        pass
        
    def on_checkpoint_loaded(self, *, algorithm: Algorithm, **kwargs, ) -> None:
        def betas_tensor_to_float(learner):
            param_grp = next(iter(learner._optimizer_parameters.keys())).param_groups[0]
            if not param_grp['capturable'] and isinstance(param_grp["betas"][0], Tensor):
                param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
        algorithm.learner_group.foreach_learner(betas_tensor_to_float)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    context = ray.init()
    print(context.dashboard_url)

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
        .environment(env="warehouse_env", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=20, num_envs_per_env_runner=1, num_cpus_per_env_runner=1)
        .resources(num_cpus_for_main_process=0)
        .learners(num_gpus_per_learner=1)
        .training(
            use_critic=True,
            gamma=0.995,
            lambda_=0.95,
            lr=1e-4,
            clip_param=0.2,
            entropy_coeff=[[0, 0.01], [2e6,0.0001]],
            kl_coeff=0.5,
            kl_target=0.005,
            train_batch_size=16_000,
            minibatch_size=2_048,
            num_epochs=10,
            vf_clip_param=1000.0, # should be bit larger than realistic mean episode return
            vf_loss_coeff=0.25,
            model={
                "fcnet_activation": "relu",
                "use_lstm": True,
                "max_seq_len": 50
            },
        )
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
    iterations = 500
    run_config = tune.RunConfig(
        name=f"parallel_warehouse_marl_train_23_05_{iterations}",
        stop={"training_iteration": iterations},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=5,
            checkpoint_at_end=True,
        )
    )
    
    if args.checkpoint:       
        config.callbacks(
            callbacks_class=AlgorithmFix
        )
        tuner = tune.Tuner.restore(
            path=args.checkpoint,
            trainable=config.algo_class,
            param_space=config
        )
    else:
        tuner = tune.Tuner(
            config.algo_class,
            run_config=run_config,
            param_space=config
        )

    tuner.fit()
