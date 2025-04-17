import configparser
from pathlib import Path
from typing import Union, List
import ast
import numpy as np

from src.map import Map
from src.environment import Environment
from src.environment_entities import Robot, Loader, Unloader
from src.classes import Position


def main(config):
    # Environment config
    config_env = config['ENVIRONMENT']
    map_yaml_path = config_env['map_yaml']
    dt = float(config_env['dt'])
    
    # Robot config
    config_robot = config['ROBOT']
    robot_footprint = ast.literal_eval(config_robot['footprint'])
    init_robot_poses = ast.literal_eval(config_robot['init_poses'])
    dist_tolerance = float(config_robot['distance_tolerance'])
    head_tolerance = float(config_robot['heading_tolerance'])
    map_scale_factor = int(config_robot['plan_map_scale_factor'])

    # Loader config
    config_loader = config['LOADER']
    init_loader_poses = ast.literal_eval(config_loader['init_poses'])
    load_delay = float(config_loader['loading_delay'])
    max_tasks = int(config_loader['max_tasks'])

    # Unloader config
    config_unloader = config['UNLOADER']
    init_unloader_poses = ast.literal_eval(config_unloader['init_poses'])
    unload_delay = float(config_unloader['unloading_delay'])
    
    
    
    
    # create map
    map = Map(map_yaml_path)

    # create robots
    footprint = robot_footprint
    robots = [
        Robot(robot_id=f"R{i}",
              start_pos=Position(init_robot_poses[i-1][0], init_robot_poses[i-1][1], init_robot_poses[i-1][2]),
              footprint=footprint,
              dist_tolerance=dist_tolerance,
              heading_tolerance=head_tolerance,
              map=map, 
              map_scale_factor=map_scale_factor,
              logging=False)
              for i in range(1, len(init_robot_poses) + 1)
    ]

    # create loaders
    loaders = [
        Loader(loader_id=f"L{i}",
               pos=Position(init_loader_poses[i-1][0], init_loader_poses[i-1][1], init_loader_poses[i-1][2]),
               load_time=load_delay,
               max_tasks=max_tasks)
               for i in range(1, len(init_loader_poses) + 1)
    ]
    # create unloaders
    unloaders = [
        Unloader(unloader_id=f"U{i}",
                 pos=Position(init_unloader_poses[i-1][0], init_unloader_poses[i-1][1], init_unloader_poses[i-1][2]),
                 unload_time=unload_delay)
                 for i in range(1, len(init_unloader_poses) + 1)
    ]

    # create environment
    env = Environment(map=map,
                    robots=robots,
                    loaders=loaders,
                    unloaders=unloaders,
                    visualize=True,
                    logging=False)

    while True:
        try:            
            env.step(dt=dt)
            env.visualize()
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    main(config)