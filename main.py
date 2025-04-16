import configparser
from pathlib import Path
from typing import Union, List
import ast
import numpy as np

from src.map import Map
from src.environment import Environment
from src.environment_entities import Robot
from src.classes import Position


def main(map_yaml_path: Union[Path, str], robot_footprint):
    # create map
    map = Map(map_yaml_path)

    pos = [
        (10.0, 6.0, -1.7),
        (10.0, -6.0, 1.7),
        (-11.0, -5.0, 0.0),
        (-7.0, 0.5, 0.0),
        (2.75, -0.6, 3.14)
    ]

    # pos = [ 
    #     (10.0, 6.0, -1.7),
    #     (10.0, 4.2, 1.7),
    #     (-11.0, -5.0, 0.0),
    #     (-10.0, -6.0, 3.14),
    #     (2.75, -0.6, 3.14)
    # ]

    # create robots
    footprint = robot_footprint
    robots = [
        Robot(robot_id=f"R{i}",
              start_pos=Position(pos[i-1][0], pos[i-1][1], pos[i-1][2]),
              footprint=footprint,
              map=map)
              for i in range(1, len(pos) + 1)
    ]

    #robots[3].goal = Position(-10.0, -6.0, 3.14)

    # create environment
    env = Environment(map, robots, visualize=True)

    while True:
        try:            
            env.step(dt=0.5)
            env.visualize()
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Environment config
    config_env = config['ENVIRONMENT']
    map_yaml_path = config_env['map_yaml']
    
    # Robot config
    config_robot = config['ROBOT']
    robot_footprint = ast.literal_eval(config_robot['footprint'])

    main(
        map_yaml_path,
        robot_footprint
        )