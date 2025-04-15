from typing import List, Tuple, Dict, Optional, Any, Union
import gym
import numpy as np

import yaml
from PIL import Image, ImageOps
from pathlib import Path


class Map:
    OCCUPIED = 0
    FREE = 255
    #UNKNOWN = 255
    OCCUPIED_THRESHOLD = 128
    UNACESSIBLE = 205 # gray area that is not accessible

    def __init__(self, map_yaml: Union[str, Path]):
        
        mr = self.__load_map(map_yaml)
        self.map:np.ndarray = mr[0]
        self.resolution:float = mr[1]
        self.origin:Tuple[float, float] = mr[2]
        #self.map_image:Image = mr[3]

        self.height, self.width = self.map.shape

        # TODO find other fix
        #self.width, self.height = self.height, self.width

        
        
        print(f"Loaded map with size: height={self.height}, width={self.width}")
        # other properties?

    def __call__(self) -> np.ndarray:
        """
        Returns the occupancy grid.
        """
        return self.map
 
    def __load_map(self, yaml_path: Union[str,Path]) -> Tuple[np.ndarray, float]:
        """
        Loads a map from PGM and YAML file.

        Returns:
            np.ndarray: Occupancy grid.
        """
        if not isinstance(yaml_path, Path):
                yaml_path = Path(yaml_path)

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

            map_dir = yaml_path.parent

            map_pgm_path = map_dir / config['image'] # join paths

            resolution = float(config['resolution'])

            origin = tuple(config['origin']) # (x, y, theta)

        img = Image.open(map_pgm_path)#.convert('L')
        
        # bullshit fix for bullshit problem
        # img = img.rotate(-90, expand=True)
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        img_array = np.asarray(img)
        
        return img_array, resolution, origin

    
    def world_to_map(self, x:float, y:float) -> Tuple[int, int]:
        """
        Converts world coordinates (meters) to map grid indices.
        Accounts for map origin (x, y, theta).
        """
        dx = x - self.origin[0]# + x
        dy = y - self.origin[1]# + y

        # TODO: rotate (dx, dy) if theta ≠ 0 (rare)
        mx = int(dx / self.resolution)
        my = int(dy / self.resolution)
        # change the y coordinate to match the map
        my = self.height - my - 1
        return mx, my
    
    def map_to_world(self, mx:int, my:int) -> Tuple[float, float]:
        """
        Converts map indices to world coordinates.
        """
        x = mx * self.resolution + self.origin[0]
        y = (self.height-my) * self.resolution + self.origin[1]
        return x, y

# test code
if __name__ == "__main__":
    map_yaml_path = Path("maps/warehouse_map.yaml")
    map = Map(map_yaml_path)
    print(f"Map loaded with resolution: {map.resolution}")
    print(f"Map origin: {map.origin} of type {type(map.origin)}")
    print(f"Map shape: {map.map.shape}")
    print(f"Map data:\n{map.map}")
    print(f"max value: {map.map.max()} min_value: {map.map.min()}")
    Image.fromarray(map.map).show()
