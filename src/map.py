from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np

from shapely.geometry import Polygon
from scipy.ndimage import binary_dilation
from skimage.draw import polygon

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

        img = Image.open(map_pgm_path)
            
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

    def check_map_collisions(self, robot) -> bool:
        """
        Checks if robot collides with the envirnment.

        Args:
            robot (Robot): Robot object.

        Returns:
            True if there is a collision, False otherwise.
        """
        map_data = self.map
        # colliding = []

        #for idx, robot in enumerate(self.robots, start=1):
        poly = Polygon(robot.get_bbox())
        minx, miny, maxx, maxy = poly.bounds

        # Convert bounding rectangle to grid range
        min_col, min_row = self.world_to_map(minx, miny)
        max_col, max_row = self.world_to_map(maxx, maxy)

        row_start, row_end = min(min_row, max_row), max(min_row, max_row)
        col_start, col_end = min(min_col, max_col), max(min_col, max_col)

        # TODO this checks if a axis aligned rectangle is out of bounds
        # this should check if the point belongs to robot polygon
        sub_map = map_data[row_start:row_end+1, col_start:col_end+1]
        return np.any(sub_map != self.FREE)
    
    def inflate_map(self, footprint:np.ndarray) -> None:
        """
        Inflates the map based on the robot's footprint polygon (in meters).
        Marks inflated (but not originally occupied) cells as UNACESSIBLE.

        Args:
            footprint : np.ndarray of shape (4, 2) in meters. Robot's footprint.
        """
        

        # Get footprint polygon (in meters, relative to robot center)
        footprint_m = footprint  # shape (4, 2)
        resolution = self.resolution

        # Convert to grid coordinates relative to center
        footprint_cells = footprint_m / resolution

        # Shift so it's all positive (for rasterizing)
        min_x = int(np.floor(np.min(footprint_cells[:, 0])))
        min_y = int(np.floor(np.min(footprint_cells[:, 1])))
        shifted = footprint_cells - np.array([min_x, min_y])
        h = int(np.ceil(np.max(shifted[:, 1]))) + 1
        w = int(np.ceil(np.max(shifted[:, 0]))) + 1

        rr, cc = polygon(shifted[:, 1], shifted[:, 0], shape=(h, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[rr, cc] = 1  # structuring element

        # Binary obstacle map
        obstacle_mask = (self.map <= self.OCCUPIED_THRESHOLD).astype(bool)

        # Dilate using the footprint
        inflated = binary_dilation(obstacle_mask, structure=mask)

        # Update map: mark inflated (but not original) as UNACESSIBLE
        self.inflated_map =np.copy(self.map)
        self.inflated_map[(inflated == 1) & (obstacle_mask == 0)] = self.UNACESSIBLE
        print("Inflation complete.")
        

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
