from typing import Optional, Tuple
import numpy as np
from PIL import Image
import random
import torch
from numba import jit, float32, int32
import math

def conv_pil_tensor(img):
	return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)

@jit(nopython=True)
def _remap(val: float, min_val: float, max_val: float, min_map: float, max_map: float) -> float:
    return (val - min_val) / (max_val - min_val) * (max_map - min_map) + min_map

@jit(nopython=True)
def _adjust(pixmap: np.ndarray, xa: int, ya: int, x: int, y: int, xb: int, yb: int, 
           roughness: float) -> None:
    if pixmap[x, y] == 0:
        d = abs(xa - xb) + abs(ya - yb)
        # Using numpy's random number generator compatible with Numba
        noise = np.random.random() - 0.555
        v = (pixmap[xa, ya] + pixmap[xb, yb]) / 2.0 + noise * d * roughness
        rand_offset = (np.random.random() * 96) - 48
        c = int(abs(v + rand_offset))
        pixmap[x, y] = min(max(c, 0), 255)

@jit(nopython=True)
def _subdivide(pixmap: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
               roughness: float) -> None:
    if not ((x2 - x1 < 2.0) and (y2 - y1 < 2.0)):
        x = int((x1 + x2) / 2.0)
        y = int((y1 + y2) / 2.0)
        
        # Diamond step
        if pixmap[x, y] == 0:
            v = int((pixmap[x1, y1] + pixmap[x2, y1] + 
                    pixmap[x2, y2] + pixmap[x1, y2]) / 4.0)
            pixmap[x, y] = v
        
        # Square step
        _adjust(pixmap, x1, y1, x, y1, x2, y1, roughness)  # Top
        _adjust(pixmap, x2, y1, x2, y, x2, y2, roughness)  # Right
        _adjust(pixmap, x1, y2, x, y2, x2, y2, roughness)  # Bottom
        _adjust(pixmap, x1, y1, x1, y, x1, y2, roughness)  # Left
        
        # Recursive subdivide
        _subdivide(pixmap, x1, y1, x, y, roughness)
        _subdivide(pixmap, x, y1, x2, y, roughness)
        _subdivide(pixmap, x, y, x2, y2, roughness)
        _subdivide(pixmap, x1, y, x, y2, roughness)

@jit(nopython=True)
def _generate_channel(size: int, roughness: float, seed: int) -> np.ndarray:
    # Initialize the random number generator with the seed
    np.random.seed(seed)
    pixmap = np.zeros((size, size), dtype=np.float32)
    
    # Initialize corners with random values
    corners = [(0, 0), (size-1, 0), (size-1, size-1), (0, size-1)]
    for x, y in corners:
        pixmap[x, y] = np.random.random() * 255
    
    _subdivide(pixmap, 0, 0, size-1, size-1, roughness)
    return pixmap

class NumbaPlasmaGenerator:
    def __init__(self, width: int, height: int, turbulence: float, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.size = max(width, height)
        self.roughness = turbulence
        self.seed = seed if seed is not None else random.randint(0, 2**32-1)
        
    def generate(self, 
                value_range: Tuple[int, int] = (-1, -1),
                red_range: Tuple[int, int] = (-1, -1),
                green_range: Tuple[int, int] = (-1, -1),
                blue_range: Tuple[int, int] = (-1, -1)) -> Image.Image:
        """
        Generate plasma effect with specified color ranges using Numba optimization.
        
        Args:
            value_range: Global value range (min, max)
            red_range: Red channel range (min, max)
            green_range: Green channel range (min, max)
            blue_range: Blue channel range (min, max)
            
        Returns:
            PIL Image with plasma effect
        """
        # Generate color channels using Numba-optimized functions
        channels = [_generate_channel(self.size, self.roughness, self.seed + i) 
                   for i in range(3)]
        
        # Handle value ranges
        v_min, v_max = (0, 255) if value_range == (-1, -1) else value_range
        ranges = [
            (v_min if r[0] == -1 else r[0], v_max if r[1] == -1 else r[1])
            for r in [red_range, green_range, blue_range]
        ]
        
        # Create output array
        output = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Apply color mapping
        for i in range(3):
            channel = channels[i][:self.width, :self.height]
            mapped = _remap(channel, 0, 255, ranges[i][0], ranges[i][1])
            output[:, :, i] = np.clip(mapped, 0, 255).T
        
        return Image.fromarray(output)

class PlasmaNoise_v2:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"width": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),
				"height": ("INT", {
					"default": 512,
					"min": 128,
					"max": 8192,
					"step": 8
				}),

				"turbulence": ("FLOAT", {
					"default": 0.5,
					"min": 0,
					"max": 32,
					"step": 0.00001
				}),
				
				"value_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"value_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),

				"red_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"red_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"green_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_min": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
				"blue_max": ("INT", {
					"default": -1,
					"min": -1,
					"max": 255,
					"step": 1
				}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "generate_plasma"
	CATEGORY = "image/noise"

	def generate_plasma(self, width, height, turbulence, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max):
		vmin = value_min if value_min != -1 else 0
		vmax = value_max if value_max != -1 else 255
		rmin = red_min if red_min != -1 else vmin
		rmax = red_max if red_max != -1 else vmax
		gmin = green_min if green_min != -1 else vmin
		gmax = green_max if green_max != -1 else vmax
		bmin = blue_min if blue_min != -1 else vmin
		bmax = blue_max if blue_max != -1 else vmax
		seed = random.randint(0, 4294967292)
		plasma = NumbaPlasmaGenerator(width, height, turbulence, seed)
		# Generate image with custom color ranges
		img = plasma.generate(
			value_range=(vmin, vmax),
			red_range=(rmin, rmax),
			green_range=(gmin, gmax),
			blue_range=(bmin, bmax)
		)
		return conv_pil_tensor(img)

	@classmethod
	def IS_CHANGED(self, width, height, turbulence, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max):
		return random.randint(0, 4294967292)

NODE_CLASS_MAPPINGS = {
	"JDC_PlasmaNoise_v2": PlasmaNoise_v2
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"JDC_PlasmaNoise_v2": "Plasma Noise v2"
}