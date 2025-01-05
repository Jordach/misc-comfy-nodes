import numpy as np
from PIL import Image
import random
import torch
import math

def conv_pil_tensor(img):
	return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)

class PlasmaGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def _generate_displacement_map(self, size, roughness, seed):
        """Generate a displacement map for the entire grid at once"""
        torch.manual_seed(seed)
        # Create log2 steps for the total iterations needed
        steps = int(math.log2(size - 1))
        grid = torch.zeros((size, size), device=self.device, dtype=torch.float32)
        
        # Set initial corners
        grid[0, 0] = random.randint(0, 255)
        grid[0, -1] = random.randint(0, 255)
        grid[-1, 0] = random.randint(0, 255)
        grid[-1, -1] = random.randint(0, 255)
        
        # For each step, divide grid into smaller squares
        for step in range(steps):
            stride = size // (2 ** step)
            half_stride = stride // 2
            
            if half_stride < 1:
                break
                
            # Diamond step
            y_coords, x_coords = torch.meshgrid(
                torch.arange(half_stride, size, stride, device=self.device),
                torch.arange(half_stride, size, stride, device=self.device),
                indexing='ij'
            )
            
            # Calculate average of corners for each diamond
            top_left = grid[y_coords-half_stride, x_coords-half_stride]
            top_right = grid[y_coords-half_stride, x_coords+half_stride]
            bottom_left = grid[y_coords+half_stride, x_coords-half_stride]
            bottom_right = grid[y_coords+half_stride, x_coords+half_stride]
            
            # Add random displacement scaled by stride and roughness
            random_offset = (torch.rand_like(top_left) - 0.5) * stride * roughness
            center_values = (top_left + top_right + bottom_left + bottom_right) / 4.0 + random_offset
            
            # Update centers
            grid[y_coords, x_coords] = center_values
            
            # Square step
            # Handle edges specially to avoid index out of bounds
            for offset_y, offset_x in [(0, half_stride), (half_stride, 0)]:
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(offset_y, size, stride, device=self.device),
                    torch.arange(offset_x, size, stride, device=self.device),
                    indexing='ij'
                )
                
                # Calculate valid neighbor indices
                counts = torch.ones_like(y_coords, dtype=torch.float32) * 4
                values = torch.zeros_like(y_coords, dtype=torch.float32)
                
                # Add up valid neighbors
                for dy, dx in [(-half_stride, 0), (half_stride, 0), (0, -half_stride), (0, half_stride)]:
                    y_idx = y_coords + dy
                    x_idx = x_coords + dx
                    
                    valid_mask = (y_idx >= 0) & (y_idx < size) & (x_idx >= 0) & (x_idx < size)
                    values[valid_mask] += grid[y_idx[valid_mask], x_idx[valid_mask]]
                    counts[~valid_mask] -= 1
                
                # Add random displacement and update grid
                random_offset = (torch.rand_like(values) - 0.5) * stride * roughness
                grid[y_coords, x_coords] = (values / counts) + random_offset
        
        return torch.clamp(grid, 0, 255).to(torch.uint8)

    def generate_plasma(self, width, height, turbulence, value_min, value_max,
                       red_min, red_max, green_min, green_max,
                       blue_min, blue_max, seed):
        # Set dimensions to power of 2 + 1 for proper diamond-square
        size = 2 ** math.ceil(math.log2(max(width, height) - 1)) + 1
        
        # Generate three color channels
        r = self._generate_displacement_map(size, turbulence, seed)
        g = self._generate_displacement_map(size, turbulence, seed + 1)
        b = self._generate_displacement_map(size, turbulence, seed + 2)
        
        # Handle clamping with vectorized operations
        def get_clamp(min_val, max_val, global_min, global_max):
            return (global_min if min_val == -1 else min_val,
                   global_max if max_val == -1 else max_val)
        
        v_min, v_max = get_clamp(value_min, value_max, 0, 255)
        r_min, r_max = get_clamp(red_min, red_max, v_min, v_max)
        g_min, g_max = get_clamp(green_min, green_max, v_min, v_max)
        b_min, b_max = get_clamp(blue_min, blue_max, v_min, v_max)
        
        # Remap values using vectorized operations
        def remap_tensor(x, old_min, old_max, new_min, new_max):
            x = x.float()
            old_range = float(old_max - old_min)
            new_range = float(new_max - new_min)
            return ((x - old_min) * new_range / old_range + new_min).clamp(new_min, new_max).to(torch.uint8)
        
        # Remap and crop to desired dimensions
        r_mapped = remap_tensor(r[:height, :width], 0, 255, r_min, r_max)
        g_mapped = remap_tensor(g[:height, :width], 0, 255, g_min, g_max)
        b_mapped = remap_tensor(b[:height, :width], 0, 255, b_min, b_max)
        
        # Combine channels and convert to PIL Image
        rgb = torch.stack([r_mapped, g_mapped, b_mapped], dim=-1)
        return Image.fromarray(rgb.cpu().numpy())

plasma = PlasmaGenerator()

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
				# Does nothing because ComfyUI doesn't understand "static" output nodes
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
			}
		}

	RETURN_TYPES = ("IMAGE",)
	FUNCTION = "generate_plasma"
	CATEGORY = "image/noise"

	def generate_plasma(self, width, height, turbulence, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max, seed):
		vmin = value_min if value_min != -1 else 0
		vmax = value_max if value_max != -1 else 255
		rmin = red_min if red_min != -1 else vmin
		rmax = red_max if red_max != -1 else vmax
		gmin = green_min if green_min != -1 else vmin
		gmax = green_max if green_max != -1 else vmax
		bmin = blue_min if blue_min != -1 else vmin
		bmax = blue_max if blue_max != -1 else vmax
		img = plasma.generate_plasma(width, height, turbulence, vmin, vmax, rmin, rmax, gmin, gmax, bmin, bmax, seed)
		return conv_pil_tensor(img)

NODE_CLASS_MAPPINGS = {
	"JDC_PlasmaNoise_v2_GPU": PlasmaNoise_v2
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"JDC_PlasmaNoise_v2_GPU": "Plasma Noise v2 (GPU)"
}