import math

class AutoImageSize:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"aspect_ratio": ("FLOAT", {"default": 1, "min": 1, "max": 8, "step": 0.01}),
				"orientation": (["portrait", "landscape"],),
				"target_resolution": ("INT", {"default": 1024, "min": 256, "max": 1024*8, "step": 1}),
				"base_resolution": ("INT", {"default": 1024, "min": 256, "max": 1024*8, "step": 1}),
				"compression_factor": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
			}
		}

	RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
	RETURN_NAMES = ("WIDTH", "HEIGHT", "DOWNSCALE_FACTOR", "DENOISE_STRENGTH")
	FUNCTION = "create_res"

	CATEGORY = "utils"

	def calculate_denoise_strength(self, scale_factor):
		"""
		Calculate appropriate denoise strength based on resolution scale factor.
		Uses exponential decay curve fitted to known good values:
		- 1.0x (1024px) → 0.75
		- 1.5x (1536px) → 0.45
		- 2.0x (2048px) → 0.2
		"""
		# Base denoise value for 1024px (scale_factor = 1.0)
		base_denoise = 0.95
		
		# Calculate denoise strength using exponential decay
		# Formula: denoise = base_denoise * e^(-k * (scale_factor - 1))
		# where k is calculated to fit our known points
		# Decay constant fitted to match reference points
		k = 1.55
		
		denoise = base_denoise * math.exp(-k * (scale_factor - 1))
		d_min = 0.1
		d_max = 0.65
		# Clamp the result between 0.1 and 0.6
		return max(d_min, min(d_max, denoise))

	def create_res(self, aspect_ratio, orientation, target_resolution, base_resolution, compression_factor):
		# Prevent cases where DOWNSCALE_FACTOR can be < 1
		if target_resolution < base_resolution:
			target_resolution = base_resolution

		w, h = target_resolution, target_resolution
		if orientation == "portrait":
			w = int((((target_resolution**2)/aspect_ratio)**0.5)//compression_factor)*compression_factor
			h = int((((target_resolution**2)*aspect_ratio)**0.5)//compression_factor)*compression_factor
		elif orientation == "landscape":
			w = int((((target_resolution**2)*aspect_ratio)**0.5)//compression_factor)*compression_factor
			h = int((((target_resolution**2)/aspect_ratio)**0.5)//compression_factor)*compression_factor
		
		scale_factor = target_resolution/base_resolution
		denoise_strength = self.calculate_denoise_strength(scale_factor)
		
		return (w, h, scale_factor, denoise_strength)

NODE_CLASS_MAPPINGS = {
	"JDC_AutoImageSize": AutoImageSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"JDC_AutoImageSize": "Easy Aspect Ratios"
}