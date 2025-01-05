from colorsys import rgb_to_hsv
from collections import defaultdict
from io import BytesIO
from skimage import color as skcolor
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import comfy

def conv_pil_tensor(img):
	return (torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0),)

def conv_tensor_pil(tsr):
	return Image.fromarray(np.clip(255. * tsr.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def get_perceptual_palette(image, kernel_size=3):
	"""
	Extract a perceptual color palette from an image using block sampling,
	organizing colors into different hue ranges.
	
	Args:
		image: Input image array
		kernel_size: Size of sampling kernel (e.g. 8 for 8x8 blocks)
	"""
	img_array = np.array(image)
	height, width = img_array.shape[:2]
	
	# Sample the image using kernel_size x kernel_size blocks
	y_indices = np.arange(0, height, kernel_size)
	x_indices = np.arange(0, width, kernel_size)
	pixels = img_array[y_indices[:, np.newaxis], x_indices]
	pixels = pixels.reshape(-1, 3)
	
	# Define color ranges in HSV (hue values from 0-1)
	color_ranges = {
		'reds': (0.97, 0.035),
		'oranges': (0.035, 0.09),
		'yellows': (0.09, 0.195),
		'greens': (0.195, 0.40),
		'blues': (0.40, 0.65),
		'indigos': (0.65, 0.75),
		'violets': (0.75, 0.97)
	}
	
	# Initialize containers for categorized colors
	categorized_colors = defaultdict(list)
	
	# Vectorized conversion to HSV
	pixels_normalized = pixels / 255.0
	hsv_pixels = np.array([rgb_to_hsv(r, g, b) for r, g, b in pixels_normalized])
	h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
	
	# Identify grayscale pixels (vectorized)
	gray_mask = s < 0.03
	if np.any(gray_mask):
		categorized_colors['grayscale'] = pixels[gray_mask]
	
	# Identify skin tones (vectorized)
	skin_mask = (0.05 <= h) & (h <= 0.1) & (0.05 <= s) & (s <= 0.6) & (0.3 <= v) & (v <= 0.9)
	if np.any(skin_mask):
		categorized_colors['skintones'] = pixels[skin_mask]
	
	# Process remaining pixels
	remaining_mask = ~(gray_mask | skin_mask)
	remaining_h = h[remaining_mask]
	remaining_pixels = pixels[remaining_mask]
	
	# Categorize remaining pixels by hue range
	for category, (start, end) in color_ranges.items():
		if start > end:  # Handle red wrap-around
			mask = (remaining_h >= start) | (remaining_h <= end)
		else:
			mask = (remaining_h >= start) & (remaining_h <= end)
		if np.any(mask):
			categorized_colors[category] = remaining_pixels[mask]
	
	return {k: v for k, v in categorized_colors.items() if len(v) > 0}

def create_multi_row_palette_strip(palettes, max_colors, dpi=100):
	"""
	Create a multi-row palette strip visualization where each row represents
	a different color category. Colors are first filtered by perceptual distance,
	then uniformly sampled across the filtered range.
	
	Args:
		palettes: Dictionary of color categories and their pixels
		max_colors: Maximum number of colors to display per category
		dpi: Output DPI for the figure
	"""
	category_info = {
		'reds': 'R',
		'oranges': 'O', 
		'yellows': 'Y',
		'greens': 'G',
		'blues': 'B',
		'indigos': 'I',
		'violets': 'V',
		'skintones': 'S',
		'grayscale': 'N'  # N for Neutral
	}
	
	category_order = list(category_info.keys())
	active_categories = [cat for cat in category_order if cat in palettes and len(palettes[cat]) > 0]
	
	if not active_categories:
		return None
	
	fig, axes = plt.subplots(
		nrows=len(active_categories),
		ncols=1,
		figsize=(max_colors * 0.5 + 0.15, len(active_categories) * 0.5),
		dpi=dpi
	)
	
	if len(active_categories) == 1:
		axes = [axes]

	min_distance = 3.5  # Minimum perceptual distance in LAB space
	
	for idx, category in enumerate(active_categories):
		colors = palettes[category]
		if len(colors) == 0:
			continue
			
		# Convert to LAB space once
		lab_colors = skcolor.rgb2lab(colors.reshape(1, -1, 3) / 255.0)[0]
		
		# Efficient perceptual distance filtering using vectorized operations
		filtered_indices = [0]  # Always keep first color
		filtered_lab = [lab_colors[0]]
		
		# Pre-compute squared distances for efficiency
		for i in range(1, len(colors)):
			current_lab = lab_colors[i]
			# Compute distances to all previously filtered colors at once
			distances = np.sqrt(np.sum((np.array(filtered_lab) - current_lab) ** 2, axis=1))
			if np.all(distances >= min_distance):
				filtered_indices.append(i)
				filtered_lab.append(current_lab)
		
		# Get filtered colors
		filtered_colors = colors[filtered_indices]
		
		# Uniform sampling
		if len(filtered_colors) > max_colors:
			stride = len(filtered_colors) / max_colors
			sample_indices = [int(i * stride) for i in range(max_colors)]
			colors_final = filtered_colors[sample_indices]
		else:
			colors_final = filtered_colors
		
		# Sort by lightness for display
		lab_final = skcolor.rgb2lab(colors_final.reshape(1, -1, 3) / 255.0)[0]
		lightness = lab_final[:, 0]
		sorted_indices = np.argsort(lightness)
		colors_final = colors_final[sorted_indices]
		
		# Pad if necessary
		if len(colors_final) < max_colors:
			padding = np.full((max_colors - len(colors_final), 3), np.nan)
			colors_final = np.vstack([colors_final, padding])
		
		# Add label and display
		label = category_info[category]
		axes[idx].text(-0.02, 0.5, label,
					transform=axes[idx].transAxes,
					verticalalignment='center',
					horizontalalignment='right',
					fontsize=8,
					fontweight='bold')
		
		axes[idx].imshow(colors_final[np.newaxis, :, :] / 255.0)
		axes[idx].axis('off')
	
	plt.tight_layout(pad=0.3)
	return fig

def get_perceptual_palette_wrapper(image, kernel_size, max_colors):
	# Extract perceptual palettes
	palettes = get_perceptual_palette(image, kernel_size)
	
	# Create visualization
	fig = create_multi_row_palette_strip(palettes, max_colors)
	
	return fig

def create_fig_byte_stream(fig):
	buf = BytesIO()
	fig.savefig(buf, format="png", bbox_inches="tight")
	buf.seek(0)
	return buf

class PerceptualPalette:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"image": ("IMAGE",),
				"kernel_size": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
				"max_colours": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
			}
		}
	
	RETURN_TYPES = ("IMAGE",)

	FUNCTION = "process_palette"
	CATEGORY = "image"

	def process_palette(self, image, kernel_size, max_colours):
		pil_input = conv_tensor_pil(image)

		fig = get_perceptual_palette_wrapper(pil_input, kernel_size, max_colours)
		buffer = create_fig_byte_stream(fig)
		pil_result = Image.open(buffer).convert("RGB")
		print(pil_result.size)
		tsr_result = conv_pil_tensor(pil_result)

		return tsr_result

NODE_CLASS_MAPPINGS = {
	"JDC_PerceptualPalette": PerceptualPalette,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"JDC_PerceptualPalette": "Perceptual Palette"
}