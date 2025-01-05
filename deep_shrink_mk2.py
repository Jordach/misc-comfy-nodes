import torch
import comfy.utils

class PatchModelAddDownscale_v2:
    upscale_methods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "block_number": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
            "downscale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 9.0, "step": 0.001}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
            "gradual_percent": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.001}),
            "downscale_after_skip": ("BOOLEAN", {"default": True}),
            "downscale_method": (s.upscale_methods,),
            "upscale_method": (s.upscale_methods,),
        }}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    def calculate_upscale_factor(self, current_percent, end_percent, gradual_percent, downscale_factor):
        """Calculate the upscale factor during the gradual resize phase"""
        if current_percent <= end_percent:
            return 1.0 / downscale_factor  # Still fully downscaled
        elif current_percent >= gradual_percent:
            return 1.0  # Fully back to original size
        else:
            # Linear interpolation between downscaled and original size
            progress = (current_percent - end_percent) / (gradual_percent - end_percent)
            scale_diff = 1.0 - (1.0 / downscale_factor)
            return (1.0 / downscale_factor) + (scale_diff * progress)

    def patch(self, model, block_number, downscale_factor, start_percent, end_percent,
             gradual_percent, downscale_after_skip, downscale_method, upscale_method):
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)
        sigma_rescale = model_sampling.percent_to_sigma(gradual_percent)
        
        def input_block_patch(h, transformer_options):
            if downscale_factor == 1:
                return h

            if transformer_options["block"][1] == block_number:
                sigma = transformer_options["sigmas"][0].item()
                
                # Normal downscale behavior between start_percent and end_percent
                if sigma <= sigma_start and sigma >= sigma_end:
                    h = comfy.utils.common_upscale(
                        h, 
                        round(h.shape[-1] * (1.0 / downscale_factor)), 
                        round(h.shape[-2] * (1.0 / downscale_factor)), 
                        downscale_method,
                        "disabled"
                    )
                # Gradually upscale latent after end_percent until gradual_percent
                elif sigma < sigma_end and sigma >= sigma_rescale:
                    scale_factor = self.calculate_upscale_factor(
                        sigma, sigma_rescale, sigma_end, downscale_factor
                    )
                    h = comfy.utils.common_upscale(
                        h,
                        round(h.shape[-1] * scale_factor),
                        round(h.shape[-2] * scale_factor),
                        upscale_method,
                        "disabled"
                    )
            return h

        def output_block_patch(h, hsp, transformer_options):
            if h.shape[2] != hsp.shape[2]:
                h = comfy.utils.common_upscale(
                    h, hsp.shape[-1], hsp.shape[-2], 
                    upscale_method, "disabled"
                )
            return h, hsp

        m = model.clone()
        if downscale_after_skip:
            m.set_model_input_block_patch_after_skip(input_block_patch)
        else:
            m.set_model_input_block_patch(input_block_patch)
        m.set_model_output_block_patch(output_block_patch)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "PatchModelAddDownscale_v2": PatchModelAddDownscale_v2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Sampling
    "PatchModelAddDownscale_v2": "PatchModelAddDownscale v2",
}