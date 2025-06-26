import torch
from diffusers import AudioLDMPipeline

class ai4artsed_audio_ldm2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0}),
                "num_inference_steps": ("INT", {"default": 200, "min": 50, "max": 500}),
                "seed": ("INT", {"default": 42, "min": 0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "AI4ArtsEd"

    def __init__(self):
        # Pipeline einmalig laden (float16 auf GPU)
        self.pipe = AudioLDMPipeline.from_pretrained(
            "cvssp/audioldm2-music", torch_dtype=torch.float16
        ).to("cuda")

    def run(self, prompt, guidance_scale, num_inference_steps, seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        audio_tensor = result.audios[0]
        audio_array = audio_tensor.cpu().numpy()
        return (audio_array,)
