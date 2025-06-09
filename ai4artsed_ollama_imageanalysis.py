import base64
import requests
import io
from PIL import Image

class ai4artsed_ollama_imageanalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": ([
                    "llava",
                    "llava:7b",
                    "llava-phi",
                    "bakllava",
                    "bakllava:7b"
                ],),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "analyze"
    CATEGORY = "AI4ArtsEd"

    def analyze(self, image, prompt, model="llava", system_prompt=None):
        image_np = image[0].detach().cpu().numpy()
        img_pil = Image.fromarray((image_np * 255).astype("uint8"))
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()
            output = response.json().get("response", "")
        except Exception as e:
            output = f"[Error from Ollama] {str(e)}"

        return (output,)
