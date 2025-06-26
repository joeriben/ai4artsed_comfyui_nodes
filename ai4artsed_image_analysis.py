import os
import json
import requests
import re
import base64
import io
from PIL import Image

class ai4artsed_image_analysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (cls.get_combined_model_list(), {"default": "local/llava:13b"}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"multiline": False, "password": True}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("output_str", "output_float", "output_int", "output_binary")
    FUNCTION = "run"
    CATEGORY = "AI4ArtsEd"

    @staticmethod
    def get_combined_model_list():
        # OpenRouter models
        openrouter_models = ["llava:7b", "llava:13b", "llava:34b"]
        openrouter_models = [f"openrouter/{m}" for m in openrouter_models]
        # Ollama local models
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            ollama_models = [f"local/{m.get('name','')}" for m in response.json().get("models",[])]
        except Exception:
            ollama_models = []
        return openrouter_models + ollama_models

    def run(self, image, prompt, model, system_prompt=None, api_key=""):
        # Convert image tensor to JPEG base64
        array = image[0].detach().cpu().numpy()
        pil = Image.fromarray((array * 255).astype("uint8"))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Prepare payload
        full_prompt = prompt.strip()
        payload = {"model": model.split("/",1)[1],
                   "prompt": full_prompt,
                   "images": [img_b64],
                   "stream": False}
        if system_prompt:
            payload["system"] = system_prompt

        # Call chosen backend
        if model.startswith("openrouter/"):
            api_url, real_key = self.get_api_credentials(api_key)
            headers = {"Authorization": f"Bearer {real_key}", "Content-Type": "application/json"}
            # OpenRouter uses messages format
            open_payload = {
                "model": payload["model"],
                "messages": [
                    {"role": "system", "content": system_prompt or "You are a multimodal assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                "images": payload["images"],
                "temperature": 0.7
            }
            try:
                r = requests.post(api_url, headers=headers, json=open_payload)
                r.raise_for_status()
                output = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                output = f"[Error from OpenRouter] {e}"

        elif model.startswith("local/"):
            try:
                r = requests.post("http://localhost:11434/api/generate", json=payload)
                r.raise_for_status()
                output = r.json().get("response","")
            except Exception as e:
                output = f"[Error from Ollama] {e}"
        else:
            output = f"[Error] Unknown model prefix: {model}"

        # Parse return values
        text = output.strip()
        # Extract float
        num = re.search(r"[-+]?\d*\.\d+", text)
        fval = float(num.group()) if num else 0.0
        # Extract int
        it = re.search(r"[-+]?\d+", text)
        ival = int(it.group()) if it else 0
        # Boolean
        bval = any(x in text.lower() for x in ["true","yes"]) or (num and float(num.group())!=0)

        return text, fval, ival, bval

    def get_api_credentials(self, key):
        if key.strip():
            return "https://openrouter
