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
                "debug": (["enable", "disable"], {"default": "disable"}),
                "unload_model": (["no", "yes"], {"default": "no"}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("output_str", "output_float", "output_int", "output_binary")
    FUNCTION = "run"
    CATEGORY = "AI4ArtsEd"

    @staticmethod
    def get_combined_model_list():
        llava_variants = ["llava:7b", "llava:13b", "llava:34b"]
        ollama_models = [f"local/{name}" for name in llava_variants]
        openrouter_models = [f"openrouter/{name}" for name in llava_variants]
        return ollama_models + openrouter_models

    def run(self, image, prompt, model, system_prompt=None, api_key="", debug="disable", unload_model="no"):
        # Convert image tensor to JPEG base64
        array = image[0].detach().cpu().numpy()
        pil_img = Image.fromarray((array * 255).astype("uint8"))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        full_prompt = prompt.strip()
        payload = {
            "model": model.split("/", 1)[1],
            "prompt": full_prompt,
            "images": [img_b64],
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt

        # Dispatch to backends
        output = ""
        if model.startswith("openrouter/"):
            api_url, real_key = self.get_api_credentials(api_key)
            headers = {"Authorization": f"Bearer {real_key}", "Content-Type": "application/json"}
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
                output = f"[Error OpenRouter] {e}"

        elif model.startswith("local/"):
            try:
                r = requests.post("http://localhost:11434/api/generate", json=payload)
                r.raise_for_status()
                output = r.json().get("response", "")
            except Exception as e:
                output = f"[Error Ollama] {e}"
            # Unload model if requested
            if unload_model == "yes":
                try:
                    unload_payload = {"model": payload["model"], "prompt": "", "keep_alive": 0, "stream": False}
                    requests.post("http://localhost:11434/api/generate", json=unload_payload, timeout=30)
                except:
                    pass

        else:
            output = f"[Error] Unknown model prefix: {model}"

        if debug == "enable":
            print("--- AI4ArtsEd Image Analysis Debug ---")
            print("Model:", model)
            print("Prompt:", full_prompt)
            print("Output:", output)

        # Parse return values
        text = output.strip()
        num_match = re.search(r"[-+]?\d*\.\d+", text)
        fval = float(num_match.group()) if num_match else 0.0
        int_match = re.search(r"[-+]?\d+", text)
        ival = int(int_match.group()) if int_match else 0
        bval = any(x in text.lower() for x in ["true", "yes"]) or (num_match and float(num_match.group()) != 0)

        return text, fval, ival, bval

    def get_api_credentials(self, key):
        if key.strip():
            return "https://openrouter.ai/api/v1/chat/completions", key.strip()
        key_path = os.path.join(os.path.dirname(__file__), "openrouter.key")
        try:
            with open(key_path, "r") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                return lines[1], lines[0]
        except:
            return "https://openrouter.ai/api/v1/chat/completions", ""
