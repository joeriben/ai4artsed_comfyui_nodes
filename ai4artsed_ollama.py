# ai4artsed_ollama.py
import requests

class ai4artsed_ollama:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (cls.get_model_list(),),
                "system_prompt": ("STRING", {"multiline": True}),
                "unload_model": (["no", "yes"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "AI4ArtsEd"

    @staticmethod
    def get_model_list():
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
            return models if models else ["<no models found>"]
        except Exception as e:
            return [f"<error loading models: {str(e)}> "]

    def run(self, prompt, model="gemma3:27b", system_prompt=None, unload_model="no"):
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()
            result = response.json().get("response", "")
        except Exception as e:
            result = f"[Error from Ollama] {str(e)}"

        if unload_model == "yes":
            try:
                unload_payload = {
                    "model": model,
                    "prompt": "",
                    "keep_alive": 0,
                    "stream": False
                }
                requests.post(
                    "http://localhost:11434/api/generate",
                    json=unload_payload,
                    timeout=30
                )
            except Exception:
                pass

        return (result,)
