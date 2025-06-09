import os
import json
import requests

class ai4artsed_prompt_interception:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_prompt": ("STRING", {"multiline": True}),
                "input_context": ("STRING", {"default": "", "multiline": True}),
                "style_prompt": ("STRING", {"default": "", "multiline": True}),
                "api_key": ("STRING", {"multiline": False, "password": True}),
                "model": (cls.get_combined_model_list(),),
                "debug": (["enable", "disable"],),
                "unload_model": (["no", "yes"],),
                "output_format": (["string", "float", "int", "binary"], {"default": "string"}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("output_str", "output_float", "output_int", "output_binary")
    FUNCTION = "run"
    CATEGORY = "AI4ArtsEd"

    @staticmethod
    def get_combined_model_list():
        openrouter_models = [
            "anthropic/claude-sonnet-4",
            "deepseek/deepseek-chat-v3-0324",
            "deepseek/deepseek-r1",
            "google/gemini-2.5-pro-preview",
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-guard-3-8b",
            "mistralai/mistral-medium-3",
            "mistralai/mistral-7b-instruct",
            "openai/o3"
        ]
        openrouter_models = [f"openrouter/{m}" for m in openrouter_models]

        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            ollama_models = [f"local/{m.get('name', '')}" for m in response.json().get("models", [])]
        except Exception:
            ollama_models = []

        return openrouter_models + ollama_models

    def get_api_credentials(self, user_input_key):
        if user_input_key.strip():
            return "https://openrouter.ai/api/v1/chat/completions", user_input_key.strip()

        key_path = os.path.join(os.path.dirname(__file__), "openrouter.key")
        try:
            with open(key_path, "r") as f:
                lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
                if len(lines) < 2:
                    raise Exception("Key-Datei unvollständig: Erwartet mindestens zwei nicht-kommentierte Zeilen.")
                api_key = lines[0]
                api_url = lines[1]
                return api_url, api_key
        except Exception as e:
            raise Exception(f"[Prompt Interception] Fehler beim Lesen der API-Zugangsdaten: {str(e)}")

    def run(self, input_prompt, input_context, style_prompt, api_key, model, debug, unload_model, output_format):
        full_prompt = (
            f"Task:\n{style_prompt.strip()}\n\n"
            f"Context:\n{input_context.strip()}\nPrompt:\n{input_prompt.strip()}"
        )

        if model.startswith("openrouter/"):
            output_text = self.call_openrouter(full_prompt, model.split("/", 1)[1], api_key, debug)[0]
        elif model.startswith("local/"):
            output_text = self.call_ollama(full_prompt, model.split("/", 1)[1], debug, unload_model)[0]
        else:
            raise Exception(f"Unbekannter Modell-Prefix in '{model}'. Erwartet 'openrouter/' oder 'local/'.")

        # Format output
        try:
            if output_format == "float":
                return output_text, float(output_text), 0, False
            elif output_format == "int":
                return output_text, 0.0, int(output_text), False
            elif output_format == "binary":
                bin_val = output_text.strip().lower() in ["1", "true", "yes"]
                return output_text, 0.0, 0, bin_val
            else:  # string
                return output_text, 0.0, 0, False
        except Exception:
            raise Exception(f"Fehler beim Konvertieren von Output '{output_text}' in das Format '{output_format}'")

    def call_openrouter(self, prompt, model, api_key, debug):
        api_url, real_api_key = self.get_api_credentials(api_key)
        headers = {"Authorization": f"Bearer {real_api_key}", "Content-Type": "application/json"}
        messages = [
            {"role": "system", "content": "You are a fresh assistant instance. Forget all previous conversation history."},
            {"role": "user", "content": prompt}
        ]
        payload = {"model": model, "messages": messages, "temperature": 0.7}

        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            raise Exception(f"[OpenRouter] API Error: {response.status_code}\n{response.text}")

        result = response.json()
        output_text = result["choices"][0]["message"]["content"]

        if debug == "enable":
            print(">>> AI4ARTSED PROMPT INTERCEPTION NODE <<<")
            print("Model:", model)
            print("Prompt sent:\n", prompt)
            print("Response received:\n", output_text)

        return (output_text,)

    def call_ollama(self, prompt, model, debug, unload_model):
        payload = {"model": model, "prompt": prompt, "system": "You are a fresh assistant instance. Forget all previous conversation history.", "stream": False}

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()
            output = response.json().get("response", "")
        except Exception as e:
            output = f"[Error from Ollama] {str(e)}"

        if unload_model == "yes":
            try:
                unload_payload = {"model": model, "prompt": "", "keep_alive": 0, "stream": False}
                requests.post("http://localhost:11434/api/generate", json=unload_payload, timeout=30)
            except Exception:
                pass

        if debug == "enable":
            print(">>> AI4ARTSED PROMPT INTERCEPTION NODE <<<")
            print("Model:", model)
            print("Prompt sent:\n", prompt)
            print("Response received:\n", output)

        return (output,)

