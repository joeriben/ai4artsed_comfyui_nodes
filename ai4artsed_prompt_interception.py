import os
import json
import requests
import re

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
                "unload_model": (["no", "yes", "enable", "disable"], {"default": "no"}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("output_str", "output_float", "output_int", "output_binary")
    FUNCTION = "run"
    CATEGORY = "AI4ArtsEd"

    @staticmethod
    def get_combined_model_list():
        openrouter_models = [
            "anthropic/claude-3-haiku",
            "anthropic/claude-sonnet-4",
            "deepseek/deepseek-chat-v3-0324",
            "deepseek/deepseek-r1",
            "google/gemini-2.5-pro-preview",
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-guard-3-8b",
            "meta-llama/llama-3.2-1b-instruct",
            "mistralai/mistral-medium-3",
            "mistralai/mistral-small-3.1-24b-instruct",
            "mistralai/ministral-8b",
            "mistralai/ministral-3b",
            "mistralai/mixtral-8x7b-instruct",
            "openai/o3",
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen-2.5-7b-instruct"
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

    def run(self, input_prompt, input_context, style_prompt, api_key, model, unload_model):
        # Legacy-Kompatibilität zu alten Flows ("enable" → "yes", "disable" → "no")
        if unload_model == "enable":
            unload_model = "yes"
        elif unload_model == "disable":
            unload_model = "no"

        full_prompt = (
            f"Task:\n{style_prompt.strip()}\n\n"
            f"Context:\n{input_context.strip()}\nPrompt:\n{input_prompt.strip()}"
        )

        if model.startswith("openrouter/"):
            output_text = self.call_openrouter(full_prompt, model.split("/", 1)[1], api_key)[0]
        elif model.startswith("local/"):
            output_text = self.call_ollama(full_prompt, model.split("/", 1)[1], unload_model)[0]
        else:
            raise Exception(f"Unbekannter Modell-Prefix in '{model}'. Erwartet 'openrouter/' oder 'local/'.")

        # Formatierung: alle vier Rückgabeformate parallel mit Failsafes
        output_str = output_text.strip()

        # Musterdefinitionen
        german_pattern = r"[-+]?\d{1,3}(?:\.\d{3})*,\d+"
        english_pattern = r"[-+]?\d*\.\d+"
        int_pattern = r"[-+]?\d+"

        # Failsafe Float: deutsche Formate zuerst
        m = re.search(german_pattern, output_str)
        if m:
            num = m.group()
            normalized = num.replace(".", "").replace(",", ".")
            try:
                output_float = float(normalized)
            except:
                output_float = 0.0
        else:
            m = re.search(english_pattern, output_str)
            if m:
                try:
                    output_float = float(m.group())
                except:
                    output_float = 0.0
            else:
                m = re.search(int_pattern, output_str)
                if m:
                    try:
                        output_float = float(m.group())
                    except:
                        output_float = 0.0
                else:
                    output_float = 0.0

        # Failsafe Int
        m_int = re.search(int_pattern, output_str)
        if m_int:
            try:
                output_int = int(round(float(m_int.group())))
            except:
                output_int = 0
        else:
            output_int = 0

        # Failsafe Binary
        lower = output_str.lower()
        num_match = re.search(english_pattern, output_str) or re.search(int_pattern, output_str)
        if (
            "true" in lower
            or re.search(r"\b1\b", lower)
            or (num_match and float(num_match.group()) != 0)
        ):
            output_binary = True
        else:
            output_binary = False

        return output_str, output_float, output_int, output_binary

    def call_openrouter(self, prompt, model, api_key):
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
        return (output_text,)

    def call_ollama(self, prompt, model, unload_model):
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
            except:
                pass

        return (output,)
