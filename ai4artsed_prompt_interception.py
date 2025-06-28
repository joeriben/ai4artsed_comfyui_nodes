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
        # Preis- und Kategorie-Mapping für OpenRouter-Modelle
        model_info = {
            "anthropic/claude-3.5-haiku":     {"price": "$0.80/$4.00", "tag": "multilingual"},
            "anthropic/claude-3-haiku":       {"price": "$0.80/$4.00", "tag": "multilingual"},
            "deepseek/deepseek-chat-v3-0324": {"price": "$0.27/$1.10", "tag": "rule-oriented"},
            "deepseek/deepseek-r1-0528":      {"price": "$0.50/$2.15", "tag": "reasoning"},
            "google/gemini-2.5-flash":        {"price": "$0.20/$2.50", "tag": "multilingual"},
            "google/gemini-2.5-flash-lite-preview-06-17": {"price": "$0.10/$0.40", "tag": "multilingual"},
            "google/gemini-2.5-flash-preview-05-20":      {"price": "$0.15/$0.60", "tag": "multilingual"},
            "google/gemini-2.5-flash-preview-05-20:thinking": {"price": "$0.15/$3.50", "tag": "multilingual"},
            "google/gemini-2.5-pro":         {"price": "$1.25/$10.00", "tag": "translator"},
            "google/gemma-3-27b-it":         {"price": "$0.10/$0.18", "tag": "translator"},
            "meta-llama/llama-3.3-70b-instruct": {"price": "$0.59/$0.79", "tag": "rule-oriented"},
            "meta-llama/llama-guard-3-8b":      {"price": "$0.05/$0.10", "tag": "rule-oriented"},
            "meta-llama/llama-3.2-1b-instruct": {"price": "$0.05/$0.10", "tag": "rule-oriented"},
            "mistralai/mistral-medium-3":        {"price": "$0.40/$2.00", "tag": "rule-oriented"},
            "mistralai/mistral-small-3.1-24b-instruct": {"price": "$0.10/$0.30", "tag": "rule-oriented"},
            "mistralai/ministral-8b":            {"price": "$0.05/$0.10", "tag": "rule-oriented"},
            "mistralai/ministral-3b":            {"price": "$0.05/$0.10", "tag": "rule-oriented"},
            "mistralai/mixtral-8x7b-instruct":   {"price": "$0.45/$0.70", "tag": "cultural-expert"},
            "nvidia/llama-3.3-nemotron-super-49b-v1": {"price": "$0.13/$0.40", "tag": "reasoning"},
            "openai/gpt-4.1-nano": {"price": "$0.10/$0.40", "tag": "reasoning"},
            "openai/o3":           {"price": "$2.00/$8.00", "tag": "rule-oriented"},
            "openai/o3-mini":      {"price": "$1.10/$4.40", "tag": "rule-oriented"},
            "qwen/qwen3-32b":      {"price": "$0.10/$0.30", "tag": "translator"},
            "qwen/qwen3-235b-a22b": {"price": "$0.13/$0.60", "tag": "multilingual"}
        }
        openrouter_models = [
            f"openrouter/{model} [{info['tag']} / {info['price']}]"
            for model, info in model_info.items()
        ]

        # Ollama-Modelle lokal & kostenlos
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            ollama_raw = [m.get('name', '') for m in response.json().get("models", [])]
        except Exception:
            ollama_raw = []
        ollama_models = [f"local/{name} [local / $0.00]" for name in ollama_raw]

        return openrouter_models + ollama_models

    def get_api_credentials(self, user_input_key):
        # aus Umgebungsvariable oder Key-Datei
        if user_input_key.strip():
            return "https://openrouter.ai/api/v1/chat/completions", user_input_key.strip()
        key_path = os.path.join(os.path.dirname(__file__), "openrouter.key")
        with open(key_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
        return lines[1] if len(lines) > 1 else "", lines[0] if lines else ""

    def run(self, input_prompt, input_context, style_prompt, api_key, model, debug, unload_model):
        full_prompt = f"""
Task:
{style_prompt.strip()}

Context:
{input_context.strip()}

Prompt:
{input_prompt.strip()}
"""

        if model.startswith("openrouter/"):
            selected = model[len("openrouter/"):].split()[0]
            output_text = self.call_openrouter(full_prompt, selected, api_key, debug)[0]
        elif model.startswith("local/"):
            selected = model[len("local/"):].split()[0]
            output_text = self.call_ollama(full_prompt, selected, debug, unload_model)[0]
        else:
            raise Exception(f"Unbekannter Modell-Prefix: {model}")

        return self._format_outputs(output_text)

    def _format_outputs(self, text):
        output_str = text.strip()
        german = re.search(r"[-+]?\d{1,3}(?:\.\d{3})*,\d+", output_str)
        eng = re.search(r"[-+]?\d*\.\d+", output_str)
        output_float = 0.0
        if german:
            norm = german.group().replace('.', '').replace(',', '.')
            output_float = float(norm)
        elif eng:
            output_float = float(eng.group())
        output_int = int(round(output_float))
        binary = bool("true" in output_str.lower() or output_float != 0)
        return output_str, output_float, output_int, binary

    def call_openrouter(self, prompt, model, api_key, debug):
        api_url, real_key = self.get_api_credentials(api_key)
        headers = {"Authorization": f"Bearer {real_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [
            {"role": "system", "content": "You are a fresh assistant instance."},
            {"role": "user", "content": prompt}
        ], "temperature": 0.7}
        resp = requests.post(api_url, headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if debug == "enable": print("[OR] Prompt->", prompt, "\nResp->", content)
        return (content,)

    def call_ollama(self, prompt, model, debug, unload_model):
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post("http://localhost:11434/api/generate", json=payload)
            resp.raise_for_status()
            out = resp.json().get("response", "")
        except Exception as e:
            out = f"[Error Ollama] {e}"
        if unload_model == "yes":
            try:
                requests.post("http://localhost:11434/api/generate", json={"model": model, "prompt": "", "keep_alive": 0, "stream": False}, timeout=10)
            except:
                pass
        if debug == "enable": print("[OL] Prompt->", prompt, "\nResp->", out)
        return (out,)

