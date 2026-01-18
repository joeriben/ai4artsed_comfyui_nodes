import os
import json
import requests
import re
import random

class ai4artsed_photo_prompt_randomizer:

    FILM_TYPES = {
        "random": None,  # Will be randomly selected at runtime
        "Kodachrome": "a Kodachrome film slide",
        "Ektachrome": "an Ektachrome film slide",
        "Portra 400": "a Kodak Portra 400 color negative",
        "Portra 800": "a Kodak Portra 800 color negative",
        "Ektar 100": "a Kodak Ektar 100 color negative",
        "Fuji Pro 400H": "a Fujifilm Pro 400H color negative",
        "Fuji Superia": "a Fujifilm Superia color negative",
        "CineStill 800T": "a CineStill 800T tungsten-balanced color negative",
        "Ilford HP5": "an Ilford HP5 Plus black and white negative",
        "Ilford Delta 400": "an Ilford Delta 400 black and white negative",
        "Ilford FP4": "an Ilford FP4 Plus black and white negative",
        "Ilford Pan F": "an Ilford Pan F Plus 50 black and white negative",
        "Ilford XP2": "an Ilford XP2 Super chromogenic black and white negative",
        "Tri-X 400": "a Kodak Tri-X 400 black and white negative",
    }

    DEFAULT_SYSTEM_PROMPT = """You are an inventive creative. Your task is to invent a REALISTIC photographic image prompt.

Think globally. Avoid cultural clichés. Avoid "retro" style descriptions.
Describe contemporary everyday motives: scenes, objects, animals, nature, tech, culture, people, homes, family, work, holiday, urban, rural, trivia, details.

Choose either unlikely, untypical or typical photographical sujets for realistic photographic images. Be verbose, provide intricate details.

Always begin your output with: "{film_description} of".
Transform the prompt strictly following the context if provided.

NO META-COMMENTS, TITLES, Remarks, dialogue WHATSOEVER, STRICTLY FOLLOW THE INSTRUCTION."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "film_type": (list(cls.FILM_TYPES.keys()),),
                "system_prompt": ("STRING", {"multiline": True, "default": cls.DEFAULT_SYSTEM_PROMPT}),
                "model": (cls.get_combined_model_list(),),
                "debug": (["disable", "enable"], {"default": "disable"}),
                "unload_model": (["no", "yes"], {"default": "no"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "password": True, "default": ""}),
                "context": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "AI4ArtsEd"

    # Force re-execution every time
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def get_combined_model_list():
        model_info = {
            "anthropic/claude-3.5-haiku": {"price": "$0.80/$4.00", "tag": "multilingual"},
            "anthropic/claude-3-haiku": {"price": "$0.80/$4.00", "tag": "multilingual"},
            "anthropic/claude-haiku-4.5": {"price": "$0.80/$4.00", "tag": "multilingual"},
            "anthropic/claude-sonnet-4.5": {"price": "$3.00/$15.00", "tag": "multilingual"},
            "deepseek/deepseek-chat-v3-0324": {"price": "$0.27/$1.10", "tag": "rule-oriented"},
            "deepseek/deepseek-r1-0528": {"price": "$0.50/$2.15", "tag": "reasoning"},
            "deepseek/deepseek-v3.2": {"price": "$0.27/$1.10", "tag": "rule-oriented"},
            "google/gemini-2.5-flash": {"price": "$0.20/$2.50", "tag": "multilingual"},
            "google/gemini-2.5-flash-lite-preview-06-17": {"price": "$0.10/$0.40", "tag": "multilingual"},
            "google/gemini-2.5-flash-preview-05-20": {"price": "$0.15/$0.60", "tag": "multilingual"},
            "google/gemini-2.5-flash-preview-05-20:thinking": {"price": "$0.15/$3.50", "tag": "multilingual"},
            "google/gemini-2.5-pro": {"price": "$1.25/$10.00", "tag": "translator"},
            "google/gemma-3-27b-it": {"price": "$0.10/$0.18", "tag": "translator"},
            "meta-llama/llama-3.3-70b-instruct": {"price": "$0.59/$0.79", "tag": "rule-oriented"},
            "meta-llama/llama-guard-3-8b": {"price": "$0.05/$0.10", "tag": "rule-oriented"},
            "meta-llama/llama-3.2-1b-instruct": {"price": "$0.05/$0.10", "tag": "reasoning"},
            "mistralai/mistral-medium-3": {"price": "$0.40/$2.00", "tag": "reasoning"},
            "mistralai/mistral-small-3.1-24b-instruct": {"price": "$0.10/$0.30", "tag": "rule-oriented, vision"},
            "mistralai/mistral-nemo": {"price": "$0.01/$0.001", "tag": "multilingual"},
            "mistralai/ministral-8b": {"price": "$0.05/$0.10", "tag": "rule-oriented"},
            "mistralai/ministral-3b": {"price": "$0.05/$0.10", "tag": "rule-oriented"},
            "mistralai/mixtral-8x7b-instruct": {"price": "$0.45/$0.70", "tag": "cultural-expert"},
            "nvidia/llama-3.3-nemotron-super-49b-v1": {"price": "$0.13/$0.40", "tag": "reasoning"},
            "openai/gpt-4.1-nano": {"price": "$0.10/$0.40", "tag": "reasoning"},
            "openai/o3": {"price": "$2.00/$8.00", "tag": "rule-oriented"},
            "openai/o3-mini": {"price": "$1.10/$4.40", "tag": "rule-oriented"},
            "openai/gpt-oss-120b": {"price": "$0.50/$2.00", "tag": "reasoning"},
            "openai/gpt-oss-20b": {"price": "$0.10/$0.40", "tag": "reasoning"},
            "qwen/qwen3-32b": {"price": "$0.10/$0.30", "tag": "translator"},
            "qwen/qwen3-235b-a22b": {"price": "$0.13/$0.60", "tag": "multilingual"}
        }

        openrouter_models = [
            f"openrouter/{model} [{info['tag']} / {info['price']}]"
            for model, info in model_info.items()
        ]

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            ollama_raw = [m.get('name', '') for m in response.json().get("models", [])]
            ollama_models = [f"local/{name} [local / $0.00]" for name in ollama_raw]
        except Exception:
            ollama_models = []

        return openrouter_models + ollama_models

    def extract_model_name(self, full_model_string):
        """Extrahiert den echten Modellnamen aus dem Dropdown-String"""
        if full_model_string.startswith("openrouter/"):
            without_prefix = full_model_string[11:]
            model_name = without_prefix.split(" [")[0]
            return model_name
        elif full_model_string.startswith("local/"):
            without_prefix = full_model_string[6:]
            model_name = without_prefix.split(" [")[0]
            return model_name
        else:
            return full_model_string

    def find_openrouter_fallback(self, failed_model, api_key, debug):
        """Sucht ähnliches OpenRouter-Modell nur bei Fehlern"""
        if debug == "enable":
            print(f"[FALLBACK] Suche Ersatz für: {failed_model}")

        model_parts = failed_model.split("/")
        if len(model_parts) >= 2:
            provider = model_parts[0]
            model_name = model_parts[1]

            size_match = re.search(r'(\d+)b', model_name.lower())
            target_size = int(size_match.group(1)) if size_match else None

            known_models = {
                "anthropic/claude-3.5-haiku": {"size": None, "provider": "anthropic"},
                "anthropic/claude-3-haiku": {"size": None, "provider": "anthropic"},
                "anthropic/claude-haiku-4.5": {"size": None, "provider": "anthropic"},
                "anthropic/claude-sonnet-4.5": {"size": None, "provider": "anthropic"},
                "deepseek/deepseek-chat-v3-0324": {"size": None, "provider": "deepseek"},
                "deepseek/deepseek-r1-0528": {"size": None, "provider": "deepseek"},
                "deepseek/deepseek-v3.2": {"size": None, "provider": "deepseek"},
                "meta-llama/llama-3.3-70b-instruct": {"size": 70, "provider": "meta-llama"},
                "meta-llama/llama-guard-3-8b": {"size": 8, "provider": "meta-llama"},
                "meta-llama/llama-3.2-1b-instruct": {"size": 1, "provider": "meta-llama"},
                "mistralai/mistral-medium-3": {"size": None, "provider": "mistralai"},
                "mistralai/mistral-small-3.1-24b-instruct": {"size": 24, "provider": "mistralai"},
                "mistralai/ministral-8b": {"size": 8, "provider": "mistralai"},
                "mistralai/ministral-3b": {"size": 3, "provider": "mistralai"},
                "mistralai/mixtral-8x7b-instruct": {"size": 56, "provider": "mistralai"},
                "qwen/qwen3-32b": {"size": 32, "provider": "qwen"},
                "qwen/qwen3-235b-a22b": {"size": 235, "provider": "qwen"},
                "openai/gpt-oss-120b": {"size": 120, "provider": "openai"},
                "openai/gpt-oss-20b": {"size": 20, "provider": "openai"},
            }

            fallback_candidates = []

            for model, info in known_models.items():
                if info["provider"] == provider:
                    fallback_candidates.append((model, 100))

            if target_size:
                for model, info in known_models.items():
                    if info["size"] and abs(info["size"] - target_size) <= 10:
                        fallback_candidates.append((model, 80))

            general_fallbacks = [
                "anthropic/claude-3.5-haiku",
                "meta-llama/llama-3.2-1b-instruct",
                "mistralai/ministral-8b"
            ]

            for model in general_fallbacks:
                fallback_candidates.append((model, 50))

            fallback_candidates.sort(key=lambda x: x[1], reverse=True)

            for candidate_model, priority in fallback_candidates:
                try:
                    if debug == "enable":
                        print(f"[FALLBACK] Teste: {candidate_model} (Priorität: {priority})")

                    if self.test_openrouter_model(candidate_model, api_key):
                        if debug == "enable":
                            print(f"[FALLBACK] Erfolgreich: {candidate_model}")
                        return candidate_model
                except:
                    continue

        return "anthropic/claude-3-haiku"

    def test_openrouter_model(self, model, api_key):
        """Testet ein Modell mit Mini-Request"""
        try:
            api_url, real_api_key = self.get_api_credentials(api_key)
            headers = {"Authorization": f"Bearer {real_api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            return response.status_code == 200
        except:
            return False

    def find_ollama_fallback(self, failed_model, debug):
        """Sucht ähnliches Ollama-Modell nur bei Fehlern"""
        if debug == "enable":
            print(f"[FALLBACK] Suche Ollama-Ersatz für: {failed_model}")

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            available_models = [m.get('name', '') for m in response.json().get("models", [])]
        except:
            if debug == "enable":
                print("[FALLBACK] Ollama nicht erreichbar")
            return None

        if not available_models:
            return None

        failed_lower = failed_model.lower()
        size_match = re.search(r'(\d+)b', failed_lower)
        target_size = int(size_match.group(1)) if size_match else None

        candidates = []

        for available in available_models:
            available_lower = available.lower()
            score = 0

            for base_name in ['llama', 'mistral', 'qwen', 'gemma', 'phi', 'codellama']:
                if base_name in failed_lower and base_name in available_lower:
                    score += 100
                    break

            if target_size:
                avail_size_match = re.search(r'(\d+)b', available_lower)
                if avail_size_match:
                    avail_size = int(avail_size_match.group(1))
                    size_diff = abs(avail_size - target_size)
                    if size_diff == 0:
                        score += 80
                    elif size_diff <= 5:
                        score += 60
                    elif size_diff <= 15:
                        score += 40

            if 'instruct' in available_lower:
                score += 20
            if any(x in available_lower for x in ['3.2', '3.1', '2.5']):
                score += 10

            candidates.append((available, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates and candidates[0][1] > 0:
            best_model = candidates[0][0]
            if debug == "enable":
                print(f"[FALLBACK] Gewählt: {best_model} (Score: {candidates[0][1]})")
            return best_model

        return available_models[0] if available_models else None

    def get_api_credentials(self, user_input_key):
        if user_input_key and user_input_key.strip():
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
            raise Exception(f"[Photo Prompt Randomizer] Fehler beim Lesen der API-Zugangsdaten: {str(e)}")

    def get_system_prompt(self, film_type, system_prompt_template):
        """Build system prompt with selected film type"""
        if film_type == "random":
            # Select random film type (excluding "random" itself)
            actual_films = [k for k in self.FILM_TYPES.keys() if k != "random"]
            film_type = random.choice(actual_films)

        film_description = self.FILM_TYPES[film_type]
        return system_prompt_template.format(film_description=film_description), film_type

    def run(self, random_seed, film_type, system_prompt, model, debug, unload_model, api_key="", context=""):
        # random_seed wird ignoriert - dient nur zum Cache-Breaking

        # Build system prompt with film type
        final_system_prompt, actual_film = self.get_system_prompt(film_type, system_prompt)

        if debug == "enable":
            print(f"[FILM TYPE] Selected: {actual_film}")

        # Prompt-Erstellung
        if context and context.strip():
            user_prompt = f"Context to incorporate: {context.strip()}\n\nGenerate a photographic image prompt based on this context."
        else:
            user_prompt = "Generate a creative photographic image prompt."

        real_model_name = self.extract_model_name(model)

        if model.startswith("openrouter/"):
            output_text = self.call_openrouter(user_prompt, real_model_name, api_key, debug, final_system_prompt)[0]
        elif model.startswith("local/"):
            output_text = self.call_ollama(user_prompt, real_model_name, debug, unload_model, final_system_prompt)[0]
        else:
            raise Exception(f"Unbekannter Modell-Prefix in '{model}'. Erwartet 'openrouter/' oder 'local/'.")

        return (output_text.strip(),)

    def call_openrouter(self, prompt, model, api_key, debug, system_prompt):
        try:
            api_url, real_api_key = self.get_api_credentials(api_key)
            headers = {"Authorization": f"Bearer {real_api_key}", "Content-Type": "application/json"}
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            payload = {"model": model, "messages": messages, "temperature": 0.9}

            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                output_text = result["choices"][0]["message"]["content"]

                if debug == "enable":
                    print("\n" + "="*60)
                    print(">>> AI4ARTSED PHOTO PROMPT RANDOMIZER <<<")
                    print("="*60)
                    print(f"Model: openrouter/{model}")
                    print("-" * 40)
                    print("User prompt:")
                    print(prompt)
                    print("-" * 40)
                    print("Generated prompt:")
                    print(output_text)
                    print("="*60)

                return (output_text,)
            else:
                raise Exception(f"API Error: {response.status_code}\n{response.text}")

        except Exception as e:
            if debug == "enable":
                print(f"[ERROR] Modell {model} fehlgeschlagen: {e}")

            fallback_model = self.find_openrouter_fallback(model, api_key, debug)
            if fallback_model != model:
                if debug == "enable":
                    print(f"[FALLBACK] Versuche {fallback_model}")
                return self.call_openrouter(prompt, fallback_model, api_key, debug, system_prompt)
            else:
                return (f"[ERROR] Alle OpenRouter-Fallbacks fehlgeschlagen: {str(e)}",)

    def call_ollama(self, prompt, model, debug, unload_model, system_prompt):
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            }

            response = requests.post("http://localhost:11434/api/generate", json=payload)

            if response.status_code == 200:
                output = response.json().get("response", "")
                if output:
                    if unload_model == "yes":
                        try:
                            unload_payload = {"model": model, "prompt": "", "keep_alive": 0, "stream": False}
                            requests.post("http://localhost:11434/api/generate", json=unload_payload, timeout=30)
                        except:
                            pass

                    if debug == "enable":
                        print("\n" + "="*60)
                        print(">>> AI4ARTSED PHOTO PROMPT RANDOMIZER <<<")
                        print("="*60)
                        print(f"Model: local/{model}")
                        print("-" * 40)
                        print("User prompt:")
                        print(prompt)
                        print("-" * 40)
                        print("Generated prompt:")
                        print(output)
                        print("="*60)

                    return (output,)

            raise Exception(f"Ollama Error: {response.status_code}")

        except Exception as e:
            if debug == "enable":
                print(f"[ERROR] Ollama-Modell {model} fehlgeschlagen: {e}")

            fallback_model = self.find_ollama_fallback(model, debug)
            if fallback_model and fallback_model != model:
                if debug == "enable":
                    print(f"[FALLBACK] Versuche {fallback_model}")
                return self.call_ollama(prompt, fallback_model, debug, unload_model, system_prompt)
            else:
                return (f"[ERROR] Ollama-Fallback fehlgeschlagen: {str(e)}",)
