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
            "anthropic/claude-3.5-haiku": {"price": "$0.80/$4.00", "tag": "multilingual"},
            "anthropic/claude-3-haiku": {"price": "$0.80/$4.00", "tag": "multilingual"},
            "deepseek/deepseek-chat-v3-0324": {"price": "$0.27/$1.10", "tag": "rule-oriented"},
            "deepseek/deepseek-r1-0528": {"price": "$0.50/$2.15", "tag": "reasoning"},
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
            "qwen/qwen3-32b": {"price": "$0.10/$0.30", "tag": "translator"},
            "qwen/qwen3-235b-a22b": {"price": "$0.13/$0.60", "tag": "multilingual"}
        }
        
        openrouter_models = [
            f"openrouter/{model} [{info['tag']} / {info['price']}]"
            for model, info in model_info.items()
        ]

        # Ollama-Modelle lokal & kostenlos
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
        
        # Extrahiere Basis-Info vom fehlgeschlagenen Modell
        model_parts = failed_model.split("/")
        if len(model_parts) >= 2:
            provider = model_parts[0]  # z.B. "anthropic", "meta-llama"
            model_name = model_parts[1]  # z.B. "claude-3-haiku", "llama-3.3-70b"
            
            # Extrahiere Größe (falls vorhanden)
            size_match = re.search(r'(\d+)b', model_name.lower())
            target_size = int(size_match.group(1)) if size_match else None
            
            # Unsere bekannte Modell-Liste
            known_models = {
                "anthropic/claude-3.5-haiku": {"size": None, "provider": "anthropic"},
                "anthropic/claude-3-haiku": {"size": None, "provider": "anthropic"},
                "deepseek/deepseek-chat-v3-0324": {"size": None, "provider": "deepseek"},
                "deepseek/deepseek-r1-0528": {"size": None, "provider": "deepseek"},
                "meta-llama/llama-3.3-70b-instruct": {"size": 70, "provider": "meta-llama"},
                "meta-llama/llama-guard-3-8b": {"size": 8, "provider": "meta-llama"},
                "meta-llama/llama-3.2-1b-instruct": {"size": 1, "provider": "meta-llama"},
                "mistralai/mistral-medium-3": {"size": None, "provider": "mistralai"},
                "mistralai/mistral-small-3.1-24b-instruct": {"size": 24, "provider": "mistralai"},
                "mistralai/ministral-8b": {"size": 8, "provider": "mistralai"},
                "mistralai/ministral-3b": {"size": 3, "provider": "mistralai"},
                "mistralai/mixtral-8x7b-instruct": {"size": 56, "provider": "mistralai"},  # 8x7b = ~56b
                "qwen/qwen3-32b": {"size": 32, "provider": "qwen"},
                "qwen/qwen3-235b-a22b": {"size": 235, "provider": "qwen"},
            }
            
            fallback_candidates = []
            
            # 1. Priorität: Gleicher Provider
            for model, info in known_models.items():
                if info["provider"] == provider:
                    fallback_candidates.append((model, 100))
            
            # 2. Priorität: Ähnliche Größe (falls Größe erkannt)
            if target_size:
                for model, info in known_models.items():
                    if info["size"] and abs(info["size"] - target_size) <= 10:
                        fallback_candidates.append((model, 80))
            
            # 3. Priorität: Allgemeine Fallbacks
            general_fallbacks = [
                "anthropic/claude-3.5-haiku",
                "meta-llama/llama-3.2-1b-instruct",
                "mistralai/ministral-8b"
            ]
            
            for model in general_fallbacks:
                fallback_candidates.append((model, 50))
            
            # Sortiere nach Priorität und teste
            fallback_candidates.sort(key=lambda x: x[1], reverse=True)
            
            for candidate_model, priority in fallback_candidates:
                try:
                    if debug == "enable":
                        print(f"[FALLBACK] Teste: {candidate_model} (Priorität: {priority})")
                    
                    # Teste das Fallback-Modell mit einem Mini-Prompt
                    if self.test_openrouter_model(candidate_model, api_key):
                        if debug == "enable":
                            print(f"[FALLBACK] Erfolgreich: {candidate_model}")
                        return candidate_model
                except:
                    continue
        
        # Letzter Fallback: Einfachstes Modell
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
        
        # Hole verfügbare Modelle
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            available_models = [m.get('name', '') for m in response.json().get("models", [])]
        except:
            if debug == "enable":
                print("[FALLBACK] Ollama nicht erreichbar")
            return None
        
        if not available_models:
            return None
        
        # Extrahiere Größe und Typ vom fehlgeschlagenen Modell
        failed_lower = failed_model.lower()
        size_match = re.search(r'(\d+)b', failed_lower)
        target_size = int(size_match.group(1)) if size_match else None
        
        # Kategorisiere verfügbare Modelle
        candidates = []
        
        for available in available_models:
            available_lower = available.lower()
            score = 0
            
            # 1. Gleicher Basis-Name (llama, mistral, etc.)
            for base_name in ['llama', 'mistral', 'qwen', 'gemma', 'phi', 'codellama']:
                if base_name in failed_lower and base_name in available_lower:
                    score += 100
                    break
            
            # 2. Ähnliche Größe
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
            
            # 3. Allgemeine Präferenzen
            if 'instruct' in available_lower:
                score += 20
            if any(x in available_lower for x in ['3.2', '3.1', '2.5']):
                score += 10
            
            candidates.append((available, score))
        
        # Sortiere und wähle bestes verfügbares Modell
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates and candidates[0][1] > 0:
            best_model = candidates[0][0]
            if debug == "enable":
                print(f"[FALLBACK] Gewählt: {best_model} (Score: {candidates[0][1]})")
            return best_model
        
        # Letzter Fallback: Erstes verfügbares Modell
        return available_models[0] if available_models else None

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

    def run(self, input_prompt, input_context, style_prompt, api_key, model, debug, unload_model):
        full_prompt = (
            f"Task:\n{style_prompt.strip()}\n\n"
            f"Context:\n{input_context.strip()}\nPrompt:\n{input_prompt.strip()}"
        )

        # Echten Modellnamen extrahieren
        real_model_name = self.extract_model_name(model)

        if model.startswith("openrouter/"):
            output_text = self.call_openrouter(full_prompt, real_model_name, api_key, debug)[0]
        elif model.startswith("local/"):
            output_text = self.call_ollama(full_prompt, real_model_name, debug, unload_model)[0]
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

    def call_openrouter(self, prompt, model, api_key, debug):
        try:
            # Normaler Aufruf
            api_url, real_api_key = self.get_api_credentials(api_key)
            headers = {"Authorization": f"Bearer {real_api_key}", "Content-Type": "application/json"}
            messages = [
                {"role": "system", "content": "You are a fresh assistant instance. Forget all previous conversation history."},
                {"role": "user", "content": prompt}
            ]
            payload = {"model": model, "messages": messages, "temperature": 0.7}

            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                output_text = result["choices"][0]["message"]["content"]
                
                if debug == "enable":
                    print("\n" + "="*60)
                    print(">>> AI4ARTSED PROMPT INTERCEPTION NODE <<<")
                    print("="*60)
                    print(f"Selected: openrouter/{model}")
                    print(f"Real Model: {model}")
                    print("-" * 40)
                    print("Prompt sent:")
                    print(prompt)
                    print("-" * 40)
                    print("Response received:")
                    print(output_text)
                    print("="*60)
                
                return (output_text,)
            else:
                raise Exception(f"API Error: {response.status_code}\n{response.text}")
                
        except Exception as e:
            if debug == "enable":
                print(f"[ERROR] Modell {model} fehlgeschlagen: {e}")
            
            # NUR JETZT Fallback aktivieren
            fallback_model = self.find_openrouter_fallback(model, api_key, debug)
            if fallback_model != model:  # Verhindere Endlosschleife
                if debug == "enable":
                    print(f"[FALLBACK] Versuche {fallback_model}")
                return self.call_openrouter(prompt, fallback_model, api_key, debug)
            else:
                return (f"[ERROR] Alle OpenRouter-Fallbacks fehlgeschlagen: {str(e)}",)

    def call_ollama(self, prompt, model, debug, unload_model):
        try:
            # Normaler Aufruf
            payload = {
                "model": model,
                "prompt": prompt,
                "system": "You are a fresh assistant instance. Forget all previous conversation history.",
                "stream": False
            }

            response = requests.post("http://localhost:11434/api/generate", json=payload)
            
            if response.status_code == 200:
                output = response.json().get("response", "")
                if output:  # Erfolg
                    if unload_model == "yes":
                        try:
                            unload_payload = {"model": model, "prompt": "", "keep_alive": 0, "stream": False}
                            requests.post("http://localhost:11434/api/generate", json=unload_payload, timeout=30)
                        except:
                            pass

                    if debug == "enable":
                        print("\n" + "="*60)
                        print(">>> AI4ARTSED PROMPT INTERCEPTION NODE <<<")
                        print("="*60)
                        print(f"Selected: local/{model} [local / $0.00]")
                        print(f"Real Model: {model}")
                        print("-" * 40)
                        print("Prompt sent:")
                        print(prompt)
                        print("-" * 40)
                        print("Response received:")
                        print(output)
                        print("="*60)

                    return (output,)
            
            raise Exception(f"Ollama Error: {response.status_code}")
            
        except Exception as e:
            if debug == "enable":
                print(f"[ERROR] Ollama-Modell {model} fehlgeschlagen: {e}")
            
            # NUR JETZT Fallback aktivieren
            fallback_model = self.find_ollama_fallback(model, debug)
            if fallback_model and fallback_model != model:
                if debug == "enable":
                    print(f"[FALLBACK] Versuche {fallback_model}")
                return self.call_ollama(prompt, fallback_model, debug, unload_model)
            else:
                return (f"[ERROR] Ollama-Fallback fehlgeschlagen: {str(e)}",)
