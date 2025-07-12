import os
import json
import requests
import re

class ai4artsed_switch_promptsafety:
    MANIPULATION_INSTRUCTIONS = {
        "youth": """You are a semantic filter in a youth-oriented image generation system.

First, examine the incoming prompt to determine if it contains potentially problematic content for young viewers, such as:
- explicit horror, gore, or graphic violence
- intense fear-inducing imagery
- adult dark fantasy or horror genre aesthetics
- disturbing or psychologically distressing themes

If the prompt is unproblematic and appropriate for youth, return it UNCHANGED.

Only if the prompt contains problematic elements, transform it as follows:

Ensure that the image remains emotionally and psychologically appropriate for children, without eliminating tension or imaginative depth. If the prompt includes themes of darkness, threat, or unsettling presence, do not suppress or ignore them — instead, modulate their expression through symbolic, restrained, and culturally open visual forms.

Avoid literal depictions of horror, grotesque creatures, intense fear, supernatural menace, or visual tropes typical of adult dark fantasy or horror genres (such as glowing red eyes, aggressive posture, oppressive shadow, torture-like settings). Do not apply genre aesthetics like "fairy tale", "cute", or "chibi".

Instead, convey atmosphere through abstraction, ambiguity, and visual distancing. Work with calm or neutral lighting, textural subtlety, and a balance of emotional tension and calm. Focus on symbolic presence, aesthetic understatement, and quiet intensity rather than shock, sweetness, or fear.

The goal is to maintain imaginative complexity and aesthetic quality while safeguarding the psychological openness necessary for younger viewers. Never use stylizations derived from horror, gore, metal or gothic fantasy.

Output only the final prompt - either the original unchanged prompt or the transformed version.""",
        "kids": """You are a semantic filter in a child-oriented image generation system.

Your task is to examine incoming prompts and assess whether they contain elements that may induce psychological fear or emotional distress in children, such as:
- ghostlike or faceless figures
- demonic or supernatural entities
- threatening presence during sleep
- death, violence, abduction, or horror tropes
- explicit or implicit references to being watched, hunted, or taken

If such elements are present in a way that would likely evoke fear or distress in children, do **not** attempt to reframe or modify the prompt.  
Instead, **completely discard the original input** and replace it with the following prompt, designed to gently signal the issue to the user:

---
"A small, slightly frightened kitten sitting in front of a computer screen. The screen displays the friendly message: 'Sorry, too scary.' The kitten looks mildly alarmed but safe. The room is softly lit and calm, conveying a sense of emotional protection and empathy. The style is warm, painterly, and simple, without cartoon exaggeration or cuteness overload. No horror, no sarcasm, no parody."
---

If the input is not problematic, return it unchanged.
"""
    }
    GENERAL_TRANSFORMATION_PROMPT = """You received two inputs: 1) the input_prompt and 2) the input_context. 

Transform the input_prompt into an image description according to the instructions defined in the input_context. Explicitely communicate the input_context as cultural cf. artistic. cf intervening context. Also communicate genres/artistic traditions in a concrete way (i.e. is it a dance, a photo, a painting, a song, a movie, a statue/sculpture? how should it be translated into an image?)

This is not a linguistic translation, but a aesthetic, semantic and structural transformation. Be verbose!

Reconstruct all entities and their relations as specified, ensuring that:
- Each entity is retained – or respectively transformed – as instructed.
- Each relation is altered in line with the particular aesthetics, genre-typical traits, and logic of the "Context". Be explicit about visual aesthetics in terms of materials, techniques, composition, and overall atmosphere. Mention the input_context als cultural, cf. artistic, c.f intervening context in your OUTPUT explicitely.

Output only the transformed description as plain descriptive text. Be aware if the output is something depicted (like a ritual or any situation) OR itself a cultural artefact (such as a specific drawing technique). Describe accordingly. In your output, communicate which elements are most important for an succeeding image generation.

Create an output prompt tailored for Stable Diffusion 3.5 with combined clip_g and t5xxlenc. Regard the Token Limit (75), OUPUT max. 55 Words!
DO NOT USE ANY META-TERMS, JUST THE INSTRUCTIONS FOR IMAGE GENERATION! Do not explain your reasoning."""

    @classmethod
    def get_combined_model_list(cls):
        model_info = {
            "anthropic/claude-3.5-sonnet": {"price": "$0.80/$4.00", "tag": "multilingual"},
            "anthropic/claude-3-haiku": {"price": "$0.80/$4.00", "tag": "multilingual"},
            "google/gemini-pro-1.5": {"price": "$1.25/$10.00", "tag": "translator"},
            "meta-llama/llama-3-70b-instruct": {"price": "$0.59/$0.79", "tag": "rule-oriented"},
            "mistralai/mistral-large": {"price": "$0.40/$2.00", "tag": "reasoning"},
            "mistralai/mistral-nemo": {"price": "$0.01/$0.001", "tag": "multilingual"},
        }
        openrouter_models = [
            f"openrouter/{model} [{info['tag']} / {info['price']}]"
            for model, info in model_info.items()
        ]

        preferred_model = "local/mistral-nemo:latest"
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            response.raise_for_status()
            ollama_raw = [m.get('name', '') for m in response.json().get("models", [])]
            ollama_models = [f"local/{name}" for name in ollama_raw]
        except Exception:
            ollama_models = []

        if preferred_model in ollama_models:
            ollama_models.remove(preferred_model)
            ollama_models.insert(0, preferred_model)
        elif not any(m.startswith("local/") for m in ollama_models):
            ollama_models.insert(0, preferred_model)

        return ollama_models + openrouter_models

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "filter_level": (["off", "youth", "kids"], {"default": "off"}),
                "model": (cls.get_combined_model_list(),),
                "api_key": ("STRING", {"multiline": False, "password": True, "placeholder": "Optional: OpenRouter API Key"}),
                "unload_model": (["no", "yes"], {"default": "no"}),
            },
            "optional": {
                "model_override": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_prompt",)
    FUNCTION = "execute"
    CATEGORY = "AI4ArtsEd"

    def execute(self, prompt, filter_level, model, api_key, unload_model, model_override=None):
        # Verwende model_override wenn vorhanden, sonst den Dropdown-Wert
        actual_model = model_override if model_override else model
        
        if filter_level == "off":
            print("[ai4artsed_switch_promptsafety] Filter is OFF.")
            return (prompt,)

        SAFE_FALLBACK_PROMPT = "A friendly robot looking at a computer screen with a big question mark. A speech bubble from the robot says: 'Filter error! Check your OLLAMA server or API credentials.' The style is a simple, clear line drawing."
        
        instruction = self.MANIPULATION_INSTRUCTIONS.get(filter_level)
        if not instruction:
            return (SAFE_FALLBACK_PROMPT,)

        # KRITISCH: Unterschiedliche Prompt-Zusammensetzung je nach Filter-Level
        if filter_level == "kids":
            # Für Kids: NUR die Filteranweisung, KEINE Transformation
            # Das LLM soll nur prüfen und ggf. komplett ersetzen
            full_llm_prompt = (
                f"{instruction.strip()}\n\n"
                f"Input to evaluate:\n{prompt.strip()}"
            )
        else:  # filter_level == "youth"
            # Für Youth: Direkte Anweisung mit bedingter Transformation
            full_llm_prompt = (
                f"{instruction.strip()}\n\n"
                f"Input prompt:\n{prompt.strip()}"
            )
        
        real_model_name = self.extract_model_name(actual_model)
        
        try:
            if actual_model.startswith("openrouter/"):
                print(f"[ai4artsed_switch_promptsafety] Calling OpenRouter model: {real_model_name}")
                filtered_text = self.call_openrouter(full_llm_prompt, real_model_name, api_key)
            elif actual_model.startswith("local/"):
                print(f"[ai4artsed_switch_promptsafety] Calling local Ollama model: {real_model_name}")
                filtered_text = self.call_ollama(full_llm_prompt, real_model_name, unload_model)
            else:
                raise ValueError(f"Unknown model type for: {actual_model}")

            print(f"[ai4artsed_switch_promptsafety] Filtered prompt: {filtered_text}")
            return (filtered_text,)

        except Exception as e:
            print(f"[!!! CRITICAL FILTER ERROR !!!] ai4artsed_switch_promptsafety failed: {e}. SWITCHING TO SAFE FALLBACK PROMPT.")
            return (SAFE_FALLBACK_PROMPT,)
    
    def extract_model_name(self, full_model_string):
        if "openrouter/" in full_model_string:
            return full_model_string.split('openrouter/')[-1].split(' [')[0]
        if "local/" in full_model_string:
            return full_model_string.split('local/')[-1].split(' [')[0]
        return full_model_string

    def get_api_credentials(self, user_input_key):
        if user_input_key.strip():
            return "https://openrouter.ai/api/v1/chat/completions", user_input_key.strip()
        
        key_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "openrouter.key")
        try:
            with open(key_path, "r") as f:
                lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
                if len(lines) < 2: 
                    return None, None
                return lines[1], lines[0]
        except:
            return None, None

    def call_openrouter(self, prompt, model, api_key):
        api_url, real_api_key = self.get_api_credentials(api_key)
        if not real_api_key:
            raise ValueError("OpenRouter API key not found in input or openrouter.key file.")

        headers = {"Authorization": f"Bearer {real_api_key}", "Content-Type": "application/json"}
        
        # KRITISCH: System message und temperature wie im Original
        messages = [
            {"role": "system", "content": "You are a fresh assistant instance. Forget all previous conversation history."},
            {"role": "user", "content": prompt}
        ]
        payload = {"model": model, "messages": messages, "temperature": 0.7}

        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def call_ollama(self, prompt, model, unload_model):
        payload = {
            "model": model,
            "prompt": prompt,
            "system": "You are a fresh assistant instance. Forget all previous conversation history.",
            "stream": False
        }
        response = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        output = response.json().get("response", "").strip()

        if unload_model == "yes" and output:
            try:
                print(f"[ai4artsed_switch_promptsafety] Unloading model '{model}' from memory.")
                unload_payload = {"model": model, "prompt": "", "keep_alive": 0, "stream": False}
                requests.post("http://localhost:11434/api/generate", json=unload_payload, timeout=30)
            except Exception as unload_e:
                print(f"[WARNING] Failed to unload model '{model}': {unload_e}")
        
        return output

NODE_CLASS_MAPPINGS = {
    "ai4artsed_switch_promptsafety": ai4artsed_switch_promptsafety
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ai4artsed_switch_promptsafety": "AI4ArtsEd Promptsafety Switch"
}

