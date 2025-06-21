from .ai4artsed_text_remix import ai4artsed_text_remix
from .ai4artsed_random_instruction_generator import ai4artsed_random_instruction_generator
from .ai4artsed_random_artform_generator import ai4artsed_random_artform_generator
from .ai4artsed_random_language_selector import ai4artsed_random_language_selector
from .ai4artsed_openrouter_key import ai4artsed_openrouter_key
from .ai4artsed_t5_clip_fusion import ai4artsed_t5_clip_fusion
from .ai4artsed_prompt_interception import ai4artsed_prompt_interception

NODE_CLASS_MAPPINGS = {
    "ai4artsed_text_remix": ai4artsed_text_remix,
    "ai4artsed_random_instruction_generator": ai4artsed_random_instruction_generator,
    "ai4artsed_random_artform_generator": ai4artsed_random_artform_generator,
    "ai4artsed_random_language_selector": ai4artsed_random_language_selector,
    "ai4artsed_openrouter_key": ai4artsed_openrouter_key,
    "ai4artsed_t5_clip_fusion": ai4artsed_t5_clip_fusion,
    "ai4artsed_prompt_interception": ai4artsed_prompt_interception,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ai4artsed_text_remix": "AI4ArtsEd Text Remix",
    "ai4artsed_random_instruction_generator": "Random Instruction Generator",
    "ai4artsed_random_artform_generator": "Random Artform Generator",
    "ai4artsed_random_language_selector": "Random Language Selector",
    "ai4artsed_openrouter_key": "Secure Access to OpenRouter API Key",
    "ai4artsed_t5_clip_fusion": "AI4ArtsEd T5‑CLIP Fusion",
    "ai4artsed_prompt_interception": "AI4ArtsEd Prompt Interception",
}
