# AI4ArtsEd ComfyUI Nodes - Architecture

## Overview

Custom ComfyUI nodes for AI-assisted art education, featuring LLM integration via OpenRouter and local Ollama.

## Node Structure

```
__init__.py                              # Node registration
ai4artsed_prompt_interception.py         # LLM prompt transformation
ai4artsed_photo_prompt_randomizer.py     # LLM photo prompt generation
ai4artsed_image_analysis.py              # Image analysis via LLM
ai4artsed_text_remix.py                  # Text remixing
ai4artsed_t5_clip_fusion.py              # T5/CLIP conditioning fusion
ai4artsed_conditioning_fusion.py         # Conditioning fusion
ai4artsed_random_instruction_generator.py
ai4artsed_random_artform_generator.py
ai4artsed_random_language_selector.py
ai4artsed_openrouter_key.py              # Secure API key handling
ai4artsed_vector_dimension_eliminator.py
ai4artsed_switch_promptsafety.py         # Safety filter
openrouter.key                           # API credentials (gitignored)
```

## LLM Integration Pattern

Nodes with LLM support share common methods:

```python
get_combined_model_list()    # OpenRouter + Ollama models with pricing
extract_model_name()         # Parse dropdown string -> real model name
get_api_credentials()        # Key from input or openrouter.key file
call_openrouter()            # API call with fallback
call_ollama()                # Local call with fallback
find_openrouter_fallback()   # Smart fallback by provider/size
find_ollama_fallback()       # Local fallback by similarity
```

## Cache-Breaking Pattern

ComfyUI only re-executes nodes when inputs change. For randomized LLM output:

```python
# Accept INT input from Integer Randomizer (value ignored)
"random_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})

# Force re-execution via IS_CHANGED
@classmethod
def IS_CHANGED(cls, **kwargs):
    return float("nan")
```

## API Key Handling

Keys stored in `openrouter.key` (gitignored):
```
<api_key>
<api_url>
```

Fallback chain: Input field -> openrouter.key file

## Model Dropdown Format

```
openrouter/<provider>/<model> [<tag> / <price>]
local/<model> [local / $0.00]
```

Examples:
- `openrouter/anthropic/claude-sonnet-4.5 [multilingual / $3.00/$15.00]`
- `local/llama3.2:3b [local / $0.00]`
