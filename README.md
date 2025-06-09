
# AI4ArtsEd Prompt Nodes for ComfyUI

Experimental prompt transformation nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), developed as part of the AI4ArtsEd research initiative. These components support stylistic and cultural translation of prompts and images for use in educational, artistic, and cultural research contexts.

More information: https://kubi-meta.de/ai4artsed

---

## 🔧 Features

- Accepts `STRING` or `IMAGE` input from any upstream node.
- Structured control of transformation via `style_prompt`, `context`, and `instruction`.
- Supports both:
  - **Local inference** using [Ollama](https://ollama.com/),
  - **Remote APIs** via [OpenRouter](https://openrouter.ai/).
- Produces output as `STRING` (for use in CLIP, chaining, or save/display).
- Optional debug logging and model unloading for Ollama.
- Optional API key file handling for OpenRouter.
- Includes an image-to-text analysis node for multimodal LLMs.
- New: **Flexible text distribution** node for routing strings to multiple downstream paths.

---

## 🧹 Available Nodes

### 1. `AI4ArtsEd Ollama Prompt`
→ Uses local Ollama models (GPU/CPU) for cultural prompt rewriting.

### 2. `AI4ArtsEd OpenRouter Prompt`
→ Uses hosted LLMs via OpenRouter for stylistic transformations.

### 3. `AI4ArtsEd OpenRouter/Ollama ImageAnalysis`
→ Sends image input (as base64-encoded JPEG) to multimodal LLMs via OpenRouter (e.g. GPT-4V) or Ollama with cultural-instruction prompts.

### 4. `AI4ArtsEd Text Remix`
→ Selectively or randomly combines 1–12 text inputs.

**Inputs:**
- `mode`: `"random"`, `"all"`, or `"1"`–`"12"`
- `text_1` to `text_12`: Optional style or instruction inputs

**Output:**
- `text`: Combined or selected text based on mode.

### 5. `AI4ArtsEd Random Instruction Generator`
→ Generates four creative, symbolic, or surreal instruction phrases to be used as part of prompt contexts.

**Input:**
- `text` (STRING) – typically a core concept or message

**Outputs:**
- `instruction_1` to `instruction_4` (STRING)

### 6. `AI4ArtsEd Random Artform Generator`
→ Generates four culturally or stylistically distinct reinterpretation prompts, suitable for LLM context use.

**Input:**
- `text` (STRING)

**Outputs:**
- `artform_1` to `artform_4` (STRING)

### 7. `AI4ArtsEd Random Language Selector`
→ Selects 12 random languages from a curated list including both major and marginalized world languages, suitable for translation-based workflows such as `stille_post`.

**Input:**
- *(none)*

**Outputs:**
- `language_1` to `language_12` (STRING): Language names selected from a diverse set of model-supported languages (including Yoruba, Māori, Guarani, etc.)

### 8. `Secure Access to OpenRouter API Key`
OpenRouter.ai is a service that provides access to LLM/generativeAI systems. This is needed because LLM such das DeepSeek R1 are much to big to be run on standalone PCs. OpenRouter.ai-based nodes require an API key to access hosted models. For security, the key should **never be saved in workflows**.
In order to safely store you OpenRouter Key:
1) Find the Folder "custom_nodes/AI4ArtsEd/ on your system.
2) Open the file "openrouter.key.template" with a text editor.
3) Replace the placeholder text with your Key. Do not add anything else.
4) Remove the ".template" part of the filename, so that it reads "openrouter.key".
5) Make sure you do not share or upload this file. If you use a Github repository, it is strongly recommended to add "openrouter.key" to the ".gitignore".

Of course, it is possible to use other service providers than OpenRouter.ai. In order to do so, the .py files have to be modified accordingly.
---

### ✅ Node: `Secure Access to OpenRouter API Key`

This node:
- provides the API key as a `STRING` output,
- retrieves it from the `OPENROUTER_KEY` environment variable,
- is intended to be connected to `ai4artsed_openrouter` or other LLM-related nodes.


---

## CREDITS

This project includes adapted code from the following third-party sources:

- **ComfyUI-Show-Text** by fairy-root  
  https://github.com/fairy-root/ComfyUI-Show-Text  
  Licensed under the MIT License.  
  Adapted in `ai4artsed_show_text.py` to enable text output display in ComfyUI workflows.

- **comfyui-ollama** by stavsap  
  https://github.com/stavsap/comfyui-ollama  
  Licensed under the Apache License 2.0.  
  Adapted in `ai4artsed_openrouter_imageanalysis.py` and related files to support multimodal LLM image analysis.  
  See `license/THIRD_PARTY_LICENSES/comfyui-ollama_APACHE.txt` for license details.
# ai4artsed_comfyui_nodes
