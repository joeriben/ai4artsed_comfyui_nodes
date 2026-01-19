# AI4ArtsEd ComfyUI Nodes - Dev Log

## 2026-01-19

### Photo Prompt Randomizer - Film Types & Editable Prompt

**Film Type Selector:**
- Dropdown with 14 analog film stocks + random option
- Slide films: Kodachrome, Ektachrome
- Color negative: Portra 400/800, Ektar 100, Fuji Pro 400H/Superia, CineStill 800T
- B&W: Ilford HP5/Delta 400/FP4/Pan F/XP2, Kodak Tri-X 400
- `{film_description}` placeholder in prompt gets replaced by selected film

**Editable System Prompt:**
- System prompt now visible and editable in node UI
- Users can customize prompt generation behavior
- `{film_description}` placeholder for film type injection

**Improved Default Prompt:**
```
You are an inventive creative. Your task is to invent a REALISTIC photographic image prompt.

Think globally. Avoid cultural clichés. Avoid "retro" style descriptions.
Describe contemporary everyday motives: scenes, objects, animals, nature, tech, culture, people, homes, family, work, holiday, urban, rural, trivia, details.

Choose either unlikely, untypical or typical photographical sujets for realistic photographic images. Be verbose, provide intricate details.

Always begin your output with: "{film_description} of".
Transform the prompt strictly following the context if provided.

NO META-COMMENTS, TITLES, Remarks, dialogue WHATSOEVER, STRICTLY FOLLOW THE INSTRUCTION.
```

**UI Cleanup:**
- Removed `debug` field (unnecessary clutter)
- Final inputs: random_seed, film_type, system_prompt, model, unload_model, api_key (opt), context (opt)

---

## 2026-01-18

### Photo Prompt Randomizer Node

**Problem:** ComfyUI executes nodes only when inputs change. Integer nodes can be randomized but don't connect to string inputs.

**Solution:** New node `ai4artsed_photo_prompt_randomizer`:
- INT input `random_seed` for cache-breaking (connected to Integer Randomizer)
- `IS_CHANGED()` returning `float("nan")` forces re-execution
- LLM generates photographic prompts with Kodachrome style

**System Prompt:**
```
You are an inventive creative. Your task is to invent a REALISTIC photographic
image prompt. Choose either unlikely, untypical or typical photographical sujets
for realistic photographic images. Be verbose, provide intricate details.
Always begin your output with: "An Kodachrome film slide of".
Transform the prompt strictly following the context.
NO META-COMMENTS, TITLES, Remarks, dialogue WHATSOEVER, STRICTLY FOLLOW THE INSTRUCTION.
```

**Workflow:**
```
[Integer Randomizer] --INT--> [Photo Prompt Randomizer] --STRING--> [T5-CLIP]
```

### Model List Update

Added new models to both `prompt_interception` and `photo_prompt_randomizer`:

| Model | Price | Tag |
|-------|-------|-----|
| anthropic/claude-sonnet-4.5 | $3.00/$15.00 | multilingual |
| openai/gpt-oss-120b | $0.50/$2.00 | reasoning |
| openai/gpt-oss-20b | $0.10/$0.40 | reasoning |

All existing models retained for backwards compatibility.

### Git SSH Fix

Switched remote from HTTPS to SSH for persistent authentication:
```bash
git remote set-url origin git@github.com:joeriben/ai4artsed_comfyui_nodes.git
```

---

## Earlier Changes

See git log for history:
```bash
git log --oneline
```
