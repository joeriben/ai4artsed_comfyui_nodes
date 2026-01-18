# AI4ArtsEd ComfyUI Nodes - Dev Log

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
