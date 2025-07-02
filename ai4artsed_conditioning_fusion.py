import torch

"""
ai4artsed_conditioning_fusion.py
--------------------------------
This node is a conceptual evolution of the T5-CLIP Fusion node, designed for more
robust and predictable blending of any two conditioning tensors.

**Purpose**
    Fuse two separate conditioning tensors (e.g., from two different T5 prompts,
    or two different CLIP prompts) by blending them over their entire common length.
    This provides a more thorough and integrated fusion than blending only a fixed
    initial segment.

Workflow:
    1. Two prompts are encoded, resulting in `conditioning_a` and `conditioning_b`.
    2. This node blends them based on an `alpha` value.

How it works:
    • It takes two conditioning inputs: `conditioning_a` and `conditioning_b`.
    • It determines the shorter of the two tensor lengths (`interp_len`).
    • The tensors are blended (interpolated) up to `interp_len`.
    • The remaining tokens from the *longer* tensor are appended to the result.
    • The `pooled_output` tensors from each conditioning are also blended.

Alpha parameter (`alpha`):
    • Controls the interpolation between A and B.
      - `alpha=0`: The result is 100% from conditioning_a.
      - `alpha=1`: The result is 100% from conditioning_b.
      - `alpha=0.5`: An equal 50/50 blend.
    • Extrapolation (values outside 0-1) is possible for creative effects.

Important:
    • This node assumes the embedding dimensions of both conditionings are identical.
      It will raise an error if they differ.
"""

NODE_NAME = "AI4ArtsEd Conditioning Fusion"

class ai4artsed_conditioning_fusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_a": ("CONDITIONING",),
                "conditioning_b": ("CONDITIONING",),
                "alpha": ("FLOAT", {"default": 0.5, "min": -5, "max": 5, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "fuse"
    CATEGORY = "AI4ArtsEd"

    def _extract_tensor_and_pooled(self, conditioning):
        """Helper function to robustly extract tensors and pooled output."""
        if not conditioning:
            raise ValueError("Input conditioning is empty.")

        if isinstance(conditioning, list) and len(conditioning) > 0 and isinstance(conditioning[0], list):
            inner_list = conditioning[0]
            if len(inner_list) == 2 and isinstance(inner_list[0], torch.Tensor):
                tokens = inner_list[0]
                pooled = inner_list[1].get('pooled_output')
                if pooled is None:
                    # Create a default pooled output if not available
                    pooled = torch.zeros(tokens.shape[0], tokens.size(-1), dtype=tokens.dtype, device=tokens.device)
                return tokens, pooled
        
        raise ValueError(f"Could not extract tensor from conditioning of type: {type(conditioning)}")

    def fuse(self, conditioning_a, conditioning_b, alpha):
        print("\n--- Running AI4ArtsEd Conditioning Fusion Node ---")
        try:
            # 1. Extract tensors and pooled outputs
            tokens_a, pooled_a = self._extract_tensor_and_pooled(conditioning_a)
            tokens_b, pooled_b = self._extract_tensor_and_pooled(conditioning_b)

            print(f"Conditioning A shape: {tokens_a.shape}")
            print(f"Conditioning B shape: {tokens_b.shape}")
            print(f"Alpha value: {alpha}")

            # 2. Validate shapes
            if tokens_a.shape[0] != tokens_b.shape[0]:
                raise ValueError(f"Batch size mismatch: A has {tokens_a.shape[0]} but B has {tokens_b.shape[0]}.")
            if tokens_a.shape[-1] != tokens_b.shape[-1]:
                raise ValueError(f"Embedding dimension mismatch: A is {tokens_a.shape[-1]}D but B is {tokens_b.shape[-1]}D.")

            # 3. Determine lengths for blending (the shorter of the two)
            len_a = tokens_a.shape[1]
            len_b = tokens_b.shape[1]
            interp_len = min(len_a, len_b)
            
            print(f"Blending the first {interp_len} tokens.")

            # 4. Slice tensors for interpolation
            interp_part_a = tokens_a[:, :interp_len, :]
            interp_part_b = tokens_b[:, :interp_len, :]

            # 5. Perform interpolation on the token embeddings
            interpolated_tokens = (1.0 - alpha) * interp_part_a + alpha * interp_part_b

            # 6. Identify the remainder of the longer prompt to append
            append_part = torch.empty((tokens_a.shape[0], 0, tokens_a.shape[2]), dtype=tokens_a.dtype, device=tokens_a.device)
            if len_a > len_b:
                append_part = tokens_a[:, interp_len:, :]
                print(f"Appending remaining {append_part.shape[1]} tokens from Conditioning A.")
            elif len_b > len_a:
                append_part = tokens_b[:, interp_len:, :]
                print(f"Appending remaining {append_part.shape[1]} tokens from Conditioning B.")

            # 7. Concatenate to form the final token tensor
            fused_tokens = torch.cat([interpolated_tokens, append_part], dim=1)

            # 8. Blend the pooled outputs for a complete fusion
            fused_pooled = (1.0 - alpha) * pooled_a + alpha * pooled_b
            
            print(f"Final fused conditioning shape: {fused_tokens.shape}")

            # 9. Repackage the result into ComfyUI's expected format
            result_conditioning = [[fused_tokens, {"pooled_output": fused_pooled}]]
            
            print("--- Fusion successful ---\n")
            return (result_conditioning,)

        except Exception as e:
            print(f"\nERROR in AI4ArtsEd Conditioning Fusion: {str(e)}")
            # Fallback to conditioning_a to prevent workflow failure
            return (conditioning_a,)

# ---- ComfyUI registration ---------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ai4artsed_conditioning_fusion": ai4artsed_conditioning_fusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ai4artsed_conditioning_fusion": NODE_NAME,
}
