"""
AI4ArtsEd OpenRouter Image Analysis Node

This file contains adapted code from:
https://github.com/stavsap/comfyui-ollama
Original code by stavsap, licensed under the Apache License 2.0.

This version is licensed under the European Union Public License (EUPL) v1.2.
See LICENSE and THIRD_PARTY_LICENSES/comfyui-ollama_APACHE.txt for details.
"""

import torch
import numpy as np
import cv2
import base64
import requests

class ai4artsed_openrouter_imageanalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "Describe the image. Detect its likely cultural context. Enrich your description with analyses of the cultural constellations and meanings, relations, values, and emotions expressed in the image. Detect the meaning also in a more abstract way: what do the depicted entities, actions, relationships imply in the given cultural context?"
                }),
                "api_key": ("STRING", {"multiline": False, "default": "sk-..."}),
                "model": (["openai/gpt-4o", "google/gemini-flash-1.5", "qwen/qwen-vl-plus", "meta-llama/llama-3.2-11b-vision-instruct"],),
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "analyze"
    CATEGORY = "AI4ArtsEd"

    def analyze(self, images, instruction, api_key, model, max_tokens, temperature):
        image = images[0]  # Expecting a list of images
        image_np = self._prepare_image_array(image)
        encoded_image = self._encode_image(image_np)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encoded_image}},
                    {"type": "text", "text": instruction}
                ]
            }
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

        response_data = response.json()
        output_text = response_data['choices'][0]['message']['content']
        return (output_text,)

    def _prepare_image_array(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        if image.ndim == 3:
            if image.shape[0] not in (1, 3, 4) and image.shape[-1] in (1, 3, 4):
                image = image.transpose(2, 0, 1)

        if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
            raise ValueError(f"Unsupported image shape: expected channels first, got shape {image.shape}")

        return image

    def _encode_image(self, image_array):
        image = image_array.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode(".jpg", image_rgb)
        if not success:
            raise RuntimeError("Failed to encode image")

        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{jpg_as_text}"

# Required for ComfyUI to discover and use the node
NODE_CLASS_MAPPINGS = {
    "ai4artsed_openrouter_imageanalysis": ai4artsed_openrouter_imageanalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ai4artsed_openrouter_imageanalysis": "AI4ArtsEd OpenRouter: ImageAnalysis",
}
