import torch
import os
import threading
from datetime import datetime
from tkinter import messagebox
from PIL import Image
from diffusers import StableDiffusionPipeline

IMAGES_DIR = "generated_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

pipeline = None

IMAGE_KEYWORDS = [
    "generate image", "create image", "draw", "make image",
    "generate a picture", "create a picture", "show me a picture",
    "generate art", "create art", "make art", "paint",
    "generate photo", "create photo", "image of", "picture of",
    "draw me", "make me a", "generate me"
]

def is_image_request(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in IMAGE_KEYWORDS)

def extract_prompt(text):
    text_lower = text.lower()
    for keyword in IMAGE_KEYWORDS:
        if keyword in text_lower:
            prompt = text_lower.split(keyword)[-1].strip()
            return prompt if prompt else text
    return text

def load_pipeline():
    global pipeline
    if pipeline is not None:
        return True
    try:
        print("Loading Stable Diffusion model...")
        print("First time download is ~4GB. Please wait...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32
        )
        pipeline = pipeline.to("cpu")
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def generate_image(prompt, callback=None):
    global pipeline

    if not load_pipeline():
        if callback:
            callback(None, "Could not load image generation model!")
        return

    try:
        print(f"Generating image for: {prompt}")

        enhanced_prompt = (
            f"{prompt}, highly detailed, beautiful, "
            "4k quality, artistic, vibrant colors"
        )

        with torch.no_grad():
            result = pipeline(
                enhanced_prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            )

        image = result.images[0]
        filename = f"riya_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(IMAGES_DIR, filename)
        image.save(filepath)

        print(f"Image saved: {filepath}")

        if callback:
            callback(image, filepath)

        return filepath

    except Exception as e:
        print(f"Error generating image: {e}")
        if callback:
            callback(None, f"Could not generate image: {e}")

if __name__ == "__main__":
    print("Riya Image Generation")
    print("=" * 40)
    print("Note: First run downloads ~4GB model")
    prompt = input("Enter image prompt: ")
    print("Generating... please wait (this takes 1-3 minutes on CPU)")
    path = generate_image(prompt)
    if path:
        print(f"Image saved to: {path}")
        img = Image.open(path)
        img.show()