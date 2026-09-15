import os
import re
import base64
import traceback
import runpod
import moviepy.editor as mpy
from gtts import gTTS
from pydub import AudioSegment, silence
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

# Global model handle (lazy loaded)
pipe = None

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def ensure_model_loaded():
    """
    Lazy-load the Stable Diffusion pipeline on first request.
    This avoids long image downloads during Docker build and prevents
    startup failures if the model can't be fetched at build time.
    """
    global pipe
    if pipe is not None:
        return

    try:
        log("Loading Stable Diffusion model (this may take a while on first run)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log("Device chosen:", device)

        # Use float16 on CUDA to reduce memory if GPU is available
        load_kwargs = {}
        if device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16

        # Download from Hugging Face at runtime and cache in the container's cache directory
        pipe_local = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            **load_kwargs
        )

        pipe_local.safety_checker = None  # optional: disable safety checker if desired
        pipe_local.enable_attention_slicing()  # reduce memory usage
        pipe = pipe_local.to(device)
        log("Model loaded successfully.")
    except Exception as e:
        log("Error loading model:", e)
        log(traceback.format_exc())
        raise

def generate_character_image(prompt, save_path):
    """
    Generate a single image from the prompt using the loaded pipeline.
    """
    ensure_model_loaded()
    # Generate image
    result = pipe(prompt, num_inference_steps=20)
    image = result.images[0]
    image.save(save_path)
    return save_path

def make_text_clip(text, duration, start):
    img = Image.new("RGB", (1280, 200), color="black")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    draw.text((50, 50), text, font=font, fill="white")
    frame = np.array(img)
    return mpy.ImageClip(frame).set_duration(duration).set_start(start).set_position(("center", "bottom"))

def make_character_clip(image_path, duration, start):
    char_img = Image.open(image_path).convert("RGBA").resize((400, 400))
    frame = np.array(char_img)
    return mpy.ImageClip(frame).set_duration(duration).set_start(start).set_position(("center", "center"))

def split_lines(script):
    lines = re.split(r'[।.!?\n]', script)
    return [line.strip() for line in lines if line.strip()]

def handler(event):
    try:
        log("Received event:", event)
        script = event.get("input", {}).get("script", None)
        if not script:
            return {"error": "No script provided."}

        workspace = "/runpod-volume"
        os.makedirs(workspace, exist_ok=True)

        # --- Step 1: Generate narration ---
        audio_path = os.path.join(workspace, "narration.mp3")
        log("Generating TTS...")
        tts = gTTS(text=script, lang='hi')
        tts.save(audio_path)
        audio = AudioSegment.from_mp3(audio_path)

        # --- Step 2: Detect speech segments ---
        log("Splitting audio into chunks...")
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=audio.dBFS - 14,
            keep_silence=250
        )
        if not chunks:
            # fallback: single chunk for entire audio
            chunks = [audio]

        # --- Step 3: Background video ---
        total_duration = audio.duration_seconds
        clip = mpy.ColorClip(size=(1280, 720), color=(0, 0, 0)).set_duration(total_duration).set_fps(24)

        # --- Step 4: Generate character image once (lazy model load) ---
        char_path = os.path.join(workspace, "character.png")
        try:
            log("Generating character image (this will load the model if not already loaded)...")
            generate_character_image("cartoon character explaining Hindi subtitles", char_path)
        except Exception as e:
            # If model generation fails, fall back to a simple placeholder image
            log("Character generation failed, creating placeholder. Error:", e)
            img = Image.new("RGB", (400, 400), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            draw.text((20, 180), "Character", font=font, fill="white")
            img.save(char_path)

        # --- Step 5: Map script lines to audio chunks ---
        lines = split_lines(script)
        overlays = []
        current_time = 0.0
        for i, chunk in enumerate(chunks):
            duration = chunk.duration_seconds if hasattr(chunk, "duration_seconds") else (len(chunk) / 1000.0)
            line = lines[i] if i < len(lines) else ""
            txt_clip = make_text_clip(line, duration, current_time)
            char_clip = make_character_clip(char_path, duration, current_time)
            overlays.extend([txt_clip, char_clip])
            current_time += duration

        # --- Step 6: Combine video + subtitles + character + audio ---
        final = mpy.CompositeVideoClip([clip, *overlays])
        final = final.set_audio(mpy.AudioFileClip(audio_path))

        output_path = os.path.join(workspace, "final_video.mp4")
        log("Writing final video to:", output_path)
        final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        # --- Step 7: Encode video as base64 ---
        log("Encoding video to base64...")
        with open(output_path, "rb") as f:
            encoded_video = base64.b64encode(f.read()).decode("utf-8")

        log("Done. Returning base64 video.")
        return {"output": {"video_base64": encoded_video}}

    except Exception as e:
        log("Handler error:", e)
        log(traceback.format_exc())
        return {"error": str(e), "trace": traceback.format_exc()}

if __name__ == "__main__":
    log("RunPod handler starting. GPU available:", torch.cuda.is_available())
    runpod.serverless.start({"handler": handler})
