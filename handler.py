# handler.py
"""
RunPod serverless handler for generating a short video from a script.
- Lazy-loads Stable Diffusion at first use (avoids long blocking model download at container start).
- Prints early import/version info so import-time failures are visible in container logs.
- Falls back to a simple placeholder character image if model generation fails.
- Returns the final .mp4 as a base64 string in the JSON response (ephemeral delivery).
Notes:
- Keep videos short to avoid response-size limits.
- Ensure Dockerfile installs torch via the correct wheel (GPU vs CPU) and pins huggingface_hub to a compatible version.
"""

# -------------------------
# Early import / version check
# -------------------------
import time
import traceback
import sys

print("Container booting — early log", flush=True)
try:
    import torch  # noqa: F401
    import diffusers  # noqa: F401
    import huggingface_hub  # noqa: F401
    print(
        "Versions:",
        "torch", getattr(torch, "__version__", None),
        "diffusers", getattr(diffusers, "__version__", None),
        "huggingface_hub", getattr(huggingface_hub, "__version__", None),
        flush=True
    )
except Exception as e:
    print("EARLY IMPORT ERROR:", e, flush=True)
    print(traceback.format_exc(), flush=True)
    # Give the platform a moment to capture logs before the process exits
    time.sleep(10)
    raise

# -------------------------
# Normal imports (after early check)
# -------------------------
import os
import re
import base64
import runpod
import moviepy.editor as mpy
from gtts import gTTS
from pydub import AudioSegment, silence
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# diffusers import will be used inside ensure_model_loaded to avoid import-time side effects
from diffusers import StableDiffusionPipeline  # safe now because early check passed

# -------------------------
# Globals and utilities
# -------------------------
pipe = None  # Stable Diffusion pipeline (lazy-loaded)

def log(*args, **kwargs):
    """Simple logger that flushes immediately for container logs."""
    print(*args, **kwargs, flush=True)

def ensure_model_loaded():
    """
    Lazy-load the Stable Diffusion pipeline on first request.
    Uses float16 on CUDA to reduce memory footprint when GPU is available.
    """
    global pipe
    if pipe is not None:
        return

    try:
        import torch  # local import to ensure availability
        from diffusers import StableDiffusionPipeline  # local import for clarity

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log("ensure_model_loaded: device =", device)

        load_kwargs = {}
        if device == "cuda":
            # Use float16 to reduce VRAM usage
            load_kwargs["torch_dtype"] = torch.float16

        log("Loading Stable Diffusion model (this may take a while on first run)...")
        # This will download and cache the model in the Hugging Face cache directory
        pipe_local = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            **load_kwargs
        )
        # Optional: disable safety checker if you accept the tradeoffs
        try:
            pipe_local.safety_checker = None
        except Exception:
            pass
        # Reduce memory usage
        try:
            pipe_local.enable_attention_slicing()
        except Exception:
            pass

        pipe = pipe_local.to(device)
        log("Model loaded successfully.")
    except Exception as e:
        log("Error while loading model:", e)
        log(traceback.format_exc())
        # Re-raise so caller can handle fallback behavior
        raise

def generate_character_image(prompt: str, save_path: str):
    """
    Generate a single image using the Stable Diffusion pipeline.
    If generation fails, raise the exception so caller can fallback.
    """
    ensure_model_loaded()
    # Use a modest number of inference steps to keep generation time reasonable
    result = pipe(prompt, num_inference_steps=20)
    image = result.images[0]
    image.save(save_path)
    return save_path

def make_text_clip(text: str, duration: float, start: float):
    """Create a simple text image clip for subtitles."""
    img = Image.new("RGB", (1280, 200), color="black")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    draw.text((50, 50), text, font=font, fill="white")
    frame = np.array(img)
    return mpy.ImageClip(frame).set_duration(duration).set_start(start).set_position(("center", "bottom"))

def make_character_clip(image_path: str, duration: float, start: float):
    """Create a clip from the generated character image."""
    char_img = Image.open(image_path).convert("RGBA").resize((400, 400))
    frame = np.array(char_img)
    return mpy.ImageClip(frame).set_duration(duration).set_start(start).set_position(("center", "center"))

def split_lines(script: str):
    """Split script into lines/sentences for subtitle mapping."""
    lines = re.split(r'[।.!?\n]', script)
    return [line.strip() for line in lines if line.strip()]

# -------------------------
# Handler
# -------------------------
def handler(event):
    """
    Expected event format:
    {
      "input": {
        "script": "Your script text here..."
      }
    }
    Returns:
    {
      "output": {
        "video_base64": "<base64 string>"
      }
    }
    """
    try:
        log("Handler invoked. Event:", event)
        script = event.get("input", {}).get("script", None)
        if not script:
            return {"error": "No script provided."}

        workspace = "/runpod-volume"
        os.makedirs(workspace, exist_ok=True)

        # --- Step 1: Generate narration (TTS) ---
        audio_path = os.path.join(workspace, "narration.mp3")
        log("Generating TTS...")
        try:
            tts = gTTS(text=script, lang='hi')
            tts.save(audio_path)
        except Exception as e:
            log("TTS generation failed:", e)
            # If TTS fails, create a short silent audio fallback
            silent_ms = 1000
            silent = AudioSegment.silent(duration=silent_ms)
            silent.export(audio_path, format="mp3")

        audio = AudioSegment.from_mp3(audio_path)

        # --- Step 2: Split audio into chunks ---
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

        # --- Step 4: Generate character image once (with fallback) ---
        char_path = os.path.join(workspace, "character.png")
        try:
            log("Attempting to generate character image (this will load the model if needed)...")
            # Example prompt; you can change or accept a prompt from the event in future
            prompt = event.get("input", {}).get("character_prompt", "cartoon character explaining Hindi subtitles")
            generate_character_image(prompt, char_path)
            log("Character image generated:", char_path)
        except Exception as e:
            log("Character generation failed; creating placeholder. Error:", e)
            # Create a simple placeholder image
            img = Image.new("RGB", (400, 400), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            except Exception:
                font = ImageFont.load_default()
            draw.text((20, 180), "Character", font=font, fill="white")
            img.save(char_path)
            log("Placeholder character saved at:", char_path)

        # --- Step 5: Map script lines to audio chunks ---
        lines = split_lines(script)
        overlays = []
        current_time = 0.0
        for i, chunk in enumerate(chunks):
            # pydub chunk may not have duration_seconds attribute in some versions; handle both
            duration = getattr(chunk, "duration_seconds", None)
            if duration is None:
                duration = len(chunk) / 1000.0
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
        # Use libx264 and aac for broad compatibility; suppress verbose logging
        final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        # --- Step 7: Encode video as base64 ---
        log("Encoding video to base64...")
        with open(output_path, "rb") as f:
            encoded_video = base64.b64encode(f.read()).decode("utf-8")

        log("Handler completed successfully.")
        return {"output": {"video_base64": encoded_video}}

    except Exception as e:
        log("Handler error:", e)
        log(traceback.format_exc())
        return {"error": str(e), "trace": traceback.format_exc()}

# -------------------------
# RunPod serverless entrypoint
# -------------------------
if __name__ == "__main__":
    # Print a clear startup message (visible in container logs)
    try:
        import torch as _torch
        gpu_available = _torch.cuda.is_available()
    except Exception:
        gpu_available = False
    log("RunPod handler starting. GPU available:", gpu_available)
    runpod.serverless.start({"handler": handler})
