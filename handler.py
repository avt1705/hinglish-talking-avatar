import os
import re
import sys
import time
import base64
import textwrap
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import runpod
import moviepy.editor as mpy
from gtts import gTTS
from pydub import AudioSegment, silence

# Global reference for lazy-loaded Stable Diffusion pipeline
pipe = None
FONT_PATH = "/app/NotoSansDevanagari.ttf"


def log(*args):
    print(*args, flush=True)


def ensure_model_loaded():
    """Lazy loads Stable Diffusion v1.5 onto GPU memory on the first invocation."""
    global pipe
    if pipe is not None:
        return

    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Initializing Stable Diffusion on device: {device}")

    load_kwargs = {}
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16

    pipe_local = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        **load_kwargs
    )
    pipe_local.safety_checker = None

    if device == "cuda":
        pipe_local.enable_attention_slicing()

    pipe = pipe_local.to(device)
    log("Stable Diffusion loaded successfully.")


def generate_instructor_image(prompt: str, save_path: str):
    """Generates an instructor portrait or falls back to a clean placeholder."""
    try:
        ensure_model_loaded()
        instructor_prompt = f"{prompt}, centered portrait, educational studio background, 8k resolution, photorealistic"
        result = pipe(instructor_prompt, num_inference_steps=25, guidance_scale=7.5)
        img = result.images[0].resize((500, 500))
        img.save(save_path)
    except Exception as e:
        log("Image generation failed, falling back to placeholder. Error:", e)
        img = Image.new("RGB", (500, 500), color=(30, 41, 59))
        draw = ImageDraw.Draw(img)
        try:
            fallback_font = ImageFont.truetype(FONT_PATH, 32)
        except Exception:
            fallback_font = ImageFont.load_default()
        draw.text((150, 230), "AI Instructor", font=fallback_font, fill=(255, 255, 255))
        img.save(save_path)


def split_sentences(text: str):
    """Splits Hindi text by common sentence delimiters (danda ।, full stop, newline)."""
    parts = re.split(r'[।.!?\n]+', text)
    return [p.strip() for p in parts if p.strip()]


def create_subtitle_frame(text: str, width=1280, height=180):
    """Draws Hindi text centered with a dark banner background."""
    img = Image.new("RGBA", (width, height), (15, 23, 42, 220))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 36)
    except Exception:
        font = ImageFont.load_default()

    wrapped_text = "\n".join(textwrap.wrap(text, width=45))
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=8)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pos_x = (width - text_w) // 2
    pos_y = (height - text_h) // 2

    draw.multiline_text(
        (pos_x, pos_y),
        wrapped_text,
        font=font,
        fill=(255, 255, 255),
        align="center",
        spacing=8
    )
    return np.array(img)


def handler(event):
    try:
        input_data = event.get("input", {})
        script = input_data.get("script", "").strip()
        char_prompt = input_data.get(
            "character_prompt",
            "Indian female teacher explaining on screen, professional, friendly"
        )

        if not script:
            return {"error": "Missing 'script' input."}

        workdir = "/tmp/runpod_job"
        os.makedirs(workdir, exist_ok=True)
        audio_path = os.path.join(workdir, "speech.mp3")
        instructor_img_path = os.path.join(workdir, "instructor.png")
        video_output_path = os.path.join(workdir, "final.mp4")

        # 1. Generate Hindi Narration via gTTS
        log("Generating Hindi TTS audio...")
        tts = gTTS(text=script, lang="hi", slow=False)
        tts.save(audio_path)

        audio_segment = AudioSegment.from_mp3(audio_path)
        total_duration = audio_segment.duration_seconds

        # 2. Split audio into phrase chunks for subtitle timing
        chunks = silence.split_on_silence(
            audio_segment,
            min_silence_len=400,
            silence_thresh=audio_segment.dBFS - 14,
            keep_silence=200
        )
        if not chunks:
            chunks = [audio_segment]

        sentences = split_sentences(script)

        # 3. Generate Instructor Image
        log("Generating instructor visual...")
        generate_instructor_image(char_prompt, instructor_img_path)

        # 4. Compose Video Layers
        # Background: Modern studio slate canvas (1280x720)
        bg_clip = mpy.ColorClip(size=(1280, 720), color=(15, 23, 42)).set_duration(total_duration)

        # Instructor avatar placed on the right side of the screen
        instructor_clip = (
            mpy.ImageClip(instructor_img_path)
            .set_duration(total_duration)
            .set_position((720, 80))
        )

        # Build Subtitle timing overlays
        subtitle_clips = []
        current_time = 0.0
        for idx, chunk in enumerate(chunks):
            duration = chunk.duration_seconds
            subtitle_text = sentences[idx] if idx < len(sentences) else ""

            if subtitle_text:
                sub_frame = create_subtitle_frame(subtitle_text)
                sub_clip = (
                    mpy.ImageClip(sub_frame)
                    .set_duration(duration)
                    .set_start(current_time)
                    .set_position(("center", 520))
                )
                subtitle_clips.append(sub_clip)

            current_time += duration

        # Combine all video elements
        final_video = mpy.CompositeVideoClip([bg_clip, instructor_clip, *subtitle_clips])
        final_video = final_video.set_audio(mpy.AudioFileClip(audio_path))

        # 5. Render Video File
        log("Encoding output MP4...")
        final_video.write_videofile(
            video_output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-crf", "26", "-preset", "fast"],
            verbose=False,
            logger=None
        )

        # 6. Return Base64 Encoded Result
        with open(video_output_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return {"output": {"video_base64": encoded}}

    except Exception as e:
        log("Error in handler execution:", traceback.format_exc())
        return {"error": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
