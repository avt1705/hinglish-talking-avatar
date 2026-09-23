import os
import re
import base64
import textwrap
import traceback
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import runpod
import moviepy.editor as mpy
from gtts import gTTS

# Hardcoded absolute paths based on the Dockerfile layout
FONT_PATH = "/app/NotoSansDevanagari.ttf"
SOURCE_IMAGE_PATH = "/app/frame1.png"

def create_subtitle_clip(text, duration):
    width, height = 1080, 180
    img = Image.new("RGBA", (width, height), (15, 23, 42, 220))
    draw = ImageDraw.Draw(img)

    # Strictly enforce the Hindi font. Do not use load_default() as it crashes on Devanagari.
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"Critical Error: Hindi font not found at {FONT_PATH}")
        
    font = ImageFont.truetype(FONT_PATH, 34)

    # Wrap the text to fit the screen
    wrapped = "\n".join(textwrap.wrap(text, width=42))
    
    # Use textbbox instead of deprecated textsize
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=6)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    draw.multiline_text(
        ((width - text_w) // 2, (height - text_h) // 2),
        wrapped,
        font=font,
        fill=(255, 255, 255),
        align="center",
        spacing=6
    )
    return mpy.ImageClip(np.array(img)).set_duration(duration).set_position(("center", "bottom"))

def handler(event):
    try:
        workdir = "/tmp/runpod_job"
        os.makedirs(workdir, exist_ok=True)

        audio_path = os.path.join(workdir, "speech.mp3")
        final_video_path = os.path.join(workdir, "final.mp4")

        input_data = event.get("input", {})
        audio_base64 = input_data.get("audio_base64", "")
        
        raw_script = input_data.get("script", "")
        if isinstance(raw_script, bytes):
            raw_script = raw_script.decode("utf-8")
        
        script = raw_script.strip()
        tts_script = script.replace('।', '.')

        # 1. Obtain Audio
        if audio_base64:
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
        elif tts_script:
            tts = gTTS(text=tts_script, lang="hi", slow=False)
            tts.save(audio_path)
        else:
            return {"error": "No script or audio provided."}

        # 2. Verify Single Image Source
        if not os.path.exists(SOURCE_IMAGE_PATH):
            return {"error": f"Missing frame1.png. Checked {SOURCE_IMAGE_PATH}"}

        # 3. Execute SadTalker AI Animation
        print("Starting SadTalker facial animation...", flush=True)
        sadtalker_cmd = [
            "python3", "/app/SadTalker/inference.py",
            "--driven_audio", audio_path,
            "--source_image", SOURCE_IMAGE_PATH,
            "--result_dir", workdir,
            "--still",
            "--enhancer", "gfpgan"
        ]
        subprocess.run(sadtalker_cmd, check=True)

        # 4. Locate the AI-generated video
        generated_video_path = None
        for root, dirs, files in os.walk(workdir):
            for file in files:
                if file.endswith(".mp4") and file != "final.mp4":
                    generated_video_path = os.path.join(root, file)
                    break
                    
        if not generated_video_path:
            return {"error": "SadTalker failed to output an MP4 file."}

        # 5. Overlay Subtitles & Encode
        print("Applying subtitles and encoding...", flush=True)
        avatar_clip = mpy.VideoFileClip(generated_video_path).resize(width=1080)
        total_dur = avatar_clip.duration
        
        if script:
            sub_clip = create_subtitle_clip(script, total_dur)
            final_clip = mpy.CompositeVideoClip([avatar_clip, sub_clip])
        else:
            final_clip = avatar_clip

        final_clip.write_videofile(
            final_video_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None,
            verbose=False
        )

        with open(final_video_path, "rb") as f:
            encoded_video = base64.b64encode(f.read()).decode("utf-8")

        return {"output": {"video_base64": encoded_video}}

    except subprocess.CalledProcessError as e:
        print(f"SadTalker Subprocess Error: {e}", flush=True)
        return {"error": "SadTalker animation failed.", "trace": traceback.format_exc()}
    except Exception as e:
        print(f"Execution Error Occurred: {e}", flush=True)
        return {"error": str(e), "trace": traceback.format_exc()}

if __name__ == "__main__":
    print("Starting AI Animation Worker...", flush=True)
    runpod.serverless.start({"handler": handler})
