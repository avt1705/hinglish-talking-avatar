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
from pydub import AudioSegment, silence
from elevenlabs.client import ElevenLabs
from elevenlabs import save as el_save

FONT_PATH = "/app/NotoSansDevanagari.ttf"

def log(*args):
    print(*args, flush=True)

# ==========================================
# 1. AUDIO GENERATION & ANALYSIS
# ==========================================
def generate_audio(script: str, output_path: str, api_key: str = None):
    if api_key:
        try:
            log("Attempting ElevenLabs TTS generation...")
            client = ElevenLabs(api_key=api_key)
            audio = client.generate(text=script, voice="Rachel", model="eleven_multilingual_v2")
            el_save(audio, output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                log("ElevenLabs audio generated successfully.")
                return
        except Exception as e:
            log(f"ElevenLabs failed: {e}. Falling back to Edge TTS.")
            
    log("Running Edge TTS generation via CLI...")
    cmd = ["edge-tts", "--text", script, "--voice", "hi-IN-MadhurNeural", "--write-media", output_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            log("Edge TTS audio successfully saved.")
            return
    except subprocess.CalledProcessError as e:
        log("Edge TTS Failed. Falling back to basic gTTS...")
        
    from gtts import gTTS
    tts = gTTS(text=script, lang='hi', slow=False)
    tts.save(output_path)
    log("Fallback gTTS audio saved.")

def get_speaking_intervals(audio_path: str):
    audio = AudioSegment.from_file(audio_path)
    nonsilent_ranges = silence.detect_nonsilent(audio, min_silence_len=150, silence_thresh=-40)
    return [(start / 1000.0, end / 1000.0) for start, end in nonsilent_ranges], audio.duration_seconds

# ==========================================
# 2. SUBTITLE GENERATION
# ==========================================
def create_subtitle_frame(text: str, width=1280, height=720):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, 55)
    except Exception:
        font = ImageFont.load_default()
        
    wrapped_text = "\n".join(textwrap.wrap(text, width=38))
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=10)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    pos_x = (width - text_w) // 2
    pos_y = height - text_h - 60  
    
    draw.multiline_text(
        (pos_x, pos_y), 
        wrapped_text, 
        font=font, 
        fill=(255, 223, 0), 
        stroke_width=6, 
        stroke_fill=(0, 0, 0),
        align="center", 
        spacing=10
    )
    return np.array(img)

# ==========================================
# 3. RUNPOD HANDLER
# ==========================================
def handler(event):
    try:
        input_data = event.get("input", {})
        script = input_data.get("script", "").strip()
        elevenlabs_key = input_data.get("elevenlabs_key", "")
        
        if not script: 
            return {"error": "Missing 'script' input."}
            
        workdir = "/tmp/runpod_job"
        os.makedirs(workdir, exist_ok=True)
        audio_path = os.path.join(workdir, "speech.mp3")
        final_video_path = os.path.join(workdir, "final.mp4")
        
        log("Generating Audio...")
        generate_audio(script, audio_path, elevenlabs_key)
        speaking_intervals, total_duration = get_speaking_intervals(audio_path)
        
        # Load the baked-in custom transparent sprites
        idle_frame = np.array(Image.open("/app/my_scene1.png").convert("RGBA"))
        talk1_frame = np.array(Image.open("/app/my_scene2.png").convert("RGBA"))
        talk2_frame = np.array(Image.open("/app/my_scene3.png").convert("RGBA"))
        
        def get_character_frame(t):
            is_speaking = any(start <= t <= end for start, end in speaking_intervals)
            if is_speaking:
                # Toggle mouth frames rapidly for a talking effect
                return talk1_frame if int(t / 0.15) % 2 == 0 else talk2_frame
            return idle_frame

        # Position character in the center and scale properly
        char_clip = mpy.VideoClip(get_character_frame, duration=total_duration)
        w, h = char_clip.size
        new_h = 600
        char_clip = char_clip.resize(height=new_h)
        char_clip = char_clip.set_position(('center', 720 - new_h))
        
        # Background Layer
        bg_clip = mpy.ColorClip(size=(1280, 720), color=(18, 18, 24)).set_duration(total_duration)
        
        # Subtitles sync
        sentences = [p.strip() for p in re.split(r'[।.!?\n]+', script) if p.strip()]
        audio_segment = AudioSegment.from_file(audio_path)
        chunks = silence.split_on_silence(audio_segment, min_silence_len=300, silence_thresh=-40, keep_silence=150)
        if not chunks: chunks = [audio_segment]
        
        text_clips = []
        current_time = 0.0
        for idx, chunk in enumerate(chunks):
            duration = chunk.duration_seconds
            text = sentences[idx] if idx < len(sentences) else ""
            if text:
                text_frame = create_subtitle_frame(text)
                txt_clip = mpy.ImageClip(text_frame).set_duration(duration).set_start(current_time)
                text_clips.append(txt_clip)
            current_time += duration

        log("Compositing final video...")
        final_video = mpy.CompositeVideoClip([bg_clip, char_clip, *text_clips])
        final_video = final_video.set_audio(mpy.AudioFileClip(audio_path))
        
        final_video.write_videofile(final_video_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        
        with open(final_video_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            
        return {"output": {"video_base64": encoded}}
        
    except Exception as e:
        log("Error:", traceback.format_exc())
        return {"error": str(e), "trace": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
