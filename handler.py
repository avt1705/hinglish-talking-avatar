import os
import re
import base64
import traceback
import subprocess

import numpy as np
import runpod

from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    AudioFileClip,
    VideoClip,
    ImageClip,
    ColorClip,
    CompositeVideoClip
)


# =====================================================
# VIDEO SETTINGS
# =====================================================

WIDTH = 1920
HEIGHT = 1080
FPS = 30

AVATAR_WIDTH = 750
AVATAR_HEIGHT = 900

AVATAR_X = 15
AVATAR_Y = 90

PANEL_X = 780
PANEL_WIDTH = 1140

SCENE1_PATH = "/app/my_scene1.png"
SCENE2_PATH = "/app/my_scene2.png"

AUDIO_PATH = "/tmp/narration.mp3"
OUTPUT_PATH = "/tmp/final_video.mp4"
RAW_AUDIO_PATH = "/tmp/audio_analysis.raw"

SAMPLE_RATE = 16000

CHUNK_DURATION = 0.05

OPEN_THRESHOLD = 0.012
CLOSE_THRESHOLD = 0.007


# =====================================================
# FONT
# =====================================================

def get_hindi_font(size=48):

    paths = [

        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",

        "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf",

        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf"

    ]

    for path in paths:

        if os.path.exists(path):

            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


# =====================================================
# AVATAR IMAGES
# =====================================================

def prepare_avatar_images():

    closed = Image.open(
        SCENE1_PATH
    ).convert("RGB")

    opened = Image.open(
        SCENE2_PATH
    ).convert("RGB")

    try:
        resampling = Image.Resampling.LANCZOS
    except AttributeError:
        resampling = Image.LANCZOS

    closed = closed.resize(
        (AVATAR_WIDTH, AVATAR_HEIGHT),
        resampling
    )

    opened = opened.resize(
        (AVATAR_WIDTH, AVATAR_HEIGHT),
        resampling
    )

    return (
        np.array(closed),
        np.array(opened)
    )


# =====================================================
# AUDIO ANALYSIS
# =====================================================

def analyze_audio(audio_clip):

    duration = audio_clip.duration

    command = [

        "ffmpeg",
        "-y",
        "-i", AUDIO_PATH,
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        RAW_AUDIO_PATH

    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    audio_data = np.fromfile(
        RAW_AUDIO_PATH,
        dtype=np.float32
    )

    samples_per_chunk = int(
        SAMPLE_RATE * CHUNK_DURATION
    )

    total_chunks = int(
        np.ceil(duration / CHUNK_DURATION)
    )

    envelope = []

    for i in range(total_chunks):

        start = i * samples_per_chunk

        end = start + samples_per_chunk

        chunk = audio_data[start:end]

        if len(chunk) == 0:

            rms = 0

        else:

            rms = np.sqrt(
                np.mean(chunk ** 2)
            )

        envelope.append(rms)

    envelope = np.array(envelope)

    if len(envelope) >= 3:

        envelope = np.convolve(
            envelope,
            np.ones(3) / 3,
            mode="same"
        )

    peak = (
        float(np.max(envelope))
        if len(envelope)
        else 0
    )

    open_threshold = max(
        OPEN_THRESHOLD,
        peak * 0.12
    )

    close_threshold = max(
        CLOSE_THRESHOLD,
        peak * 0.07
    )

    return (
        envelope,
        open_threshold,
        close_threshold
    )


# =====================================================
# CONTINUOUS AVATAR ANIMATION
# =====================================================

def create_avatar(
    closed_image,
    open_image,
    envelope,
    open_threshold,
    close_threshold,
    duration
):

    states = []

    is_open = False

    for volume in envelope:

        if not is_open and volume >= open_threshold:

            is_open = True

        elif is_open and volume <= close_threshold:

            is_open = False

        states.append(is_open)

    states = np.array(states)

    def make_frame(t):

        index = int(
            t / CHUNK_DURATION
        )

        index = min(
            index,
            len(states) - 1
        )

        if states[index]:

            return open_image

        return closed_image

    avatar = VideoClip(
        make_frame=make_frame,
        duration=duration
    )

    avatar = avatar.set_position(
        (AVATAR_X, AVATAR_Y)
    )

    return avatar


# =====================================================
# SPLIT SCRIPT INTO SENTENCES
# =====================================================

def split_script(script):

    lines = [
        line.strip()
        for line in script.splitlines()
        if line.strip()
    ]

    if len(lines) > 1:
        return lines

    sentences = re.split(
        r'(?<=[।.!?])\s+',
        script.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences if sentences else [script.strip()]


# =====================================================
# SENTENCE TIMING
# =====================================================

def calculate_sentence_timings(
    sentences,
    duration
):

    total_chars = sum(
        len(sentence)
        for sentence in sentences
    )

    timings = []

    current_time = 0

    for sentence in sentences:

        if total_chars:

            sentence_duration = (
                len(sentence) / total_chars
            ) * duration

        else:

            sentence_duration = (
                duration / len(sentences)
            )

        timings.append({

            "text": sentence,

            "start": current_time,

            "end": current_time + sentence_duration

        })

        current_time += sentence_duration

    return timings


# =====================================================
# WRAP HINDI TEXT
# =====================================================

def wrap_text(
    draw,
    text,
    font,
    max_width
):

    words = text.split()

    lines = []

    current_line = ""

    for word in words:

        test_line = (
            current_line + " " + word
        ).strip()

        box = draw.textbbox(
            (0, 0),
            test_line,
            font=font
        )

        text_width = box[2] - box[0]

        if text_width <= max_width:

            current_line = test_line

        else:

            if current_line:

                lines.append(current_line)

            current_line = word

    if current_line:

        lines.append(current_line)

    return lines


# =====================================================
# SPLIT SCREEN TEXT PANEL
# =====================================================

def create_text_panel(
    timings,
    duration
):

    font = get_hindi_font(43)

    small_font = get_hindi_font(34)

    def make_frame(t):

        image = Image.new(
            "RGB",
            (PANEL_WIDTH, HEIGHT),
            (250, 250, 250)
        )

        draw = ImageDraw.Draw(image)

        # Panel heading
        draw.text(
            (65, 75),
            "गीत का मतलब",
            font=get_hindi_font(55),
            fill=(25, 25, 25)
        )

        draw.rounded_rectangle(
            (65, 155, PANEL_WIDTH - 65, 160),
            radius=3,
            fill=(220, 220, 220)
        )

        # Find current sentence
        current_index = 0

        for i, item in enumerate(timings):

            if item["start"] <= t < item["end"]:

                current_index = i

                break

            if t >= item["end"]:

                current_index = i

        # Show recent sentences
        start_index = max(
            0,
            current_index - 3
        )

        visible = timings[
            start_index:current_index + 1
        ]

        y = 220

        for local_index, item in enumerate(visible):

            actual_index = (
                start_index + local_index
            )

            is_current = (
                actual_index == current_index
            )

            if is_current:

                box_fill = (255, 238, 190)

                text_fill = (25, 25, 25)

                text_font = font

            else:

                box_fill = (240, 240, 240)

                text_fill = (125, 125, 125)

                text_font = small_font

            wrapped = wrap_text(
                draw,
                item["text"],
                text_font,
                PANEL_WIDTH - 190
            )

            box_height = (
                max(1, len(wrapped)) * 65 + 45
            )

            draw.rounded_rectangle(
                (
                    65,
                    y,
                    PANEL_WIDTH - 65,
                    y + box_height
                ),
                radius=22,
                fill=box_fill
            )

            text_y = y + 20

            for line in wrapped:

                draw.text(
                    (100, text_y),
                    line,
                    font=text_font,
                    fill=text_fill
                )

                text_y += 65

            y += box_height + 25

        return np.array(image)

    panel = VideoClip(
        make_frame=make_frame,
        duration=duration
    )

    panel = panel.set_position(
        (PANEL_X, 0)
    )

    return panel


# =====================================================
# MAIN HANDLER
# =====================================================

def handler(event):

    audio_clip = None

    background = None

    avatar_clip = None

    panel_clip = None

    final_video = None

    try:

        print("Starting PTG video generation...")

        job_input = event.get(
            "input",
            {}
        )

        script = job_input.get(
            "script",
            ""
        ).strip()

        audio_base64 = job_input.get(
            "audio_base64",
            ""
        )

        if not script:

            return {
                "error": "Hindi script is missing."
            }

        if not audio_base64:

            return {
                "error": "Sarvam audio is missing."
            }

        if len(script) > 2500:

            return {
                "error": "Script exceeds Sarvam's 2500-character limit."
            }

        # ---------------------------------------------
        # STEP 1: SAVE SARVAM AUDIO
        # ---------------------------------------------

        print("Decoding Sarvam narration...")

        audio_bytes = base64.b64decode(
            audio_base64
        )

        with open(
            AUDIO_PATH,
            "wb"
        ) as f:

            f.write(audio_bytes)

        # ---------------------------------------------
        # STEP 2: LOAD AUDIO
        # ---------------------------------------------

        audio_clip = AudioFileClip(
            AUDIO_PATH
        )

        duration = audio_clip.duration

        print("Narration duration:", duration)

        # ---------------------------------------------
        # STEP 3: ANALYZE AUDIO
        # ---------------------------------------------

        envelope, open_threshold, close_threshold = analyze_audio(
            audio_clip
        )

        # ---------------------------------------------
        # STEP 4: LOAD AVATAR
        # ---------------------------------------------

        closed_image, open_image = prepare_avatar_images()

        # ---------------------------------------------
        # STEP 5: CREATE AVATAR ANIMATION
        # ---------------------------------------------

        avatar_clip = create_avatar(
            closed_image,
            open_image,
            envelope,
            open_threshold,
            close_threshold,
            duration
        )

        # ---------------------------------------------
        # STEP 6: CREATE SPLIT SCREEN PANEL
        # ---------------------------------------------

        sentences = split_script(script)

        timings = calculate_sentence_timings(
            sentences,
            duration
        )

        panel_clip = create_text_panel(
            timings,
            duration
        )

        # ---------------------------------------------
        # STEP 7: WHITE BACKGROUND
        # ---------------------------------------------

        background = ColorClip(
            size=(WIDTH, HEIGHT),
            color=(255, 255, 255),
            duration=duration
        )

        # ---------------------------------------------
        # STEP 8: COMBINE VIDEO
        # ---------------------------------------------

        print("Combining split-screen video...")

        final_video = CompositeVideoClip(

            [
                background,
                avatar_clip,
                panel_clip
            ],

            size=(WIDTH, HEIGHT)

        )

        final_video = final_video.set_audio(
            audio_clip
        )

        # ---------------------------------------------
        # STEP 9: EXPORT
        # ---------------------------------------------

        print("Rendering final MP4...")

        final_video.write_videofile(

            OUTPUT_PATH,

            fps=FPS,

            codec="libx264",

            audio_codec="aac",

            bitrate="4000k",

            preset="ultrafast",

            threads=4,

            logger=None

        )

        # ---------------------------------------------
        # STEP 10: ENCODE VIDEO
        # ---------------------------------------------

        if not os.path.exists(OUTPUT_PATH):

            raise Exception(
                "Video file was not created."
            )

        file_size = os.path.getsize(
            OUTPUT_PATH
        )

        if file_size == 0:

            raise Exception(
                "Generated video is empty."
            )

        with open(
            OUTPUT_PATH,
            "rb"
        ) as f:

            video_base64 = base64.b64encode(
                f.read()
            ).decode("utf-8")

        print("Video generation successful.")

        return {

            "video_base64": video_base64,

            "duration": round(duration, 2),

            "resolution": "1920x1080",

            "fps": FPS,

            "voice": "Sarvam Bulbul V3 - Shubh",

            "layout": "Split Screen",

            "file_size_mb": round(
                file_size / (1024 * 1024),
                2
            )

        }

    except Exception as e:

        print(
            "VIDEO GENERATION ERROR:",
            traceback.format_exc()
        )

        return {

            "error": str(e),

            "traceback": traceback.format_exc()

        }

    finally:

        print("Cleaning temporary resources...")

        clips = [

            final_video,

            background,

            avatar_clip,

            panel_clip,

            audio_clip

        ]

        for clip in clips:

            if clip is not None:

                try:

                    clip.close()

                except Exception:

                    pass

        for path in [

            AUDIO_PATH,

            OUTPUT_PATH,

            RAW_AUDIO_PATH

        ]:

            try:

                if os.path.exists(path):

                    os.remove(path)

            except Exception:

                pass


# =====================================================
# START RUNPOD SERVERLESS
# =====================================================

runpod.serverless.start({

    "handler": handler

})
