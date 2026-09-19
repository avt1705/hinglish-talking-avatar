import os
import base64
import traceback
import runpod
import moviepy.editor as mpy
from pydub import AudioSegment, silence

def get_speaking_intervals(audio_path):
    """
    Analyzes the audio file to find intervals where the character is speaking vs silent.
    Returns a list of (start_time, end_time) in seconds, and the total duration.
    """
    audio = AudioSegment.from_file(audio_path)
    total_dur = len(audio) / 1000.0
    
    # Detect non-silent parts (speaking)
    nonsilent_ranges = silence.detect_nonsilent(
        audio, 
        min_silence_len=200, 
        silence_thresh=audio.dBFS - 16
    )
    
    # Convert from milliseconds to seconds
    speaking_intervals = [(start / 1000.0, end / 1000.0) for start, end in nonsilent_ranges]
    return speaking_intervals, total_dur


def handler(event):
    try:
        # Dynamically get the exact folder where this script is running
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        workdir = "/tmp/runpod_job"
        os.makedirs(workdir, exist_ok=True)
        
        audio_path = os.path.join(workdir, "speech.wav")
        final_video_path = os.path.join(workdir, "final.mp4")

        input_data = event.get("input", {})
        
        # Handle the audio input (fallback to your baked-in my_voice.wav if base64 is missing)
        if "audio_base64" in input_data and input_data["audio_base64"]:
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(input_data["audio_base64"]))
        else:
            default_audio = os.path.join(base_dir, "my_voice.wav")
            if os.path.exists(default_audio):
                import shutil
                shutil.copy(default_audio, audio_path)
            else:
                return {"error": "Missing audio input and my_voice.wav not found in container"}

        # Get the timing intervals for lip-syncing
        speaking_intervals, total_dur = get_speaking_intervals(audio_path)

        # Use dynamic paths to prevent the "Missing frame asset" error
        frame_paths = [
            os.path.join(base_dir, "frame1.png"),
            os.path.join(base_dir, "frame2.png"),
            os.path.join(base_dir, "frame3.png"),
            os.path.join(base_dir, "frame4.png")
        ]

        # Verify all assets exist
        for fp in frame_paths:
            if not os.path.exists(fp):
                return {"error": f"Missing frame asset: {fp}"}

        # Load image clips
        clips = [mpy.ImageClip(f).resize(width=1080) for f in frame_paths]
        idle_clip = clips[0]
        talk_clips = clips[1:]

        # Create the video frame-by-frame based on speaking intervals
        def make_frame(t):
            is_speaking = False
            for start, end in speaking_intervals:
                if start <= t <= end:
                    is_speaking = True
                    break
            
            if is_speaking:
                # Swap between the 3 talking frames every 0.15 seconds to create animation
                frame_idx = int((t / 0.15) % len(talk_clips))
                return talk_clips[frame_idx].get_frame(0)
            else:
                # Use the idle frame during silent pauses
                return idle_clip.get_frame(0)

        # Assemble the video track
        animated_clip = mpy.VideoClip(make_frame, duration=total_dur)
        
        # Attach the audio track
        final_audio = mpy.AudioFileClip(audio_path)
        final_video = animated_clip.set_audio(final_audio)

        # Render the final MP4
        final_video.write_videofile(
            final_video_path, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac",
            logger=None,
            verbose=False
        )

        # Encode and return the final video
        with open(final_video_path, "rb") as f:
            encoded_video = base64.b64encode(f.read()).decode("utf-8")
            
        return {"output": {"video_base64": encoded_video}}

    except Exception as e:
        print("Handler Error:", e)
        print(traceback.format_exc())
        return {"error": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    print("Starting RunPod Serverless Worker...", flush=True)
    runpod.serverless.start({"handler": handler})
