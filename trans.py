import os
import subprocess
import whisper
import numpy as np
import soundfile as sf
import webrtcvad

RTSP_URL = "rtsp://admin:admin%40123@192.168.0.127:554/Streaming/Channels/101"

RAW_FILE = "/home/safepro/Desktop/opencv/raw.wav"
FINAL_FILE = "/home/safepro/Desktop/opencv/final.wav"

MODEL_NAME = "small"

# -----------------------------
# RECORD LONG AUDIO BUFFER (20 sec)
# -----------------------------
def record_audio():
    if os.path.exists(RAW_FILE):
        os.remove(RAW_FILE)

    print("\n🎤 Recording 20 seconds from CCTV camera...")

    cmd = [
        "ffmpeg",
        "-y",
        "-rtsp_transport", "tcp",
        "-rw_timeout", "5000000",
        "-i", RTSP_URL,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-t", "20",
        RAW_FILE
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.isfile(RAW_FILE):
        print("❌ ERROR: No raw audio recorded")
        return False

    size = os.path.getsize(RAW_FILE)
    print(f"📁 Raw audio recorded: {size} bytes")

    return size > 8000   # ensure some audio present


# -----------------------------
# VOICE ACTIVITY DETECTION (VAD)
# -----------------------------
def detect_voice():
    print("\n🔍 Detecting voice region...")

    audio, sr = sf.read(RAW_FILE)
    audio_float = audio.astype(np.float32)

    # convert to int16 PCM
    audio_pcm = (audio_float * 32767).astype(np.int16).tobytes()

    vad = webrtcvad.Vad()
    vad.set_mode(3)  # most aggressive for speech detection

    frame_ms = 30
    frame_size = int(sr * frame_ms / 1000)  # samples
    frame_bytes = frame_size * 2            # int16 = 2 bytes

    speech_frames = []

    for i in range(0, len(audio_pcm), frame_bytes):
        frame = audio_pcm[i:i + frame_bytes]
        if len(frame) < frame_bytes:
            break
        if vad.is_speech(frame, sr):
            speech_frames.append(i / 2)  # position in samples

    if len(speech_frames) == 0:
        print("❌ No speech detected!")
        return False

    start_sample = int(max(0, speech_frames[0] - sr * 0.5))
    end_sample = int(min(len(audio_float), speech_frames[-1] + sr * 0.5))

    print(f"🎙 Speech detected from sample {start_sample} to {end_sample}")

    final_audio = audio_float[start_sample:end_sample]
    sf.write(FINAL_FILE, final_audio, sr)

    size = os.path.getsize(FINAL_FILE)
    print(f"📁 Final speech audio saved: {size} bytes")

    return size > 1000


# -----------------------------
# TRANSCRIBE & TRANSLATE
# -----------------------------
def whisper_process():
    print("\n🧠 Loading Whisper model...")
    model = whisper.load_model(MODEL_NAME, device="cpu")

    print("\n🎧 Kannada Transcription...")
    result_kn = model.transcribe(FINAL_FILE, task="transcribe", language="kn")
    print("KANNADA TEXT:", result_kn["text"])

    print("\n🌍 English Translation...")
    result_en = model.transcribe(FINAL_FILE, task="translate", language="kn")
    print("ENGLISH:", result_en["text"])


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    if record_audio():
        if detect_voice():
            whisper_process()
