import os
import subprocess
import whisper
import soundfile as sf
import numpy as np

RTSP_URL = "rtsp://admin:admin@123@192.168.0.127:554/streaming/channels/101/"
RAW_FILE = "/home/safepro/Desktop/opencv/raw.wav"
FINAL_FILE = "/home/safepro/Desktop/opencv/final.wav"

MODEL_NAME = "small"

# -----------------------------
# Step 1: RECORD LONG BUFFER
# -----------------------------
def record_long_audio():
    if os.path.exists(RAW_FILE):
        os.remove(RAW_FILE)

    print("\n🎤 Recording 12 seconds from CCTV camera (for stability)...")

    cmd = [
        "ffmpeg",
        "-y",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-t", "12",      # <-- RECORD MORE, ALWAYS WORKS
        RAW_FILE
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.isfile(RAW_FILE):
        print("❌ ERROR: No raw audio recorded.")
        return False

    print("📁 Raw audio recorded (12 sec buffer).")
    return True


# -----------------------------
# Step 2: TRIM LAST 6 SECONDS
# -----------------------------
def trim_voice_part():
    print("\n✂️  Extracting last 6 seconds where your real voice is...")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", RAW_FILE,
        "-ss", "6",        # start at second 6
        "-t", "6",         # take last 6 seconds
        FINAL_FILE
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.isfile(FINAL_FILE):
        print("❌ ERROR trimming audio.")
        return False

    size = os.path.getsize(FINAL_FILE)
    print(f"📁 Final audio trimmed ({size} bytes).")
    return True


# -----------------------------
# Step 3: TRANSCRIBE + TRANSLATE
# -----------------------------
def whisper_translate():
    print("\n🧠 Loading Whisper model (CPU)...")
    model = whisper.load_model(MODEL_NAME, device="cpu")

    audio, sr = sf.read(FINAL_FILE)
    print(f"🔊 Audio peak: {np.max(np.abs(audio)):.4f}")

    print("\n⌛ Converting speech → English...")
    result = model.transcribe(FINAL_FILE, task="translate", language="kn")

    print("\n========= 📝 ENGLISH TRANSLATION =========")
    print(result['text'])
    print("==========================================\n")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    if record_long_audio() and trim_voice_part():
        whisper_translate()
