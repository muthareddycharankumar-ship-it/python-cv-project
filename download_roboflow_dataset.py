from roboflow import Roboflow

# --------- CONFIG ----------
API_KEY = "rWKTplbXnIeiez230IVX"
WORKSPACE = "safepro"          # your roboflow workspace
PROJECT = "cup-detection"      # your project name (URL name)
VERSION = 1                   # dataset version
# ---------------------------

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("yolov8")

print("✅ Dataset downloaded successfully!")
print("📁 Location:", dataset.location)
