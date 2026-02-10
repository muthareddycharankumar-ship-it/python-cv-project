import os
import cv2
import time
import numpy as np
from dataclasses import dataclass
from ultralytics import solutions
from collections import defaultdict


# ================== CONFIG ==================
RTSP_HOST = "137.97.110.166"
RTSP_PORT = 8554
RTSP_PATH = "main"
USERNAME = "admin"
PASSWORD = "admin@123"

RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{RTSP_HOST}:{RTSP_PORT}/{RTSP_PATH}"
OUTPUT_PATH = "rtsp_speed_output.mp4"
VEHICLE_IMAGES_DIR = "vehicle_speed_images"

os.makedirs(VEHICLE_IMAGES_DIR, exist_ok=True)


@dataclass
class SpeedConfig:

    model_name: str = "yolov8s.pt"  # NANO model for speed
    tracker: str = "botsort.yaml"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    max_speed_kmh: int = 200  # Increased max
    min_speed_kmh: float = 3.0
    max_hist: int = 10
    peak_save_threshold: float = 10.0
    show_window: bool = False
    device: str = "cuda"  # Use GPU
    # Speed smoothing parameters
    speed_smoothing_window: int = 5
    speed_outlier_threshold: float = 2.5
    meter_per_pixel: float = 0.141156382 # Initial calibration value
    manual_calibration: bool = True  # Enable manual calibration
    # FPS calculation
    fps_smoothing_window: int = 30  # Calculate FPS over last 30 frames
    manual_calibration: bool = True  # Set to True to adjust meter_per_pixel live


# ============================================


class VehicleTracker:
    """Tracks each vehicle and saves best image at peak speed"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.vehicle_data = defaultdict(lambda: {
            'speeds': [],
            'raw_speeds': [],
            'positions': [],
            'frame_first_seen': None,
            'frame_last_seen': None,
            'peak_speed': 0,
            'peak_frame': None,
            'peak_frame_image': None,  # STORE THE ACTUAL FRAME
            'saved_image': False,
            'age_frames': 0
        })
        
        # FPS tracking
        self.fps_history = []
        self.last_frame_time = None
    
    def smooth_speed(self, speeds):
        """Apply median filter to remove outliers"""
        if len(speeds) < 3:
            return speeds[-1] if speeds else 0
        
        window = speeds[-self.cfg.speed_smoothing_window:]
        median_speed = np.median(window)
        
        filtered = [s for s in window if s < median_speed * self.cfg.speed_outlier_threshold]
        
        if not filtered:
            return median_speed
        
        return np.mean(filtered)
    
    def update_fps(self):
        """Calculate actual processing FPS"""
        current_time = time.time()
        
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            if frame_time > 0:
                instant_fps = 1.0 / frame_time
                self.fps_history.append(instant_fps)
                
                # Keep only last N frames
                if len(self.fps_history) > self.cfg.fps_smoothing_window:
                    self.fps_history.pop(0)
        
        self.last_frame_time = current_time
        
        # Return smoothed FPS
        if self.fps_history:
            return np.mean(self.fps_history)
        return 5.0  # Default fallback
    
    def update_vehicle(self, track_id, speed_kmh, frame_num, current_frame):
        """Update vehicle data with new speed measurement"""
        data = self.vehicle_data[track_id]
        
        # Store raw speed
        data['raw_speeds'].append(speed_kmh)
        
        # Apply smoothing
        smoothed_speed = self.smooth_speed(data['raw_speeds'])
        data['speeds'].append(smoothed_speed)
        data['age_frames'] += 1
        
        if data['frame_first_seen'] is None:
            data['frame_first_seen'] = frame_num
        
        data['frame_last_seen'] = frame_num
        
        # Track peak speed and SAVE THE FRAME
        if smoothed_speed > data['peak_speed'] and smoothed_speed > self.cfg.peak_save_threshold:
            data['peak_speed'] = smoothed_speed
            data['peak_frame'] = frame_num
            data['peak_frame_image'] = current_frame.copy()  # CRITICAL: Save frame at peak
            print(f"🎯 NEW PEAK for ID{track_id}: {smoothed_speed:.1f}km/h (frame saved)")
        
        # Keep only last 30 speeds
        if len(data['speeds']) > 30:
            data['speeds'].pop(0)
            data['raw_speeds'].pop(0)
    
    def should_save_image(self, track_id):
        """Save image when vehicle exits AND has good data"""
        data = self.vehicle_data[track_id]
        
        min_track_time = 30
        age_ok = data['age_frames'] >= min_track_time
        peak_ok = data['peak_speed'] >= self.cfg.peak_save_threshold
        not_saved = not data['saved_image']
        has_image = data['peak_frame_image'] is not None
        
        return peak_ok and age_ok and not_saved and has_image
    
    def save_vehicle_image(self, track_id):
        """Save the BEST image for this vehicle"""
        data = self.vehicle_data[track_id]
        
        if data['peak_frame_image'] is None:
            print(f"⚠️ Cannot save ID{track_id}: No frame stored")
            return False
        
        avg_speed = np.mean(data['speeds']) if data['speeds'] else 0
        peak_speed = data['peak_speed']
        
        # Filename with all info
        timestamp = int(time.time() * 1000)
        filename = f"{VEHICLE_IMAGES_DIR}/car_{track_id}_peak{peak_speed:.1f}_avg{avg_speed:.1f}_{timestamp}.jpg"
        
        # Save the stored peak frame
        success = cv2.imwrite(filename, data['peak_frame_image'])
        
        if success:
            data['saved_image'] = True
            print(f"✅ SAVED: {filename}")
            print(f"   Peak: {peak_speed:.1f}km/h | Avg: {avg_speed:.1f}km/h")
            print(f"   Tracked {data['age_frames']} frames")
            
            # Free memory
            data['peak_frame_image'] = None
            return True
        else:
            print(f"❌ FAILED to save {filename}")
            return False
    
    def cleanup_old_tracks(self, current_frame):
        """Remove vehicles that left frame >60 frames ago"""
        expired_tracks = []
        for track_id, data in self.vehicle_data.items():
            frames_since_last = current_frame - data['frame_last_seen']
            if frames_since_last > 60:
                if self.should_save_image(track_id):
                    self.save_vehicle_image(track_id)
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.vehicle_data[track_id]
            if track_id not in [t for t, d in self.vehicle_data.items() if d['saved_image']]:
                print(f"🧹 Cleaned track {track_id} (no save - didn't meet criteria)")


def open_rtsp_stream(url: str, retry_interval: int = 5):
    while True:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print(f"[INFO] Connected to RTSP: {url}")
            return cap
        print(f"[WARN] Failed to open RTSP. Retrying in {retry_interval}s...")
        time.sleep(retry_interval)


def create_writer(cap, output_path: str):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 60
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer, width, height, fps


def main():
    cfg = SpeedConfig()
    
    print("="*70)
    print("🔧 CONFIGURATION:")
    print(f"   Device: {cfg.device.upper()}")
    print(f"   Model: {cfg.model_name}")
    print(f"   meter_per_pixel = {cfg.meter_per_pixel} m/px")
    print(f"   FPS: Will be calculated dynamically")
    print(f"   Expected speed range: 60-120 km/h")
    print(f"   Min speed to save: {cfg.peak_save_threshold} km/h")
    print("\n⚙️ CALIBRATION HELPER:")
    print(f"   If speeds are wrong, adjust meter_per_pixel:")
    print(f"   New value = {cfg.meter_per_pixel} × (actual_speed / detected_speed)")
    print(f"   Example: If 100km/h shows as 50km/h:")
    print(f"            New value = {cfg.meter_per_pixel} × (100/50) = {cfg.meter_per_pixel * 2:.3f}")
    print("="*70)
    
    # Check GPU availability
    if cfg.device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ CUDA not available, falling back to CPU")
                cfg.device = "cpu"
        except ImportError:
            print("⚠️ PyTorch not found, using CPU")
            cfg.device = "cpu"
    
    # 1) Open RTSP
    cap = open_rtsp_stream(RTSP_URL)
    writer, W, H, fps_stream = create_writer(cap, OUTPUT_PATH)
    print(f"[INFO] Stream: {W}x{H} @ {fps_stream:.1f} FPS (camera)")
    
    # 2) Initialize speed estimator with dynamic FPS (will be updated)
    speed_estimator = solutions.SpeedEstimator(
        show=False,
        model=cfg.model_name,
        fps=10,  # Start with low FPS, will update dynamically
        max_hist=cfg.max_hist,
        meter_per_pixel=cfg.meter_per_pixel,
        max_speed=cfg.max_speed_kmh,
        conf=cfg.conf_threshold,
        iou=cfg.iou_threshold,
        tracker=cfg.tracker,
        device=cfg.device,
    )
    
    # 3) Initialize vehicle tracker
    vehicle_tracker = VehicleTracker(cfg)
    
    frame_count = 0
    t0 = time.time()
    current_fps = 10.0  # Initial estimate
    
    # Calibration tracking
    speed_samples = []  # Store last 10 speed readings for calibration help
    
    print("\n🚗 LIVE VEHICLE TRACKING STARTED")
    print("📸 Saves 1 image/car at PEAK SPEED")
    print("📹 Output: rtsp_speed_output.mp4")
    print("\n💡 TIP: Note down actual vs detected speeds for calibration")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            ok, frame = cap.read()
            
            if not ok or frame is None:
                print("[WARN] RTSP reconnecting...")
                cap.release()
                time.sleep(2)
                cap = open_rtsp_stream(RTSP_URL)
                writer, W, H, fps_stream = create_writer(cap, OUTPUT_PATH)
                continue
            
            # Calculate actual FPS
            current_fps = vehicle_tracker.update_fps()
            
            # Update speed estimator FPS dynamically
            if frame_count > 30:  # After warmup
                speed_estimator.fps = current_fps
            
            # Process frame
            start = time.time()
            result = speed_estimator(frame)
            proc_time = time.time() - start
            
            annotated_frame = result.plot_im.copy()
            
            # Extract speeds and update tracker
            if hasattr(speed_estimator, 'speeds') and speed_estimator.speeds:
                for track_id, speed_kmh in speed_estimator.speeds.items():
                    if speed_kmh >= cfg.min_speed_kmh:
                        # PASS THE FRAME to update_vehicle
                        vehicle_tracker.update_vehicle(track_id, speed_kmh, frame_count, frame)
                        
                        # Store speed samples for calibration
                        speed_samples.append(speed_kmh)
                        if len(speed_samples) > 100:
                            speed_samples.pop(0)
                        
                        # Debug: Print speed updates
                        data = vehicle_tracker.vehicle_data[track_id]
                        raw = data['raw_speeds'][-1] if data['raw_speeds'] else 0
                        smooth = data['speeds'][-1] if data['speeds'] else 0
                        
                        if frame_count % 30 == 0:  # Every second or so
                            print(f"[SPEED] ID{track_id}: Raw={raw:.1f} → Smooth={smooth:.1f} km/h | Peak={data['peak_speed']:.1f} | FPS={current_fps:.1f}")
            
            # Check for savable vehicles
            vehicle_tracker.cleanup_old_tracks(frame_count)
            
            # Add stats overlay
            active_tracks = len(vehicle_tracker.vehicle_data)
            saved_count = sum(1 for v in vehicle_tracker.vehicle_data.values() if v['saved_image'])
            
            # Calculate average detected speed for calibration help
            avg_detected = np.mean(speed_samples) if speed_samples else 0
            
            stats_text = [
                f"Processing FPS: {current_fps:.1f}",
                f"Inference: {1/proc_time:.1f} fps",
                f"Active: {active_tracks} cars",
                f"Saved: {saved_count}",
                f"Calib: {cfg.meter_per_pixel:.3f}m/px",
                f"Avg Speed: {avg_detected:.1f} km/h"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(annotated_frame, text, (10, 30 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write to output
            writer.write(annotated_frame)
            
            frame_count += 1
            
            # Live stats every 2 seconds
            if frame_count % max(int(current_fps * 2), 30) == 0:
                elapsed = time.time() - t0
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                
                print(f"\n{'='*70}")
                print(f"📊 Frame {frame_count:4d} | Avg FPS: {avg_fps:4.1f} | Current FPS: {current_fps:.1f}")
                print(f"   Speed Estimator using: {speed_estimator.fps:.1f} FPS | Calib: {cfg.meter_per_pixel:.3f} m/px")
                print(f"   Active: {active_tracks} | Saved: {saved_count} | Avg Detected: {avg_detected:.1f} km/h")
                
                # Calibration helper
                if avg_detected > 0:
                    print(f"\n   🔧 CALIBRATION: If average is WRONG, calculate:")
                    for test_actual in [60, 80, 100, 120]:
                        new_calib = cfg.meter_per_pixel * (test_actual / avg_detected)
                        print(f"      If actual = {test_actual}km/h → use meter_per_pixel = {new_calib:.3f}")
                
                # Show top speeds
                print(f"\n   📊 TRACKED VEHICLES:")
                top_vehicles = sorted(vehicle_tracker.vehicle_data.items(), 
                                    key=lambda x: x[1]['peak_speed'], reverse=True)[:5]
                for tid, data in top_vehicles:
                    avg = np.mean(data['speeds']) if data['speeds'] else 0
                    has_img = "✓" if data['peak_frame_image'] is not None else "✗"
                    saved = "💾" if data['saved_image'] else ""
                    print(f"      🚗 ID{tid}: Peak={data['peak_speed']:.1f}km/h | Avg={avg:.1f}km/h | Frames={data['age_frames']} | Img={has_img} {saved}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    
    finally:
        # Save any remaining vehicles
        print("\n🧹 Final cleanup - saving remaining vehicles...")
        saved_final = 0
        for track_id in list(vehicle_tracker.vehicle_data.keys()):
            if vehicle_tracker.should_save_image(track_id):
                if vehicle_tracker.save_vehicle_image(track_id):
                    saved_final += 1
        
        print(f"💾 Saved {saved_final} vehicles in final cleanup")
        
        cap.release()
        writer.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        total_saved = sum(1 for v in vehicle_tracker.vehicle_data.values() if v['saved_image'])
        
        print(f"\n✅ FINISHED!")
        print(f"📹 Video: {OUTPUT_PATH}")
        print(f"📸 Images: {VEHICLE_IMAGES_DIR}/ ({total_saved} vehicles saved)")


if __name__ == "__main__":
    main()