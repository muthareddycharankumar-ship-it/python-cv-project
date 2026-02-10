#!/usr/bin/env python3
"""
Counter Event Detection - Records ONLY Active Periods
Detects events at counter and records ONLY while there's activity
Stops recording immediately when things go static (cup sitting still)
"""

import cv2
import numpy as np
import os
from datetime import timedelta
import argparse
from collections import deque

class SmartEventDetector:
    def __init__(self, video_path, output_dir='/home/safepro/Desktop/opencv/event_out',
                 motion_threshold=500, 
                 static_timeout=2,  # Stop recording after 2 seconds of no movement
                 pre_event_sec=5, 
                 post_event_sec=5):
        """
        Detect and record ONLY active periods at counter
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save event clips
            motion_threshold: Threshold for detecting change (lower = more sensitive)
            static_timeout: Seconds of no activity before STOPPING recording (default: 2)
            pre_event_sec: Seconds to include before event starts
            post_event_sec: Seconds to include after last activity
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.motion_threshold = motion_threshold
        self.static_timeout = static_timeout
        self.pre_event_sec = pre_event_sec
        self.post_event_sec = post_event_sec
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Video properties
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.width = None
        self.height = None
        
    def initialize_video(self):
        """Open video and get properties"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        duration = self.total_frames / self.fps
        print(f"Video loaded: {self.width}x{self.height} @ {self.fps} fps")
        print(f"Duration: {timedelta(seconds=int(duration))}")
        print(f"Total frames: {self.total_frames}")
        print(f"Output directory: {self.output_dir}")
        
    def detect_change(self, frame1, frame2):
        """
        Detect change/movement between two frames
        Returns score indicating amount of change
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
        
        # Compute absolute difference
        frame_diff = cv2.absdiff(gray1, gray2)
        
        # Threshold
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Calculate change score
        change_score = np.sum(thresh) / 255
        
        return change_score
    
    def process(self):
        """
        Main processing:
        - Detect when activity starts
        - Record ONLY while activity continues
        - STOP recording when things go static
        - Skip static periods completely
        """
        try:
            self.initialize_video()
            
            print("\n=== Recording ONLY active periods ===")
            print(f"Motion threshold: {self.motion_threshold}")
            print(f"Static timeout: {self.static_timeout}s (stops recording after this)")
            print(f"Pre-event buffer: {self.pre_event_sec}s")
            print(f"Post-event buffer: {self.post_event_sec}s\n")
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            ret, prev_frame = self.cap.read()
            if not ret:
                raise ValueError("Could not read first frame")
            
            # State tracking
            is_recording = False
            event_number = 0
            video_writer = None
            
            # Pre-buffer for capturing frames before event starts
            pre_buffer = deque(maxlen=int(self.fps * self.pre_event_sec))
            
            # Track last time we saw activity
            last_activity_frame = 0
            
            # Post-event buffer tracking
            post_buffer_frames = []
            max_post_buffer_frames = int(self.fps * self.post_event_sec)
            
            frame_count = 0
            recording_start_time = 0
            
            while True:
                ret, curr_frame = self.cap.read()
                if not ret:
                    # End of video - close any open recording
                    if is_recording and video_writer:
                        # Write remaining post-buffer frames
                        for frame in post_buffer_frames:
                            video_writer.write(frame)
                        video_writer.release()
                        duration = (frame_count - recording_start_time) / self.fps
                        end_time = frame_count / self.fps
                        print(f"  ✓ Saved event{event_number}.mp4 "
                              f"(duration: {duration:.1f}s, ended at {timedelta(seconds=int(end_time))})")
                    break
                
                frame_count += 1
                current_time = frame_count / self.fps
                
                # Always add to pre-buffer
                pre_buffer.append(curr_frame.copy())
                
                # Detect change/movement
                change_score = self.detect_change(prev_frame, curr_frame)
                has_activity = change_score > self.motion_threshold
                
                # Calculate time since last activity
                if has_activity:
                    last_activity_frame = frame_count
                
                frames_since_activity = frame_count - last_activity_frame
                time_since_activity = frames_since_activity / self.fps
                
                # START RECORDING - Activity detected and not currently recording
                if has_activity and not is_recording:
                    is_recording = True
                    event_number += 1
                    recording_start_time = frame_count
                    
                    output_path = os.path.join(self.output_dir, f"event{event_number}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                                   (self.width, self.height))
                    
                    # Write pre-buffer frames
                    for buffered_frame in pre_buffer:
                        video_writer.write(buffered_frame)
                    
                    post_buffer_frames = []
                    
                    print(f"Event {event_number} started at {timedelta(seconds=int(current_time))} "
                          f"(change score: {change_score:.0f})")
                
                # CONTINUE RECORDING - Already recording
                elif is_recording:
                    if has_activity:
                        # Activity is still happening
                        # Write current frame and any buffered post frames
                        if post_buffer_frames:
                            for frame in post_buffer_frames:
                                video_writer.write(frame)
                            post_buffer_frames = []
                        video_writer.write(curr_frame)
                    else:
                        # No activity - add to post buffer
                        post_buffer_frames.append(curr_frame.copy())
                        
                        # Check if we should STOP recording
                        if time_since_activity >= self.static_timeout:
                            # Static period exceeded - STOP recording
                            # Only write post_event_sec worth of post-buffer
                            frames_to_write = min(len(post_buffer_frames), max_post_buffer_frames)
                            for i in range(frames_to_write):
                                video_writer.write(post_buffer_frames[i])
                            
                            video_writer.release()
                            duration = (frame_count - recording_start_time) / self.fps
                            print(f"  ✓ Saved event{event_number}.mp4 "
                                  f"(duration: {duration:.1f}s, ended at {timedelta(seconds=int(current_time))}) "
                                  f"- Activity stopped")
                            
                            is_recording = False
                            video_writer = None
                            post_buffer_frames = []
                
                prev_frame = curr_frame
                
                # Progress indicator every 30 seconds
                if frame_count % (self.fps * 30) == 0:
                    progress = (frame_count / self.total_frames) * 100
                    status = "RECORDING" if is_recording else "scanning"
                    print(f"Progress: {progress:.1f}% ({timedelta(seconds=int(current_time))}) "
                          f"- {event_number} events recorded - {status}")
            
            self.cap.release()
            if video_writer:
                video_writer.release()
            
            print(f"\n✓ Complete! {event_number} events recorded in '{self.output_dir}/'")
            print(f"Only active periods were recorded - static periods were skipped")
            
        except Exception as e:
            print(f"Error: {e}")
            if self.cap:
                self.cap.release()
            if video_writer:
                video_writer.release()


def main():
    parser = argparse.ArgumentParser(
        description='Record ONLY active periods at counter - skip static scenes'
    )
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', default='/home/safepro/Desktop/opencv/event_out',
                        help='Output directory for event clips')
    parser.add_argument('--threshold', type=float, default=500,
                        help='Motion detection threshold (default: 500, lower = more sensitive)')
    parser.add_argument('--static-timeout', type=float, default=2,
                        help='Seconds of no activity before STOPPING recording (default: 2)')
    parser.add_argument('--pre-sec', type=int, default=5,
                        help='Seconds before event (default: 5)')
    parser.add_argument('--post-sec', type=int, default=5,
                        help='Seconds after last activity (default: 5)')
    
    args = parser.parse_args()
    
    detector = SmartEventDetector(
        video_path=args.video_path,
        output_dir=args.output_dir,
        motion_threshold=args.threshold,
        static_timeout=args.static_timeout,
        pre_event_sec=args.pre_sec,
        post_event_sec=args.post_sec
    )
    
    detector.process()


if __name__ == '__main__':
    main()
