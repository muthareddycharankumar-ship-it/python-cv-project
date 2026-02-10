#!/usr/bin/env python3
"""
Simple Video Splitter
Splits a long video into smaller event clips of specified duration
"""

import cv2
import os
from datetime import timedelta
import argparse

class VideoSplitter:
    def __init__(self, video_path, output_dir='events', clip_duration=10):
        """
        Initialize the video splitter
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save clips
            clip_duration: Duration of each clip in seconds (default: 10)
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.clip_duration = clip_duration
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def split_video(self):
        """Split video into clips"""
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        duration = total_frames / fps
        print(f"Video loaded: {width}x{height} @ {fps} fps")
        print(f"Total duration: {timedelta(seconds=int(duration))}")
        print(f"Clip duration: {self.clip_duration} seconds")
        print(f"\n=== Splitting video into clips ===\n")
        
        # Calculate frames per clip
        frames_per_clip = int(fps * self.clip_duration)
        
        event_number = 1
        frame_count = 0
        video_writer = None
        clip_start_time = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Close last clip if open
                if video_writer:
                    video_writer.release()
                    print(f"✓ Saved event{event_number}.mp4 "
                          f"({timedelta(seconds=int(clip_start_time))} - "
                          f"{timedelta(seconds=int(frame_count/fps))})")
                break
            
            # Start new clip if needed
            if frame_count % frames_per_clip == 0:
                # Close previous clip
                if video_writer:
                    video_writer.release()
                    print(f"✓ Saved event{event_number}.mp4 "
                          f"({timedelta(seconds=int(clip_start_time))} - "
                          f"{timedelta(seconds=int(frame_count/fps))})")
                    event_number += 1
                
                # Start new clip
                output_path = os.path.join(self.output_dir, f"event{event_number}.mp4")
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                clip_start_time = frame_count / fps
                print(f"Creating event{event_number}.mp4 starting at "
                      f"{timedelta(seconds=int(clip_start_time))}...")
            
            # Write frame to current clip
            video_writer.write(frame)
            frame_count += 1
            
            # Progress indicator every 5%
            if frame_count % (total_frames // 20) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.0f}%")
        
        cap.release()
        
        total_clips = event_number
        print(f"\n✓ Complete! Split into {total_clips} clips in '{self.output_dir}/' directory")


def main():
    parser = argparse.ArgumentParser(
        description='Split a long video into smaller clips'
    )
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', default='events', 
                        help='Output directory for clips (default: events)')
    parser.add_argument('--duration', type=int, default=10,
                        help='Duration of each clip in seconds (default: 10)')
    
    args = parser.parse_args()
    
    # Create splitter and process
    splitter = VideoSplitter(
        video_path=args.video_path,
        output_dir=args.output_dir,
        clip_duration=args.duration
    )
    
    splitter.split_video()


if __name__ == '__main__':
    main()
