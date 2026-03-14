import argparse
import os
import subprocess
import sys
import json
import re
import yaml
import cv2
from datetime import datetime
from tqdm import tqdm
from .core.physics import (
    KinematicsEngine, RadiometricEngine, MorphologicalEngine, 
    ArtifactEngine, VisualTracker, StackEngine
)
# New forensic image processing module
from .core.image_proc import ImageProcessor 

def get_video_metadata(file_path):
    """Sniffs video metadata using ffprobe for auto-population."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", file_path]
    try:
        result = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(result)
        tags = data.get('format', {}).get('tags', {})
        return {
            "location": tags.get('com.apple.quicktime.location.ISO6709', "Unknown"),
            "make": tags.get('com.apple.quicktime.make', "Unknown"),
            "model": tags.get('com.apple.quicktime.model', "Unknown")
        }
    except Exception:
        return {}

def main():
    # Robust GPU Handshake for M4 Pro and Intel/eGPU setups
    cv2.ocl.setUseOpenCL(True)
    gpu_name = "CPU (Fall-back)"
    for i in range(5):
        try:
            dev = cv2.ocl.Device_getDevice(i)
            if any(x in dev.name() for x in ["AMD", "Radeon", "Apple", "M4"]):
                cv2.ocl.Device.setDefault(dev)
                gpu_name = dev.name()
                break
        except Exception:
            break

    parser = argparse.ArgumentParser(description="LUMICRON: Forensic UAP Suite")
    subparsers = parser.add_subparsers(dest="command")

    # 1. INIT
    init_p = subparsers.add_parser('init', help='Initialize project and extract frames')
    init_p.add_argument('project')
    init_p.add_argument('--source', required=True)
    init_p.add_argument('--auto', action='store_true', help='Auto-detect metadata')
    init_p.add_argument('--start_time', help='Start (HH:MM:SS)')
    init_p.add_argument('--duration', help='Duration in seconds')

    # 2. ANALYSIS
    subparsers.add_parser('radiate', help='Luminance/FFT analysis').add_argument('project')
    subparsers.add_parser('morph', help='Shape/Rigidity analysis').add_argument('project')
    subparsers.add_parser('noise', help='Sensor artifact audit').add_argument('project')
    
    # 3. VISUALIZATION & STACKING
    stack_p = subparsers.add_parser('stack', help='Generate Streak Map')
    stack_p.add_argument('project')
    stack_p.add_argument('--start', type=int, default=None)
    stack_p.add_argument('--end', type=int, default=None)
    # New "Switches" for visual enhancement
    stack_p.add_argument('--enhance', action='store_true', help='Apply CLAHE contrast stretching')
    stack_p.add_argument('--false_color', action='store_true', help='Apply heatmap to visualize intensity')
    stack_p.add_argument('--isolate', action='store_true', help='Black out sky to isolate the streak')
    stack_p.add_argument('--threshold', type=int, default=210, help='Brightness threshold (0-255)')

    vis_p = subparsers.add_parser('visualize', help='Manual point-tracking')
    vis_p.add_argument('project')
    vis_p.add_argument('--start', type=int, default=None)
    vis_p.add_argument('--end', type=int, default=None)

    # 4. REPORT
    report_p = subparsers.add_parser('report', help='Generate forensic dossier')
    report_p.add_argument('project')
    report_p.add_argument('--fps', type=int, default=240)
    report_p.add_argument('--distance', type=float, default=5000.0)
    report_p.add_argument('--focal', type=float, default=24.0)

    args = parser.parse_args()
    if not args.command:
        return

    project_path = os.path.abspath(args.project)

    if args.command == "init":
        # ... (Existing init logic preserved)
        for f in ["01_RAW", "02_FRAMES", "03_DATA", "04_REPORTS"]:
            os.makedirs(os.path.join(project_path, f), exist_ok=True)
        
        dest = os.path.join(project_path, "01_RAW", os.path.basename(args.source))
        if not os.path.exists(dest):
            subprocess.run(["cp", args.source, dest])
        
        manifest = {"project_id": os.path.basename(project_path), "sensor": {"fps": 240, "type": "iPhone 16 Pro"}}
        if args.auto:
            meta = get_video_metadata(args.source)
            manifest["location"] = meta.get("location", "Hamilton, VIC")
            manifest["sensor"]["type"] = f"{meta.get('make', 'Apple')} {meta.get('model', 'iPhone')}"
        
        with open(os.path.join(project_path, "lumicron.yaml"), 'w') as f:
            yaml.dump(manifest, f)

        extract_cmd = ["ffmpeg", "-y", "-i", dest]
        if args.start_time: extract_cmd.extend(["-ss", args.start_time])
        if args.duration: extract_cmd.extend(["-t", args.duration])
        extract_cmd.extend(["-vsync", "0", "-q:v", "2", "-progress", "pipe:1", os.path.join(project_path, "02_FRAMES", "frame_%05d.png")])
        
        print(f"Extracting frames on: {gpu_name}")
        subprocess.run(extract_cmd)

    elif args.command == "radiate":
        # 1. Generate the raw luminance data (The CSV)
        RadiometricEngine.analyze_luminance(project_path)
        
        # 2. Run the FFT to find the "Beat" (The Evidence)
        # fps is usually 240 for these iPhone captures
        beat_hz = RadiometricEngine.analyze_beat_frequency(project_path, fps=240)
        
        # 3. Generate the visual plots (The Reports)
        RadiometricEngine.generate_plots(project_path)
        
        print(f"\nFORENSIC ANALYSIS COMPLETE")
        print(f"Propulsion Signature: {beat_hz} Hz")

    elif args.command == "morph":
        MorphologicalEngine.analyze_shape_stability(project_path)

    elif args.command == "noise":
        ArtifactEngine.detect_noise_signature(project_path)

    elif args.command == "stack":
        streak_map_path = StackEngine.generate_streak_map(project_path, args.start, args.end)
        
        if any([args.enhance, args.false_color, args.isolate]):
            img = cv2.imread(streak_map_path)
            if args.enhance:
                img = ImageProcessor.stretch_contrast(img)
            if args.isolate:
                # Using the surgical switch we defined in image_proc.py
                img = ImageProcessor.apply_binary_isolation(img, args.threshold)
            if args.false_color:
                img = ImageProcessor.apply_false_color(img)
            
            output_path = streak_map_path.replace(".png", "_ANALYTIC.png")
            cv2.imwrite(output_path, img)
            print(f"Processed streak map saved: {output_path}")

    elif args.command == "visualize":
        data = VisualTracker.manual_track(project_path, False, False, args.start, args.end)
        if data:
            shifts, f_delta = data
            t_path = os.path.join(project_path, "03_DATA", "tracking.json")
            # We'll need to manually add bg_shifts to this JSON if using an anchor for now
            with open(t_path, 'w') as f:
                json.dump({"pixel_shifts": shifts, "frame_delta": f_delta}, f)
            print(f"SUCCESS: Tracking saved. Segments: {len(shifts)}")

    elif args.command == "report":
        t_path = os.path.join(project_path, "03_DATA", "tracking.json")
        if not os.path.exists(t_path):
            print("Error: No tracking data found.")
            return

        with open(t_path, 'r') as f:
            t_data = json.load(f)
        
        # Pull bg_shifts if they exist in the JSON, else None
        bg_s = t_data.get('bg_shifts', None)
        
        kin = KinematicsEngine.calculate_telemetry(
            t_data['pixel_shifts'], 
            args.distance, 
            args.focal, 
            36.0, # Sensor Width
            3840, # Pixel Width
            args.fps,
            bg_shifts=bg_s # Pass the anchor data to the engine
        )
        
        print("\n" + "="*50)
        print(f"--- KINEMATIC REPORT: {os.path.basename(project_path)} ---")
        print("="*50)
        print(f"Status:      {kin['classification']}")
        print(f"Top Speed:   {kin['top_speed_ms']} m/s (Approx. Mach {round(kin['top_speed_ms']/343, 1)})")
        print(f"Max G-Force: {kin['max_g']} Gs")
        if bg_s: print(f"Parallax:    CORRECTED (Active Anchor)")
        print("="*50)

if __name__ == "__main__":
    main()
