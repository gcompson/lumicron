import argparse
import os
import subprocess
import sys
import json
from datetime import datetime
import cv2
from .core.physics import KinematicsEngine, RadiometricEngine, MorphologicalEngine, ArtifactEngine, VisualTracker, StackEngine

def main():
    # Force eGPU Handshake
    cv2.ocl.setUseOpenCL(True)
    gpu_name = "CPU (Fall-back)"
    for i in range(5):
        try:
            dev = cv2.ocl.Device_getDevice(i)
            if "AMD" in dev.name() or "Radeon" in dev.name():
                cv2.ocl.Device.setDefault(dev)
                gpu_name = dev.name()
                break
            elif "Intel" in dev.name() and gpu_name == "CPU (Fall-back)":
                gpu_name = dev.name()
        except: break

    parser = argparse.ArgumentParser(description="LUMICRON: Hypersonic UAP Forensic Suite")
    subparsers = parser.add_subparsers(dest="command")

    # 1. INIT (High-Speed Support)
    init_p = subparsers.add_parser('init', help='Initialize project and extract all frames')
    init_p.add_argument('project')
    init_p.add_argument('--source', required=True)

    # 2. RADIATE
    subparsers.add_parser('radiate', help='Analyze luminance and frequency').add_argument('project')

    # 3. MORPH
    subparsers.add_parser('morph', help='Analyze shape stability').add_argument('project')

    # 4. NOISE
    subparsers.add_parser('noise', help='Run artifact rejection').add_argument('project')

    # 5. STACK (The Streak Map)
    stack_p = subparsers.add_parser('stack', help='Generate Long Exposure Streak Map')
    stack_p.add_argument('project')
    stack_p.add_argument('--start', type=int, default=None)
    stack_p.add_argument('--end', type=int, default=None)

    # 6. VISUALIZE (Range & Filter Support)
    vis_p = subparsers.add_parser('visualize', help='Manual point-track with range and filters')
    vis_p.add_argument('project')
    vis_p.add_argument('--start', type=int, default=None)
    vis_p.add_argument('--end', type=int, default=None)
    vis_p.add_argument('--mask', action='store_true')
    vis_p.add_argument('--filter', action='store_true')

    # 7. REPORT
    report_p = subparsers.add_parser('report', help='Generate complete dossier')
    report_p.add_argument('project')
    report_p.add_argument('--distance', type=float, default=5000.0)
    report_p.add_argument('--focal', type=float, default=200.0)
    report_p.add_argument('--fps', type=int, default=240) # Defaulted to 240 for your iPhone 16

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    project_path = os.path.abspath(args.project)

    if args.command == "init":
        print(f"--- INITIALIZING: {args.project} (HIGH-SPEED MODE) ---")
        for folder in ["01_RAW", "02_FRAMES", "03_DATA", "04_REPORTS"]:
            os.makedirs(os.path.join(project_path, folder), exist_ok=True)
        
        original_filename = os.path.basename(args.source)
        dest_video = os.path.join(project_path, "01_RAW", original_filename)
        
        print(f"Archiving raw file: {original_filename}")
        subprocess.run(["cp", args.source, dest_video])
        
        print("Extracting ALL frames (240fps hardware scan)...")
        # -vsync 0 is the key for iPhone Variable Frame Rate
        extract_cmd = ["ffmpeg", "-i", dest_video, "-vsync", "0", "-q:v", "2", os.path.join(project_path, "02_FRAMES", "frame_%05d.png")]
        subprocess.run(extract_cmd)

    elif args.command == "stack":
        StackEngine.generate_streak_map(project_path, args.start, args.end)

    elif args.command == "visualize":
        data = VisualTracker.manual_track(project_path, args.mask, args.filter, args.start, args.end)
        if data:
            dist, f_count = data
            with open(os.path.join(project_path, "03_DATA", "tracking.json"), 'w') as f:
                json.dump({"pixel_dist": dist, "frame_count": f_count}, f)
            print(f"\nSUCCESS: Tracking data saved for {f_count} frames.")

    elif args.command == "report":
        print(f"--- HARDWARE: {gpu_name} ---")
        r = RadiometricEngine.analyze_luminance(project_path)
        m = MorphologicalEngine.analyze_shape_stability(project_path)
        n = ArtifactEngine.detect_noise_signature(project_path)
        
        t_path = os.path.join(project_path, "03_DATA", "tracking.json")
        if os.path.exists(t_path):
            with open(t_path, 'r') as f: t_data = json.load(f)
            # Use the specified FPS (default 240) for high-speed calculation
            k = KinematicsEngine.calculate_telemetry([t_data['pixel_dist']], args.distance, args.focal, 36.0, 3840, args.fps)
            mode = "REAL (High-Speed)"
        else:
            k = KinematicsEngine.calculate_telemetry([0], 1, 1, 1, 1, 1)
            mode = "MOCK"
        
        print(f"\n[Dossier for {args.project} ({mode})]\nSpeed: {k['top_speed_ms']} m/s\nPulse: {r['peak_intensity']} units\nVerdict: {n}")

if __name__ == "__main__":
    main()
