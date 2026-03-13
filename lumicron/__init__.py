import argparse
import os
import subprocess
import sys
import json
from datetime import datetime
import cv2
from .core.physics import KinematicsEngine, RadiometricEngine, MorphologicalEngine, ArtifactEngine, VisualTracker

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

    # 1. INIT
    init_p = subparsers.add_parser('init', help='Initialize project and extract frames')
    init_p.add_argument('project')
    init_p.add_argument('--source', required=True)

    # 2. RADIATE
    subparsers.add_parser('radiate', help='Analyze luminance and frequency').add_argument('project')

    # 3. MORPH
    subparsers.add_parser('morph', help='Analyze shape stability').add_argument('project')

    # 4. NOISE
    subparsers.add_parser('noise', help='Run artifact rejection').add_argument('project')

    # 5. VISUALIZE (The New Tracker)
    subparsers.add_parser('visualize', help='Manual point-track for real speed').add_argument('project')

    # 6. REPORT
    report_p = subparsers.add_parser('report', help='Generate complete dossier')
    report_p.add_argument('project')
    report_p.add_argument('--distance', type=float, default=5000.0)
    report_p.add_argument('--focal', type=float, default=200.0)
    report_p.add_argument('--fps', type=int, default=60)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    project_path = os.path.abspath(args.project)

    if args.command == "init":
        print(f"--- INITIALIZING: {args.project} ---")
        for folder in ["01_RAW", "02_FRAMES", "03_DATA", "04_REPORTS"]:
            os.makedirs(os.path.join(project_path, folder), exist_ok=True)
        source_ext = os.path.splitext(args.source)[1]
        dest_video = os.path.join(project_path, "01_RAW", f"source{source_ext}")
        subprocess.run(["cp", args.source, dest_video])
        print("Extracting frames via FFmpeg...")
        subprocess.run(["ffmpeg", "-i", dest_video, "-q:v", "2", os.path.join(project_path, "02_FRAMES", "frame_%03d.png")])

    elif args.command == "visualize":
        data = VisualTracker.manual_track(project_path)
        if data:
            dist, frames = data
            tracking_file = os.path.join(project_path, "03_DATA", "tracking.json")
            with open(tracking_file, 'w') as f:
                json.dump({"pixel_dist": dist, "frame_count": frames}, f)
            print(f"\nSUCCESS: {dist:.2f}px distance saved to 03_DATA.")

    elif args.command == "radiate":
        res = RadiometricEngine.analyze_luminance(project_path)
        print(f"PEAK INTENSITY: {res['peak_intensity']} | PULSE: {res['pulse_freq']} Hz")

    elif args.command == "morph":
        res = MorphologicalEngine.analyze_shape_stability(project_path)
        print(f"SHAPE VARIANCE: {res:.6f} ({'Rigid' if res < 0.001 else 'Dynamic'})")

    elif args.command == "noise":
        res = ArtifactEngine.detect_noise_signature(project_path)
        print(f"AUDIT VERDICT: {res}")

    elif args.command == "report":
        print(f"--- HARDWARE: {gpu_name} ---")
        r = RadiometricEngine.analyze_luminance(project_path)
        m = MorphologicalEngine.analyze_shape_stability(project_path)
        n = ArtifactEngine.detect_noise_signature(project_path)
        
        # Look for real tracking data
        tracking_file = os.path.join(project_path, "03_DATA", "tracking.json")
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r') as f:
                t_data = json.load(f)
            # Use real delta-x/y for kinematics
            k = KinematicsEngine.calculate_telemetry([t_data['pixel_dist']], args.distance, args.focal, 36.0, 3840, args.fps)
            mode = "REAL"
        else:
            k = KinematicsEngine.calculate_telemetry([10, 20, 50], args.distance, args.focal, 36.0, 3840, args.fps)
            mode = "MOCK"
        
        print(f"\n[Dossier for {args.project} ({mode} DATA)]\nSpeed: {k['top_speed_ms']}m/s\nPulse: {r['pulse_freq']}Hz\nVerdict: {n}")

if __name__ == "__main__":
    main()
