import argparse
import os
import subprocess
import sys
from datetime import datetime
import cv2
from .core.physics import KinematicsEngine, RadiometricEngine, MorphologicalEngine, ArtifactEngine

def main():
    # Force eGPU Handshake
    cv2.ocl.setUseOpenCL(True)
    for i in range(5):
        try:
            dev = cv2.ocl.Device_getDevice(i)
            if "AMD" in dev.name() or "Radeon" in dev.name():
                cv2.ocl.Device.setDefault(dev)
                break
        except: break

    parser = argparse.ArgumentParser(description="LUMICRON: Hypersonic UAP Forensic Suite")
    subparsers = parser.add_subparsers(dest="command")

    # INIT Command
    init_p = subparsers.add_parser('init', help='Initialize a new project from a video')
    init_p.add_argument('project', help='Name of the new project')
    init_p.add_argument('--source', required=True, help='Path to raw video file')

    # REPORT Command
    report_p = subparsers.add_parser('report', help='Run full forensic dossier')
    report_p.add_argument('project', help='Project folder')
    report_p.add_argument('--distance', type=float, default=5000.0)
    report_p.add_argument('--sensor', type=float, default=36.0)
    report_p.add_argument('--focal', type=float, default=200.0)
    report_p.add_argument('--fps', type=int, default=60)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    project_path = os.path.abspath(args.project)

    if args.command == "init":
        print(f"--- INITIALIZING PROJECT: {args.project} ---")
        # Create hierarchy
        for folder in ["01_RAW", "02_FRAMES", "03_DATA", "04_REPORTS"]:
            os.makedirs(os.path.join(project_path, folder), exist_ok=True)
        
        # Copy Source
        source_ext = os.path.splitext(args.source)[1]
        dest_video = os.path.join(project_path, "01_RAW", f"source{source_ext}")
        subprocess.run(["cp", args.source, dest_video])
        
        # Extract Frames via FFmpeg
        print("Extracting frames... this may take a moment.")
        extract_cmd = [
            "ffmpeg", "-i", dest_video, 
            "-q:v", "2", 
            os.path.join(project_path, "02_FRAMES", "frame_%03d.png")
        ]
        result = subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        if result.returncode == 0:
            frame_count = len(os.listdir(os.path.join(project_path, "02_FRAMES")))
            print(f"SUCCESS: {frame_count} frames extracted to 02_FRAMES.")
        else:
            print("ERROR: FFmpeg failed. Ensure ffmpeg is installed (brew install ffmpeg).")

    elif args.command == "report":
        active_dev = cv2.ocl.Device.getDefault()
        print(f"\n--- HARDWARE ACCELERATION: {active_dev.name()} ---")
        
        # Run Engines
        # (Using mock shifts for kinematics until we implement the point-tracker)
        mock_shifts = [10.5, 12.2, 45.8, 110.2] 
        k = KinematicsEngine.calculate_telemetry(mock_shifts, args.distance, args.focal, args.sensor, 3840, args.fps)
        r = RadiometricEngine.analyze_luminance(project_path)
        m = MorphologicalEngine.analyze_shape_stability(project_path)
        n = ArtifactEngine.detect_noise_signature(project_path)

        report_content = f"""
========================================
 LUMICRON FORENSIC DOSSIER: {args.project}
 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================

[KINEMATICS]
- Top Speed: {k['top_speed_ms']} m/s
- Max Accel: {k['max_g']} Gs

[RADIOMETRICS]
- Intensity: {r['peak_intensity'] if r else 'N/A'}/255
- Frequency: {r['pulse_freq'] if r else 'N/A'} Hz

[MORPHOLOGY]
- Shape Var: {m:.6f}
- Structure: {"Rigid/Solid" if m < 0.001 else "Fluctuating"}

[ARTIFACT REJECTION]
- Verdict: {n}

========================================
"""
        print(report_content)
        
        report_dir = os.path.join(project_path, "04_REPORTS")
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"dossier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w') as f:
            f.write(report_content)
        print(f"REPORT SAVED TO: {report_file}")

if __name__ == "__main__":
    main()
