import argparse
import os
import subprocess
import sys
import json
import re
import shutil
import cv2
from tqdm import tqdm

# LOCKED-IN: Forensic Engine Imports
from .core.physics import (
    KinematicsEngine, 
    RadiometricEngine, 
    MorphologicalEngine, 
    ArtifactEngine, 
    VisualTracker, 
    StackEngine
)
from .core.stabilize import stabilize_project

def get_video_metadata(file_path):
    """Audits source video to ensure temporal resolution matches project specs."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", file_path]
    try:
        result = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(result)
        duration = float(data.get('format', {}).get('duration', 0))
        fps_str = data['streams'][0].get('avg_frame_rate', "0/0")
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den > 0 else 240.0
        else:
            fps = float(fps_str) if fps_str != "0" else 240.0
        return {"duration": duration, "fps": fps}
    except Exception:
        return {"duration": 0, "fps": 240.0}

def generate_review_video(source_path, project_path, start=None, duration=None, debug=False):
    """
    MISSION CRITICAL: Burns frame numbers into a review copy.
    Optimized for iPhone 16 (1080p @ 240fps capture).
    """
    output_path = os.path.join(project_path, "03_DATA", "review_telemetry.mp4")
    
    # Forensic Standards: Yellow text, Courier New (Monospace), Black Box
    # fontsize=35 is optimized for 1080p legibility.
    drawtext_filter = (
        "drawtext=fontfile=/System/Library/Fonts/Supplemental/Courier\\ New.tty: "
        "text='FRAME\\: %{frame_num}': start_number=0: x=(w-tw)/2: y=h-(2*lh): "
        "fontcolor=yellow: fontsize=35: box=1: boxcolor=black@0.4"
    )

    # REPAIR: Ensure the review video matches the extracted frame window
    cmd = ['ffmpeg', '-y']
    if start: cmd.extend(['-ss', start])
    if duration: cmd.extend(['-t', duration])
    
    cmd.extend([
        '-i', source_path,
        '-vf', drawtext_filter,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
        output_path
    ])

    if debug: print(f"Executing Burn-In: {' '.join(cmd)}")
    print(f"Progress: Generating Review Telemetry (1080p/240fps)...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"✅ Review video locked: {output_path}")

def main():
    # HARDWARE HANDSHAKE: Optimized for Mac Mini M4 Pro
    cv2.ocl.setUseOpenCL(True)
    
    parser = argparse.ArgumentParser(description="LUMICRON: UAP Forensic Analysis Engine")
    parser.add_argument("--debug", action="store_true", help="Verbose hardware and logic logs")
    subparsers = parser.add_subparsers(dest="command")

    # --- INIT: Folder Prep & Frame Extraction ---
    init_p = subparsers.add_parser("init")
    init_p.add_argument("project")
    init_p.add_argument("--source", required=True)
    init_p.add_argument("--fps", type=float)
    init_p.add_argument("--start")
    init_p.add_argument("--duration")

    # --- RADIATE: Energy Signature & Luma Analysis ---
    rad_p = subparsers.add_parser("radiate")
    rad_p.add_argument("project")

    # --- STACK: Persistence / Streak Mapping ---
    stack_p = subparsers.add_parser("stack")
    stack_p.add_argument("project")
    stack_p.add_argument("--mode", choices=['max', 'min', 'diff'], default='max')

    # --- STABILIZE: ECC Registration ---
    subparsers.add_parser("stabilize").add_argument("project")

    # --- MORPH: Shape Stability Index (SSI) ---
    subparsers.add_parser("morph").add_argument("project")

    # --- NOISE: Sensor Artifact Audit ---
    subparsers.add_parser("noise").add_argument("project")

    # --- VISUALIZE: Manual Point-Tracking ---
    vis_p = subparsers.add_parser("visualize")
    vis_p.add_argument("project")
    vis_p.add_argument("--anchor", action="store_true")
    vis_p.add_argument("--mask", action="store_true")
    vis_p.add_argument("--filter", action="store_true")

    # --- REPORT: Forensic Dossier Generation ---
    rep_p = subparsers.add_parser("report")
    rep_p.add_argument("project")
    rep_p.add_argument("--distance", type=float, required=True)
    rep_p.add_argument("--focal", type=float, default=24.0)
    rep_p.add_argument("--fps", type=float, default=240.0) # Updated for iPhone 16

    args = parser.parse_args()
    if not args.command: return

    # DATA PERSISTENCE: Standardize Project Root
    home = os.path.expanduser("~")
    project_path = os.path.join(home, "Projects/UAP_Data", args.project)

    # --- EXECUTION LOGIC ---
    if args.command == "init":
        frame_dir = os.path.join(project_path, "02_FRAMES")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(os.path.join(project_path, "03_DATA"), exist_ok=True)
        
        # 1. RAW EXTRACTION: vsync 0 to prevent frame dropping
        cmd = ['ffmpeg', '-i', args.source, '-vsync', '0', '-q:v', '2']
        if args.start: cmd.extend(['-ss', args.start])
        if args.duration: cmd.extend(['-t', args.duration])
        cmd.append(os.path.join(frame_dir, "%05d.png")) # 5-digit padding
        
        if args.debug: print(f"Executing Extraction: {' '.join(cmd)}")
        subprocess.run(cmd)

        # 2. TELEMETRY BURN-IN: Generate Review Video (Corrected Windowing)
        generate_review_video(
            args.source, 
            project_path, 
            start=args.start, 
            duration=args.duration, 
            debug=args.debug
        )

    elif args.command == "radiate":
        RadiometricEngine(project_path).analyze()

    elif args.command == "stack":
        StackEngine(project_path).generate(mode=args.mode)

    elif args.command == "stabilize":
        stabilize_project(project_path)

    elif args.command == "morph":
        MorphologicalEngine(project_path).analyze()

    elif args.command == "noise":
        ArtifactEngine(project_path).audit()

    elif args.command == "visualize":
        tracker = VisualTracker(project_path)
        pixel_shifts, f_delta = tracker.manual_track(
            use_anchor=args.anchor, 
            use_mask=args.mask, 
            use_filter=args.filter
        )
        
        # LOCK-IN: Ensure tracking data is saved to disk
        if pixel_shifts:
            tracking_file = os.path.join(project_path, "03_DATA", "tracking.json")
            with open(tracking_file, 'w') as f:
                json.dump({"pixel_shifts": pixel_shifts, "frame_delta": f_delta}, f, indent=4)
            print(f"✅ Flight path saved to {tracking_file}")

    elif args.command == "report":
        # Check for necessary data components
        t_path = os.path.join(project_path, "03_DATA", "tracking.json")
        if not os.path.exists(t_path):
            print(f"❌ ERROR: Missing tracking data. Run 'visualize' for {args.project} first.")
            return

        engine = KinematicsEngine(project_path)
        dossier = engine.generate_markdown_dossier(args.distance, args.focal, args.fps)
        print("\n" + dossier + "\n")

if __name__ == "__main__":
    main()
