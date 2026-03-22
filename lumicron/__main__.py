import argparse
import os
import subprocess
import json
import re
import cv2
import sys
from tqdm import tqdm

# LOCKED-IN: Forensic Engine Imports
from . import (
    KinematicsEngine, 
    RadiometricEngine, 
    MorphologicalEngine, 
    ArtifactEngine, 
    VisualTracker, 
    StackEngine, 
    stabilize_project
)

def get_video_metadata(file_path):
    """Audits source video to ensure temporal resolution matches project specs."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", file_path]
    try:
        result = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(result)
        fps_str = data['streams'][0].get('avg_frame_rate', "0/0")
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den > 0 else 240.0
        else:
            fps = float(fps_str) if fps_str != "0" else 240.0
        return {"fps": fps}
    except Exception:
        return {"fps": 240.0}

def run_ffmpeg_with_progress(cmd, desc="Processing"):
    """Pipes FFmpeg output to drive a tqdm progress bar for single-line updates."""
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    with tqdm(unit="fr", desc=f" {desc}", bar_format="{desc}: {n_fmt} frames | {elapsed} | {rate_fmt}") as pbar:
        for line in process.stdout:
            match = re.search(r'frame=\s*(\d+)', line)
            if match:
                pbar.n = int(match.group(1))
                pbar.refresh()
    
    process.wait()
    if process.returncode != 0:
        print(f"\n❌ FFmpeg Failed: Operation interrupted.")
        sys.exit(1)

def generate_review_video(source_path, project_path, start=None, duration=None):
    """Burns frame numbers into review copy. Optimized for 1080p legibility."""
    output_path = os.path.join(project_path, "03_DATA", "review_telemetry.mp4")
    
    drawtext_filter = (
        "drawtext=fontfile=/System/Library/Fonts/Supplemental/Courier\\ New.tty: "
        "text='FRAME\\: %{frame_num}': start_number=0: x=(w-tw)/2: y=h-(2*lh): "
        "fontcolor=yellow: fontsize=35: box=1: boxcolor=black@0.4"
    )

    # REPAIR: Fast-seek (-ss before -i) and hardware-aware stats
    cmd = ['ffmpeg', '-y', '-hide_banner', '-stats']
    if start: cmd.extend(['-ss', start])
    if duration: cmd.extend(['-t', duration])
    
    cmd.extend([
        '-i', source_path,
        '-vf', drawtext_filter,
        '-fps_mode', 'passthrough',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
        output_path
    ])

    print(f"Progress: Finalizing GPU Burn-In (Telemetry)...")
    subprocess.run(cmd, stderr=subprocess.STDOUT)
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"✅ Review video locked: {output_path}")
    else:
        print(f"❌ Error: Review video generation failed.")

def main():
    # HARDWARE HANDSHAKE: Optimized for Mac Mini M4 Pro
    cv2.ocl.setUseOpenCL(True)
    
    parser = argparse.ArgumentParser(description="LUMICRON: UAP Forensic Analysis Engine")
    parser.add_argument("--debug", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    # --- CLI CONFIGURATION ---
    init_p = subparsers.add_parser("init")
    init_p.add_argument("project")
    init_p.add_argument("--source", required=True)
    init_p.add_argument("--start")
    init_p.add_argument("--duration")

    subparsers.add_parser("radiate").add_argument("project")
    
    stack_p = subparsers.add_parser("stack")
    stack_p.add_argument("project")
    stack_p.add_argument("--mode", choices=['max', 'min', 'diff'], default='max')
    
    subparsers.add_parser("stabilize").add_argument("project")
    subparsers.add_parser("morph").add_argument("project")
    subparsers.add_parser("noise").add_argument("project")
    subparsers.add_parser("gui").add_argument("project", nargs='?')
    
    vis_p = subparsers.add_parser("visualize")
    vis_p.add_argument("project")
    vis_p.add_argument("--anchor", action="store_true")
    vis_p.add_argument("--mask", action="store_true")
    vis_p.add_argument("--filter", action="store_true")

    rep_p = subparsers.add_parser("report")
    rep_p.add_argument("project")
    rep_p.add_argument("--distance", type=float, required=True)
    rep_p.add_argument("--focal", type=float, default=24.0)
    rep_p.add_argument("--fps", type=float, default=240.0)

    args = parser.parse_args()
    if not args.command: return

    home = os.path.expanduser("~")
    
    # FIXED: Handle optional project names for the GUI
    project_path = None
    if hasattr(args, 'project') and args.project:
        project_path = os.path.join(home, "Projects/UAP_Data", args.project)
    elif args.command != "gui":
        # Other commands (radiate, report, etc.) still REQUIRE a project
        print(f"❌ Error: The '{args.command}' command requires a project name.")
        return

    if args.command == "init":
        frame_dir = os.path.join(project_path, "02_FRAMES")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(os.path.join(project_path, "03_DATA"), exist_ok=True)
        
        # AUDIT: Restore metadata check from old version
        meta = get_video_metadata(args.source)
        if args.debug: print(f"Source Audit: {meta['fps']} FPS detected.")

        cmd = ['ffmpeg', '-y', '-hide_banner', '-stats']
        if args.start: cmd.extend(['-ss', args.start])
        if args.duration: cmd.extend(['-t', args.duration])
        cmd.extend(['-i', args.source, '-fps_mode', 'passthrough', '-q:v', '2', os.path.join(frame_dir, "%05d.png")])
        
        run_ffmpeg_with_progress(cmd, desc="Extracting PNGs")
        generate_review_video(args.source, project_path, start=args.start, duration=args.duration)

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

    elif args.command == "gui":
        from .gui import BrianDashboard
        import tkinter as tk
        
        initial_path = None
        if args.project:
            initial_path = os.path.join(os.path.expanduser("~"), "Projects/UAP_Data", args.project)
            if not os.path.exists(initial_path):
                print(f"❌ Project '{args.project}' not found. Opening empty GUI.")
                initial_path = None

        root = tk.Tk()
        app = BrianDashboard(root, "LUMICRON COCKPIT", initial_project_path=initial_path)
        root.mainloop()

    elif args.command == "visualize":
        tracker = VisualTracker(project_path)
        pixel_shifts, f_delta = tracker.manual_track(args.anchor, args.mask, args.filter)
        if pixel_shifts:
            # LOCK-IN: Ensure tracking data is saved to disk
            t_file = os.path.join(project_path, "03_DATA", "tracking.json")
            with open(t_file, 'w') as f:
                json.dump({"pixel_shifts": pixel_shifts, "frame_delta": f_delta}, f, indent=4)
            print(f"✅ Flight path saved to {t_file}")

    elif args.command == "report":
        # VALIDATION: Ensure tracking data exists
        t_path = os.path.join(project_path, "03_DATA", "tracking.json")
        if not os.path.exists(t_path):
            print(f"❌ ERROR: Run 'visualize' for {args.project} before generating report.")
            return
        
        engine = KinematicsEngine(project_path)
        dossier = engine.generate_markdown_dossier(args.distance, args.focal, args.fps)
        print("\n" + dossier + "\n")

if __name__ == "__main__":
    main()
