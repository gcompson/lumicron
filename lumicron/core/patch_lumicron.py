import os

file_path = "lumicron.py"
backup_path = "lumicron.py.bak"

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        original_content = f.read()
    with open(backup_path, "w") as f:
        f.write(original_content)
    print(f"Created backup at {backup_path}")

import_line = "from core.physics import KinematicsEngine, RadiometricEngine, MorphologicalEngine, ArtifactEngine\n"

parsers_code = """
    # --- Physics & Forensic Commands ---
    track_p = subparsers.add_parser('track', help='Calculate velocity and G-Force')
    track_p.add_argument('project', help='Project folder')
    track_p.add_argument('--distance', type=float, default=5000.0)
    track_p.add_argument('--sensor', type=float, default=36.0)
    track_p.add_argument('--focal', type=float, default=200.0)
    track_p.add_argument('--fps', type=int, default=60)

    subparsers.add_parser('radiate', help='Luminance/Pulse analysis')
    subparsers.add_parser('morph', help='Shape stability analysis')
    subparsers.add_parser('noise', help='Artifact rejection')
"""

logic_code = """
    elif args.command == "track":
        mock_shifts = [10.5, 12.2, 45.8, 110.2] 
        results = KinematicsEngine.calculate_telemetry(
            mock_shifts, args.distance, args.focal, args.sensor, 3840, args.fps
        )
        print(f"\\n--- KINEMATIC TELEMETRY: {args.project} ---")
        print(f"Top Speed: {results['top_speed_ms']} m/s")
        print(f"Max Acceleration: {results['max_g']} Gs")
        if results['max_g'] > 50:
            print("WARNING: Kinematic anomaly detected (High-G maneuver).")
"""

lines = original_content.splitlines()
if "from core.physics" not in original_content:
    lines.insert(0, import_line)

new_content = "\n".join(lines)

if "subparsers.add_parser('track'" not in new_content:
    target = 'subparsers = parser.add_subparsers(dest="command")'
    if target in new_content:
        new_content = new_content.replace(target, target + parsers_code)

if 'elif args.command == "track":' not in new_content:
    marker = 'if __name__ == "__main__":'
    new_content = new_content.replace(marker, logic_code + "\n" + marker)

with open(file_path, "w") as f:
    f.write(new_content)
print("Patch applied successfully.")
