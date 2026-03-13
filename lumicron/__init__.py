import cv2
cv2.ocl.setUseOpenCL(True)
gpu_status = "ACTIVE (eGPU)" if cv2.ocl.useOpenCL() else "INACTIVE (CPU Only)"
import argparse
import os
from datetime import datetime
from .core.physics import KinematicsEngine, RadiometricEngine, MorphologicalEngine, ArtifactEngine

def main():
    # Force OpenCV to use OpenCL (eGPU)
    cv2.ocl.setUseOpenCL(True)

# Attempt to find and set the AMD device
    for i in range(10):
        try:
            dev = cv2.ocl.Device_getDevice(i)
            if "AMD" in dev.name() or "Radeon" in dev.name():
                cv2.ocl.Device.setDefault(dev)
                break
        except:
            break
            
    active_dev = cv2.ocl.Device.getDefault()
    print(f"--- HARDWARE: {active_dev.name()} ---")

    print(f"Hardware Acceleration: {'ENABLED' if cv2.ocl.useOpenCL() else 'DISABLED'}")
    parser = argparse.ArgumentParser(description="LUMICRON: Hypersonic UAP Forensic Suite")
    subparsers = parser.add_subparsers(dest="command")

    for cmd in ['track', 'radiate', 'morph', 'noise', 'report']:
        p = subparsers.add_parser(cmd, help=f'{cmd.capitalize()} analysis')
        p.add_argument('project', help='Project folder')
        if cmd in ['track', 'report']:
            p.add_argument('--distance', type=float, default=5000.0)
            p.add_argument('--sensor', type=float, default=36.0)
            p.add_argument('--focal', type=float, default=200.0)
            p.add_argument('--fps', type=int, default=60)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    project_path = os.path.abspath(args.project)
    
    # Pre-calculate common results
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

    if args.command == "report":
        # Save to file
        report_dir = os.path.join(project_path, "04_REPORTS")
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"dossier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w') as f:
            f.write(report_content)
        print(f"REPORT SAVED TO: {report_file}")

if __name__ == "__main__":
    main()
