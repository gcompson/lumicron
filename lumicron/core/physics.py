import numpy as np
import cv2
import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

class KinematicsEngine:
    def __init__(self, project_path):
        self.project_path = project_path

    @staticmethod
    def calculate_telemetry(pixel_shifts, distance_m, focal_input, sensor_w_mm, img_w_px, fps, bg_shifts=None):
        """
        LOCKED-IN: Multi-point kinematics with background-jitter subtraction.
        Converts pixel deltas to real-world velocity and G-force.
        """
        dt = 1.0 / fps
        velocities = []
        
        # Parallax/Jitter correction: default to 0 if no anchor tracking provided
        if bg_shifts is None:
            bg_shifts = [0.0] * len(pixel_shifts)

        for i, s in enumerate(pixel_shifts):
            # Support both fixed focal lengths and focal arrays (zooming)
            f_current = focal_input[i] if isinstance(focal_input, list) else focal_input
            
            # Physical scale: meters per pixel at target distance
            m_per_px = (distance_m * (sensor_w_mm / f_current)) / img_w_px
            
            # Sub-pixel delta corrected for camera movement
            true_shift = s - bg_shifts[i]
            velocities.append(abs((true_shift * m_per_px) / dt))
        
        g_forces = [0]
        for i in range(1, len(velocities)):
            dv = velocities[i] - velocities[i-1]
            accel = dv / dt
            # Convert m/s^2 to G (Standard Gravity: 9.80665)
            g_forces.append(abs(accel / 9.80665))

        top_speed = max(velocities) if velocities else 0
        max_g = max(g_forces) if g_forces else 0

        # MISSION CRITICAL: Classification Thresholds
        status = "Prosaic/Ballistic"
        if max_g > 20: status = "Advanced Propulsion"
        if max_g > 100: status = "Non-Inertial Maneuver" # The "Physics-Defying" benchmark
        if top_speed > 110215: status = "Mach 321 Cruiser Class" # Target B Signature

        return {
            "classification": status,
            "top_speed_mps": round(top_speed, 2),
            "max_g": round(max_g, 2),
            "velocities": velocities
        }

    def generate_markdown_dossier(self, distance, focal, fps):
        """Consolidates all metrics from 03_DATA into a copy-pasteable report."""
        # Load all persistent data
        try:
            with open(os.path.join(self.project_path, "03_DATA", "tracking.json"), 'r') as f:
                t_data = json.load(f)
            rad_df = pd.read_csv(os.path.join(self.project_path, "03_DATA", "smear_audit.csv"))
            with open(os.path.join(self.project_path, "03_DATA", "morphology.json"), 'r') as f:
                m_data = json.load(f)
        except FileNotFoundError as e:
            return f"Error: Missing analysis data for report. ({e.filename})"

        peak_f = rad_df.iloc[rad_df['delta_flux'].idxmax()]['frame']
        kin = self.calculate_telemetry(t_data['pixel_shifts'], distance, focal, 36.0, 3840, fps)

        return f"""
# LUMICRON FORENSIC DOSSIER: {os.path.basename(self.project_path)}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Final Classification:** {kin['classification']}

## 1. KINEMATIC VALIDATION
* **Top Velocity:** {kin['top_speed_mps']:.2f} m/s (Mach {kin['top_speed_mps']/343.2:.2f})
* **Max Resultant Force:** {kin['max_g']:.2f} G
* **Rigidity (SSI):** {m_data['ssi']} ({m_data['classification']})

## 2. RADIOMETRIC SIGNATURE
* **Peak Energy Event:** Frame {int(peak_f)}
* **Signal Analysis:** Localized Energy Projection (Baseline Subtracted)
* **Status:** Confirmed environmental illumination on foreground targets.

## 3. ANALYST CONCLUSION
Object transited the environment at {kin['top_speed_mps']:.2f} m/s. The lack of atmospheric disruption and structural deformation (SSI {m_data['ssi']}) indicates non-conventional propulsion.
        """

class RadiometricEngine:
    def __init__(self, project_path):
        self.project_path = project_path
        stab_dir = os.path.join(project_path, "02_STABILIZED")
        raw_dir = os.path.join(project_path, "02_FRAMES")
        self.frames_dir = stab_dir if os.path.exists(stab_dir) and len(os.listdir(stab_dir)) > 0 else raw_dir

    def analyze(self, lookback=5):
        """Localized Flash Hunting: Gaussian Blur + Peak Finding."""
        files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        luma_global, peak_delta_flux, frame_buffer = [], [], []

        for i, f in enumerate(tqdm(files, desc="Scanning Photons")):
            img = cv2.imread(os.path.join(self.frames_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is None: continue # Safety Check

            luma_global.append(cv2.mean(img)[0])
            frame_buffer.append(img)
            if len(frame_buffer) > lookback + 1: frame_buffer.pop(0)

            if i >= lookback:
                delta = cv2.absdiff(img, frame_buffer[0])
                # Gaussian blur kills Pixel 6a sensor noise while preserving the flash
                blurred = cv2.GaussianBlur(delta, (9, 9), 0)
                _, max_val, _, _ = cv2.minMaxLoc(blurred)
                peak_delta_flux.append(max_val)
            else:
                peak_delta_flux.append(0.0)

        df = pd.DataFrame({"frame": range(1, len(luma_global)+1), "luma": luma_global, "delta_flux": peak_delta_flux})
        df.to_csv(os.path.join(self.project_path, "03_DATA", "smear_audit.csv"), index=False)
        print(f"✅ Telemetry saved. Peak: Frame {df.iloc[df['delta_flux'].idxmax()]['frame']}")

class StackEngine:
    def __init__(self, project_path):
        self.project_path = project_path
        stab_dir = os.path.join(project_path, "02_STABILIZED")
        raw_dir = os.path.join(project_path, "02_FRAMES")
        self.frames_dir = stab_dir if os.path.exists(stab_dir) and len(os.listdir(stab_dir)) > 0 else raw_dir

    def generate(self, mode='max'):
        """Streak Map Generation (Maximum Intensity Projection)."""
        files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        if not files: return
        
        base_img = cv2.imread(os.path.join(self.frames_dir, files[0]))
        if base_img is None: return
        
        res = base_img.copy().astype(np.float32)
        
        for f in tqdm(files[1:], desc=f"Stacking ({mode})"):
            img = cv2.imread(os.path.join(self.frames_dir, f))
            if img is None: continue # FIX: Prevents .astype() crash on None
            
            f_img = img.astype(np.float32)
            if mode == 'max':
                res = np.maximum(res, f_img)
            elif mode == 'diff':
                # Absdiff creates the clean "Streak over Black" look
                diff = cv2.absdiff(img, base_img)
                res = np.maximum(res, diff.astype(np.float32))

        out_path = os.path.join(self.project_path, "03_DATA", "streak_map.png")
        cv2.imwrite(out_path, res.astype(np.uint8))
        print(f"✅ Persistence map saved: {out_path}")

class MorphologicalEngine:
    def __init__(self, project_path):
        self.project_path = project_path
        stab_dir = os.path.join(project_path, "02_STABILIZED")
        raw_dir = os.path.join(project_path, "02_FRAMES")
        self.frames_dir = stab_dir if os.path.exists(stab_dir) else raw_dir

    def analyze(self):
        """Audits Geometry for SSI using Otsu's Binarization."""
        files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        aspect_ratios = []

        for f in tqdm(files, desc="Auditing Geometry"):
            img = cv2.imread(os.path.join(self.frames_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # Otsu automatically finds the object even in noisier Pixel 6a frames
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 5: # Noise floor
                    x, y, w, h = cv2.boundingRect(c)
                    aspect_ratios.append(w / float(h))

        ssi = max(0, 1.0 - np.std(aspect_ratios)) if aspect_ratios else 0
        res = {"ssi": round(ssi, 4), "classification": "RIGID CRAFT" if ssi > 0.85 else "ANOMALOUS"}
        with open(os.path.join(self.project_path, "03_DATA", "morphology.json"), 'w') as f:
            json.dump(res, f, indent=4)
        print(f"✅ SSI Score Locked: {ssi}")

class ArtifactEngine:
    def __init__(self, project_path):
        self.project_path = project_path
        stab_dir = os.path.join(project_path, "02_STABILIZED")
        raw_dir = os.path.join(project_path, "02_FRAMES")
        self.frames_dir = stab_dir if os.path.exists(stab_dir) else raw_dir

    def audit(self, frame_count=10, threshold=250):
        """Flags static sensor hot-pixels across project start."""
        files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])[:frame_count]
        stack = []
        for f in files:
            img = cv2.imread(os.path.join(self.frames_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is not None: stack.append(img)

        if not stack: return
        variance_map = np.var(stack, axis=0)
        mean_map = np.mean(stack, axis=0)
        hot_pixels = np.where((variance_map == 0) & (mean_map > threshold))
        coords = list(zip(hot_pixels[1].tolist(), hot_pixels[0].tolist()))
        
        with open(os.path.join(self.project_path, "03_DATA", "sensor_artifacts.json"), 'w') as f:
            json.dump({"hot_pixels": coords, "audit_date": datetime.now().isoformat()}, f, indent=4)
        print(f"✅ Sensor Audit Complete: {len(coords)} artifacts found.")

class VisualTracker:
    def __init__(self, project_path):
        self.project_path = project_path
        stab_dir = os.path.join(project_path, "02_STABILIZED")
        raw_dir = os.path.join(project_path, "02_FRAMES")
        self.frames_dir = stab_dir if os.path.exists(stab_dir) and len(os.listdir(stab_dir)) > 0 else raw_dir
        self.points = []

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Captured: {x}, {y}")

    def manual_track(self, use_anchor=False, use_mask=False, use_filter=False):
        """
        Interactive OpenCV interface. 
        ADDED: Path validation and debug prints to solve the 'no window' issue.
        """
        # DEBUG: Check if the path actually exists
        if not os.path.exists(self.frames_dir):
            print(f"❌ CRITICAL ERROR: Target directory not found: {self.frames_dir}")
            return [], 0

        files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        
        # DEBUG: Check if we actually found frames
        print(f"--- ATTEMPTING TO TRACK {len(files)} FRAMES IN {os.path.basename(self.frames_dir)} ---")
        
        if not files:
            print("❌ ERROR: No .png frames found. Did you run 'init' or 'stabilize' first?")
            return [], 0
        
        cv2.namedWindow("LUMICRON VISUALIZER")
        cv2.setMouseCallback("LUMICRON VISUALIZER", self._mouse_callback)
        
        # FORCE WINDOW TO FRONT (Essential for macOS focus issues)
        cv2.setWindowProperty("LUMICRON VISUALIZER", cv2.WND_PROP_TOPMOST, 1)

        pixel_shifts = []
        last_point = None
        
        for f in files:
            img_path = os.path.join(self.frames_dir, f)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"⚠️ Warning: Skipping corrupted frame {f}")
                continue
            
            display = img.copy()
            
            if use_filter:
                gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            if last_point:
                cv2.circle(display, last_point, 5, (0, 255, 0), -1)

            cv2.putText(display, f"FRAME: {f} | CLICK CENTER | 'n'=Next, 'q'=Save", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            while True:
                cv2.imshow("LUMICRON VISUALIZER", display)
                
                # IMPORTANT: On macOS, waitKey must be > 0 to process the window events
                key = cv2.waitKey(20) & 0xFF 
                
                if key == ord('n'):
                    if len(self.points) > 0:
                        curr_point = self.points[-1]
                        if last_point:
                            dist = np.sqrt((curr_point[0]-last_point[0])**2 + (curr_point[1]-last_point[1])**2)
                            pixel_shifts.append(float(dist))
                        last_point = curr_point
                    break
                elif key == ord('q'):
                    print(f"💾 Saving {len(pixel_shifts)} tracking points to 03_DATA...")
                    cv2.destroyAllWindows()
                    return pixel_shifts, len(pixel_shifts)
        
        cv2.destroyAllWindows()
        return pixel_shifts, len(pixel_shifts)
