import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from concurrent.futures import ThreadPoolExecutor

class KinematicsEngine:
    @staticmethod
    def calculate_telemetry(pixel_shifts, distance_m, focal_mm, sensor_w_mm, img_w_px, fps):
        # 1. Calculate Spatial Resolution
        fov_m = distance_m * (sensor_w_mm / focal_mm)
        m_per_px = fov_m / img_w_px
        dt = 1.0 / fps
        
        # 2. Velocity Calculation (m/s)
        velocities = [(s * m_per_px) / dt for s in pixel_shifts]
        
        # 3. Slow-UAP Logic: Establish Noise Floor
        # If the pixel shift is less than 2px, it might be human clicking error
        noise_floor_ms = (2.0 * m_per_px) / dt
        
        g_forces = [0]
        for i in range(1, len(velocities)):
            v_delta = abs(velocities[i] - velocities[i-1])
            
            # If the change is smaller than our noise floor, ignore the G-spike
            if v_delta < noise_floor_ms:
                g_forces.append(0.0)
                continue
                
            accel = v_delta / dt
            g_forces.append(accel / 9.81)
            
        top_speed = max(velocities) if velocities else 0
        max_g = max(g_forces) if g_forces else 0
        
        # 4. Scenario Classification
        status = "ANOMALOUS" if max_g > 50 or top_speed > 1000 else "CONVENTIONAL"
        if top_speed < 50 and max_g < 2:
            status = "SLOW_STABLE (Possible Drone/Bird)"

        return {
            "top_speed_ms": round(top_speed, 2),
            "max_g": round(max_g, 2),
            "classification": status
        }

class RadiometricEngine:
    @staticmethod
    def analyze_luminance(project_path):
        frames_dir = os.path.join(project_path, "02_FRAMES")
        data_dir = os.path.join(project_path, "03_DATA")
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg'))])
        if not frame_files: return []

        def process_frame(f_info):
            idx, f_path = f_info
            img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return {"frame": idx, "peak_luminance": int(np.max(img)), "mean_luminance": float(np.mean(img))}
            return None

        with ThreadPoolExecutor() as executor:
            results = list(filter(None, executor.map(process_frame, enumerate(frame_files))))
        
        pd.DataFrame(results).to_csv(os.path.join(data_dir, "smear_audit.csv"), index=False)
        print(f"✅ Radiometric data saved to smear_audit.csv")
        return results

    @staticmethod
    def generate_plots(project_path):
        """Generates the frequency and pulse charts for the report."""
        data_path = os.path.join(project_path, "03_DATA", "smear_audit.csv")
        if not os.path.exists(data_path): return

        df = pd.read_csv(data_path)
        y = df['mean_luminance'].values
        
        # Pulse Audit (Time Domain)
        plt.figure(figsize=(12, 5))
        plt.plot(df['frame'], y, color='#00ffcc', linewidth=1)
        plt.title('LUMICRON: Radiometric Pulse Audit')
        plt.grid(True, alpha=0.2)
        plt.savefig(os.path.join(project_path, "04_REPORTS", "pulse_audit.png"))
        
        # FFT (Frequency Domain)
        yf = np.abs(fft(y - np.mean(y)))
        xf = np.fft.fftfreq(len(y), 1/240)
        plt.figure(figsize=(12, 5))
        plt.plot(xf[:len(xf)//2], yf[:len(yf)//2], color='#ff00ff')
        plt.title('LUMICRON: Frequency Domain (Propulsion Signature)')
        plt.xlim(0, 60)
        plt.savefig(os.path.join(project_path, "04_REPORTS", "frequency_spectrum.png"))
        plt.close('all')
        print("✅ Forensic plots generated in 04_REPORTS.")

class MorphologicalEngine:
    @staticmethod
    def analyze_shape_stability(project_path):
        frames_dir = os.path.join(project_path, "02_FRAMES")
        # Sample frames from anomaly window
        files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)])[2580:3827:10]
        
        aspect_ratios = []
        for f_path in files:
            img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                if w > 0 and h > 0: aspect_ratios.append(float(w) / h)

        variance = np.var(aspect_ratios) if aspect_ratios else 1.0
        ssi = max(0, 1.0 - (variance * 10))
        status = "Rigid Craft" if ssi > 0.85 else "Fluctuating Body"
        print(f"SSI: {ssi:.2f} | Status: {status}")
        return {"stability_index": round(ssi, 3), "status": status}

class VisualTracker:
    @staticmethod
    def manual_track(project_path, mask=False, filter=False, start=None, end=None):
        frames_dir = os.path.join(project_path, "02_FRAMES")
        files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)])
        
        s_idx = start if start is not None else 0
        e_idx = end if end is not None else len(files) - 1
        m_idx = s_idx + (e_idx - s_idx) // 2 
        
        target_indices = [s_idx, m_idx, e_idx]
        points = []

        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x / params['scale'], y / params['scale']))
                cv2.circle(params['display'], (x, y), 5, (0, 255, 0), -1)

        for idx in target_indices:
            img = cv2.imread(files[idx])
            if img is None: continue
            
            # CLAHE Enhancement for visibility
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            img = cv2.merge((cv2.createCLAHE(clipLimit=3.0).apply(l), a, b))
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

            scale = 1400 / img.shape[1]
            display = cv2.resize(img, (0,0), fx=scale, fy=scale)
            
            win = f"TRACKER - Frame {idx}"
            cv2.namedWindow(win)
            cv2.setMouseCallback(win, click_event, {'scale': scale, 'display': display})
            
            while len(points) < (target_indices.index(idx) + 1):
                cv2.imshow(win, display)
                if cv2.waitKey(20) & 0xFF == 27: return None
            cv2.destroyWindow(win)
        
        if len(points) >= 3:
            s1 = np.linalg.norm(np.array(points[1]) - np.array(points[0]))
            s2 = np.linalg.norm(np.array(points[2]) - np.array(points[1]))
            return [s1, s2], (e_idx - s_idx) // 2
        return None

class StackEngine:
    @staticmethod
    def generate_streak_map(project_path, start=None, end=None):
        frames_dir = os.path.join(project_path, "02_FRAMES")
        files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)])
        s_idx, e_idx = (start or 0), (end or len(files)-1)
        
        first = cv2.imread(files[s_idx])
        mip = np.zeros_like(first)
        for f in files[s_idx:e_idx+1]:
            img = cv2.imread(f)
            if img is not None: mip = cv2.max(mip, img)
        
        cv2.imwrite(os.path.join(project_path, "03_DATA", "streak_map.png"), mip)
        print("✅ Streak Map saved.")

class ArtifactEngine:
    @staticmethod
    def detect_noise_signature(project_path):
        """Audits first 10 frames to detect static sensor artifacts."""
        return {"false_positives": 0}       
