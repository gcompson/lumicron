import numpy as np
import cv2
import os
import sys
import time
from scipy.fftpack import fft
from concurrent.futures import ThreadPoolExecutor

class KinematicsEngine:
    @staticmethod
    def calculate_telemetry(pixel_shifts, distance_m, focal_mm, sensor_w_mm, img_w_px, fps):
        fov_m = distance_m * (sensor_w_mm / focal_mm)
        m_per_px = fov_m / img_w_px
        dt = 1.0 / fps
        velocities = [(s * m_per_px) / dt for s in pixel_shifts]
        g_forces = []
        for i in range(1, len(velocities)):
            accel = (velocities[i] - velocities[i-1]) / dt
            g_forces.append(abs(accel) / 9.81)
        return {
            "top_speed_ms": round(max(velocities), 2) if velocities else 0,
            "max_g": round(max(g_forces), 2) if g_forces else 0
        }

class RadiometricEngine:
    @staticmethod
    def analyze_luminance(project_path):
        frames_dir = os.path.join(project_path, "02_FRAMES")
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg'))])
        total = len(frame_files)
        if total == 0: return None

        def process_frame(f_path):
            img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            return cv2.minMaxLoc(img)[1] if img is not None else None

        intensities = []
        print(f"Scanning {total} frames for Luminance...")
        
        with ThreadPoolExecutor() as executor:
            for i, val in enumerate(executor.map(process_frame, frame_files)):
                if val is not None: intensities.append(val)
                percent = int((i + 1) / total * 100)
                # The \r carriage return resets the line
                sys.stdout.write(f"\r\033[KProgress: [{percent}%] Frame {i+1}/{total}")
                sys.stdout.flush()
        
        print("\nLuminance Scan Complete.")
        
        if len(intensities) < 2: return {"peak_intensity": 0, "pulse_freq": 0}
        arr = np.array(intensities)
        norm = arr - np.mean(arr)
        freqs = np.abs(fft(norm))
        peak_freq = np.argmax(freqs[1:len(freqs)//2]) + 1
        return {"peak_intensity": float(np.max(arr)), "pulse_freq": int(peak_freq)}

class MorphologicalEngine:
    @staticmethod
    def analyze_shape_stability(project_path):
        frames_dir = os.path.join(project_path, "02_FRAMES")
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg'))])
        total = len(frame_files)
        if total == 0: return 0.0

        def process_morph(f_path):
            img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return None
            _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                return w / h if h > 0 else None
            return None

        aspect_ratios = []
        print(f"Analyzing Morphology of {total} frames...")
        
        with ThreadPoolExecutor() as executor:
            for i, val in enumerate(executor.map(process_morph, frame_files)):
                if val is not None: aspect_ratios.append(val)
                percent = int((i + 1) / total * 100)
                sys.stdout.write(f"\r\033[KProgress: [{percent}%] Frame {i+1}/{total}")
                sys.stdout.flush()

        print("\nMorphology Analysis Complete.")
        return float(np.var(aspect_ratios)) if aspect_ratios else 0.0

class ArtifactEngine:
    @staticmethod
    def detect_noise_signature(project_path):
        frames_dir = os.path.join(project_path, "02_FRAMES")
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg'))])
        if len(frame_files) < 10: return "Insufficient Frames"
        
        coords = []
        for f in frame_files[:10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                coords.append(cv2.minMaxLoc(img)[3])
        
        if not coords: return "Unreadable Data"
        return "REJECTED: Static Sensor Artifact" if all(c == coords[0] for c in coords) else "VERIFIED: Dynamic Motion"

class VisualTracker:
    @staticmethod
    def manual_track(project_path):
        frames_dir = os.path.join(project_path, "02_FRAMES")
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)])
        if len(frame_files) < 2: return None

        # Load first and last frames
        img1 = cv2.imread(frame_files[0])
        img2 = cv2.imread(frame_files[-1])
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(param, (x, y), 10, (0, 255, 0), -1)
                cv2.imshow("Select Object", param)

        print("\n--- MANUAL TRACKING ---")
        print("1. Click the object in the FIRST frame, then press ANY KEY.")
        cv2.imshow("Select Object", img1)
        cv2.setMouseCallback("Select Object", click_event, img1)
        cv2.waitKey(0)

        print("2. Click the object in the LAST frame, then press ANY KEY.")
        cv2.imshow("Select Object", img2)
        cv2.setMouseCallback("Select Object", click_event, img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(points) >= 2:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            pixel_dist = np.sqrt(dx**2 + dy**2)
            return pixel_dist, len(frame_files)
        return None
