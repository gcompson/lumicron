import numpy as np
import cv2
import os
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
        if not os.path.exists(frames_dir): return None
        
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg'))])
        if not frame_files: return None

        def get_intensity(f_path):
            img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Use OpenCL-accelerated minMaxLoc if enabled
                _, max_val, _, _ = cv2.minMaxLoc(img)
                return max_val
            return None

        # ThreadPool for parallel I/O
        with ThreadPoolExecutor() as executor:
            intensities = list(filter(None, executor.map(get_intensity, frame_files)))
        
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
        if not os.path.exists(frames_dir): return 0.0
        
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg'))])
        
        def get_aspect_ratio(f_path):
            img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return None
            _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                if h > 0: return w / h
            return None

        with ThreadPoolExecutor() as executor:
            aspect_ratios = list(filter(None, executor.map(get_aspect_ratio, frame_files)))
            
        return float(np.var(aspect_ratios)) if aspect_ratios else 0.0

class ArtifactEngine:
    @staticmethod
    def detect_noise_signature(project_path):
        frames_dir = os.path.join(project_path, "02_FRAMES")
        if not os.path.exists(frames_dir): return "Folder Missing"
        
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg'))])
        if len(frame_files) < 10: return "Insufficient Frames"
        
        def get_max_loc(f_path):
            img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, _, _, max_loc = cv2.minMaxLoc(img)
                return max_loc
            return None

        with ThreadPoolExecutor() as executor:
            coords = list(filter(None, executor.map(get_max_loc, frame_files[:10])))
        
        if not coords: return "Unreadable Data"
        if all(c == coords[0] for c in coords):
            return "REJECTED: Static Sensor Artifact"
        
        return "VERIFIED: Dynamic Motion"

class ThermalEngine:
    @staticmethod
    def analyze_heat_gradient(project_path):
        # Placeholder for FLIR/Radiometric JPEG processing
        return "Thermal Module: Earmarked for Future Implementation"

class SpectralEngine:
    @staticmethod
    def extract_wavelengths(project_path, grating_lines_mm=300):
        # Placeholder for diffraction grating spectroscopy
        return "Spectral Module: Earmarked for Future Implementation"

class ThermalEngine:
    @staticmethod
    def analyze_heat_gradient(project_path):
        # Placeholder for FLIR/Radiometric JPEG processing
        return "Thermal Module: Earmarked for Future Implementation"

class SpectralEngine:
    @staticmethod
    def extract_wavelengths(project_path, grating_lines_mm=300):
        # Placeholder for diffraction grating spectroscopy
        return "Spectral Module: Earmarked for Future Implementation"

class ThermalEngine:
    @staticmethod
    def analyze_heat_gradient(project_path):
        # Placeholder for FLIR/Radiometric JPEG processing
        return "Thermal Module: Earmarked for Future Implementation"

class SpectralEngine:
    @staticmethod
    def extract_wavelengths(project_path, grating_lines_mm=300):
        # Placeholder for diffraction grating spectroscopy
        return "Spectral Module: Earmarked for Future Implementation"
