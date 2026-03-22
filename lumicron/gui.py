import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import os
import json
import numpy as np

class BrianDashboard:
    def __init__(self, window, window_title, initial_project_path=None):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1600x950")
        self.window.configure(bg="#1e1e1e")

        # --- FORENSIC CONSTANTS (Hamilton Baseline) ---
        self.src_w, self.src_h = 1920, 1080
        self.aspect_ratio = self.src_w / self.src_h
        self.dist_m, self.fps, self.focal_mm, self.sensor_w_mm = 175.0, 240.0, 24.0, 36.0
        
        # State
        self.vid = None
        self.current_frame = 0
        self.points = {} # {frame_index: (x, y)}
        self.project_path = initial_project_path
        self.total_frames = 0
        self.render_w, self.render_h = 1280, 720
        self.off_x, self.off_y = 0, 0

        self.setup_ui()
        
        # Bindings
        self.window.bind("<Left>", lambda e: self.step_frame(-1))
        self.window.bind("<Right>", lambda e: self.step_frame(1))
        self.window.bind("<Configure>", self.on_window_resize)
        
        if self.project_path:
            self.load_project_data(self.project_path)

    def setup_ui(self):
        # --- SIDEBAR ---
        self.sidebar = tk.Frame(self.window, width=320, bg="#252526")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.sidebar, text="LUMICRON v5.9", font=("Courier", 22, "bold"), bg="#252526", fg="fuchsia").pack(pady=20)
        
        # --- PART 3: TEMPORAL DILATION PANE ---
        tk.Label(self.sidebar, text="TRANSIENT ZOOM (4x)", font=("Arial", 10, "bold"), bg="#252526", fg="#888").pack()
        self.zoom_canvas = tk.Canvas(self.sidebar, width=250, height=250, bg="black", highlightthickness=1, highlightbackground="fuchsia")
        self.zoom_canvas.pack(pady=10, padx=25)

        # --- TELEMETRY ---
        self.lbl_mach = tk.Label(self.sidebar, text="MACH: 0.00", font=("Courier", 16), bg="#252526", fg="white")
        self.lbl_mach.pack(pady=5)
        self.lbl_gforce = tk.Label(self.sidebar, text="G-FORCE: 0.0", font=("Courier", 16), bg="#252526", fg="orange")
        self.lbl_gforce.pack(pady=5)

        # FIXED: Style dictionary no longer contains 'bg' to avoid the TypeError
        btn_s = {"font": ("Arial", 12, "bold"), "fg": "white", "padx": 20, "pady": 12, "cursor": "hand2"}
        
        self.btn_load = tk.Label(self.sidebar, text="LOAD PROJECT", bg="#3e3e3e", **btn_s)
        self.btn_load.pack(pady=10, fill=tk.X, padx=30)
        self.btn_load.bind("<Button-1>", lambda e: self.load_project())

        self.btn_save = tk.Label(self.sidebar, text="SAVE & EXPORT", bg="#007acc", **btn_s)
        self.btn_save.pack(pady=10, fill=tk.X, padx=30)
        self.btn_save.bind("<Button-1>", lambda e: self.save_and_exit())

        self.status_lbl = tk.Label(self.sidebar, text="READY", font=("Arial", 10), bg="#252526", fg="#888")
        self.status_lbl.pack(side=tk.BOTTOM, pady=20)

        # --- MAIN DISPLAY ---
        self.display_frame = tk.Frame(self.window, bg="#1e1e1e")
        self.display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.display_frame, bg="black", highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=15, pady=15)
        
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-2>", self.on_right_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        self.scrub = tk.Scale(self.display_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                              command=self.on_scrub, bg="#1e1e1e", fg="white", 
                              highlightthickness=0, troughcolor="#333", length=1000)
        self.scrub.pack(pady=20)

    def on_window_resize(self, event):
        if event.widget == self.canvas:
            cw, ch = event.width, event.height
            if cw < 100 or ch < 100: return
            if (cw / ch) > self.aspect_ratio:
                self.render_h = ch
                self.render_w = int(ch * self.aspect_ratio)
            else:
                self.render_w = cw
                self.render_h = int(cw / self.aspect_ratio)
            self.off_x, self.off_y = (cw - self.render_w) // 2, (ch - self.render_h) // 2
            self.render_frame()

    def step_frame(self, delta):
        if not self.vid: return
        new_f = self.current_frame + delta
        if 0 <= new_f < self.total_frames:
            self.current_frame = new_f
            self.scrub.set(new_f)

    def load_project_data(self, path):
        self.project_path = path
        video_path = os.path.join(self.project_path, "03_DATA", "review_telemetry.mp4")
        if os.path.exists(video_path):
            self.vid = cv2.VideoCapture(video_path)
            self.total_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.scrub.config(to=self.total_frames - 1)
            self.render_frame()
            self.status_lbl.config(text=f"ACTIVE: {os.path.basename(path)}", fg="lime")

    def load_project(self):
        path = filedialog.askdirectory(initialdir="~/Projects/UAP_Data")
        if path: self.load_project_data(path)

    def render_frame(self):
        if self.vid:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.vid.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_zoom(frame_rgb)
                display_img = cv2.resize(frame_rgb, (self.render_w, self.render_h))
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(display_img))
                self.canvas.delete("all")
                self.canvas.create_image(self.off_x, self.off_y, image=self.photo, anchor=tk.NW)
                self.draw_overlay()

    def update_zoom(self, frame_rgb):
        target_pos = self.points.get(self.current_frame) or (self.src_w // 2, self.src_h // 2)
        tx, ty = int(target_pos[0]), int(target_pos[1])
        sz = 60
        y1, y2 = max(0, ty-sz), min(self.src_h, ty+sz)
        x1, x2 = max(0, tx-sz), min(self.src_w, tx+sz)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
            crop_enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)
            crop_zoom = cv2.resize(crop_enhanced, (250, 250), interpolation=cv2.INTER_CUBIC)
            self.zoom_photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(crop_zoom))
            self.zoom_canvas.delete("all")
            self.zoom_canvas.create_image(0, 0, image=self.zoom_photo, anchor=tk.NW)

    def on_left_click(self, event):
        vx, vy = event.x - self.off_x, event.y - self.off_y
        if 0 <= vx <= self.render_w and 0 <= vy <= self.render_h:
            rx, ry = vx * (self.src_w / self.render_w), vy * (self.src_h / self.render_h)
            self.points[self.current_frame] = (rx, ry)
            self.auto_bridge(); self.render_frame(); self.calculate_live_physics()

    def on_right_click(self, event):
        if self.current_frame in self.points:
            del self.points[self.current_frame]
            self.auto_bridge(); self.render_frame(); self.calculate_live_physics()

    def draw_overlay(self):
        for f, (px, py) in self.points.items():
            cx = self.off_x + (px * (self.render_w / self.src_w))
            cy = self.off_y + (py * (self.render_h / self.src_h))
            color = "lime" if f == self.current_frame else "red"
            self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill=color, outline="white", width=2)

    def auto_bridge(self):
        if not self.total_frames: return
        shifts = [0.0] * self.total_frames
        sorted_fs = sorted(self.points.keys())
        if len(sorted_fs) >= 2:
            for i in range(len(sorted_fs)-1):
                f1, f2 = sorted_fs[i], sorted_fs[i+1]
                p1, p2 = self.points[f1], self.points[f2]
                dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                step = dist / (f2 - f1)
                for f in range(f1, f2): shifts[f] = step
        t_file = os.path.join(self.project_path, "03_DATA", "tracking.json")
        with open(t_file, 'w') as f: json.dump({"pixel_shifts": shifts, "frame_delta": 1}, f, indent=4)

    def calculate_live_physics(self):
        sorted_fs = sorted(self.points.keys())
        if len(sorted_fs) < 2: return
        total_px = sum(np.sqrt((self.points[sorted_fs[i+1]][0]-self.points[sorted_fs[i]][0])**2 + (self.points[sorted_fs[i+1]][1]-self.points[sorted_fs[i]][1])**2) for i in range(len(sorted_fs)-1))
        time_sec = (sorted_fs[-1] - sorted_fs[0]) / self.fps
        fov_h = 2 * np.degrees(np.arctan(self.sensor_w_mm / (2 * self.focal_mm)))
        total_deg = total_px * (fov_h / 1920)
        real_dist_m = 2 * self.dist_m * np.sin(np.radians(total_deg / 2))
        velocity_ms = real_dist_m / time_sec
        self.lbl_mach.config(text=f"MACH: {velocity_ms / 340.29:.2f}")
        self.lbl_gforce.config(text=f"G-FORCE: {(velocity_ms**2 / (real_dist_m / 2)) / 9.81 if real_dist_m > 0 else 0:.1f}")

    def on_scrub(self, val):
        self.current_frame = int(val); self.render_frame()

    def save_and_exit(self):
        self.auto_bridge(); self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk(); app = BrianDashboard(root, "LUMICRON FORENSIC COCKPIT v5.9"); root.mainloop()
