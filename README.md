# 🛸 LUMICRON: Hypersonic UAP Forensic Suite

**Lumicron** is a high-speed video analysis toolkit designed to identify, isolate, and audit kinematic anomalies (UAPs/UFOs) from high-frame-rate digital sensors. It provides a standardized forensic pipeline from raw video ingestion to AI-assisted physics reporting.

## 🔬 Core Philosophy
Lumicron focuses on **Kinematic Discontinuity**. By leveraging Background Subtraction (MOG2) and Smear-Ratio Analysis, it isolates objects moving at speeds or trajectories that defy conventional biological or aeronautical profiles.

---

## 🛠️ Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** and **FFmpeg** (with `ffprobe`) installed on your system.

### 2. Clone and Install
```zsh
git clone [https://github.com/your-username/lumicron.git](https://github.com/your-username/lumicron.git)
cd lumicron
pip install -e .

---

## 🚀 Standard Forensic Workflow

### 🏗️ 1. Initialize
Create a localized project folder. This generates a `lumicron.yaml` manifest which stores your locale and sensor specs.
```zsh
lumicron init Project_Alpha --location "Hamilton, Australia"

### 🎞️ 2. Extract
Ingest your video into the **01_RAW** folder and extract frames. Use time-slicing to save space and focus specifically on the "event window" where the anomaly occurs.
```zsh
lumicron extract ~/Downloads/capture.mov Project_Alpha --start 00:00:15 --end 00:00:20

### 🎞️ 2. Extract
Ingest your video into the **01_RAW** folder and extract frames. Use time-slicing to save space and focus specifically on the "event window."

```bash
lumicron extract ~/Downloads/capture.mov Project_Alpha --start 00:00:15 --end 00:00:20

### 🔍 3. Scan
Run the motion detection engine with Smear Filtering. Use the `--limit` option to test filters on a small batch before committing to the full sequence.

```bash
lumicron scan Project_Alpha --sensitivity 70 --min-smear 2.5

### 📊 4. Report
Generate the evidence packet, heatmaps, and AI-ready prompts for final auditing.

```bash
# Generate a density map of detections
lumicron report heatmap Project_Alpha

# Create high-contrast zoomed-in crops of top targets
lumicron report crops Project_Alpha --enhance

# Generate a dynamic prompt for LLM analysis
lumicron report prompt Project_Alpha

# Export the final Forensic PDF brief
lumicron report brief Project_Alpha

## 🏛️ Directory Structure
Lumicron enforces a strict forensic folder hierarchy to ensure data provenance and scientific repeatability:

* **`01_RAW/`**: Master source video and ingestion logs.
* **`02_FRAMES/`**: Extracted high-speed PNGs (The "Source of Truth").
* **`03_ANALYSIS/`**: CSV transit logs and kinematic telemetry.
* **`04_REPORTS/`**: PDF and Markdown Forensic Briefs.
* **`05_FIGURES/`**: Heatmaps, annotated reels, and sub-pixel crops.
* **`99_LEGACY/`**: Archive for superseded data versions.

## 🤖 AI Integration
Lumicron creates a **Dynamic Context Window** to bridge the gap between raw data and expert analysis. 

The `report prompt` command pulls data from the `lumicron.yaml` (locale, sensor FPS) and the `transit_log.csv` (top smear ratios) to create a scientific briefing. Paste this output into a Large Language Model (LLM) to receive an objective, physics-based assessment of the detected targets.

## 🤝 Contributing
Lumicron is an open-source project for the global research community. Contributions are welcome for:

* **New reporting modules**: Trajectory plots and velocity graphs.
* **Sensor Integration**: FLIR and multispectral support.
* **Enhanced Filtering**: Advanced morphological filters for noise reduction.

**Standardize the search. Verify the math. Find the needles.**


