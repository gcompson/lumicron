# LUMICRON: Hypersonic UAP Forensic Suite

![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

**Lumicron** is a high-speed video analysis toolkit designed to identify, isolate, and audit kinematic anomalies (UAPs/UFOs) from high-frame-rate digital sensors. It provides a standardized forensic pipeline from raw video ingestion to AI-assisted physics reporting.

---

## Core Philosophy
Lumicron focuses on **Kinematic Discontinuity**. By leveraging Background Subtraction (MOG2) and Smear-Ratio Analysis, it isolates objects moving at speeds or trajectories that defy conventional biological or aeronautical profiles.

---

## Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** and **FFmpeg** (with `ffprobe`) installed on your system.

### 2. Clone and Install
```bash
git clone [https://github.com/gcompson/lumicron.git](https://github.com/gcompson/lumicron.git)
cd lumicron
pip install -e .
```

---

## Standard Forensic Workflow

### 1. Initialize
Create a localized project folder and manifest.
```bash
lumicron init Project_Alpha --location "Hamilton, Australia"
```

### 2. Extract
Ingest video into the **01_RAW** folder and extract high-speed frames.
```bash
lumicron extract ~/Downloads/capture.mov Project_Alpha --start 00:00:15 --end 00:00:20
```

### 3. Scan
Run the motion detection engine with Smear Filtering.
```bash
lumicron scan Project_Alpha --sensitivity 70 --min-smear 2.5
```

### 4. Report
Generate evidence packets, heatmaps, and AI-ready prompts.
```bash
lumicron report heatmap Project_Alpha
lumicron report crops Project_Alpha --enhance
lumicron report prompt Project_Alpha
lumicron report brief Project_Alpha
```

---

## Directory Structure
* **01_RAW/**: Master source video and ingestion logs.
* **02_FRAMES/**: Extracted high-speed PNGs (The "Source of Truth").
* **03_ANALYSIS/**: CSV transit logs and kinematic telemetry.
* **04_REPORTS/**: PDF and Markdown Forensic Briefs.
* **05_FIGURES/**: Heatmaps, annotated reels, and sub-pixel crops.

---

## AI Integration
Lumicron creates a **Dynamic Context Window** to bridge the gap between raw data and expert analysis. The \`report prompt\` command pulls data from the \`lumicron.yaml\` and the \`transit_log.csv\` to create a scientific briefing. Paste this into an LLM for an objective assessment.

---

## Contributing
Lumicron is an open-source project for the global research community. Contributions are welcome for:
* **New reporting modules**: Trajectory plots and velocity graphs.
* **Sensor Integration**: FLIR and multispectral support.
* **Enhanced Filtering**: Advanced morphological filters for noise reduction.

**Standardize the search. Verify the math. Find the needles.**
