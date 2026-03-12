#!/usr/bin/env python3
import click
import os
import yaml
import pandas as pd
import shutil
import zipfile
import cv2
import numpy as np
import subprocess
import logging
import tempfile
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from fpdf import FPDF

# Standard Project Structure
FOLDERS = ["01_RAW", "02_FRAMES", "03_ANALYSIS", "04_REPORTS", "05_FIGURES", "99_LEGACY"]

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

def get_config(project_dir):
    config_path = os.path.join(project_dir, "lumicron.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable debug logging.')
def cli(verbose):
    """LUMICRON: Open-Source Hypersonic UAP Forensic Suite"""
    setup_logging(verbose)

@cli.command()
@click.argument('project_path', default='.')
@click.option('--location', default='Unknown', help='Observer location for the manifest.')
def init(project_path, location):
    """Initialize a forensic project structure."""
    base_path = os.path.abspath(project_path)
    if os.path.exists(base_path) and os.path.isdir(base_path):
        contents = [f for f in os.listdir(base_path) if f != '.DS_Store']
        if contents:
            click.echo(f"❌ Error: Directory '{project_path}' is not empty.")
            return
    elif not os.path.exists(base_path):
        os.makedirs(base_path)

    for folder in FOLDERS:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    
    config = {
        "project_id": os.path.basename(base_path),
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        "location": location,
        "sensor": {"fps": 240, "make": "High-Speed", "model": "Camera"},
        "baseline_meters": 100.0
    }
    with open(os.path.join(base_path, "lumicron.yaml"), 'w') as f:
        yaml.dump(config, f)
    click.echo(f"✅ Project initialized for {location}.")

@cli.command()
@click.argument('video_source', type=click.Path(exists=True))
@click.argument('project_dir', type=click.Path(exists=True))
@click.option('--start', help="Start time (HH:MM:SS).")
@click.option('--end', help="End time (HH:MM:SS).")
def extract(video_source, project_dir, start, end):
    """Ingest and extract frames with forensic slicing."""
    raw_dir = os.path.join(project_dir, "01_RAW")
    dest_frames = os.path.join(project_dir, "02_FRAMES")
    local_video_path = os.path.join(raw_dir, os.path.basename(video_source))
    
    if not os.path.exists(local_video_path):
        shutil.copy2(video_source, local_video_path)

    ffmpeg_cmd = ['ffmpeg']
    if start: ffmpeg_cmd.extend(['-ss', start])
    if end: ffmpeg_cmd.extend(['-to', end])
    ffmpeg_cmd.extend(['-i', local_video_path, '-vsync', '0', '-q:v', '2', 
                       os.path.join(dest_frames, 'frame_%05d.png'), '-hide_banner', '-loglevel', 'error'])
    
    click.echo("🎞️  Extracting forensic slice...")
    subprocess.run(ffmpeg_cmd)
    click.echo(f"✅ Extraction complete.")

@cli.command()
@click.argument('project_dir', type=click.Path(exists=True))
@click.option('--sensitivity', default=65)
@click.option('--min-area', default=50)
@click.option('--min-smear', default=1.8)
@click.option('--range', 'f_range', nargs=2, type=int, help="Start and end frame index.")
def scan(project_dir, sensitivity, min_area, min_smear, f_range):
    """Deep scan for kinematic anomalies."""
    frames_dir = os.path.join(project_dir, "02_FRAMES")
    analysis_dir = os.path.join(project_dir, "03_ANALYSIS")
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if f_range:
        frames = frames[f_range[0]:f_range[1]]

    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=sensitivity, detectShadows=False)
    results = []

    for frame_name in tqdm(frames, desc="Scanning"):
        img = cv2.imread(os.path.join(frames_dir, frame_name))
        mask = back_sub.apply(cv2.GaussianBlur(img, (5, 5), 0))
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            x, y, w, h = cv2.boundingRect(cnt)
            smear = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if smear >= min_smear:
                results.append({"frame": frame_name, "x": x+w//2, "y": y+h//2, "area": area, "smear_ratio": round(smear, 2)})

    pd.DataFrame(results).to_csv(os.path.join(analysis_dir, "transit_log.csv"), index=False)
    click.echo("✅ Scan complete. Results in 03_ANALYSIS.")

# --- REPORTING GROUP ---
@cli.group()
def report():
    """Visualization and AI-assisted reporting."""
    pass

@report.command()
@click.argument('project_dir', type=click.Path(exists=True))
@click.option('--top', default=5, help="Number of targets to crop.")
@click.option('--enhance', is_flag=True, help="Apply Jet heatmap to crops.")
def crops(project_dir, top, enhance):
    """Generate high-contrast sub-pixel crops of top candidates."""
    log_path = os.path.join(project_dir, "03_ANALYSIS/transit_log.csv")
    frames_dir = os.path.join(project_dir, "02_FRAMES")
    figures_dir = os.path.join(project_dir, "05_FIGURES/CROPS")
    
    if not os.path.exists(log_path): return
    os.makedirs(figures_dir, exist_ok=True)
    df = pd.read_csv(log_path).sort_values(by='smear_ratio', ascending=False).head(top)
    
    for idx, row in df.iterrows():
        img = cv2.imread(os.path.join(frames_dir, row['frame']))
        x, y = int(row['x']), int(row['y'])
        crop = img[max(0, y-100):min(img.shape[0], y+100), max(0, x-100):min(img.shape[1], x+100)]
        
        if enhance:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.applyColorMap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)
        
        cv2.imwrite(os.path.join(figures_dir, f"hvt_{idx+1}_{row['frame']}"), crop)
    click.echo(f"🏆 {top} crops saved.")

@report.command()
@click.argument('project_dir', type=click.Path(exists=True))
def heatmap(project_dir):
    """Generate a detection density map."""
    log_path = os.path.join(project_dir, "03_ANALYSIS/transit_log.csv")
    frames_dir = os.path.join(project_dir, "02_FRAMES")
    df = pd.read_csv(log_path)
    base = cv2.imread(os.path.join(frames_dir, sorted(os.listdir(frames_dir))[0]))
    
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(base, cv2.COLOR_BGR2RGB), alpha=0.5)
    plt.hexbin(df['x'], df['y'], gridsize=30, cmap='inferno', mincnt=1)
    plt.axis('off')
    plt.savefig(os.path.join(project_dir, "05_FIGURES/heatmap.png"), bbox_inches='tight')
    click.echo("🏆 Heatmap generated.")

@report.command()
@click.argument('project_dir', type=click.Path(exists=True))
def prompt(project_dir):
    """Generate a dynamic, locale-specific LLM Forensic Prompt."""
    config = get_config(project_dir)
    log_path = os.path.join(project_dir, "03_ANALYSIS/transit_log.csv")
    df = pd.read_csv(log_path).sort_values(by='smear_ratio', ascending=False).head(3)
    
    hvt_text = "\n".join([f"- Frame {r['frame']}: Smear {r['smear_ratio']}, Area {r['area']}" for _, r in df.iterrows()])
    
    dynamic_prompt = f"""
FORENSIC DATA FOR AI ANALYSIS:
Location: {config.get('location')} | Sensor FPS: {config.get('sensor', {}).get('fps')}
Project ID: {config.get('project_id')}

Top 3 Kinematic Targets:
{hvt_text}

MISSION: As a physics-based UAP forensic analyst, evaluate these targets for hypersonic potential. 
Ignore biologics with smear < 2.0. Focus on high-smear linear transits.
    """
    click.echo("-" * 30 + "\n" + dynamic_prompt + "\n" + "-" * 30)
    click.echo("📝 Copy the text above into your LLM.")

@report.command()
@click.argument('project_dir', type=click.Path(exists=True))
def brief(project_dir):
    """Generate professional PDF forensic brief."""
    log_path = os.path.join(project_dir, "03_ANALYSIS/transit_log.csv")
    df = pd.read_csv(log_path)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"LUMICRON BRIEF: {os.path.basename(project_dir)}", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Total Detections: {len(df)}", ln=True)
    pdf.cell(0, 10, f"Max Smear: {df['smear_ratio'].max()}", ln=True)
    pdf.output(os.path.join(project_dir, "04_REPORTS/brief.pdf"))
    click.echo("📄 PDF Brief saved.")

if __name__ == '__main__':
    cli()
