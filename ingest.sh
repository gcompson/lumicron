#!/bin/bash

VIDEO_FILE=$1
PROJECT_NAME=$2

if [ -z "$VIDEO_FILE" ] || [ -z "$PROJECT_NAME" ]; then
    echo "Usage: ./ingest.sh <video_file> <project_name>"
    exit 1
fi

echo "--- INGESTING: $VIDEO_FILE into $PROJECT_NAME ---"

# 1. Create Folder Structure
mkdir -p "$PROJECT_NAME/01_RAW"
mkdir -p "$PROJECT_NAME/02_FRAMES"
mkdir -p "$PROJECT_NAME/03_DATA"
mkdir -p "$PROJECT_NAME/04_REPORTS"

# 2. Copy the original video for archival
cp "$VIDEO_FILE" "$PROJECT_NAME/01_RAW/"

# 3. Extract Frames using FFmpeg
# -q:v 2 ensures high quality
# %03d.png creates frame_001.png, frame_002.png...
ffmpeg -i "$VIDEO_FILE" -q:v 2 "$PROJECT_NAME/02_FRAMES/frame_%03d.png"

echo "--- INGEST COMPLETE ---"
echo "Total frames extracted: $(ls "$PROJECT_NAME/02_FRAMES" | wc -l)"
