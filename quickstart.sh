#!/bin/bash

# LUMICRON Quick Start Script
# This script automates environment setup and project initialization.

set -e

echo "🛸 Starting Lumicron Setup..."

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ Error: FFmpeg is not installed. FFmpeg is required for video extraction."
    exit 1
fi

# Install the package and dependencies
echo "📦 Installing Lumicron and its dependencies..."
pip install -e .

# Initialize a default project
PROJECT_NAME="Forensic_Project_1"
echo "🏗️  Initializing project: $PROJECT_NAME..."
lumicron init "$PROJECT_NAME"

echo ""
echo "✅ Installation and Initialization Successful!"
echo "--------------------------------------------------------"
echo "Workflow Summary:"
echo "1. Add video to: $PROJECT_NAME/01_RAW/"
echo "2. Extract frames: lumicron extract [video_path] $PROJECT_NAME"
echo "3. Run analysis:   lumicron scan $PROJECT_NAME"
echo "4. Generate brief: lumicron report brief $PROJECT_NAME"
echo "--------------------------------------------------------"
