#!/bin/bash

# Render build script for NASA Exoplanet AI
echo "🚀 Starting NASA Exoplanet AI build..."

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p static/plots

echo "✅ Build complete!"
