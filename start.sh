#!/bin/bash

# Render start script for NASA Exoplanet AI
echo "🌟 Starting NASA Exoplanet AI server..."

# Set environment variables
export FLASK_ENV=production
export PYTHONPATH=/opt/render/project/src:$PYTHONPATH

# Start the application with gunicorn
echo "🚀 Launching with gunicorn..."
exec gunicorn --bind 0.0.0.0:$PORT app:app --timeout 120 --workers 1 --preload
