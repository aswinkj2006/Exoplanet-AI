"""
Quick Start Script for NASA Exoplanet Detection System
Run this to automatically set up and launch the system
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'tensorflow', 
        'flask', 'matplotlib', 'astropy', 'lightkurve'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_models():
    """Check if models are trained"""
    return os.path.exists('models') and len(os.listdir('models')) > 0

def train_models():
    """Train models if not already trained"""
    print("Training AI models... This may take 10-15 minutes.")
    print("☕ Perfect time for a coffee break!")
    
    try:
        subprocess.check_call([sys.executable, 'train_models.py'])
        print("✓ Models trained successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Model training failed")
        return False

def launch_app():
    """Launch the web application"""
    print("🚀 Launching NASA Exoplanet Detection System...")
    print("🌐 Opening web interface at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 Thanks for using NASA Exoplanet Detection System!")

def main():
    """Main startup sequence"""
    print("🚀 NASA EXOPLANET DETECTION SYSTEM")
    print("🌟 Automated Setup & Launch")
    print("=" * 50)
    
    # Check dependencies
    print("1. Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        if not install_dependencies():
            print("Please install dependencies manually: pip install -r requirements.txt")
            return
    else:
        print("✓ All dependencies found")
    
    # Check models
    print("\n2. Checking trained models...")
    if not check_models():
        print("No trained models found. Starting training...")
        if not train_models():
            print("Please run 'python train_models.py' manually")
            return
    else:
        print("✓ Trained models found")
    
    # Launch application
    print("\n3. Starting web application...")
    time.sleep(1)
    launch_app()

if __name__ == "__main__":
    main()
