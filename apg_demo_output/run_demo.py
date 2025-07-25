#!/usr/bin/env python3
"""
APG Demo Application Runner
===========================

Runs the APG-generated Flask-AppBuilder application with setup instructions.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_and_run():
    """Set up and run the APG demo application"""
    
    print("🚀 APG Functional Output Demo")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found!")
        print("   Make sure you're in the apg_demo_output directory")
        return False
    
    print("📦 Setting up the application...")
    
    # Install requirements if possible
    try:
        print("   Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("   ✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠ Warning: Could not install requirements automatically")
        print(f"   Please run: pip install -r requirements.txt")
        print()
    
    print("🌟 Starting APG Flask-AppBuilder Application...")
    print()
    print("📱 Features available:")
    print("   • Interactive agent dashboard")
    print("   • Real-time agent control (start/stop)")
    print("   • Method execution via web interface")
    print("   • Live status monitoring")
    print("   • Activity logging")
    print("   • Database table management")
    print()
    print("🔑 Default login credentials:")
    print("   Username: admin")
    print("   Password: admin")
    print()
    print("🌐 Access the application at: http://localhost:8080")
    print("   Navigate to: Agents > TaskManagerAgent")
    print()
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Run the Flask application
        os.system("python app.py")
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped successfully")
        return True

if __name__ == "__main__":
    setup_and_run()
