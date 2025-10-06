#!/usr/bin/env python3
"""
Simple runner for Streamlit Resume Processing UI
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit application"""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    streamlit_app_path = script_dir / "streamlit_app.py"
    
    # Check if streamlit_app.py exists
    if not streamlit_app_path.exists():
        print("❌ Error: streamlit_app.py not found!")
        return 1
    
    try:
        print("🚀 Starting Streamlit Resume Processing UI...")
        print("📱 The app will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app_path),
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, cwd=script_dir)
        
    except KeyboardInterrupt:
        print("\n⏹️  Streamlit server stopped by user")
        return 0
    except FileNotFoundError:
        print("❌ Error: Streamlit is not installed!")
        print("💡 Install it with: pip install streamlit")
        return 1
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        return 1

if __name__ == "__main__":
    exit(main())