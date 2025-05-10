#!/usr/bin/env python
"""
Launcher script for HIPO Streamlit UI
"""
import os
import sys
import subprocess
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent.parent.absolute()


def main():
    """Run the Streamlit UI app"""
    print(f"Starting HIPO Streamlit UI from {project_root}")

    # Add project root to Python path
    sys.path.append(str(project_root))

    # Run the Streamlit app
    streamlit_path = os.path.join(project_root, "src", "ui", "app.py")
    cmd = ["streamlit", "run", streamlit_path, "--server.port=8501"]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down Streamlit UI")
    except Exception as e:
        print(f"Error running Streamlit UI: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
