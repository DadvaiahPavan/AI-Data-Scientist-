"""
Main entry point for the Streamlit application.
This file imports and runs the UI component from src/ui/streamlit_app.py
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the main function from the UI module
from src.ui.streamlit_app import main

if __name__ == "__main__":
    main()
