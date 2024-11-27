import streamlit.cli as stcli
import sys
import os

if __name__ == "__main__":
    # Add src directory to Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    
    # Run the Streamlit app
    sys.argv = ["streamlit", "run", "src/ui/streamlit_app.py"]
    sys.exit(stcli.main())
