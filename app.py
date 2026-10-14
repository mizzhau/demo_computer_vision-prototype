import os
import sys
from pathlib import Path
import runpy

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute main Streamlit app
main_py_path = ROOT_DIR / "app" / "main.py"
runpy.run_path(str(main_py_path), run_name="__main__")
