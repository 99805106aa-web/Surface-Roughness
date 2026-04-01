import os
import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def find_app_dir() -> Path:
    for path in ROOT_DIR.iterdir():
        if not path.is_dir():
            continue
        if not (path / "app.py").exists():
            continue
        if not (path / "logic").is_dir():
            continue
        if not (path / "ISO 22514-7 Study 1_Rev02.py").exists():
            continue
        return path
    raise FileNotFoundError("Could not find the MSA Streamlit app directory.")


APP_DIR = find_app_dir()
APP_FILE = APP_DIR / "app.py"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

os.chdir(APP_DIR)
runpy.run_path(str(APP_FILE), run_name="__main__")
