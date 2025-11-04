# src/slither_tools/runner.py
import subprocess
from pathlib import Path

def run_slither_once(sol_path: Path, report_path: Path):
    cmd = [
        "python", "-m", "slither",
        str(sol_path),
        "--json", str(report_path),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
