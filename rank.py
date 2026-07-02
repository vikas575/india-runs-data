#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    # Get directory of current script
    current_dir = Path(__file__).parent.resolve()
    main_script = current_dir / "main.py"
    
    # Delegate to main.py
    cmd = [sys.executable, str(main_script)] + sys.argv[1:]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)
