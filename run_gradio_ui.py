#!/usr/bin/env python3
"""
Entrypoint to run the Gradio web UI. Use this when "python -m demo.gradio_ui" fails (e.g. in Docker).
  python run_gradio_ui.py --model-dir output_model --host 0.0.0.0
"""
import sys
from pathlib import Path

# Ensure project root is on path (this script lives at project root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo.gradio_ui import main

if __name__ == "__main__":
    main()
