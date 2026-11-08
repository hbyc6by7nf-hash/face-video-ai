#!/usr/bin/env python3
"""python run.py  -> CLI (see --help);  python app.py -> web interface."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from facevideo.cli import main  # noqa: E402

raise SystemExit(main())
