from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "shared_checkpoints"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from vlm_gui_app import main


if __name__ == "__main__":
    main()
