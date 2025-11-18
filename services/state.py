import json
import shutil
from typing import Optional, Dict, Any

from config import DATA_DIR, STATE_FILE


def reset_workspace() -> None:
    """Delete all data for a fresh run (called on new upload)."""
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_state() -> Optional[Dict[str, Any]]:
    if not STATE_FILE.exists():
        return None
    with STATE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)
