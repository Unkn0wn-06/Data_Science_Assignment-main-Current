"""Make the repository package importable for direct ``python scripts/...`` runs."""

from pathlib import Path
import sys


# Direct script execution places ``scripts`` rather than the repository on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)
if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)

