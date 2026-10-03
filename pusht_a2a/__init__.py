"""PushT adapters for the pinned upstream A2A policy."""
import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1] / "vendor" / "A2A_Flow_Matching"
sys.path.insert(0, str(UPSTREAM))
