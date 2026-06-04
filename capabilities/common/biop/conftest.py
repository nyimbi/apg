"""Make biop importable as top-level package for relative imports in tests."""
import sys
from pathlib import Path

# Add parent dir so "biop" is importable as a top-level package
# This allows tests/ci/ to use: from ..models import ...
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
