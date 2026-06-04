"""Configure pytest for capability-level test runs."""
import sys
from pathlib import Path

# Add capability root to path so imports work both as package and standalone
_cap_root = Path(__file__).parent.parent
if str(_cap_root) not in sys.path:
    sys.path.insert(0, str(_cap_root))

# Also add project root
_project_root = _cap_root.parent.parent
while _project_root.name != 'capabilities' and _project_root != _project_root.parent:
    _project_root = _project_root.parent
_project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
