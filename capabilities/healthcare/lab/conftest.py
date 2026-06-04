"""Pytest conftest: make this capability importable as a package."""
import sys
import os

# Insert the parent of this capability dir so `import ana` (etc.) resolves
# the package, allowing relative imports inside service.py to work.
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
