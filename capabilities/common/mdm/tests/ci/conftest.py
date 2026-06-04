import sys
from pathlib import Path
_cap = Path(__file__).parent.parent.parent
_proj = _cap
while _proj.name != "capabilities" and _proj != _proj.parent:
    _proj = _proj.parent
_proj = _proj.parent
for p in [str(_cap), str(_proj)]:
    if p not in sys.path:
        sys.path.insert(0, p)
