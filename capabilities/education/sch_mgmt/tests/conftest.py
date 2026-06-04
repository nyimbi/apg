import sys
import os

# Ensure the sch_mgmt package directory is first on path for bare imports
_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg not in sys.path:
	sys.path.insert(0, _pkg)
