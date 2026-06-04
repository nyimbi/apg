"""Make the parent capability package importable for relative-import services."""
import sys, os, importlib

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_cap_dir = os.path.dirname(_tests_dir)
_healthcare_dir = os.path.dirname(_cap_dir)

# Add healthcare/ parent so the cap can be imported as a package
if _healthcare_dir not in sys.path:
    sys.path.insert(0, _healthcare_dir)

# Force the capability package to load so relative imports resolve
_cap_name = os.path.basename(_cap_dir)
if _cap_name not in sys.modules:
    importlib.import_module(_cap_name)
