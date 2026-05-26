"""Pytest configuration for IMEX capability tests."""

from __future__ import annotations

import sys
import types
import importlib
from pathlib import Path


IMEX_ROOT = Path(__file__).resolve().parents[1]
if str(IMEX_ROOT) not in sys.path:
	sys.path.insert(0, str(IMEX_ROOT))

for _short_name in ("models",):
	_package_name = f"capabilities.common.imex.{_short_name}"
	sys.modules[_short_name] = importlib.import_module(_package_name)

if "requests" not in sys.modules:
	requests_stub = types.ModuleType("requests")
	requests_stub.get = lambda *args, **kwargs: types.SimpleNamespace(status_code=200, json=lambda: {})
	requests_stub.post = lambda *args, **kwargs: types.SimpleNamespace(status_code=200, json=lambda: {})
	sys.modules["requests"] = requests_stub

try:
	import flask_appbuilder
except ImportError:
	flask_appbuilder = types.ModuleType("flask_appbuilder")
	flask_appbuilder.AppBuilder = lambda *args, **kwargs: types.SimpleNamespace()
	sys.modules["flask_appbuilder"] = flask_appbuilder

if not hasattr(flask_appbuilder, "SQLA"):
	class SQLA:
		def __init__(self, app=None):
			self.app = app
			self.session = types.SimpleNamespace()

	flask_appbuilder.SQLA = SQLA
