"""Compatibility package for the APG source-tree layout.

The repository currently keeps its core packages at the project root
(`compiler`, `templates`, `language_server`, and related modules).  Public
entry points and tests import them through the `apg.*` namespace, so this
module exposes those root packages under that namespace while the codebase is
being consolidated into a regular package layout.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
import sys

try:
	__version__ = version("apg")
except PackageNotFoundError:
	__version__ = "0.1.0"

_ALIASES = (
	"capabilities",
	"cli",
	"common",
	"compiler",
	"language_server",
	"marketplace",
	"templates",
)

for _name in _ALIASES:
	try:
		sys.modules[f"{__name__}.{_name}"] = importlib.import_module(_name)
	except ModuleNotFoundError:
		# Some optional surfaces may be absent in trimmed installs.
		pass

__all__ = ["__version__", *_ALIASES]
