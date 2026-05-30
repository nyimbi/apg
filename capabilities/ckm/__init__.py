"""Collaboration and Knowledge Management capability namespace."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


__version__ = "1.0.0"

_SUBMODULES = {
	"doc": "Document Collaboration",
	"ecn": "Enterprise Content Management",
	"kbs": "Knowledge Base System",
	"kno": "Knowledge Management",
	"lea": "Learning and Training",
	"not": "Notification System",
	"rtc": "Real-Time Collaboration",
	"soc": "Social Collaboration",
	"tct": "Team Collaboration Tools",
	"tra": "Translation Services",
	"wfa": "Workflow Automation",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name: str) -> ModuleType:
	if name in _SUBMODULES:
		module = import_module(f"{__name__}.{name}")
		globals()[name] = module
		return module
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
