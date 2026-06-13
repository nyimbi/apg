"""
APG Common Tax Engine

Horizontal tax capability covering VAT/GST, Withholding Tax, Excise, PAYE,
and Corporate Tax across 54 African jurisdictions plus global treaty rules.

Subcapabilities:
  calc  - Core tax calculation service (horizontal integration point)
  vat   - VAT/GST country rule packs
  wht   - Withholding Tax certificates and returns

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

__version__ = "1.0.0"

import importlib
import logging

logger = logging.getLogger(__name__)

_SUBCAPABILITY_MODULES = ["calc", "vat", "wht"]
_UNAVAILABLE_SUBCAPABILITIES: dict[str, str] = {}


def _safe_export(module_name: str) -> None:
	try:
		module = importlib.import_module(f"{__name__}.{module_name}")
	except Exception as exc:
		_UNAVAILABLE_SUBCAPABILITIES[module_name] = f"{type(exc).__name__}: {exc}"
		logger.debug("Tax subcapability %s unavailable: %s", module_name, exc)
		return
	globals()[module_name] = module
	for exported_name in getattr(module, "__all__", []):
		globals()[exported_name] = getattr(module, exported_name)


for _module_name in _SUBCAPABILITY_MODULES:
	_safe_export(_module_name)

del _module_name

__all__ = ["calc", "vat", "wht"]
