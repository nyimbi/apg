"""
APG Common ERP Capabilities

Enterprise-grade common capabilities following canonical ERP architecture
with standardized 4-character codes for maximum interoperability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

__version__ = "2.0.0"

import importlib
import logging

logger = logging.getLogger(__name__)

_SUBCAPABILITY_MODULES = [
	"agnt", "aicr", "audl", "conf", "mten", "usrm",
	"conn", "apig", "mqeb", "etlp", "mdm", "meta", "imex", "regy", "dvrl",
	"srch", "ragn", "grag", "grph", "kngr", "onto",
	"cvsn", "frec", "pose", "nlpc", "recs", "anom", "pred", "mlcm", "fedl",
	"colb", "chat", "vidc", "ntfy", "help", "esgn",
	"wflo", "schd", "scpt", "ncod",
	"cach", "moni", "logt", "hlth", "depl", "dist", "edge", "bkup", "cicd",
	"envm", "shdn",
	"geos", "i18n", "walt", "wsbl", "scrp", "mchn", "them", "accs", "cons",
	"plgn", "sbox", "audp",
	"dtwn", "iotd", "bclg", "esgc", "quan",
	"seop", "plfd", "tens",
]

_UNAVAILABLE_SUBCAPABILITIES: dict[str, str] = {}


def _safe_export(module_name: str) -> None:
	"""Import a common subcapability without breaking unrelated imports."""

	try:
		module = importlib.import_module(f"{__name__}.{module_name}")
	except Exception as exc:
		_UNAVAILABLE_SUBCAPABILITIES[module_name] = f"{type(exc).__name__}: {exc}"
		logger.debug("Common subcapability %s unavailable: %s", module_name, exc)
		return

	globals()[module_name] = module
	for exported_name in getattr(module, "__all__", []):
		globals()[exported_name] = getattr(module, exported_name)


for _module_name in _SUBCAPABILITY_MODULES:
	_safe_export(_module_name)

del _module_name

__all__ = [
    # Core Infrastructure
    "agnt", "aicr", "auth", "audl", "conf", "mten", "usrm",
    
    # Security & Compliance
    "secu", "mfau", "biop", "encr", "keym", "comp", "idfd", "dlpd", "ztna",
    
    # Data & Integration
    "conn", "apig", "mqeb", "etlp", "mdm", "meta", "imex", "regy", "dvrl",
    
    # Search & Knowledge
    "srch", "ragn", "grag", "grph", "kngr", "onto",
    
    # AI & Machine Learning
    "cvsn", "frec", "pose", "nlpc", "recs", "anom", "pred", "mlcm", "fedl",
    
    # Collaboration & Communication
    "colb", "chat", "vidc", "ntfy", "help", "esgn",
    
    # Workflow & Automation
    "wflo", "schd", "scpt", "ncod",
    
    # Infrastructure & Operations
    "cach", "moni", "logt", "hlth", "depl", "dist", "edge", "bkup", "cicd", 
    "envm", "shdn",
    
    # Specialized Services
    "geos", "i18n", "walt", "wsbl", "scrp", "mchn", "them", "accs", "cons", 
    "plgn", "sbox", "audp",
    
    # Emerging Technologies
    "dtwn", "iotd", "bclg", "esgc", "quan",
    
    # Legacy/Special
    "seop", "plfd", "tens",
]
