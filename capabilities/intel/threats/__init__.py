"""APG Threat Intelligence executable capability package."""

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, get_capability_contract
from .service import IntelThreatsService, ThreatIntelligenceService

__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"ThreatIntelligenceService",
	"IntelThreatsService",
	"get_capability_contract",
]

__version__ = CAPABILITY_VERSION
__status__ = "Executable"
