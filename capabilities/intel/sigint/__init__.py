"""APG Signals Intelligence executable capability package."""

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, get_capability_contract
from .service import IntelSIGINTService, SignalsIntelligenceService

__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"IntelSIGINTService",
	"SignalsIntelligenceService",
	"get_capability_contract",
]

__version__ = CAPABILITY_VERSION
__status__ = "Executable"
