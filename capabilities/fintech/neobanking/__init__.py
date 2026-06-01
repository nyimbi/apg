"""APG Digital Neobanking executable capability package."""

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, get_capability_contract
from .service import DigitalNeobankingService, NeobankingService

__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"DigitalNeobankingService",
	"NeobankingService",
	"get_capability_contract",
]
