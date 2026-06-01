"""APG Intelligence Analytics executable capability package."""

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, get_capability_contract
from .service import IntelligenceAnalyticsService, IntelAnalyticsService

__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"IntelligenceAnalyticsService",
	"IntelAnalyticsService",
	"get_capability_contract",
]

__version__ = CAPABILITY_VERSION
__status__ = "Executable"
