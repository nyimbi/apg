"""APG Predictive Intelligence executable capability package."""

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, get_capability_contract
from .service import IntelPredictionService, PredictiveIntelligenceService

__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"PredictiveIntelligenceService",
	"IntelPredictionService",
	"get_capability_contract",
]

__version__ = CAPABILITY_VERSION
__status__ = "Executable"
