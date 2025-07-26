"""
Predictive Maintenance & Analytics Capability

Predictive maintenance using AI and sensor analytics
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "PREDICTIVE_MAINTENANCE"
__capability_name__ = "Predictive Maintenance & Analytics"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_predictive_maintenance",
	"integrates_with_predictive_maintenance",
	"uses_predictive_maintenance"
]

# Import views if they exist
try:
	from .views import *
except ImportError:
	pass

try:
	from .enhanced_views import *
except ImportError:
	pass
