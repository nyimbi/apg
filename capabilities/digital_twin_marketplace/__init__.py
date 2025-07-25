"""
Digital Twin Marketplace Capability

Marketplace for digital twin assets, templates, and services
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "DIGITAL_TWIN_MARKETPLACE"
__capability_name__ = "Digital Twin Marketplace"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_digital_twin_marketplace",
	"integrates_with_digital_twin_marketplace",
	"uses_digital_twin_marketplace"
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
