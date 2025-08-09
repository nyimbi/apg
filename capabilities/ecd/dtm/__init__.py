"""
Digital Twin Management Capability

Digital twin creation, management, and real-time synchronization
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "DIGITAL_TWIN"
__capability_name__ = "Digital Twin Management"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_digital_twin",
	"integrates_with_digital_twin",
	"uses_digital_twin"
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
