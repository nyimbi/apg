"""
IoT Device Management Capability

Comprehensive IoT device lifecycle and data management
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "IOT_MANAGEMENT"
__capability_name__ = "IoT Device Management"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_iot_management",
	"integrates_with_iot_management",
	"uses_iot_management"
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
