"""
Edge Computing & IoT Capability

Edge computing infrastructure and IoT device management
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "EDGE_COMPUTING"
__capability_name__ = "Edge Computing & IoT"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_edge_computing",
	"integrates_with_edge_computing",
	"uses_edge_computing"
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
