"""
Real-Time Collaboration Capability

Real-time collaboration tools and synchronized workspaces
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "REAL_TIME_COLLABORATION"
__capability_name__ = "Real-Time Collaboration"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_real_time_collaboration",
	"integrates_with_real_time_collaboration",
	"uses_real_time_collaboration"
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
