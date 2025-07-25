"""
Intelligent System Orchestration Capability

Intelligent orchestration of complex enterprise systems
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "INTELLIGENT_ORCHESTRATION"
__capability_name__ = "Intelligent System Orchestration"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_intelligent_orchestration",
	"integrates_with_intelligent_orchestration",
	"uses_intelligent_orchestration"
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
