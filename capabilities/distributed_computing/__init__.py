"""
Distributed Computing & Processing Capability

Distributed computing infrastructure and parallel processing
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "DISTRIBUTED_COMPUTING"
__capability_name__ = "Distributed Computing & Processing"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_distributed_computing",
	"integrates_with_distributed_computing",
	"uses_distributed_computing"
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
