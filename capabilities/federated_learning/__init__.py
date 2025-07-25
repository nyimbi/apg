"""
Federated Learning & AI Capability

Federated machine learning and distributed AI training
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "FEDERATED_LEARNING"
__capability_name__ = "Federated Learning & AI"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_federated_learning",
	"integrates_with_federated_learning",
	"uses_federated_learning"
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
