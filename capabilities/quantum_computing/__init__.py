"""
Quantum Computing Integration Capability

Quantum computing integration and quantum algorithm support
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "QUANTUM_COMPUTING"
__capability_name__ = "Quantum Computing Integration"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_quantum_computing",
	"integrates_with_quantum_computing",
	"uses_quantum_computing"
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
