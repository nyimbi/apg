"""
Blockchain Security & Trust Capability

Blockchain-based security, verification, and decentralized trust systems
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "BLOCKCHAIN_SECURITY"
__capability_name__ = "Blockchain Security & Trust"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_blockchain_security",
	"integrates_with_blockchain_security",
	"uses_blockchain_security"
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
