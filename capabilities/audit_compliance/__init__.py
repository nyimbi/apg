"""
Audit & Compliance Management Capability

Comprehensive audit logging and regulatory compliance management
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "AUDIT_COMPLIANCE"
__capability_name__ = "Audit & Compliance Management"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_audit_compliance",
	"integrates_with_audit_compliance",
	"uses_audit_compliance"
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
