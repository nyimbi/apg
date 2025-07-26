"""
Multi-Tenant Enterprise Platform Capability

Multi-tenant enterprise architecture and tenant management
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "MULTI_TENANT_ENTERPRISE"
__capability_name__ = "Multi-Tenant Enterprise Platform"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_multi_tenant_enterprise",
	"integrates_with_multi_tenant_enterprise",
	"uses_multi_tenant_enterprise"
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
