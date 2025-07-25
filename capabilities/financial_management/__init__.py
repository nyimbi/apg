"""
Financial Management & Accounting Capability

Comprehensive financial management, accounting, and reporting
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "FINANCIAL_MANAGEMENT"
__capability_name__ = "Financial Management & Accounting"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_financial_management",
	"integrates_with_financial_management",
	"uses_financial_management"
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
