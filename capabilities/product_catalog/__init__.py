"""
Advanced Product Catalog Capability

Advanced product catalog with AI-powered recommendations
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "PRODUCT_CATALOG"
__capability_name__ = "Advanced Product Catalog"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_product_catalog",
	"integrates_with_product_catalog",
	"uses_product_catalog"
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
