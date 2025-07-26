"""
Time Series Analytics Capability

Advanced time series analysis and forecasting
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "TIME_SERIES_ANALYTICS"
__capability_name__ = "Time Series Analytics"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_time_series_analytics",
	"integrates_with_time_series_analytics",
	"uses_time_series_analytics"
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
