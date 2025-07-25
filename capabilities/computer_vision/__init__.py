"""
Computer Vision & Image Analysis Capability

Advanced computer vision, image processing, and visual analysis
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "COMPUTER_VISION"
__capability_name__ = "Computer Vision & Image Analysis"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_computer_vision",
	"integrates_with_computer_vision",
	"uses_computer_vision"
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
