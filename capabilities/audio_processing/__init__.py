"""
Audio Processing & Analysis Capability

Advanced audio processing, analysis, and transformation capabilities
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "AUDIO_PROCESSING"
__capability_name__ = "Audio Processing & Analysis"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_audio_processing",
	"integrates_with_audio_processing",
	"uses_audio_processing"
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
