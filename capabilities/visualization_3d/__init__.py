"""
3D Visualization & Rendering Capability

3D visualization, rendering, and interactive graphics
"""

from .service import *

__version__ = "1.0.0"
__capability_code__ = "VISUALIZATION_3D"
__capability_name__ = "3D Visualization & Rendering"

# Capability composition keywords for integration with other capabilities
__composition_keywords__ = [
	"requires_visualization_3d",
	"integrates_with_visualization_3d",
	"uses_visualization_3d"
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
