"""
Singer.io Tap for Sage ERP Systems
Comprehensive Sage ERP data extraction

Supported Sage Systems:
- Sage X3
- Sage 100 (MAS 90/200)
- Sage 300 (AccPac)
- Sage Intacct
- Sage People
- Sage Fixed Assets

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from .tap import TapSage
from .client import SageClient
from .streams import STREAM_MAPS

__version__ = "1.0.0"
__all__ = ["TapSage", "SageClient", "STREAM_MAPS"]