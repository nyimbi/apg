"""
Singer.io Tap for Workday HCM
Comprehensive Workday data extraction

Supported Workday Modules:
- Workday Human Capital Management
- Workday Financial Management
- Workday Planning
- Workday Analytics
- Workday Student
- Workday Adaptive Planning

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from .tap import TapWorkday
from .client import WorkdayClient
from .streams import STREAM_MAPS

__version__ = "1.0.0"
__all__ = ["TapWorkday", "WorkdayClient", "STREAM_MAPS"]