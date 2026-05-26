"""
Singer.io Tap for Microsoft Dynamics Systems
Comprehensive Microsoft Dynamics data extraction

Supported Dynamics Systems:
- Dynamics 365 Finance & Operations
- Dynamics 365 Business Central
- Dynamics 365 Sales (CRM)
- Dynamics 365 Customer Service
- Dynamics 365 Marketing
- Dynamics 365 Supply Chain Management
- Dynamics AX (Legacy)
- Dynamics NAV (Legacy)

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from .tap import TapDynamics
from .client import DynamicsClient
from .streams import STREAM_MAPS

__version__ = "1.0.0"
__all__ = ["TapDynamics", "DynamicsClient", "STREAM_MAPS"]