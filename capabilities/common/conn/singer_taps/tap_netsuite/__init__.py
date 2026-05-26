"""
Singer.io Tap for NetSuite ERP
Comprehensive NetSuite data extraction

Supported NetSuite Modules:
- NetSuite ERP
- NetSuite CRM
- NetSuite Ecommerce
- NetSuite SuiteCommerce
- NetSuite Analytics
- NetSuite Planning & Budgeting

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from .tap import TapNetSuite
from .client import NetSuiteClient
from .streams import STREAM_MAPS

__version__ = "1.0.0"
__all__ = ["TapNetSuite", "NetSuiteClient", "STREAM_MAPS"]