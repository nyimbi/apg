"""
Singer.io Tap for Oracle ERP Systems
Comprehensive Oracle ERP data extraction

Supported Oracle Systems:
- Oracle Cloud ERP
- Oracle Fusion Applications
- Oracle E-Business Suite
- Oracle JD Edwards EnterpriseOne
- Oracle PeopleSoft Enterprise

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from .tap import TapOracle
from .client import OracleClient
from .streams import STREAM_MAPS

__version__ = "1.0.0"
__all__ = ["TapOracle", "OracleClient", "STREAM_MAPS"]