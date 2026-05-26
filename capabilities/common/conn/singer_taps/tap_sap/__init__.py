"""
Singer.io Tap for SAP Systems
Comprehensive SAP data extraction supporting multiple SAP products

Supported SAP Systems:
- SAP ERP (ECC)
- SAP S/4HANA
- SAP Business One
- SAP SuccessFactors
- SAP Concur
- SAP Ariba
- SAP Fieldglass

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

from .tap import TapSAP
from .client import SAPClient
from .streams import STREAM_MAPS

__version__ = "1.0.0"
__all__ = ["TapSAP", "SAPClient", "STREAM_MAPS"]