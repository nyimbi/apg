"""
Governance, Risk & Compliance (GRC) Capabilities

Comprehensive GRC modules for enterprise compliance and risk management
"""

__version__ = "1.0.0"

from .rcm import *
from .pol import *
from .rsa import *
from .icm import *
from .aud import *
from .doc import *

__all__ = [
    "rcm",
    "pol",
    "rsa",
    "icm",
    "aud",
    "doc",
]
