"""
Collaboration & Knowledge Management (CKM) Capabilities

Enterprise collaboration and knowledge management systems
"""

__version__ = "1.0.0"

from .ecn import *  # Enterprise Content Management
from .kbs import *  # Knowledge Base System
from .wfa import *  # Workflow Automation
from .tct import *  # Team Collaboration Tools
from .not import *  # Notification System
from .rtc import *  # Real-Time Collaboration
from .kno import *  # Knowledge Management
from .doc import *  # Document Collaboration
from .soc import *  # Social Collaboration
from .lea import *  # Learning & Training
from .tra import *  # Translation Services

__all__ = [
    "ecn",  # Enterprise Content Management
    "kbs",  # Knowledge Base System
    "wfa",  # Workflow Automation
    "tct",  # Team Collaboration Tools
    "not",  # Notification System
    "rtc",  # Real-Time Collaboration
    "kno",  # Knowledge Management
    "doc",  # Document Collaboration
    "soc",  # Social Collaboration
    "lea",  # Learning & Training
    "tra",  # Translation Services
]
