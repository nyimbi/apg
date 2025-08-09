"""
Telecommunications (TEL) Industry Vertical

Telecommunications industry capabilities including network management,
customer service, billing systems, and service provisioning.
"""

__version__ = "1.0.0"

# Telecommunications sub-modules with 3-character naming
from .net import *  # Network Management
from .cus import *  # Customer Management
from .bil import *  # Billing & Rating
from .pro import *  # Service Provisioning
from .ord import *  # Order Management
from .inv import *  # Inventory Management
from .per import *  # Performance Monitoring
from .sec import *  # Security Management
from .qos import *  # Quality of Service
from .ana import *  # Network Analytics

__all__ = [
    "net",  # Network Management
    "cus",  # Customer Management
    "bil",  # Billing & Rating
    "pro",  # Service Provisioning
    "ord",  # Order Management
    "inv",  # Inventory Management
    "per",  # Performance Monitoring
    "sec",  # Security Management
    "qos",  # Quality of Service
    "ana",  # Network Analytics
]
