"""
Transportation & Logistics (TRL) Industry Vertical

Transportation and logistics management including fleet operations,
route optimization, cargo tracking, and supply chain coordination.
"""

__version__ = "1.0.0"

# Transportation sub-modules with 3-character naming
from .fle import *  # Fleet Management
from .rou import *  # Route Optimization
from .dis import *  # Dispatch Management
from .war import *  # Warehouse Operations
from .tra import *  # Cargo Tracking
from .car import *  # Carrier Management
from .del import *  # Delivery Management
from .sch import *  # Scheduling System
from .fue import *  # Fuel Management
from .mai import *  # Maintenance Planning

__all__ = [
    "fle",  # Fleet Management
    "rou",  # Route Optimization
    "dis",  # Dispatch Management
    "war",  # Warehouse Operations
    "tra",  # Cargo Tracking
    "car",  # Carrier Management
    "del",  # Delivery Management
    "sch",  # Scheduling System
    "fue",  # Fuel Management
    "mai",  # Maintenance Planning
]
