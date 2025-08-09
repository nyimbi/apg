"""
Manufacturing & Production (MFG) Capabilities

Advanced manufacturing execution and planning systems
"""

__version__ = "1.0.0"

from .plm import *
from .bom import *
from .ppl import *
from .sfc import *
from .mes import *
from .qms import *
from .mro import *
from .mrp import *
from .cap import *
from .rfm import *
from .aps import *  # Advanced Planning & Scheduling
from .cam import *  # Computer-Aided Manufacturing
from .lmt import *  # Lean Manufacturing Tools
from .pco import *  # Product Costing

__all__ = [
    "plm",  # Product Lifecycle Management
    "bom",  # Bill of Materials
    "ppl",  # Production Planning
    "sfc",  # Shop Floor Control
    "mes",  # Manufacturing Execution System
    "qms",  # Quality Management System
    "mro",  # Maintenance, Repair, Operations
    "mrp",  # Material Requirements Planning
    "cap",  # Capacity Planning
    "rfm",  # Recipe/Formula Management
    "aps",  # Advanced Planning & Scheduling
    "cam",  # Computer-Aided Manufacturing
    "lmt",  # Lean Manufacturing Tools
    "pco",  # Product Costing
]
