"""
Government & Public Sector (GPS) Industry Vertical

Government and public sector capabilities including citizen services,
case management, regulatory compliance, and public administration.
"""

__version__ = "1.0.0"

# Government sub-modules with 3-character naming
from .csr import *  # Citizen Services
from .cas import *  # Case Management
from .lic import *  # Licensing & Permits
from .tax import *  # Tax Administration
from .ele import *  # Elections Management
from .bud import *  # Public Budgeting
from .con import *  # Contract Management
from .law import *  # Law Enforcement
from .per import *  # Personnel Management
from .eme import *  # Emergency Management

__all__ = [
    "csr",  # Citizen Services
    "cas",  # Case Management
    "lic",  # Licensing & Permits
    "tax",  # Tax Administration
    "ele",  # Elections Management
    "bud",  # Public Budgeting
    "con",  # Contract Management
    "law",  # Law Enforcement
    "per",  # Personnel Management
    "eme",  # Emergency Management
]
