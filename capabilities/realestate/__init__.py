"""
Real Estate & Facilities (REF) Industry Vertical

Real estate and facility management including property management,
leasing operations, maintenance scheduling, and tenant services.
"""

__version__ = "1.0.0"

# Real Estate sub-modules with 3-character naming
from .prm import *  # Property Management
from .lea import *  # Leasing Management
from .ten import *  # Tenant Services
from .mai import *  # Maintenance & Repairs
from .spa import *  # Space Management
from .acc import *  # Accounting & Billing
from .ren import *  # Rent Roll Management
from .val import *  # Property Valuation
from .con import *  # Construction Management
from .ins import *  # Insurance Management

__all__ = [
    "prm",  # Property Management
    "lea",  # Leasing Management
    "ten",  # Tenant Services
    "mai",  # Maintenance & Repairs
    "spa",  # Space Management
    "acc",  # Accounting & Billing
    "ren",  # Rent Roll Management
    "val",  # Property Valuation
    "con",  # Construction Management
    "ins",  # Insurance Management
]
