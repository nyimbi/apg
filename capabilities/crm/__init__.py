"""
Customer Relationship Management (CRM) Capabilities

Complete customer lifecycle and relationship management
"""

from importlib import import_module

__version__ = "1.0.0"

from .cdp import *
from .mkt import *
from .can import *
from .css import *
from .fsm import *
from .csm import *
from .sfa import *
from .adv import *
from .ord import *  # Order Entry
from .pro import *  # Order Processing
from .pri import *  # Pricing & Discounts
from .quo import *  # Quotations
_sales_forecasting = import_module(".for", __name__)
for _name in getattr(_sales_forecasting, "__all__", ()):
    globals()[_name] = getattr(_sales_forecasting, _name)
globals().pop("_name", None)
del _sales_forecasting

__all__ = [
    "cdp",  # Customer Data Platform
    "mkt",  # Marketing Automation
    "can",  # Customer Analytics
    "css",  # Customer Service/Support
    "fsm",  # Field Service Management
    "csm",  # Contract & Subscription Management
    "sfa",  # Sales Force Automation
    "adv",  # Advanced CRM
    "ord",  # Order Entry
    "pro",  # Order Processing
    "pri",  # Pricing & Discounts
    "quo",  # Quotations
    "for",  # Sales Forecasting
]
