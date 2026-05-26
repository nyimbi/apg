"""
APG Billing Capability

Comprehensive enterprise billing system with AI-powered intelligence,
real-time usage tracking, and revenue optimization.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from .service import get_billing_service, BillingService
from .models import (
    BLCustomer, BLPlan, BLSubscription, BLUsage, BLInvoice, BLPayment,
    BLPricingRule, BLTax, BLDiscount, BLRevenue,
    CreateSubscriptionRequest, UsageSubmissionRequest, InvoiceGenerationRequest
)

_VIEW_EXPORTS = {
    "BillingCustomerModelView",
    "BillingPlanModelView",
    "BillingSubscriptionModelView",
    "BillingInvoiceModelView",
    "BillingPaymentModelView",
    "BillingUsageModelView",
    "BillingDashboardView",
    "BillingReportsView",
    "BillingCustomerPortalView",
}


def __getattr__(name: str):
    """Lazy-load Flask-AppBuilder views only when the UI layer asks for them."""
    if name in _VIEW_EXPORTS:
        from . import views
        return getattr(views, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0"
__author__ = "Nyimbi Odero <nyimbi@gmail.com>"
__copyright__ = "© 2025 Datacraft"

__all__ = [
    # Service
    "get_billing_service",
    "BillingService",
    
    # Models
    "BLCustomer",
    "BLPlan", 
    "BLSubscription",
    "BLUsage",
    "BLInvoice",
    "BLPayment",
    "BLPricingRule",
    "BLTax",
    "BLDiscount",
    "BLRevenue",
    "CreateSubscriptionRequest",
    "UsageSubmissionRequest", 
    "InvoiceGenerationRequest",
    
    # Views
    "BillingCustomerModelView",
    "BillingPlanModelView",
    "BillingSubscriptionModelView", 
    "BillingInvoiceModelView",
    "BillingPaymentModelView",
    "BillingUsageModelView",
    "BillingDashboardView",
    "BillingReportsView",
    "BillingCustomerPortalView"
]
