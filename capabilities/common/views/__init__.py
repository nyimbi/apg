"""Cross-capability view composition for APG.

Provides ComposedView: a base class that assembles screen models from
multiple capabilities in parallel using asyncio.gather(), equivalent to
Salesforce's 360-degree customer view.

Usage::

    from capabilities.common.views import ComposedView

    class PaymentDetailView(ComposedView):
        async def build(self, payment_id: str, tenant_id: str) -> dict:
            return await self.compose(
                tenant_id=tenant_id,
                sources=[
                    ("fintech_gateway", "get_payment",   {"payment_id": payment_id}),
                    ("fintech_fraud",   "get_alerts",    {"transaction_id": payment_id}),
                    ("fintech_kyc",     "get_profile",   {}),   # canonical_id auto-resolved
                    ("fintech_aml",     "get_flags",     {"entity_id": payment_id}),
                ],
                canonical_id_source=("fintech_gateway", "get_payment", "customer_id"),
            )
"""
from .composition import ComposedView, ComposedSource

__all__ = ["ComposedView", "ComposedSource"]
