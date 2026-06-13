"""APG Developer Portal capability — common/devp.

Provides API key self-service, OpenAPI browser, usage analytics,
API product monetization, and webhook management for the APG ecosystem.

Extends common/apig (API gateway).
"""

from __future__ import annotations

from .capability_contract import CAPABILITY_ID, CAPABILITY_VERSION, get_capability_contract
from .models import (
	DpApiKey,
	DpApiProduct,
	DpDeveloperApp,
	DpSubscription,
	DpUsageStats,
	DpWebhookEndpoint,
)
from .service import DeveloperPortalService

__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_VERSION",
	"DeveloperPortalService",
	"DpApiKey",
	"DpApiProduct",
	"DpDeveloperApp",
	"DpSubscription",
	"DpUsageStats",
	"DpWebhookEndpoint",
	"get_capability_contract",
]
