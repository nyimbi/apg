"""Domain adapters for SASRA Regulatory Reporting.

Provides adapters that connect the regulatory service to external data sources:
  - NATS event adapter for audit trail publishing
  - Ledger adapter for pulling live balances from dep/lnd services
  - SASRA portal adapter stub for direct XML submission
"""
from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Any

_log = logging.getLogger(__name__)


def get_audit_adapter(capability_id: str = "fintech_sacco_reg"):
	"""Return NATS event adapter if NATS_URL env var is set, else None."""
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	return None


class LedgerAdapter:
	"""Pulls balance sheet and income data from SACCO dep/lnd services.

	In production, replace _fetch_* with live queries to SaccoDepositsService
	and SaccoLendingService.  In tests, use SACCARegulatoryService.seed_ledger().
	"""

	def __init__(self, tenant_id: str) -> None:
		self.tenant_id = tenant_id

	async def fetch_snapshot(self, as_of_date: str) -> dict[str, Any]:
		"""Return a ledger snapshot dict compatible with SACCARegulatoryService.seed_ledger()."""
		try:
			from capabilities.fintech.sacco.dep.service import SaccoDepositsService
			from capabilities.fintech.sacco.lnd.service import SaccoLendingService

			dep_svc = SaccoDepositsService(self.tenant_id)
			lnd_svc = SaccoLendingService(self.tenant_id)

			dep_summary = await dep_svc.portfolio_summary(self.tenant_id)
			lnd_summary = await lnd_svc.portfolio_summary(self.tenant_id)

			# Map dep summary → ledger fields
			snapshot: dict[str, Any] = {
				"member_deposits": Decimal(str(dep_summary.get("total_balance", 0))),
				"gross_loan_portfolio": Decimal(str(lnd_summary.get("total_outstanding_balance", 0))),
				"total_arrears_amount": Decimal(str(lnd_summary.get("total_arrears_amount", 0))),
			}
			return snapshot
		except Exception as exc:
			_log.debug("LedgerAdapter.fetch_snapshot unavailable: %s", exc)
			return {}


class SASRAPortalAdapter:
	"""Stub for SASRA online portal XML submission.

	The actual SASRA portal (https://portal.sasra.go.ke) requires manual
	authentication and file upload. This adapter prepares the payload.
	"""

	def __init__(self, portal_url: str | None = None) -> None:
		self.portal_url = portal_url or os.environ.get(
			"SASRA_PORTAL_URL", "https://portal.sasra.go.ke"
		)

	async def submit_xml(self, xml_content: str, sacco_name: str, period: str) -> dict[str, Any]:
		"""Prepare submission payload. Actual upload requires SASRA portal credentials."""
		_log.info("[SASRA Portal] Would submit %s bytes for %s period=%s to %s",
			len(xml_content), sacco_name, period, self.portal_url)
		return {
			"status": "prepared",
			"portal_url": self.portal_url,
			"sacco_name": sacco_name,
			"period": period,
			"payload_bytes": len(xml_content),
			"note": "Upload via SASRA portal manually or via portal API when credentials available",
		}
