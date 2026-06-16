"""push_journal — transaction journal posting to external core-banking or audit API.

Reads configuration from environment variables:
  JOURNAL_BASE_URL   — base URL of the journal endpoint (required for live posting)
  JOURNAL_ENDPOINT   — path appended to base URL (default: /api/journal/push)
  JOURNAL_API_KEY    — bearer token for Authorization header
  JOURNAL_TIMEOUT    — request timeout in seconds (default: 10)
  JOURNAL_MAX_RETRIES — max retry attempts on transient errors (default: 3)
  JOURNAL_RETRY_BACKOFF — backoff seconds between retries (default: 1.0)

When JOURNAL_BASE_URL is not set the call is a no-op log — safe for development.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)


@dataclass
class APIConfig:
	base_url: str = field(default_factory=lambda: os.environ.get("JOURNAL_BASE_URL", ""))
	endpoint: str = field(default_factory=lambda: os.environ.get("JOURNAL_ENDPOINT", "/api/journal/push"))
	api_key: str = field(default_factory=lambda: os.environ.get("JOURNAL_API_KEY", ""))
	timeout: float = field(default_factory=lambda: float(os.environ.get("JOURNAL_TIMEOUT", "10")))
	max_retries: int = field(default_factory=lambda: int(os.environ.get("JOURNAL_MAX_RETRIES", "3")))
	retry_backoff: float = field(default_factory=lambda: float(os.environ.get("JOURNAL_RETRY_BACKOFF", "1.0")))


def send_push_journal(
	*,
	rrn: str,
	stan: str,
	amount: str,
	account_number: str,
	pan: str,
	status: str,
	terminal_id: str,
	comment: str = "",
	error: str = "",
) -> dict:
	"""Post a transaction journal entry to the configured endpoint.

	Returns the API response dict on success, or a local ack dict when
	JOURNAL_BASE_URL is not configured.
	"""
	config = APIConfig()
	payload = {
		"rrn": rrn,
		"stan": stan,
		"amount": amount,
		"account_number": account_number,
		"pan": pan[-4:] if pan else "",  # never log full PAN
		"status": status,
		"terminal_id": terminal_id,
		"comment": comment,
		"error": error,
	}

	if not config.base_url:
		_log.info("JOURNAL_BASE_URL not set — journal entry logged locally: rrn=%s stan=%s status=%s", rrn, stan, status)
		return {"ack": "local", "rrn": rrn, "stan": stan, "status": status}

	import httpx

	url = config.base_url.rstrip("/") + config.endpoint
	headers = {"Content-Type": "application/json"}
	if config.api_key:
		headers["Authorization"] = f"Bearer {config.api_key}"

	last_exc: Exception | None = None
	for attempt in range(1, config.max_retries + 1):
		try:
			resp = httpx.post(url, json=payload, headers=headers, timeout=config.timeout)
			resp.raise_for_status()
			return resp.json()
		except Exception as exc:
			last_exc = exc
			_log.warning("Journal post attempt %d/%d failed: %s", attempt, config.max_retries, exc)
			if attempt < config.max_retries:
				time.sleep(config.retry_backoff)

	raise RuntimeError(f"Journal post failed after {config.max_retries} attempts: {last_exc}") from last_exc
