"""Federation runtime helpers for the APG IDFD capability."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .models import CertificateRecord, FederatedSession, FederationProvider, ProviderStatus, SessionStatus


def iso_hours_ago(hours: int | float, now: datetime | None = None) -> str:
	now = now or datetime.now(timezone.utc)
	return (now - timedelta(hours=float(hours))).isoformat()


def iso_hours_from_now(hours: int | float, now: datetime | None = None) -> str:
	now = now or datetime.now(timezone.utc)
	return (now + timedelta(hours=float(hours))).isoformat()


def parse_iso_timestamp(value: str) -> datetime:
	parsed = datetime.fromisoformat(value)
	if parsed.tzinfo is None:
		parsed = parsed.replace(tzinfo=timezone.utc)
	return parsed


class MetadataFreshnessInspector:
	"""Inspect federation-provider metadata freshness."""

	def metadata_age_hours(self, provider: FederationProvider, now: datetime | None = None) -> float:
		now = now or datetime.now(timezone.utc)
		try:
			refreshed_at = parse_iso_timestamp(provider.metadata_refreshed_at)
		except ValueError:
			return 0.0
		return max(0.0, (now - refreshed_at).total_seconds() / 3600)

	def stale_providers(
		self,
		providers: list[FederationProvider],
		tenant_id: str,
		threshold_hours: int,
		now: datetime | None = None,
	) -> list[FederationProvider]:
		return [
			provider
			for provider in providers
			if provider.tenant_id == tenant_id
			and provider.status != ProviderStatus.DISABLED
			and self.metadata_age_hours(provider, now) > threshold_hours
		]


class FederationSessionIssuer:
	"""Issue bounded in-process sessions for generated APG applications."""

	def issue(
		self,
		session_id: str,
		tenant_id: str,
		provider_id: str,
		subject_id: str,
		session_privilege: str,
		mfa_completed: bool,
		max_session_hours: int,
		risk_score: float = 0.0,
		now: datetime | None = None,
	) -> FederatedSession:
		now = now or datetime.now(timezone.utc)
		return FederatedSession(
			id=session_id,
			tenant_id=tenant_id,
			provider_id=provider_id,
			subject_id=subject_id,
			session_privilege=session_privilege,
			mfa_completed=mfa_completed,
			issued_at=now.isoformat(),
			expires_at=(now + timedelta(hours=max_session_hours)).isoformat(),
			status=SessionStatus.ACTIVE,
			risk_score=float(risk_score),
		)

	def effective_status(self, session: FederatedSession, now: datetime | None = None) -> SessionStatus:
		if session.status == SessionStatus.REVOKED:
			return SessionStatus.REVOKED
		now = now or datetime.now(timezone.utc)
		try:
			if parse_iso_timestamp(session.expires_at) <= now:
				return SessionStatus.EXPIRED
		except ValueError:
			return SessionStatus.EXPIRED
		return SessionStatus.ACTIVE


class FederationHealthInspector:
	"""Summarize federation operating health for dashboards and audits."""

	def __init__(self) -> None:
		self._metadata = MetadataFreshnessInspector()
		self._sessions = FederationSessionIssuer()

	def summarize(
		self,
		tenant_id: str,
		providers: list[FederationProvider],
		sessions: list[FederatedSession],
		certificates: list[CertificateRecord],
		metadata_refresh_hours: int,
		certificate_rotation_days: int,
		now: datetime | None = None,
	) -> dict[str, int]:
		now = now or datetime.now(timezone.utc)
		stale = self._metadata.stale_providers(providers, tenant_id, metadata_refresh_hours, now)
		active_sessions = [
			session
			for session in sessions
			if session.tenant_id == tenant_id and self._sessions.effective_status(session, now) == SessionStatus.ACTIVE
		]
		expiring_certificates = [
			certificate
			for certificate in certificates
			if certificate.tenant_id == tenant_id
			and certificate.active
			and self._expires_within_days(certificate, certificate_rotation_days, now)
		]
		return {
			"stale_provider_count": len(stale),
			"active_session_count": len(active_sessions),
			"expiring_certificate_count": len(expiring_certificates),
			"metadata_refresh_required_count": len(stale),
		}

	def _expires_within_days(self, certificate: CertificateRecord, days: int, now: datetime) -> bool:
		try:
			expires_at = parse_iso_timestamp(certificate.expires_at)
		except ValueError:
			return True
		return expires_at <= now + timedelta(days=days)
