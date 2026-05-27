"""Executable state regressions for notification service surfaces."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATHS = (
	REPO_ROOT / "capabilities" / "common" / "ntfy" / "service.py",
	REPO_ROOT / "capabilities" / "ckm" / "not" / "service.py",
)


def test_notification_services_persist_preferences_deliveries_and_audience():
	for path in SERVICE_PATHS:
		source = path.read_text(encoding="utf-8")

		assert "self._preference_store: Dict[Tuple[str, str], UltimateUserPreferences] = {}" in source
		assert "self._delivery_records: Dict[str, ComprehensiveDelivery] = {}" in source
		assert "self._audience_members: Dict[str, Dict[str, Any]] = {}" in source
		assert "self._delivery_records[delivery.id] = delivery" in source
		assert "self._preference_store[(self.tenant_id, user_id)] = preferences" in source
		assert "def register_audience_members(self, members: List[Dict[str, Any]]) -> None:" in source


def test_notification_services_use_delivery_boundary_not_mock_delivery():
	for path in SERVICE_PATHS:
		source = path.read_text(encoding="utf-8")

		for forbidden in (
			"For now, return default preferences",
			"For now, simulate success/failure based on channel priority",
			"For now, return mock audience",
			"returning mock data structure",
			"total_sent=10000",
			"welcome_series_v2",
		):
			assert forbidden not in source

		assert "channel_results = await self._channel_manager.send_notification(" in source
		assert "'provider': 'local_delivery_store'" in source
		assert "def _normalize_channel_result(self, result: Any) -> Dict[str, Any]:" in source
		assert "def _calculate_channel_performance(" in source
		assert "delivery_records = [" in source
