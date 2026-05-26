"""Focused tests for marketplace view catalog integration."""

from capabilities.common.conn.marketplace import CapabilityType, MarketplaceSearchQuery
from capabilities.common.conn.marketplace_views import MarketplaceDashboardView


def test_marketplace_view_search_uses_backend_catalog():
	view = MarketplaceDashboardView()
	results = view._search_catalog(
		MarketplaceSearchQuery(
			query="postgres",
			capability_type=CapabilityType.CONNECTOR,
			tags=["database"],
			limit=5
		)
	)

	assert results["source"] == "local_catalog"
	assert results["total"] == 1
	assert results["capabilities"][0]["id"] == "postgres-connector"


def test_marketplace_view_detail_uses_backend_catalog_versions():
	view = MarketplaceDashboardView()
	capability = view._get_catalog_capability_detail("postgres-connector")

	assert capability["id"] == "postgres-connector"
	assert capability["current_version"] == "2.1.0"
	assert capability["versions"][0]["version"] == "2.1.0"
	assert capability["changelog"][0]["changes"]


def test_marketplace_view_trending_categories_reflect_catalog():
	view = MarketplaceDashboardView()
	categories = view._get_trending_categories()

	category_names = {category["name"] for category in categories}
	assert "Connector" in category_names
	assert "Database" in category_names
