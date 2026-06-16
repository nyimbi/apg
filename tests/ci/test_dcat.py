# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025
#
# Async tests for the dcat (Data Catalog) capability.
# Run with: uv run pytest -vxs tests/ci/test_dcat.py

from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from capabilities.common.dcat.models import (
	Dataset,
	DatasetSearch,
	DatasetTag,
	DatasetStatus,
	LineageEdge,
	LineageEdgeType,
)
from capabilities.common.dcat.service import DataCatalogService

TENANT = "test-tenant"


async def test_register_dataset():
	svc = DataCatalogService()

	ds = Dataset(
		tenant_id=TENANT,
		name="Sales Transactions",
		qualified_name="warehouse.sales.transactions",
		description="Daily sales transaction records",
		owner="data-engineering",
		location="s3://dw/sales/transactions/",
	)

	returned_id = await svc.register_dataset(ds)
	assert returned_id == ds.id

	fetched = await svc.get_dataset(TENANT, ds.id)
	assert fetched is not None
	assert fetched.name == "Sales Transactions"
	assert fetched.qualified_name == "warehouse.sales.transactions"
	assert fetched.owner == "data-engineering"
	assert fetched.status == DatasetStatus.ACTIVE

	# Upsert: update the name and re-register
	ds.name = "Sales Transactions v2"
	await svc.register_dataset(ds)
	updated = await svc.get_dataset(TENANT, ds.id)
	assert updated.name == "Sales Transactions v2"


async def test_add_lineage():
	svc = DataCatalogService()

	raw = Dataset(tenant_id=TENANT, name="Raw Events", qualified_name="lake.raw.events")
	cleaned = Dataset(tenant_id=TENANT, name="Cleaned Events", qualified_name="lake.cleaned.events")
	aggregated = Dataset(tenant_id=TENANT, name="Agg Events", qualified_name="warehouse.agg.events")

	for ds in (raw, cleaned, aggregated):
		await svc.register_dataset(ds)

	edge1 = LineageEdge(
		tenant_id=TENANT,
		source_id=raw.id,
		target_id=cleaned.id,
		edge_type=LineageEdgeType.TRANSFORMS,
		process_name="clean_events_job",
	)
	edge2 = LineageEdge(
		tenant_id=TENANT,
		source_id=cleaned.id,
		target_id=aggregated.id,
		edge_type=LineageEdgeType.AGGREGATES,
		process_name="agg_events_job",
	)

	await svc.add_lineage(edge1)
	await svc.add_lineage(edge2)

	# Duplicate should be silently skipped
	await svc.add_lineage(edge1)

	graph = await svc.get_lineage(raw.id, TENANT, depth=3)

	assert graph["baseEntityGuid"] == raw.id
	node_ids = {n["guid"] for n in graph["nodes"]}
	assert raw.id in node_ids
	assert cleaned.id in node_ids
	assert aggregated.id in node_ids

	edge_pairs = {(e["fromEntityId"], e["toEntityId"]) for e in graph["edges"]}
	assert (raw.id, cleaned.id) in edge_pairs
	assert (cleaned.id, aggregated.id) in edge_pairs

	# Exactly 2 unique edges despite the duplicate submission
	assert len(graph["edges"]) == 2


async def test_search_datasets():
	svc = DataCatalogService()

	ds_a = Dataset(
		tenant_id=TENANT,
		name="Customer Profiles",
		qualified_name="crm.customers.profiles",
		owner="crm-team",
		tags=[DatasetTag(key="domain", value="crm", tenant_id=TENANT)],
	)
	ds_b = Dataset(
		tenant_id=TENANT,
		name="Product Catalogue",
		qualified_name="ecom.products.catalogue",
		owner="ecom-team",
		tags=[DatasetTag(key="domain", value="ecom", tenant_id=TENANT)],
	)
	ds_c = Dataset(
		tenant_id=TENANT,
		name="Order History",
		qualified_name="ecom.orders.history",
		owner="ecom-team",
		tags=[DatasetTag(key="domain", value="ecom", tenant_id=TENANT)],
	)

	for ds in (ds_a, ds_b, ds_c):
		await svc.register_dataset(ds)

	# Free-text search
	results = await svc.search_datasets(DatasetSearch(tenant_id=TENANT, query="customer"))
	names = [r.name for r in results]
	assert "Customer Profiles" in names
	assert "Product Catalogue" not in names

	# Tag filter
	ecom_results = await svc.search_datasets(
		DatasetSearch(tenant_id=TENANT, tag_key="domain", tag_value="ecom")
	)
	assert len(ecom_results) == 2
	assert all(r.owner == "ecom-team" for r in ecom_results)

	# Owner filter
	crm_results = await svc.search_datasets(DatasetSearch(tenant_id=TENANT, owner="crm-team"))
	assert len(crm_results) == 1
	assert crm_results[0].name == "Customer Profiles"

	# Pagination
	page = await svc.search_datasets(DatasetSearch(tenant_id=TENANT, limit=2, offset=0))
	assert len(page) == 2

	# Cross-tenant isolation: different tenant sees nothing
	other = await svc.search_datasets(DatasetSearch(tenant_id="other-tenant"))
	assert other == []


if __name__ == "__main__":
	loop = asyncio.get_event_loop()
	loop.run_until_complete(test_register_dataset())
	loop.run_until_complete(test_add_lineage())
	loop.run_until_complete(test_search_datasets())
	print("All dcat tests passed.")
