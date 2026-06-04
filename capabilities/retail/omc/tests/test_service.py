"""Service tests for retail_omc capability."""

import asyncio
import pytest
from datetime import datetime, timedelta

from ..service import OmcService
from ..models import (
	OmcChannelCreate, OmcCatalogueItemCreate, OmcInventoryRecord,
	OmcCartCreate, OmcCartLineItem, OmcOrderCreate, OmcOrderUpdate,
	OmcReturnCreate, OmcJourneyEventCreate, OmcPricingRuleCreate,
)


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


@pytest.fixture
def svc():
	return OmcService()


@pytest.fixture
def channel(svc):
	return run(svc.create_channel(OmcChannelCreate(
		tenant_id="t1", name="Main Store", channel_type="store", created_by="admin",
	)))


@pytest.fixture
def item(svc):
	return run(svc.create_catalogue_item(OmcCatalogueItemCreate(
		tenant_id="t1", sku="SKU-001", name="Blue T-Shirt",
		base_price=29.99, currency_code="USD", created_by="admin",
	)))


def test_create_channel(svc):
	ch = run(svc.create_channel(OmcChannelCreate(
		tenant_id="t1", name="Online", channel_type="ecommerce", created_by="admin",
	)))
	assert ch.id
	assert ch.channel_type == "ecommerce"


def test_list_channels_empty(svc):
	assert run(svc.list_channels("unknown")) == []


def test_create_catalogue_item(svc):
	item = run(svc.create_catalogue_item(OmcCatalogueItemCreate(
		tenant_id="t1", sku="SKU-XYZ", name="Red Hat", base_price=15.0, created_by="admin",
	)))
	assert item.sku == "SKU-XYZ"
	assert item.base_price == 15.0


def test_get_catalogue_item_by_sku(svc):
	run(svc.create_catalogue_item(OmcCatalogueItemCreate(
		tenant_id="t1", sku="FIND-ME", name="Findable", base_price=9.99, created_by="admin",
	)))
	found = run(svc.get_catalogue_item_by_sku("t1", "FIND-ME"))
	assert found is not None
	assert found.name == "Findable"


def test_set_channel_price(svc, item, channel):
	updated = run(svc.set_channel_price("t1", item.id, channel.id, 24.99))
	assert updated.channel_prices[channel.id] == 24.99


def test_upsert_inventory(svc, item):
	inv = run(svc.upsert_inventory(OmcInventoryRecord(
		tenant_id="t1", sku="SKU-001", location_id="store-01",
		channel_id="ch-01", on_hand_qty=100, reserved_qty=0,
		available_qty=100, safety_stock_qty=10, updated_by="wms",
	)))
	assert inv.available_qty == 100


def test_reserve_inventory(svc, item):
	run(svc.upsert_inventory(OmcInventoryRecord(
		tenant_id="t1", sku="SKU-001", location_id="store-01",
		channel_id="ch-01", on_hand_qty=50, reserved_qty=0,
		available_qty=50, updated_by="wms",
	)))
	ok = run(svc.reserve_inventory("t1", "SKU-001", "store-01", "ch-01", 10))
	assert ok is True
	recs = run(svc.get_inventory("t1", "SKU-001", "store-01"))
	assert recs[0].available_qty == 40


def test_reserve_insufficient_stock(svc):
	run(svc.upsert_inventory(OmcInventoryRecord(
		tenant_id="t1", sku="SCARCE", location_id="s1",
		channel_id="c1", on_hand_qty=5, reserved_qty=0,
		available_qty=5, updated_by="wms",
	)))
	ok = run(svc.reserve_inventory("t1", "SCARCE", "s1", "c1", 99))
	assert ok is False


def test_create_and_get_order(svc, channel):
	items = [OmcCartLineItem(sku="X", quantity=2, unit_price=10.0, line_total=20.0)]
	order = run(svc.create_order(OmcOrderCreate(
		tenant_id="t1", channel_id=channel.id, fulfilment_mode="ship_to_home",
		items=items, payment_method="card", created_by="web",
	)))
	assert order.id
	assert order.status == "confirmed"
	fetched = run(svc.get_order("t1", order.id))
	assert fetched.order_number == order.order_number


def test_click_and_collect_requires_store(svc, channel):
	with pytest.raises(AssertionError, match="store"):
		run(svc.create_order(OmcOrderCreate(
			tenant_id="t1", channel_id=channel.id,
			fulfilment_mode="click_and_collect",
			items=[], payment_method="card", created_by="web",
		)))


def test_cancel_order(svc, channel):
	order = run(svc.create_order(OmcOrderCreate(
		tenant_id="t1", channel_id=channel.id, fulfilment_mode="ship_to_home",
		items=[], payment_method="card", created_by="web",
	)))
	cancelled = run(svc.cancel_order("t1", order.id, "customer request", "cs"))
	assert cancelled.status == "cancelled"


def test_ship_order(svc, channel):
	order = run(svc.create_order(OmcOrderCreate(
		tenant_id="t1", channel_id=channel.id, fulfilment_mode="ship_to_home",
		items=[], payment_method="card", created_by="web",
	)))
	shipped = run(svc.mark_order_shipped("t1", order.id, "TRACK-123", "warehouse"))
	assert shipped.status == "shipped"
	assert shipped.carrier_tracking_number == "TRACK-123"


def test_initiate_return(svc, channel):
	order = run(svc.create_order(OmcOrderCreate(
		tenant_id="t1", channel_id=channel.id, fulfilment_mode="ship_to_home",
		items=[], payment_method="card", created_by="web",
	)))
	ret = run(svc.initiate_return(OmcReturnCreate(
		tenant_id="t1", order_id=order.id, channel_id=channel.id,
		return_reason="damaged", items=[], refund_method="card", created_by="cs",
	)))
	assert ret.id
	assert ret.status == "pending"


def test_pricing_rule_applied(svc):
	run(svc.create_pricing_rule(OmcPricingRuleCreate(
		tenant_id="t1", name="10% off", rule_type="promotional",
		sku_pattern="SKU-001", adjustment_type="percentage",
		adjustment_value=10.0, priority=1,
		valid_from=datetime.utcnow(), created_by="admin",
	)))
	price = run(svc.apply_pricing_rules("t1", "SKU-001", 100.0, "any"))
	assert price == 90.0


def test_tenant_isolation_orders(svc, channel):
	order = run(svc.create_order(OmcOrderCreate(
		tenant_id="t1", channel_id=channel.id, fulfilment_mode="ship_to_home",
		items=[], payment_method="card", created_by="web",
	)))
	assert run(svc.get_order("t2", order.id)) is None
