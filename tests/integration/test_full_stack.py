"""Full-stack integration tests: real HTTP against a spawned generated app."""

from __future__ import annotations

import requests


def test_livez(running_app):
	r = requests.get(f"{running_app}/livez", timeout=5)
	assert r.status_code == 200
	assert r.json().get("status") == "ok"


def test_readyz_after_request(running_app):
	requests.get(f"{running_app}/livez", timeout=5)
	r = requests.get(f"{running_app}/readyz", timeout=5)
	assert r.status_code == 200
	assert r.json().get("status") == "ready"


def test_create_and_list_product(running_app):
	payload = {"name": "IntTestWidget", "price": 9.99}
	r = requests.post(f"{running_app}/records/Product", json=payload, timeout=5)
	assert r.status_code in (200, 201)
	r2 = requests.get(f"{running_app}/records/Product", timeout=5)
	assert r2.status_code == 200
	body = r2.json()
	records = body.get("data") or body.get("records") or []
	assert any(str(rec.get("name")) == "IntTestWidget" for rec in records)


def test_openapi_spec(running_app):
	r = requests.get(f"{running_app}/openapi.json", timeout=5)
	assert r.status_code == 200
	spec = r.json()
	assert "paths" in spec
	assert "/records/Product" in spec["paths"]


def test_pagination_real_http(running_app):
	for i in range(60):
		requests.post(
			f"{running_app}/records/Product",
			json={"name": f"PageProd{i}", "price": float(i)},
			timeout=5,
		)
	r = requests.get(f"{running_app}/records/Product?limit=10", timeout=5)
	assert r.status_code == 200
	body = r.json()
	records = body.get("data") or body.get("records") or []
	assert len(records) == 10
	assert body.get("next_cursor") or body.get("cursor") or body.get("next")


def test_search_endpoint(running_app):
	requests.post(
		f"{running_app}/records/Product",
		json={"name": "Acme Widget", "price": 1.0},
		timeout=5,
	)
	r = requests.get(f"{running_app}/records/Product/search?q=Acme", timeout=5)
	assert r.status_code == 200
	body = r.json()
	records = body if isinstance(body, list) else (body.get("data") or body.get("records") or [])
	assert any("Acme" in str(rec.get("name", "")) for rec in records)


def test_csv_export(running_app):
	r = requests.get(f"{running_app}/records/Product?format=csv", timeout=5)
	assert r.status_code == 200
	assert "text/csv" in r.headers.get("Content-Type", "").lower()
	first_line = r.text.splitlines()[0] if r.text else ""
	assert "," in first_line  # CSV header row
