"""Generated smoke, CRUD, and search tests for this APG app."""

from __future__ import annotations

import pytest


def test_smoke_livez(generated_app_client):
	assert generated_app_client.get('/livez').status_code == 200


def test_crud_inventoryitem(generated_app_client):
	payload = {'sku': 'InventoryItem Wave K Needle', 'description': 'sample description', 'barcode': 'sample barcode', 'notes': 'sample notes', 'quantity_on_hand': 7, 'quantity_reserved': 7, 'quantity_on_order': 7, 'unit_price': 7.5, 'cost_price': 7.5, 'weight_kg': 7.5, 'reorder_point': 7, 'reorder_quantity': 7, 'is_active': True, 'is_serialised': True, 'is_hazardous': True, 'requires_refrigeration': True, 'created_at': 'sample created_at', 'last_counted_at': 'sample last_counted_at', 'best_before': 'sample best_before', 'reorder_time': 'sample reorder_time', 'image_thumbnail': 'sample image_thumbnail', 'categories': [], 'attributes': {}, 'dimensions': {}, 'metadata': {}, 'status': 'sample status', 'warehouse_location': 'sample warehouse_location'}
	created_response = generated_app_client.post('/records/InventoryItem', json={'record': payload})
	assert created_response.status_code == 201, created_response.get_json()
	created = created_response.get_json()['record']
	list_response = generated_app_client.get('/records/InventoryItem')
	assert list_response.status_code == 200, list_response.get_json()
	listed = list_response.get_json()
	records = listed.get('data', listed.get('records', []))
	assert any(str(record.get('id')) == str(created['id']) for record in records)
	delete_response = generated_app_client.delete('/records/InventoryItem/' + str(created['id']))
	assert delete_response.status_code == 200, delete_response.get_json()
	after_delete_response = generated_app_client.get('/records/InventoryItem')
	assert after_delete_response.status_code == 200, after_delete_response.get_json()
	after_delete = after_delete_response.get_json()
	after_records = after_delete.get('data', after_delete.get('records', []))
	assert all(str(record.get('id')) != str(created['id']) for record in after_records)


def test_search_inventoryitem(generated_app_client):
	payload = {'sku': 'InventoryItem Wave K Needle', 'description': 'sample description', 'barcode': 'sample barcode', 'notes': 'sample notes', 'quantity_on_hand': 7, 'quantity_reserved': 7, 'quantity_on_order': 7, 'unit_price': 7.5, 'cost_price': 7.5, 'weight_kg': 7.5, 'reorder_point': 7, 'reorder_quantity': 7, 'is_active': True, 'is_serialised': True, 'is_hazardous': True, 'requires_refrigeration': True, 'created_at': 'sample created_at', 'last_counted_at': 'sample last_counted_at', 'best_before': 'sample best_before', 'reorder_time': 'sample reorder_time', 'image_thumbnail': 'sample image_thumbnail', 'categories': [], 'attributes': {}, 'dimensions': {}, 'metadata': {}, 'status': 'sample status', 'warehouse_location': 'sample warehouse_location'}
	created_response = generated_app_client.post('/records/InventoryItem', json={'record': payload})
	assert created_response.status_code == 201, created_response.get_json()
	created = created_response.get_json()['record']
	search_response = generated_app_client.get('/records/InventoryItem/search', query_string={'q': 'Needle', 'limit': '5'})
	assert search_response.status_code == 200, search_response.get_json()
	matches = search_response.get_json()
	assert isinstance(matches, list)
	assert any(str(record.get('id')) == str(created['id']) for record in matches)
