"""Generated smoke, CRUD, and search tests for this APG app."""

from __future__ import annotations

import pytest


def test_smoke_livez(generated_app_client):
	assert generated_app_client.get('/livez').status_code == 200


def test_crud_workorder(generated_app_client):
	payload = {'work_order_id': 'WorkOrder Wave K Needle', 'product_code': 'sample product_code', 'quantity_planned': 7, 'quantity_produced': 7, 'quantity_rejected': 7, 'status': 'sample status', 'started_at': 'sample started_at', 'completed_at': 'sample completed_at', 'production_line': 'sample production_line'}
	created_response = generated_app_client.post('/records/WorkOrder', json={'record': payload})
	assert created_response.status_code == 201, created_response.get_json()
	created = created_response.get_json()['record']
	list_response = generated_app_client.get('/records/WorkOrder')
	assert list_response.status_code == 200, list_response.get_json()
	listed = list_response.get_json()
	records = listed.get('data', listed.get('records', []))
	assert any(str(record.get('id')) == str(created['id']) for record in records)
	delete_response = generated_app_client.delete('/records/WorkOrder/' + str(created['id']))
	assert delete_response.status_code == 200, delete_response.get_json()
	after_delete_response = generated_app_client.get('/records/WorkOrder')
	assert after_delete_response.status_code == 200, after_delete_response.get_json()
	after_delete = after_delete_response.get_json()
	after_records = after_delete.get('data', after_delete.get('records', []))
	assert all(str(record.get('id')) != str(created['id']) for record in after_records)


def test_search_workorder(generated_app_client):
	payload = {'work_order_id': 'WorkOrder Wave K Needle', 'product_code': 'sample product_code', 'quantity_planned': 7, 'quantity_produced': 7, 'quantity_rejected': 7, 'status': 'sample status', 'started_at': 'sample started_at', 'completed_at': 'sample completed_at', 'production_line': 'sample production_line'}
	created_response = generated_app_client.post('/records/WorkOrder', json={'record': payload})
	assert created_response.status_code == 201, created_response.get_json()
	created = created_response.get_json()['record']
	search_response = generated_app_client.get('/records/WorkOrder/search', query_string={'q': 'Needle', 'limit': '5'})
	assert search_response.status_code == 200, search_response.get_json()
	matches = search_response.get_json()
	assert isinstance(matches, list)
	assert any(str(record.get('id')) == str(created['id']) for record in matches)
