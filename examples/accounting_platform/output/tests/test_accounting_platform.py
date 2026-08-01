"""Generated smoke, CRUD, and search tests for this APG app."""

from __future__ import annotations

import pytest


def test_smoke_livez(generated_app_client):
	assert generated_app_client.get('/livez').status_code == 200


def test_crud_chartofaccounts(generated_app_client):
	payload = {'account_code': 'ChartOfAccounts Wave K Needle', 'account_name': 'sample account_name', 'account_type': 'sample account_type', 'normal_balance': 'sample normal_balance', 'currency': 'sample currency', 'allow_posting': True, 'parent_code': 'sample parent_code'}
	created_response = generated_app_client.post('/records/ChartOfAccounts', json={'record': payload})
	assert created_response.status_code == 201, created_response.get_json()
	created = created_response.get_json()['record']
	list_response = generated_app_client.get('/records/ChartOfAccounts')
	assert list_response.status_code == 200, list_response.get_json()
	listed = list_response.get_json()
	records = listed.get('data', listed.get('records', []))
	assert any(str(record.get('id')) == str(created['id']) for record in records)
	delete_response = generated_app_client.delete('/records/ChartOfAccounts/' + str(created['id']))
	assert delete_response.status_code == 200, delete_response.get_json()
	after_delete_response = generated_app_client.get('/records/ChartOfAccounts')
	assert after_delete_response.status_code == 200, after_delete_response.get_json()
	after_delete = after_delete_response.get_json()
	after_records = after_delete.get('data', after_delete.get('records', []))
	assert all(str(record.get('id')) != str(created['id']) for record in after_records)


def test_search_chartofaccounts(generated_app_client):
	payload = {'account_code': 'ChartOfAccounts Wave K Needle', 'account_name': 'sample account_name', 'account_type': 'sample account_type', 'normal_balance': 'sample normal_balance', 'currency': 'sample currency', 'allow_posting': True, 'parent_code': 'sample parent_code'}
	created_response = generated_app_client.post('/records/ChartOfAccounts', json={'record': payload})
	assert created_response.status_code == 201, created_response.get_json()
	created = created_response.get_json()['record']
	search_response = generated_app_client.get('/records/ChartOfAccounts/search', query_string={'q': 'Needle', 'limit': '5'})
	assert search_response.status_code == 200, search_response.get_json()
	matches = search_response.get_json()
	assert isinstance(matches, list)
	assert any(str(record.get('id')) == str(created['id']) for record in matches)
