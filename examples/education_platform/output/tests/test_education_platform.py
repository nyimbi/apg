"""Generated smoke, CRUD, and search tests for this APG app."""

from __future__ import annotations

import pytest


def test_smoke_livez(generated_app_client):
	assert generated_app_client.get('/livez').status_code == 200


def test_crud_student(generated_app_client):
	payload = {'student_id': 'Student Wave K Needle', 'admission_number': 'sample admission_number', 'first_name': 'sample first_name', 'last_name': 'sample last_name', 'date_of_birth': 'sample date_of_birth', 'gender': 'sample gender', 'national_id': 'sample national_id', 'guardian_name': 'sample guardian_name', 'guardian_phone': 'sample guardian_phone', 'guardian_email': 'sample guardian_email', 'class_id': 'sample class_id', 'stream_id': 'sample stream_id', 'enrolment_date': 'sample enrolment_date', 'status': 'sample status', 'special_needs': []}
	created_response = generated_app_client.post('/records/Student', json={'record': payload})
	assert created_response.status_code == 201, created_response.get_json()
	created = created_response.get_json()['record']
	list_response = generated_app_client.get('/records/Student')
	assert list_response.status_code == 200, list_response.get_json()
	listed = list_response.get_json()
	records = listed.get('data', listed.get('records', []))
	assert any(str(record.get('id')) == str(created['id']) for record in records)
	delete_response = generated_app_client.delete('/records/Student/' + str(created['id']))
	assert delete_response.status_code == 200, delete_response.get_json()
	after_delete_response = generated_app_client.get('/records/Student')
	assert after_delete_response.status_code == 200, after_delete_response.get_json()
	after_delete = after_delete_response.get_json()
	after_records = after_delete.get('data', after_delete.get('records', []))
	assert all(str(record.get('id')) != str(created['id']) for record in after_records)


def test_search_student(generated_app_client):
	payload = {'student_id': 'Student Wave K Needle', 'admission_number': 'sample admission_number', 'first_name': 'sample first_name', 'last_name': 'sample last_name', 'date_of_birth': 'sample date_of_birth', 'gender': 'sample gender', 'national_id': 'sample national_id', 'guardian_name': 'sample guardian_name', 'guardian_phone': 'sample guardian_phone', 'guardian_email': 'sample guardian_email', 'class_id': 'sample class_id', 'stream_id': 'sample stream_id', 'enrolment_date': 'sample enrolment_date', 'status': 'sample status', 'special_needs': []}
	created_response = generated_app_client.post('/records/Student', json={'record': payload})
	assert created_response.status_code == 201, created_response.get_json()
	created = created_response.get_json()['record']
	search_response = generated_app_client.get('/records/Student/search', query_string={'q': 'Needle', 'limit': '5'})
	assert search_response.status_code == 200, search_response.get_json()
	matches = search_response.get_json()
	assert isinstance(matches, list)
	assert any(str(record.get('id')) == str(created['id']) for record in matches)
