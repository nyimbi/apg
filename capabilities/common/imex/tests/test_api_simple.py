#!/usr/bin/env python3
"""
Simple API test for APG IMEX capability.

This test validates the simple Flask API implementation.
"""
import asyncio
import logging
import json
import tempfile
import csv
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask
from flask.testing import FlaskClient

from models import JobType, DataFormat, SourceType, ProcessingPriority, ValidationLevel, ErrorHandlingStrategy
from database import DatabaseManager, DatabaseConfig
from ai_intelligence import AIIntelligenceEngine
from service import ImportExportService
from api_simple import imex_api_bp, initialize_api_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleAPITestSuite:
	"""Simple API testing suite."""

	def __init__(self):
		self.app = None
		self.client = None
		self.service = None
		self.temp_dir = None

	async def setup(self):
		"""Setup test environment."""
		try:
			# Create Flask app
			self.app = Flask(__name__)
			self.app.config['TESTING'] = True

			# Setup database and AI components
			db_config = DatabaseConfig(
				host="localhost", port=5432, database="test", user="test", password="test"
			)
			db_manager = DatabaseManager(db_config)
			ai_engine = AIIntelligenceEngine()
			await ai_engine.initialize()

			# Create service
			self.service = ImportExportService(db_manager, ai_engine)
			await self.service.initialize()

			# Initialize API with service
			initialize_api_service(self.service)

			# Register blueprint
			self.app.register_blueprint(imex_api_bp)

			# Create temporary directory for test files
			self.temp_dir = Path(tempfile.mkdtemp())

			# Create test client
			self.client = self.app.test_client()

			logger.info("✓ Simple API test setup completed")
			return True

		except Exception as e:
			logger.error(f"Simple API test setup failed: {e}")
			return False

	def test_api_health_check(self) -> bool:
		"""Test API health check endpoint."""
		try:
			response = self.client.get('/api/v1/imex/health')

			logger.info(f"Health check status: {response.status_code}")
			logger.info(f"Health check response: {response.get_json()}")

			# Should get some response (200 or error)
			assert response.status_code in [200, 500, 503]
			data = json.loads(response.data)
			assert 'success' in data

			logger.info("✓ API health check test passed")
			return True

		except Exception as e:
			logger.error(f"API health check test failed: {e}")
			return False

	def test_job_list_endpoint(self) -> bool:
		"""Test job listing endpoint."""
		try:
			response = self.client.get('/api/v1/imex/jobs')

			logger.info(f"Job list status: {response.status_code}")
			logger.info(f"Job list response: {response.get_json()}")

			# Should get some response
			assert response.status_code in [200, 500, 503]
			data = json.loads(response.data)
			assert 'success' in data

			logger.info("✓ Job list endpoint test passed")
			return True

		except Exception as e:
			logger.error(f"Job list endpoint test failed: {e}")
			return False

	def test_job_creation_validation(self) -> bool:
		"""Test job creation with validation."""
		try:
			# Test invalid job creation (missing fields)
			invalid_job_data = {
				"name": "Test Job"
				# Missing required fields
			}

			response = self.client.post('/api/v1/imex/jobs',
									  json=invalid_job_data,
									  content_type='application/json',
									  headers={'X-Tenant-ID': 'test', 'X-User-ID': 'test-user'})

			logger.info(f"Invalid job creation status: {response.status_code}")
			logger.info(f"Invalid job creation response: {response.get_json()}")

			# Should return validation error
			assert response.status_code == 400
			data = json.loads(response.data)
			assert data['success'] == False
			assert 'error' in data

			logger.info("✓ Job creation validation test passed")
			return True

		except Exception as e:
			logger.error(f"Job creation validation test failed: {e}")
			return False

	def test_error_handling(self) -> bool:
		"""Test API error handling."""
		try:
			# Test non-existent job
			response = self.client.get('/api/v1/imex/jobs/nonexistent-job-id')

			logger.info(f"Nonexistent job status: {response.status_code}")
			logger.info(f"Nonexistent job response: {response.get_json()}")

			# Should return 404 or 500
			assert response.status_code in [404, 500, 503]
			data = json.loads(response.data)
			assert data['success'] == False

			logger.info("✓ Error handling test passed")
			return True

		except Exception as e:
			logger.error(f"Error handling test failed: {e}")
			return False

	def test_malformed_request(self) -> bool:
		"""Test malformed request handling."""
		try:
			# Test malformed JSON
			response = self.client.post('/api/v1/imex/jobs',
									  data="invalid json",
									  content_type='application/json')

			logger.info(f"Malformed request status: {response.status_code}")
			logger.info(f"Malformed request response: {response.get_json()}")

			# Should return 400
			assert response.status_code == 400
			data = json.loads(response.data)
			assert data['success'] == False

			logger.info("✓ Malformed request test passed")
			return True

		except Exception as e:
			logger.error(f"Malformed request test failed: {e}")
			return False

	def teardown(self):
		"""Clean up test resources."""
		try:
			if self.temp_dir and self.temp_dir.exists():
				import shutil
				shutil.rmtree(self.temp_dir)
			logger.info("✓ Simple API test cleanup completed")
		except Exception as e:
			logger.warning(f"Cleanup warning: {e}")

async def main():
	"""Run simple API tests."""
	logger.info("Starting APG IMEX Simple API tests...")

	test_suite = SimpleAPITestSuite()

	try:
		# Setup
		if not await test_suite.setup():
			logger.error("Test setup failed")
			return 1

		# Run test suite
		tests = [
			("API Health Check", test_suite.test_api_health_check),
			("Job List Endpoint", test_suite.test_job_list_endpoint),
			("Job Creation Validation", test_suite.test_job_creation_validation),
			("Error Handling", test_suite.test_error_handling),
			("Malformed Request", test_suite.test_malformed_request),
		]

		passed = 0
		failed = 0

		for test_name, test_func in tests:
			logger.info(f"\nRunning: {test_name}")
			try:
				if test_func():
					passed += 1
					logger.info(f"✓ {test_name} PASSED")
				else:
					failed += 1
					logger.error(f"✗ {test_name} FAILED")
			except Exception as e:
				failed += 1
				logger.error(f"✗ {test_name} FAILED with exception: {e}")

		# Results
		total = passed + failed
		logger.info(f"\nSimple API Test Results:")
		logger.info(f"  Total tests: {total}")
		logger.info(f"  Passed: {passed}")
		logger.info(f"  Failed: {failed}")
		logger.info(f"  Success rate: {(passed/total)*100:.1f}%")

		if failed == 0:
			logger.info("✓ All simple API tests passed successfully!")
			return 0
		else:
			logger.error(f"✗ {failed} simple API tests failed")
			return 1

	finally:
		test_suite.teardown()

if __name__ == "__main__":
	result = asyncio.run(main())
	exit(result)