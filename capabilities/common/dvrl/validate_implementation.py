#!/usr/bin/env python3
"""
DVRL Implementation Validation Script
Validates all components work correctly in isolation

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import sys
import traceback
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch


def test_imports():
	"""Test that all core components can be imported"""
	print("Testing imports...")
	
	try:
		# Test NLP integration imports
		from nlp_integration import APGNLPProcessor, QuerySuggestionEngine, SemanticQueryMatcher
		print("✓ NLP integration imports successful")
	except Exception as e:
		print(f"✗ NLP integration import failed: {e}")
		return False
	
	try:
		# Test Singer integration imports
		from singer_integration import SingerTapConnector, SingerTapManager
		print("✓ Singer integration imports successful")
	except Exception as e:
		print(f"✗ Singer integration import failed: {e}")
		return False
	
	try:
		# Test views imports
		from views import DVRLDashboardView, DataSourceForm
		print("✓ Views imports successful")
	except Exception as e:
		print(f"✗ Views import failed: {e}")
		return False
	
	try:
		# Test API imports
		from api import DVRLAPIController
		print("✓ API imports successful")
	except Exception as e:
		print(f"✗ API import failed: {e}")
		return False
	
	try:
		# Test service imports
		from service import DVRLService
		print("✓ Service imports successful")
	except Exception as e:
		print(f"✗ Service import failed: {e}")
		return False
	
	try:
		# Test models imports
		from models import DataSource, DataSourceType, DataSourceStatus
		print("✓ Models imports successful")
	except Exception as e:
		print(f"✗ Models import failed: {e}")
		return False
	
	try:
		# Test connectors imports
		from connectors import SQLDatabaseConnector, ConnectorFactory
		print("✓ Connectors imports successful")
	except Exception as e:
		print(f"✗ Connectors import failed: {e}")
		return False
	
	return True


async def test_nlp_integration():
	"""Test NLP integration functionality"""
	print("\nTesting NLP integration...")
	
	try:
		from nlp_integration import APGNLPProcessor
		
		# Mock Ollama
		with patch('ollama.list') as mock_list, patch('ollama.generate') as mock_generate:
			mock_list.return_value = {'models': [{'name': 'llama3.2:latest'}]}
			mock_generate.side_effect = [
				{'response': 'SELECT COUNT(*) FROM users;'},
				{'response': 'This counts all users'}
			]
			
			processor = APGNLPProcessor('test_tenant', 'test_user')
			await processor._initialize_ollama()
			
			result = await processor.process_natural_language_query(
				"How many users are there?",
				{'tables': {'users': ['id', 'name']}}
			)
			
			assert 'generated_sql' in result
			assert 'SELECT COUNT(*)' in result['generated_sql']
			print("✓ NLP query processing successful")
			
			# Test suggestions
			suggestions = await processor.get_query_suggestions({'tables': {'users': ['id']}})
			assert isinstance(suggestions, list)
			print("✓ NLP query suggestions successful")
			
		return True
		
	except Exception as e:
		print(f"✗ NLP integration test failed: {e}")
		traceback.print_exc()
		return False


async def test_singer_integration():
	"""Test Singer integration functionality"""
	print("\nTesting Singer integration...")
	
	try:
		from singer_integration import SingerTapManager
		from models import DataSource, DataSourceType, DataSourceStatus
		
		# Mock HTTP requests and subprocess
		with patch('httpx.AsyncClient.get') as mock_get, \
			 patch('subprocess.create_subprocess_exec') as mock_subprocess:
			
			# Mock Meltano Hub response
			mock_response = Mock()
			mock_response.status_code = 200
			mock_response.json.return_value = {
				'plugins': [
					{
						'name': 'tap-postgres',
						'description': 'PostgreSQL tap',
						'category': 'database'
					}
				]
			}
			mock_get.return_value.__aenter__.return_value = mock_response
			
			# Mock subprocess
			mock_process = AsyncMock()
			mock_process.communicate.return_value = (b'Version: 1.0.0', b'')
			mock_process.returncode = 0
			mock_subprocess.return_value = mock_process
			
			manager = SingerTapManager()
			await manager.initialize()
			
			assert 'tap-postgres' in manager.available_taps
			print("✓ Singer tap discovery successful")
			
			# Test installation
			result = await manager.install_tap('tap-postgres')
			assert result is True
			print("✓ Singer tap installation successful")
			
		return True
		
	except Exception as e:
		print(f"✗ Singer integration test failed: {e}")
		traceback.print_exc()
		return False


def test_forms_validation():
	"""Test form validation"""
	print("\nTesting form validation...")
	
	try:
		from views import DataSourceForm, NaturalLanguageQueryForm, SQLQueryForm
		
		# Test DataSourceForm
		valid_data = {
			'name': 'Test DB',
			'type': 'postgresql',
			'host': 'localhost',
			'port': '5432'
		}
		form = DataSourceForm(data=valid_data)
		assert form.validate() is True
		print("✓ DataSourceForm validation successful")
		
		# Test NLForm
		nl_data = {
			'query': 'Show me all users created last week'
		}
		nl_form = NaturalLanguageQueryForm(data=nl_data)
		assert nl_form.validate() is True
		print("✓ NaturalLanguageQueryForm validation successful")
		
		# Test SQLForm
		sql_data = {
			'sql': 'SELECT * FROM users WHERE created_at > CURRENT_DATE - INTERVAL 7 DAY'
		}
		sql_form = SQLQueryForm(data=sql_data)
		assert sql_form.validate() is True
		print("✓ SQLQueryForm validation successful")
		
		return True
		
	except Exception as e:
		print(f"✗ Form validation test failed: {e}")
		traceback.print_exc()
		return False


def test_models():
	"""Test model classes"""
	print("\nTesting models...")
	
	try:
		from models import DataSource, DataSourceType, DataSourceStatus
		from datetime import datetime, timezone
		
		# Test DataSource creation
		data_source = DataSource(
			id='test_id',
			name='Test DB',
			type=DataSourceType.POSTGRESQL,
			connection_config={'host': 'localhost'},
			status=DataSourceStatus.ACTIVE
		)
		
		assert data_source.name == 'Test DB'
		assert data_source.type == DataSourceType.POSTGRESQL
		assert data_source.status == DataSourceStatus.ACTIVE
		print("✓ DataSource model creation successful")
		
		return True
		
	except Exception as e:
		print(f"✗ Models test failed: {e}")
		traceback.print_exc()
		return False


async def test_api_controller():
	"""Test API controller functionality"""
	print("\nTesting API controller...")
	
	try:
		from api import DVRLAPIController
		
		# Mock service
		mock_service = Mock()
		mock_service.get_health_status = AsyncMock(return_value={'status': 'healthy'})
		mock_service.data_sources = {}
		
		controller = DVRLAPIController(mock_service)
		
		# Test health endpoint
		mock_request = Mock()
		response = await controller.get_health_status(mock_request)
		
		assert response.status_code == 200
		assert response.data['status'] == 'healthy'
		print("✓ API health endpoint successful")
		
		return True
		
	except Exception as e:
		print(f"✗ API controller test failed: {e}")
		traceback.print_exc()
		return False


def test_dashboard_view():
	"""Test dashboard view functionality"""
	print("\nTesting dashboard view...")
	
	try:
		from views import DVRLDashboardView
		
		# Mock service
		mock_service = Mock()
		mock_service.data_sources = {}
		mock_service.get_health_status = AsyncMock(return_value={'status': 'ok'})
		mock_service.get_performance_metrics = AsyncMock(return_value={'metrics': 'ok'})
		
		view = DVRLDashboardView(mock_service)
		
		# Test async helper
		async def test_coro():
			return "test_result"
		
		result = view._run_async(test_coro())
		assert result == "test_result"
		print("✓ Dashboard view async helper successful")
		
		return True
		
	except Exception as e:
		print(f"✗ Dashboard view test failed: {e}")
		traceback.print_exc()
		return False


async def run_all_tests():
	"""Run all validation tests"""
	print("DVRL Implementation Validation")
	print("=" * 50)
	
	tests_passed = 0
	total_tests = 7
	
	# Test imports
	if test_imports():
		tests_passed += 1
	
	# Test NLP integration
	if await test_nlp_integration():
		tests_passed += 1
	
	# Test Singer integration
	if await test_singer_integration():
		tests_passed += 1
	
	# Test forms
	if test_forms_validation():
		tests_passed += 1
	
	# Test models
	if test_models():
		tests_passed += 1
	
	# Test API controller
	if await test_api_controller():
		tests_passed += 1
	
	# Test dashboard view
	if test_dashboard_view():
		tests_passed += 1
	
	print("\n" + "=" * 50)
	print(f"VALIDATION RESULTS: {tests_passed}/{total_tests} tests passed")
	
	if tests_passed == total_tests:
		print("🎉 ALL TESTS PASSED! DVRL implementation is validated.")
		return True
	else:
		print("❌ Some tests failed. Review the implementation.")
		return False


if __name__ == '__main__':
	# Run validation
	try:
		result = asyncio.run(run_all_tests())
		sys.exit(0 if result else 1)
	except KeyboardInterrupt:
		print("\nValidation interrupted by user")
		sys.exit(1)
	except Exception as e:
		print(f"Validation failed with error: {e}")
		traceback.print_exc()
		sys.exit(1)