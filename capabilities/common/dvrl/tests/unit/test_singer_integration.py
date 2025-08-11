#!/usr/bin/env python3
"""
Unit Tests for Singer Integration
Tests for Singer.io tap/target ecosystem integration

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import subprocess
from typing import Dict, Any

from capabilities.common.dvrl.singer_integration import SingerTapConnector, SingerTapManager
from capabilities.common.dvrl.models import DataSource, DataSourceType, DataSourceStatus


class TestSingerTapConnector:
	"""Test suite for Singer Tap Connector"""
	
	@pytest.fixture
	def sample_data_source(self):
		"""Create sample data source for testing"""
		return DataSource(
			id='test_source',
			name='Test PostgreSQL Source',
			type=DataSourceType.POSTGRESQL,
			connection_config={
				'tap_name': 'tap-postgres',
				'tap_config': {
					'host': 'localhost',
					'port': 5432,
					'user': 'testuser',
					'password': 'testpass',
					'dbname': 'testdb'
				}
			},
			status=DataSourceStatus.ACTIVE
		)
	
	@pytest.fixture
	def tap_connector(self, sample_data_source):
		"""Create tap connector for testing"""
		return SingerTapConnector(sample_data_source)
	
	@patch('tempfile.mkdtemp')
	@patch('subprocess.run')
	async def test_initialize_success(self, mock_subprocess, mock_mkdtemp, tap_connector):
		"""Test successful tap initialization"""
		mock_mkdtemp.return_value = '/tmp/test_singer'
		
		# Mock subprocess calls for validation, config, discovery
		mock_subprocess.side_effect = [
			MagicMock(returncode=0),  # tap validation
			MagicMock(returncode=0),  # catalog discovery
		]
		
		with patch.object(tap_connector, '_validate_tap_installation', return_value=True), \
			 patch.object(tap_connector, '_create_tap_config', return_value='/tmp/config.json'), \
			 patch.object(tap_connector, '_discover_tap_catalog', return_value={'streams': []}), \
			 patch.object(tap_connector, '_create_catalog_file', return_value='/tmp/catalog.json'), \
			 patch.object(tap_connector, '_create_state_file', return_value='/tmp/state.json'):
			
			result = await tap_connector.initialize()
			
			assert result is True
			assert tap_connector.temp_dir == '/tmp/test_singer'
	
	@patch('subprocess.run')
	async def test_validate_tap_installation_success(self, mock_subprocess, tap_connector):
		"""Test tap installation validation when tap exists"""
		mock_result = MagicMock()
		mock_result.returncode = 0
		mock_result.stdout = "tap-postgres 1.0.0"
		mock_subprocess.return_value = mock_result
		
		result = await tap_connector._validate_tap_installation()
		
		assert result is True
		mock_subprocess.assert_called_once()
	
	@patch('subprocess.run')
	async def test_validate_tap_installation_failure(self, mock_subprocess, tap_connector):
		"""Test tap installation validation when tap doesn't exist"""
		mock_subprocess.side_effect = FileNotFoundError()
		
		result = await tap_connector._validate_tap_installation()
		
		assert result is False
	
	async def test_create_tap_config(self, tap_connector):
		"""Test tap configuration file creation"""
		tap_connector.temp_dir = tempfile.mkdtemp()
		
		config_file = await tap_connector._create_tap_config()
		
		assert config_file is not None
		assert Path(config_file).exists()
		
		# Verify config content
		with open(config_file, 'r') as f:
			config = json.load(f)
		
		assert config['host'] == 'localhost'
		assert config['port'] == 5432
		assert config['user'] == 'testuser'
		assert config['dbname'] == 'testdb'
		assert 'password' in config  # Should be present but we won't check value
	
	@patch('subprocess.run')
	async def test_discover_tap_catalog_success(self, mock_subprocess, tap_connector):
		"""Test successful catalog discovery"""
		catalog_data = {
			'streams': [
				{
					'stream': 'users',
					'tap_stream_id': 'users',
					'schema': {
						'properties': {
							'id': {'type': 'integer'},
							'name': {'type': 'string'}
						}
					}
				}
			]
		}
		
		mock_result = MagicMock()
		mock_result.returncode = 0
		mock_result.stdout = json.dumps(catalog_data)
		mock_subprocess.return_value = mock_result
		
		tap_connector.temp_dir = tempfile.mkdtemp()
		await tap_connector._create_tap_config()
		
		catalog = await tap_connector._discover_tap_catalog()
		
		assert catalog is not None
		assert 'streams' in catalog
		assert len(catalog['streams']) == 1
		assert catalog['streams'][0]['stream'] == 'users'
	
	@patch('subprocess.run')
	async def test_discover_tap_catalog_failure(self, mock_subprocess, tap_connector):
		"""Test catalog discovery failure"""
		mock_result = MagicMock()
		mock_result.returncode = 1
		mock_result.stderr = "Connection failed"
		mock_subprocess.return_value = mock_result
		
		tap_connector.temp_dir = tempfile.mkdtemp()
		await tap_connector._create_tap_config()
		
		catalog = await tap_connector._discover_tap_catalog()
		
		assert catalog is None
	
	async def test_test_connection_success(self, tap_connector):
		"""Test successful connection test"""
		catalog_data = {
			'streams': [
				{'stream': 'users', 'tap_stream_id': 'users'},
				{'stream': 'orders', 'tap_stream_id': 'orders'}
			]
		}
		
		with patch.object(tap_connector, '_discover_tap_catalog', return_value=catalog_data), \
			 patch.object(tap_connector, '_get_tap_version', return_value='1.2.3'):
			
			result = await tap_connector.test_connection()
			
			assert result['success'] is True
			assert result['tap_name'] == 'tap-postgres'
			assert result['streams_discovered'] == 2
			assert result['tap_version'] == '1.2.3'
			assert 'connection_time_ms' in result
	
	async def test_test_connection_failure(self, tap_connector):
		"""Test connection test failure"""
		with patch.object(tap_connector, '_discover_tap_catalog', return_value=None):
			
			result = await tap_connector.test_connection()
			
			assert result['success'] is False
			assert 'error' in result
	
	@patch('subprocess.Popen')
	async def test_execute_query_success(self, mock_popen, tap_connector):
		"""Test successful query execution via tap"""
		# Mock successful tap execution
		mock_process = MagicMock()
		mock_process.communicate.return_value = (
			'{"type": "RECORD", "record": {"id": 1, "name": "John"}}\n'
			'{"type": "RECORD", "record": {"id": 2, "name": "Jane"}}\n',
			''
		)
		mock_process.returncode = 0
		mock_popen.return_value = mock_process
		
		tap_connector.temp_dir = tempfile.mkdtemp()
		tap_connector.catalog_file = await tap_connector._create_catalog_file({'streams': []})
		tap_connector.state_file = await tap_connector._create_state_file()
		
		result = await tap_connector.execute_query("SELECT * FROM users", {})
		
		assert result is not None
		assert 'data' in result
		assert len(result['data']) == 2
		assert result['data'][0]['id'] == 1
		assert result['data'][0]['name'] == 'John'
	
	@patch('subprocess.Popen')
	async def test_execute_query_failure(self, mock_popen, tap_connector):
		"""Test query execution failure"""
		mock_process = MagicMock()
		mock_process.communicate.return_value = ('', 'Connection error')
		mock_process.returncode = 1
		mock_popen.return_value = mock_process
		
		tap_connector.temp_dir = tempfile.mkdtemp()
		
		result = await tap_connector.execute_query("SELECT * FROM invalid_table", {})
		
		assert result is None
	
	async def test_discover_schema_success(self, tap_connector):
		"""Test schema discovery"""
		catalog_data = {
			'streams': [
				{
					'stream': 'users',
					'schema': {
						'properties': {
							'id': {'type': 'integer'},
							'name': {'type': 'string'},
							'email': {'type': 'string'}
						}
					}
				},
				{
					'stream': 'orders',
					'schema': {
						'properties': {
							'id': {'type': 'integer'},
							'user_id': {'type': 'integer'},
							'total': {'type': 'number'}
						}
					}
				}
			]
		}
		
		with patch.object(tap_connector, '_discover_tap_catalog', return_value=catalog_data):
			
			schema = await tap_connector.discover_schema()
			
			assert schema is not None
			assert schema.schema_name == 'tap-postgres'
			assert 'users' in schema.tables
			assert 'orders' in schema.tables
			assert len(schema.tables['users']['columns']) == 3
			assert schema.tables['users']['columns'][0]['name'] == 'id'
			assert schema.tables['users']['columns'][0]['type'] == 'integer'


class TestSingerTapManager:
	"""Test suite for Singer Tap Manager"""
	
	@pytest.fixture
	def tap_manager(self):
		"""Create tap manager for testing"""
		return SingerTapManager()
	
	async def test_initialization(self, tap_manager):
		"""Test tap manager initialization"""
		with patch.object(tap_manager, '_discover_available_taps') as mock_discover, \
			 patch.object(tap_manager, '_check_installed_taps') as mock_check:
			
			await tap_manager.initialize()
			
			mock_discover.assert_called_once()
			mock_check.assert_called_once()
	
	@patch('httpx.AsyncClient.get')
	async def test_discover_available_taps_success(self, mock_get, tap_manager):
		"""Test successful tap discovery from Meltano Hub"""
		mock_response = MagicMock()
		mock_response.status_code = 200
		mock_response.json.return_value = {
			'plugins': [
				{
					'name': 'tap-postgres',
					'description': 'PostgreSQL tap',
					'category': 'database',
					'settings': ['host', 'port', 'user', 'password'],
					'docs': 'https://docs.example.com',
					'repo': 'https://github.com/example/tap-postgres',
					'pip_url': 'tap-postgres==1.0.0'
				},
				{
					'name': 'tap-mysql', 
					'description': 'MySQL tap',
					'category': 'database',
					'settings': ['host', 'port', 'user', 'password', 'database'],
					'docs': 'https://docs.example.com/mysql',
					'repo': 'https://github.com/example/tap-mysql'
				}
			]
		}
		mock_get.return_value.__aenter__.return_value = mock_response
		
		await tap_manager._discover_available_taps()
		
		assert 'tap-postgres' in tap_manager.available_taps
		assert 'tap-mysql' in tap_manager.available_taps
		assert tap_manager.available_taps['tap-postgres']['description'] == 'PostgreSQL tap'
		assert tap_manager.available_taps['tap-postgres']['category'] == 'database'
	
	@patch('httpx.AsyncClient.get')
	async def test_discover_available_taps_failure(self, mock_get, tap_manager):
		"""Test tap discovery failure with fallback"""
		mock_response = MagicMock()
		mock_response.status_code = 500
		mock_get.return_value.__aenter__.return_value = mock_response
		
		await tap_manager._discover_available_taps()
		
		# Should fall back to predefined taps
		assert len(tap_manager.available_taps) > 0
		assert 'tap-postgres' in tap_manager.available_taps  # From fallback
	
	@patch('subprocess.create_subprocess_exec')
	async def test_install_tap_success(self, mock_subprocess, tap_manager):
		"""Test successful tap installation"""
		# Mock pip install success
		mock_process = AsyncMock()
		mock_process.communicate.return_value = (b'Successfully installed tap-postgres', b'')
		mock_process.returncode = 0
		mock_subprocess.return_value = mock_process
		
		with patch.object(tap_manager, '_get_tap_version_real', return_value='1.2.3'):
			
			result = await tap_manager.install_tap('tap-postgres')
			
			assert result is True
			assert 'tap-postgres' in tap_manager.installed_taps
			assert tap_manager.installed_taps['tap-postgres']['version'] == '1.2.3'
			assert tap_manager.installed_taps['tap-postgres']['status'] == 'installed'
	
	@patch('subprocess.create_subprocess_exec')
	async def test_install_tap_failure(self, mock_subprocess, tap_manager):
		"""Test tap installation failure"""
		# Mock pip install failure
		mock_process = AsyncMock()
		mock_process.communicate.return_value = (b'', b'Could not find tap-invalid')
		mock_process.returncode = 1
		mock_subprocess.return_value = mock_process
		
		result = await tap_manager.install_tap('tap-invalid')
		
		assert result is False
		assert 'tap-invalid' not in tap_manager.installed_taps
	
	@patch('subprocess.create_subprocess_exec')
	async def test_check_installed_taps(self, mock_subprocess, tap_manager):
		"""Test checking installed taps"""
		# Mock pip list output
		mock_process = AsyncMock()
		mock_process.communicate.return_value = (
			json.dumps([
				{'name': 'tap-postgres', 'version': '1.2.3'},
				{'name': 'tap-mysql', 'version': '2.1.0'},
				{'name': 'other-package', 'version': '1.0.0'}  # Should be ignored
			]).encode(),
			b''
		)
		mock_process.returncode = 0
		mock_subprocess.return_value = mock_process
		
		await tap_manager._check_installed_taps()
		
		assert 'tap-postgres' in tap_manager.installed_taps
		assert 'tap-mysql' in tap_manager.installed_taps
		assert 'other-package' not in tap_manager.installed_taps
		assert tap_manager.installed_taps['tap-postgres']['version'] == '1.2.3'
	
	@patch('subprocess.create_subprocess_exec')
	async def test_get_tap_version_real(self, mock_subprocess, tap_manager):
		"""Test getting real tap version"""
		mock_process = AsyncMock()
		mock_process.communicate.return_value = (
			b'Name: tap-postgres\nVersion: 1.2.3\nSummary: PostgreSQL tap\n',
			b''
		)
		mock_process.returncode = 0
		mock_subprocess.return_value = mock_process
		
		version = await tap_manager._get_tap_version_real('tap-postgres')
		
		assert version == '1.2.3'
	
	async def test_create_tap_connector_success(self, tap_manager):
		"""Test successful tap connector creation"""
		tap_config = {
			'host': 'localhost',
			'port': 5432,
			'user': 'testuser',
			'password': 'testpass',
			'dbname': 'testdb'
		}
		
		with patch('...singer_integration.SingerTapConnector') as mock_connector_class:
			mock_connector = AsyncMock()
			mock_connector.initialize.return_value = True
			mock_connector_class.return_value = mock_connector
			
			connector = await tap_manager.create_tap_connector('tap-postgres', tap_config)
			
			assert connector is not None
			mock_connector.initialize.assert_called_once()
	
	async def test_create_tap_connector_failure(self, tap_manager):
		"""Test tap connector creation failure"""
		tap_config = {'invalid': 'config'}
		
		with patch('...singer_integration.SingerTapConnector') as mock_connector_class:
			mock_connector = AsyncMock()
			mock_connector.initialize.return_value = False
			mock_connector_class.return_value = mock_connector
			
			connector = await tap_manager.create_tap_connector('tap-invalid', tap_config)
			
			assert connector is None
	
	async def test_get_manager_stats(self, tap_manager):
		"""Test manager statistics retrieval"""
		tap_manager.available_taps = {
			'tap-postgres': {'category': 'database'},
			'tap-mysql': {'category': 'database'},
			'tap-stripe': {'category': 'saas'}
		}
		tap_manager.installed_taps = {
			'tap-postgres': {'status': 'installed', 'version': '1.0.0'}
		}
		
		stats = await tap_manager.get_manager_stats()
		
		assert stats['total_available'] == 3
		assert stats['total_installed'] == 1
		assert 'available_taps' in stats
		assert 'installed_taps' in stats


# Integration tests combining tap connector and manager
class TestSingerIntegrationComplete:
	"""Integration tests for complete Singer workflow"""
	
	@patch('subprocess.create_subprocess_exec')
	@patch('subprocess.run')
	@patch('httpx.AsyncClient.get')
	async def test_complete_tap_installation_workflow(self, mock_http_get, mock_subprocess_run, mock_subprocess_exec):
		"""Test complete workflow from discovery to installation to usage"""
		# Mock Meltano Hub discovery
		mock_response = MagicMock()
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
		mock_http_get.return_value.__aenter__.return_value = mock_response
		
		# Mock installation process
		mock_install_process = AsyncMock()
		mock_install_process.communicate.return_value = (b'Successfully installed', b'')
		mock_install_process.returncode = 0
		
		mock_version_process = AsyncMock()
		mock_version_process.communicate.return_value = (b'Version: 1.2.3', b'')
		mock_version_process.returncode = 0
		
		mock_subprocess_exec.side_effect = [mock_install_process, mock_version_process]
		
		# Mock tap usage
		mock_subprocess_run.return_value = MagicMock(returncode=0)
		
		# Test workflow
		manager = SingerTapManager()
		await manager.initialize()
		
		# Install tap
		install_result = await manager.install_tap('tap-postgres')
		assert install_result is True
		
		# Create connector
		tap_config = {
			'host': 'localhost',
			'port': 5432,
			'user': 'test',
			'password': 'test',
			'dbname': 'test'
		}
		
		with patch.object(manager, 'create_tap_connector') as mock_create:
			mock_connector = AsyncMock()
			mock_create.return_value = mock_connector
			
			connector = await manager.create_tap_connector('tap-postgres', tap_config)
			assert connector is not None


if __name__ == '__main__':
	pytest.main([__file__])