#!/usr/bin/env python3
"""
Tests for Data Source Adapters
Specialized adapters for file systems, cloud storage, and data formats

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
from typing import Dict, Any

from ..models import DataSource, DataSourceType, DataSourceStatus
from ..adapters import (
	FileSystemConnector, CloudStorageConnector, 
	DistributedFileSystemConnector, DataWarehouseConnector
)


@pytest.fixture
def sample_filesystem_config() -> Dict[str, Any]:
	"""Sample file system data source configuration"""
	return {
		'name': 'test_filesystem',
		'description': 'Test file system',
		'type': DataSourceType.FILE_CSV,
		'connection_config': {
			'base_path': '/tmp/test_data',
			'access_mode': 'read',
			'recursive': True
		}
	}


@pytest.fixture
def sample_s3_config() -> Dict[str, Any]:
	"""Sample S3 data source configuration"""
	return {
		'name': 'test_s3_bucket',
		'description': 'Test S3 bucket',
		'type': DataSourceType.S3,
		'connection_config': {
			'bucket': 'my-test-bucket',
			'region': 'us-east-1',
			'access_key_id': 'AKIA...',
			'secret_access_key': 'secret...',
			'prefix': 'data/'
		}
	}


@pytest.fixture
def sample_hdfs_config() -> Dict[str, Any]:
	"""Sample HDFS data source configuration"""
	return {
		'name': 'test_hdfs',
		'description': 'Test HDFS cluster',
		'type': DataSourceType.HDFS,
		'connection_config': {
			'namenode': 'hdfs://namenode:9000',
			'user': 'hadoop',
			'replication': 3
		}
	}


@pytest.fixture
def sample_snowflake_config() -> Dict[str, Any]:
	"""Sample Snowflake data warehouse configuration"""
	return {
		'name': 'test_snowflake',
		'description': 'Test Snowflake warehouse',
		'type': DataSourceType.SNOWFLAKE,
		'connection_config': {
			'account': 'myaccount.snowflakecomputing.com',
			'user': 'testuser',
			'password': 'testpass',
			'database': 'TESTDB',
			'warehouse': 'COMPUTE_WH',
			'role': 'SYSADMIN'
		}
	}


async def test_filesystem_connector_creation():
	"""Test file system connector creation and basic functionality"""
	config = {
		'name': 'test_filesystem',
		'type': DataSourceType.FILE_CSV,
		'connection_config': {'base_path': '/tmp'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = FileSystemConnector(data_source, 'test-tenant', 'test-user')
	
	assert isinstance(connector, FileSystemConnector)
	assert connector.data_source.name == 'test_filesystem'
	assert len(connector.supported_formats) > 0
	assert 'csv' in connector.supported_formats


async def test_filesystem_connection_lifecycle():
	"""Test file system connector connection lifecycle"""
	config = {
		'name': 'test_filesystem',
		'type': DataSourceType.FILE_CSV,
		'connection_config': {'base_path': '/tmp'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = FileSystemConnector(data_source, 'test-tenant', 'test-user')
	
	# Test connection
	connected = await connector.connect()
	assert connected == True
	
	# Test capabilities
	capabilities = await connector.get_capabilities()
	assert len(capabilities) > 0
	
	# Test disconnection
	disconnected = await connector.disconnect()
	assert disconnected == True


async def test_cloud_storage_connector():
	"""Test cloud storage connector functionality"""
	config = {
		'name': 'test_s3',
		'type': DataSourceType.S3,
		'connection_config': {
			'bucket': 'test-bucket',
			'region': 'us-east-1'
		}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = CloudStorageConnector(data_source, 'test-tenant', 'test-user')
	
	# Test connection
	connected = await connector.connect()
	assert connected == True
	
	# Test cloud provider detection
	assert connector.connection_metadata['cloud_provider'] == 'aws_s3'
	
	# Test schema discovery
	schema = await connector.discover_schema()
	assert schema is not None
	assert len(schema.tables) > 0
	
	# Test query execution
	result = await connector.execute_query("SELECT * FROM events")
	assert 'results' in result
	assert 'object_count' in result


async def test_hdfs_connector():
	"""Test HDFS connector functionality"""
	config = {
		'name': 'test_hdfs',
		'type': DataSourceType.HDFS,
		'connection_config': {
			'namenode': 'hdfs://localhost:9000',
			'user': 'hadoop'
		}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = DistributedFileSystemConnector(data_source, 'test-tenant', 'test-user')
	
	# Test connection
	connected = await connector.connect()
	assert connected == True
	
	# Test schema discovery
	schema = await connector.discover_schema()
	assert schema is not None
	assert schema.discovery_method == "hdfs_directory_scan"
	
	# Test capabilities
	capabilities = await connector.get_capabilities()
	assert len(capabilities) >= 4  # Should have multiple capabilities


async def test_data_warehouse_connector():
	"""Test data warehouse connector functionality"""
	config = {
		'name': 'test_snowflake',
		'type': DataSourceType.SNOWFLAKE,
		'connection_config': {
			'account': 'testaccount',
			'database': 'TESTDB'
		}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = DataWarehouseConnector(data_source, 'test-tenant', 'test-user')
	
	# Test connection
	connected = await connector.connect()
	assert connected == True
	
	# Test warehouse type detection
	assert connector.connection_metadata['warehouse_type'] == 'snowflake'
	
	# Test schema discovery
	schema = await connector.discover_schema()
	assert schema is not None
	assert 'snowflake' in schema.discovery_method
	
	# Test query execution
	result = await connector.execute_query("SELECT COUNT(*) FROM customers")
	assert 'compute_credits_used' in result
	assert 'bytes_scanned' in result
	
	# Test capabilities
	capabilities = await connector.get_capabilities()
	assert len(capabilities) >= 8  # Should have many advanced capabilities


async def test_file_schema_inference():
	"""Test file format schema inference"""
	config = {
		'name': 'test_files',
		'type': DataSourceType.FILE_CSV,
		'connection_config': {'base_path': '/tmp'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = FileSystemConnector(data_source, 'test-tenant', 'test-user')
	await connector.connect()
	
	# Test CSV schema inference
	csv_schema = await connector._infer_file_schema(None, 'csv')
	assert len(csv_schema) > 0
	assert any(col['name'] == 'id' for col in csv_schema)
	
	# Test JSON schema inference
	json_schema = await connector._infer_file_schema(None, 'json')
	assert len(json_schema) > 0
	assert any(col['type'] == 'json' for col in json_schema)
	
	# Test Parquet schema inference
	parquet_schema = await connector._infer_file_schema(None, 'parquet')
	assert len(parquet_schema) > 0
	assert any('timestamp' in col['type'] for col in parquet_schema)


async def test_cloud_object_grouping():
	"""Test cloud storage object grouping logic"""
	config = {
		'name': 'test_s3',
		'type': DataSourceType.S3,
		'connection_config': {'bucket': 'test'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = CloudStorageConnector(data_source, 'test-tenant', 'test-user')
	await connector.connect()
	
	# Mock objects with different patterns
	objects = [
		{'key': 'data/year=2024/month=01/events.parquet', 'format': 'parquet'},
		{'key': 'data/year=2024/month=02/events.parquet', 'format': 'parquet'},
		{'key': 'logs/app.log', 'format': 'text'},
		{'key': 'exports/users.csv', 'format': 'csv'}
	]
	
	grouped = connector._group_objects_by_pattern(objects)
	
	# Should group partitioned parquet files together
	assert len(grouped) >= 3  # At least 3 different patterns
	
	# Check partitioned pattern exists
	partitioned_patterns = [k for k in grouped.keys() if 'partitioned' in k]
	assert len(partitioned_patterns) > 0


async def test_warehouse_compute_resources():
	"""Test data warehouse compute resource detection"""
	# Test Snowflake
	snowflake_config = {
		'name': 'test_snowflake',
		'type': DataSourceType.SNOWFLAKE,
		'connection_config': {'account': 'test'}
	}
	
	snowflake_ds = DataSource(tenant_id='test', created_by='test', **snowflake_config)
	snowflake_conn = DataWarehouseConnector(snowflake_ds, 'test', 'test')
	
	snowflake_resources = await snowflake_conn._get_compute_resources()
	assert 'warehouse_size' in snowflake_resources
	assert 'auto_suspend' in snowflake_resources
	
	# Test BigQuery
	bq_config = {
		'name': 'test_bigquery',
		'type': DataSourceType.BIGQUERY,
		'connection_config': {'project': 'test'}
	}
	
	bq_ds = DataSource(tenant_id='test', created_by='test', **bq_config)
	bq_conn = DataWarehouseConnector(bq_ds, 'test', 'test')
	
	bq_resources = await bq_conn._get_compute_resources()
	assert 'slot_allocation' in bq_resources
	assert 'location' in bq_resources


if __name__ == "__main__":
	# Run basic adapter tests
	loop = asyncio.get_event_loop()
	
	print("Testing File System Connector...")
	loop.run_until_complete(test_filesystem_connector_creation())
	print("✓ File System Connector test passed")
	
	print("Testing File System Connection Lifecycle...")
	loop.run_until_complete(test_filesystem_connection_lifecycle())
	print("✓ File System Connection Lifecycle test passed")
	
	print("Testing Cloud Storage Connector...")
	loop.run_until_complete(test_cloud_storage_connector())
	print("✓ Cloud Storage Connector test passed")
	
	print("Testing HDFS Connector...")
	loop.run_until_complete(test_hdfs_connector())
	print("✓ HDFS Connector test passed")
	
	print("Testing Data Warehouse Connector...")
	loop.run_until_complete(test_data_warehouse_connector())
	print("✓ Data Warehouse Connector test passed")
	
	print("Testing File Schema Inference...")
	loop.run_until_complete(test_file_schema_inference())
	print("✓ File Schema Inference test passed")
	
	print("Testing Cloud Object Grouping...")
	loop.run_until_complete(test_cloud_object_grouping())
	print("✓ Cloud Object Grouping test passed")
	
	print("Testing Warehouse Compute Resources...")
	loop.run_until_complete(test_warehouse_compute_resources())
	print("✓ Warehouse Compute Resources test passed")
	
	print("\n🎉 All adapter tests passed!")