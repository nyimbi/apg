#!/usr/bin/env python3
"""
APG Metadata Management - Data Source Connectors
Universal connectors for discovering metadata from various data sources

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from .base_connector import BaseConnector, ConnectorConfig, DiscoveryResult
from .database_connectors import (
	PostgreSQLConnector,
	MySQLConnector,
	OracleConnector,
	SQLServerConnector,
	MongoDBConnector,
	RedisConnector,
	SnowflakeConnector,
	BigQueryConnector
)
try:
	from .file_connectors import (
		CSVConnector,
		JSONConnector,
		ParquetConnector,
		AvroConnector,
		S3Connector,
		GCSConnector
	)
except ImportError:  # pragma: no cover - optional file connector dependencies.
	CSVConnector = JSONConnector = ParquetConnector = AvroConnector = S3Connector = GCSConnector = None
try:
	from .api_connectors import (
		RESTAPIConnector,
		GraphQLConnector,
		BytewaxConnector
	)
except ImportError:  # pragma: no cover - optional API connector dependencies.
	RESTAPIConnector = GraphQLConnector = BytewaxConnector = None
try:
	from .ml_connectors import (
		MLflowConnector,
		KubeflowConnector,
		SageMakerConnector,
		JupyterConnector
	)
except ImportError:  # pragma: no cover - optional ML connector dependencies.
	MLflowConnector = KubeflowConnector = SageMakerConnector = JupyterConnector = None
from .connector_registry import ConnectorRegistry

__all__ = [
	# Base classes
	'BaseConnector',
	'ConnectorConfig', 
	'DiscoveryResult',
	
	# Database connectors
	'PostgreSQLConnector',
	'MySQLConnector',
	'OracleConnector',
	'SQLServerConnector',
	'MongoDBConnector',
	'RedisConnector',
	'SnowflakeConnector',
	'BigQueryConnector',
	
	# File connectors
	'CSVConnector',
	'JSONConnector',
	'ParquetConnector',
	'AvroConnector',
	'S3Connector',
	'GCSConnector',
	
	# API connectors
	'RESTAPIConnector',
	'GraphQLConnector',
	'BytewaxConnector',
	
	# ML connectors
	'MLflowConnector',
	'KubeflowConnector',
	'SageMakerConnector',
	'JupyterConnector',
	
	# Registry
	'ConnectorRegistry'
]
