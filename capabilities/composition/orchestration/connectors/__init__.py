"""
APG Workflow Orchestration External System Connectors

Comprehensive integration layer for external systems including REST/GraphQL APIs,
databases, cloud services, message queues, and file systems.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from .base_connector import BaseConnector, ConnectorStatus, ConnectorMetrics
from .connector_registry import ConnectorRegistry, get_connector_registry, CONNECTORS_MANIFEST

try:
	from .rest_connector import RESTConnector, GraphQLConnector
except ImportError:
	RESTConnector = None  # type: ignore[assignment,misc]
	GraphQLConnector = None  # type: ignore[assignment,misc]

try:
	from .database_connector import DatabaseConnector, PostgreSQLAdapter, MongoDBAdapter
except ImportError:
	DatabaseConnector = PostgreSQLAdapter = MongoDBAdapter = None  # type: ignore[assignment,misc]

try:
	from .cloud_connector import AWSConnector, AzureConnector, GCPConnector
except ImportError:
	AWSConnector = AzureConnector = GCPConnector = None  # type: ignore[assignment,misc]

try:
	from .message_queue_connector import BytewaxConnector, RabbitMQConnector, RedisQueueConnector
except ImportError:
	BytewaxConnector = RabbitMQConnector = RedisQueueConnector = None  # type: ignore[assignment,misc]

try:
	from .file_connector import FileSystemConnector, FTPConnector, S3Connector
except ImportError:
	FileSystemConnector = FTPConnector = S3Connector = None  # type: ignore[assignment,misc]

__all__ = [
	"BaseConnector",
	"ConnectorStatus",
	"ConnectorMetrics",
	"ConnectorRegistry",
	"get_connector_registry",
	"CONNECTORS_MANIFEST",
	"RESTConnector",
	"GraphQLConnector",
	"DatabaseConnector",
	"PostgreSQLAdapter",
	"MongoDBAdapter",
	"AWSConnector",
	"AzureConnector",
	"GCPConnector",
	"BytewaxConnector",
	"RabbitMQConnector",
	"RedisQueueConnector",
	"FileSystemConnector",
	"FTPConnector",
	"S3Connector",
]
