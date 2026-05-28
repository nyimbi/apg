"""Dependency-light coverage for orchestration cloud connector uploads."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONNECTOR_DIR = REPO_ROOT / "capabilities" / "composition" / "orchestration" / "connectors"
TEST_PACKAGE = "_apg_cloud_connector_under_test"


def _module(name: str) -> types.ModuleType:
	module = types.ModuleType(name)
	sys.modules[name] = module
	return module


def _install_sdk_stubs() -> None:
	for name in [
		"boto3",
		"botocore",
		"botocore.exceptions",
		"aiobotocore",
		"aiobotocore.session",
		"azure",
		"azure.identity",
		"azure.storage",
		"azure.storage.blob",
		"azure.cosmos",
		"azure.servicebus",
		"google",
		"google.cloud",
		"google.cloud.storage",
		"google.cloud.pubsub_v1",
		"google.oauth2",
		"google.oauth2.service_account",
		"google.auth",
		"google.auth.exceptions",
	]:
		sys.modules.setdefault(name, types.ModuleType(name))

	sys.modules["botocore.exceptions"].ClientError = type("ClientError", (Exception,), {})
	sys.modules["botocore.exceptions"].BotoCoreError = type("BotoCoreError", (Exception,), {})
	sys.modules["aiobotocore.session"].get_session = lambda: object()
	sys.modules["azure.identity"].DefaultAzureCredential = object
	sys.modules["azure.identity"].ClientSecretCredential = object
	sys.modules["azure.storage.blob"].BlobServiceClient = object
	sys.modules["azure.cosmos"].CosmosClient = object
	sys.modules["azure.servicebus"].ServiceBusClient = object
	sys.modules["google.cloud"].storage = sys.modules["google.cloud.storage"]
	sys.modules["google.cloud"].pubsub_v1 = sys.modules["google.cloud.pubsub_v1"]
	sys.modules["google.oauth2"].service_account = sys.modules["google.oauth2.service_account"]
	sys.modules["google.auth.exceptions"].GoogleAuthError = type("GoogleAuthError", (Exception,), {})


def _load_cloud_connector() -> types.ModuleType:
	_install_sdk_stubs()
	package = types.ModuleType(TEST_PACKAGE)
	package.__path__ = [str(CONNECTOR_DIR)]
	sys.modules[TEST_PACKAGE] = package

	for module_name, file_name in [
		(f"{TEST_PACKAGE}.base_connector", "base_connector.py"),
		(f"{TEST_PACKAGE}.cloud_connector", "cloud_connector.py"),
	]:
		spec = importlib.util.spec_from_file_location(module_name, CONNECTOR_DIR / file_name)
		assert spec is not None
		assert spec.loader is not None
		module = importlib.util.module_from_spec(spec)
		sys.modules[module_name] = module
		spec.loader.exec_module(module)

	return sys.modules[f"{TEST_PACKAGE}.cloud_connector"]


class FakeBlob:
	generation = "generation-1"
	md5_hash = "md5-1"
	public_url = "https://storage.example/bucket/path/item.txt"

	def __init__(self) -> None:
		self.payload: bytes | None = None
		self.content_type: str | None = None

	def upload_from_string(self, payload: bytes, content_type: str | None = None) -> None:
		self.payload = payload
		self.content_type = content_type


class FakeBucket:
	def __init__(self) -> None:
		self.blobs: dict[str, FakeBlob] = {}

	def blob(self, name: str) -> FakeBlob:
		blob = FakeBlob()
		self.blobs[name] = blob
		return blob


class FakeStorageClient:
	def __init__(self) -> None:
		self.bucket_calls: list[str] = []
		self.buckets: dict[str, FakeBucket] = {}

	def bucket(self, name: str) -> FakeBucket:
		self.bucket_calls.append(name)
		bucket = FakeBucket()
		self.buckets[name] = bucket
		return bucket


def _configured_connector(module: types.ModuleType) -> tuple[Any, FakeStorageClient]:
	config = module.GCPConfiguration(
		name="gcp-test",
		tenant_id="tenant-1",
		user_id="user-1",
		project_id="project-1",
		services=[],
		environment="development",
	)
	connector = module.GCPConnector(config)
	storage_client = FakeStorageClient()
	connector.clients["storage"] = storage_client
	return connector, storage_client


def test_gcp_storage_upload_blob_uploads_string_payload_and_returns_metadata() -> None:
	module = _load_cloud_connector()
	connector, storage_client = _configured_connector(module)

	result = asyncio.run(
		connector.storage_upload_blob(
			"bucket",
			"path/item.txt",
			"hello",
			content_type="text/plain",
		)
	)

	blob = storage_client.buckets["bucket"].blobs["path/item.txt"]
	assert storage_client.bucket_calls == ["bucket"]
	assert blob.payload == b"hello"
	assert blob.content_type == "text/plain"
	assert result == {
		"bucket": "bucket",
		"blob": "path/item.txt",
		"content_type": "text/plain",
		"size_bytes": 5,
		"generation": "generation-1",
		"md5_hash": "md5-1",
		"public_url": "https://storage.example/bucket/path/item.txt",
		"uploaded_at": result["uploaded_at"],
		"service": "storage",
		"method": "upload_from_string",
	}
	assert result["uploaded_at"].endswith("+00:00")


def test_gcp_storage_upload_blob_initializes_missing_storage_client() -> None:
	module = _load_cloud_connector()
	config = module.GCPConfiguration(
		name="gcp-test",
		tenant_id="tenant-1",
		user_id="user-1",
		project_id="project-1",
		services=[],
		environment="development",
	)
	connector = module.GCPConnector(config)
	storage_client = FakeStorageClient()
	initialized_services: list[str] = []

	async def initialize_service(service: str) -> None:
		initialized_services.append(service)
		connector.clients[service] = storage_client

	connector._initialize_service_client = initialize_service

	result = asyncio.run(
		connector.storage_upload_blob(
			"bucket",
			"path/item.bin",
			b"\x00\x01",
			content_type="application/octet-stream",
		)
	)

	blob = storage_client.buckets["bucket"].blobs["path/item.bin"]
	assert initialized_services == ["storage"]
	assert blob.payload == b"\x00\x01"
	assert result["size_bytes"] == 2
