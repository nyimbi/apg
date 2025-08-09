"""
APG Document Service Test Configuration

Pytest configuration and fixtures for APG-integrated testing
with async patterns and real object fixtures.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, AsyncGenerator
from unittest.mock import Mock, AsyncMock
from uuid_extensions import uuid7str

from ..config import APGDocumentConfig, override_config
from ..models import DSDocument, DSTemplate, DSProcessingJob, ProcessingStatus


@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session"""
	loop = asyncio.get_event_loop()
	yield loop
	loop.close()


@pytest.fixture
async def temp_storage():
	"""Create temporary storage directory for testing"""
	temp_dir = tempfile.mkdtemp()
	yield Path(temp_dir)
	shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_storage: Path):
	"""Test configuration with temporary storage"""
	return override_config(
		storage_backend="filesystem",
		storage_path=str(temp_storage),
		database_url="sqlite:///test.db",
		tenant_mode="single",
		max_file_size_mb=10,
		processing_timeout_seconds=30,
		cache_enabled=False,
		notifications_enabled=False
	)


@pytest.fixture
def tenant_id():
	"""Test tenant ID"""
	return uuid7str()


@pytest.fixture
def user_id():
	"""Test user ID"""
	return uuid7str()


@pytest.fixture
def mock_apg_context():
	"""Mock APG context with capability services"""
	context = Mock()
	
	# Mock auth_rbac service
	auth_service = AsyncMock()
	auth_service.authorize_action = AsyncMock(return_value=True)
	auth_service.get_user_permissions = AsyncMock(return_value=["document:read", "document:write"])
	auth_service.evaluate_access = AsyncMock(return_value=Mock(allowed=True))
	context.get_capability = Mock(return_value=auth_service)
	
	# Mock audit_compliance service
	audit_service = AsyncMock()
	audit_service.log_event = AsyncMock(return_value=True)
	audit_service.log_access_attempt = AsyncMock(return_value=True)
	context.audit_compliance = audit_service
	
	# Mock computer_vision service
	vision_service = AsyncMock()
	vision_service.extract_text = AsyncMock(return_value=Mock(
		text="Sample extracted text",
		confidence=0.95,
		regions=[]
	))
	vision_service.analyze_layout = AsyncMock(return_value=Mock(
		regions=[],
		elements=[]
	))
	vision_service.assess_image_quality = AsyncMock(return_value=0.9)
	context.computer_vision = vision_service
	
	# Mock nlp service
	nlp_service = AsyncMock()
	nlp_service.extract_entities = AsyncMock(return_value=[])
	nlp_service.analyze_sentiment = AsyncMock(return_value={"score": 0.5, "label": "neutral"})
	nlp_service.generate_summary = AsyncMock(return_value="Sample summary")
	nlp_service.identify_topics = AsyncMock(return_value=[{"topic": "general", "confidence": 0.8}])
	context.nlp = nlp_service
	
	# Mock ai_orchestration service
	ai_service = AsyncMock()
	workflow_mock = Mock()
	workflow_mock.complete = AsyncMock(return_value=True)
	ai_service.create_workflow = AsyncMock(return_value=workflow_mock)
	context.ai_orchestration = ai_service
	
	return context


@pytest.fixture
def sample_document(tenant_id: str, user_id: str):
	"""Sample document for testing"""
	return DSDocument(
		tenant_id=tenant_id,
		created_by=user_id,
		title="Test Document",
		description="A test document for unit testing",
		content="This is sample content for testing purposes.",
		classification="public",
		status="draft"
	)


@pytest.fixture
def sample_template(tenant_id: str, user_id: str):
	"""Sample document template for testing"""
	return DSTemplate(
		tenant_id=tenant_id,
		created_by=user_id,
		name="Test Template",
		description="A test template",
		template_content="{{title}}\n\n{{content}}",
		template_variables={"title": "string", "content": "text"},
		category="general"
	)


@pytest.fixture
def sample_processing_job(tenant_id: str, user_id: str):
	"""Sample processing job for testing"""
	return DSProcessingJob(
		tenant_id=tenant_id,
		job_name="Test Processing Job",
		processing_type="document_analysis",
		input_file_path="/tmp/test.pdf",
		created_by=user_id,
		status=ProcessingStatus.PENDING
	)


@pytest.fixture
def sample_pdf_content():
	"""Sample PDF content as bytes for testing"""
	# Minimal PDF content for testing
	return b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Test Document) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000207 00000 n
trailer<</Size 5/Root 1 0 R>>
startxref
299
%%EOF"""


@pytest.fixture
def sample_image_content():
	"""Sample image content as bytes for testing"""
	# Minimal PNG content for testing
	return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'


@pytest.fixture 
async def document_files(temp_storage: Path, sample_pdf_content: bytes, sample_image_content: bytes):
	"""Create sample document files for testing"""
	pdf_file = temp_storage / "test.pdf"
	image_file = temp_storage / "test.png"
	text_file = temp_storage / "test.txt"
	
	pdf_file.write_bytes(sample_pdf_content)
	image_file.write_bytes(sample_image_content)
	text_file.write_text("This is a test text document.")
	
	return {
		"pdf": str(pdf_file),
		"image": str(image_file),
		"text": str(text_file)
	}


@pytest.fixture
def mock_database_session():
	"""Mock database session for testing"""
	session = Mock()
	session.add = Mock()
	session.commit = AsyncMock()
	session.rollback = AsyncMock()
	session.close = AsyncMock()
	session.execute = AsyncMock()
	session.query = Mock()
	return session


class MockOllamaService:
	"""Mock Ollama service for testing"""
	
	def __init__(self):
		self.session = None
	
	async def extract_text(self, image_path: str):
		"""Mock text extraction"""
		return Mock(
			text="Extracted text from image",
			confidence=0.95,
			language="en"
		)
	
	async def analyze_content(self, content: str):
		"""Mock content analysis"""
		return Mock(
			entities=["test", "document"],
			sentiment="neutral",
			topics=["general"],
			summary="Test document content"
		)
	
	async def close(self):
		"""Mock close method"""
		pass


@pytest.fixture
def mock_ollama_service():
	"""Mock Ollama service fixture"""
	return MockOllamaService()


# Performance testing fixtures
@pytest.fixture
def performance_test_documents():
	"""Generate multiple documents for performance testing"""
	documents = []
	for i in range(100):
		doc = DSDocument(
			tenant_id=uuid7str(),
			created_by=uuid7str(),
			title=f"Performance Test Document {i}",
			content=f"Content for document {i}" * 100,  # Make content substantial
			classification="public",
			status="draft"
		)
		documents.append(doc)
	return documents


# Integration testing fixtures
@pytest.fixture
def integration_test_config():
	"""Configuration for integration tests"""
	return {
		"database_url": "postgresql://test:test@localhost/test_documents",
		"ollama_base_url": "http://localhost:11434",
		"enable_real_services": False,  # Set to True for full integration tests
		"test_timeout": 30
	}


# Helper functions for tests
def assert_document_valid(document: DSDocument):
	"""Assert that a document is valid"""
	assert document.document_id
	assert document.tenant_id
	assert document.created_by
	assert document.title
	assert document.created_at
	assert document.status in ["draft", "published", "archived", "deleted"]


def assert_processing_successful(result: Dict[str, Any]):
	"""Assert that processing completed successfully"""
	assert result is not None
	assert result.get("success", False) is True
	assert "processing_time" in result
	assert result["processing_time"] > 0