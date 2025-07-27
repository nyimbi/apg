"""
APG Multi-language Localization - API Tests

API endpoint tests for the localization service including authentication,
validation, performance, and integration testing.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025 Datacraft. All rights reserved.
"""

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import json

from ..api import create_localization_api, LocalizationClient
from ..models import (
	MLLanguage, MLLocale, MLNamespace, MLTranslationKey, MLTranslation,
	MLTranslationStatus, MLTranslationType, MLContentType,
	MLLanguageCreate, MLLocaleCreate, MLTranslationKeyCreate
)

@pytest.fixture
def api_client():
	"""Create test client for the API"""
	app = create_localization_api()
	return TestClient(app)

@pytest.fixture
def mock_services():
	"""Create mock services for testing"""
	return {
		"translation_service": AsyncMock(),
		"language_service": AsyncMock(),
		"formatting_service": Mock(),
		"content_service": AsyncMock(),
		"user_preference_service": AsyncMock()
	}

class TestTranslationEndpoints:
	"""Test translation-related API endpoints"""
	
	def test_get_translation_success(self, api_client, mock_services):
		"""Test successful translation retrieval"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			mock_services["translation_service"].get_translation.return_value = "Hola"
			
			response = api_client.get("/api/v1/translate?key=hello&language=es")
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["translation"] == "Hola"
			assert data["meta"]["key"] == "hello"
			assert data["meta"]["language"] == "es"
	
	def test_get_translation_not_found(self, api_client, mock_services):
		"""Test translation not found"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			mock_services["translation_service"].get_translation.return_value = None
			
			response = api_client.get("/api/v1/translate?key=missing&language=es")
			
			assert response.status_code == 404
			data = response.json()
			assert data["success"] is False
			assert "not found" in data["message"]
	
	def test_get_translation_missing_parameters(self, api_client):
		"""Test translation request with missing parameters"""
		response = api_client.get("/api/v1/translate?key=hello")  # Missing language
		assert response.status_code == 422
		
		response = api_client.get("/api/v1/translate?language=es")  # Missing key
		assert response.status_code == 422
	
	def test_get_translations_batch_success(self, api_client, mock_services):
		"""Test successful batch translation retrieval"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			mock_services["translation_service"].get_translations.return_value = {
				"hello": "Hola",
				"goodbye": "Adiós",
				"missing": None
			}
			
			request_data = {
				"keys": ["hello", "goodbye", "missing"],
				"language": "es",
				"namespace": "test"
			}
			
			response = api_client.post("/api/v1/translate/batch", json=request_data)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["translations"]["hello"] == "Hola"
			assert data["data"]["translations"]["goodbye"] == "Adiós"
			assert data["data"]["translations"]["missing"] is None
			assert data["meta"]["total_keys"] == 3
			assert data["meta"]["found_translations"] == 2
	
	def test_get_translations_batch_validation(self, api_client):
		"""Test batch translation validation"""
		# Empty keys list
		request_data = {
			"keys": [],
			"language": "es"
		}
		response = api_client.post("/api/v1/translate/batch", json=request_data)
		assert response.status_code == 422
		
		# Missing language
		request_data = {
			"keys": ["hello", "goodbye"]
		}
		response = api_client.post("/api/v1/translate/batch", json=request_data)
		assert response.status_code == 422
	
	def test_create_translation_success(self, api_client, mock_services):
		"""Test successful translation creation"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			mock_services["translation_service"].set_translation.return_value = "translation_123"
			
			request_data = {
				"translation_key_id": "key_123",
				"language_id": "lang_es",
				"content": "Hola mundo",
				"translation_type": "human"
			}
			
			response = api_client.post("/api/v1/translations", json=request_data)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["translation_id"] == "translation_123"
			assert "created successfully" in data["message"]
	
	def test_detect_language_success(self, api_client, mock_services):
		"""Test successful language detection"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			mock_services["translation_service"].detect_language.return_value = "fr"
			
			request_data = {"text": "Bonjour le monde"}
			response = api_client.post("/api/v1/detect-language", json=request_data)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["detected_language"] == "fr"
			assert data["meta"]["text_length"] == len("Bonjour le monde")
	
	def test_detect_language_failed(self, api_client, mock_services):
		"""Test failed language detection"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			mock_services["translation_service"].detect_language.return_value = None
			
			request_data = {"text": "xyz123"}
			response = api_client.post("/api/v1/detect-language", json=request_data)
			
			assert response.status_code == 422
			data = response.json()
			assert data["success"] is False
			assert "Unable to detect" in data["message"]

class TestLanguageManagementEndpoints:
	"""Test language management API endpoints"""
	
	def test_get_languages_success(self, api_client, mock_services):
		"""Test successful language retrieval"""
		with patch('integration_api_management.api.get_language_service', return_value=mock_services["language_service"]):
			mock_languages = [
				Mock(model_dump=lambda: {"code": "en", "name": "English", "native_name": "English"}),
				Mock(model_dump=lambda: {"code": "es", "name": "Spanish", "native_name": "Español"})
			]
			mock_services["language_service"].get_supported_languages.return_value = mock_languages
			
			response = api_client.get("/api/v1/languages")
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert len(data["data"]["languages"]) == 2
			assert data["meta"]["total"] == 2
	
	def test_get_languages_include_inactive(self, api_client, mock_services):
		"""Test language retrieval including inactive languages"""
		with patch('integration_api_management.api.get_language_service', return_value=mock_services["language_service"]):
			mock_services["language_service"].get_supported_languages.return_value = []
			
			response = api_client.get("/api/v1/languages?include_inactive=true")
			
			mock_services["language_service"].get_supported_languages.assert_called_once_with(True)
	
	def test_create_language_success(self, api_client, mock_services):
		"""Test successful language creation"""
		with patch('integration_api_management.api.get_language_service', return_value=mock_services["language_service"]):
			mock_language = Mock(model_dump=lambda: {
				"id": "lang_123",
				"code": "pt",
				"name": "Portuguese",
				"native_name": "Português"
			})
			mock_services["language_service"].create_language.return_value = mock_language
			
			request_data = {
				"code": "pt",
				"name": "Portuguese",
				"native_name": "Português",
				"script": "Latn",
				"direction": "ltr"
			}
			
			response = api_client.post("/api/v1/languages", json=request_data)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["language"]["code"] == "pt"
			assert "created successfully" in data["message"]
	
	def test_create_language_validation_error(self, api_client):
		"""Test language creation with validation errors"""
		# Missing required fields
		request_data = {
			"code": "pt"
			# Missing name, native_name, script
		}
		
		response = api_client.post("/api/v1/languages", json=request_data)
		assert response.status_code == 422
	
	def test_create_language_duplicate_error(self, api_client, mock_services):
		"""Test creating duplicate language"""
		with patch('integration_api_management.api.get_language_service', return_value=mock_services["language_service"]):
			mock_services["language_service"].create_language.side_effect = ValueError("Language already exists")
			
			request_data = {
				"code": "en",
				"name": "English",
				"native_name": "English",
				"script": "Latn"
			}
			
			response = api_client.post("/api/v1/languages", json=request_data)
			
			assert response.status_code == 400
			data = response.json()
			assert data["success"] is False
			assert "already exists" in data["message"]
	
	def test_get_locales_success(self, api_client, mock_services):
		"""Test successful locale retrieval"""
		with patch('integration_api_management.api.get_language_service', return_value=mock_services["language_service"]):
			mock_locales = [
				{"locale_code": "en-US", "currency_code": "USD"},
				{"locale_code": "en-GB", "currency_code": "GBP"}
			]
			mock_services["language_service"].get_locales.return_value = mock_locales
			
			response = api_client.get("/api/v1/locales")
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert len(data["data"]["locales"]) == 2
	
	def test_get_locales_filtered_by_language(self, api_client, mock_services):
		"""Test locale retrieval filtered by language"""
		with patch('integration_api_management.api.get_language_service', return_value=mock_services["language_service"]):
			mock_services["language_service"].get_locales.return_value = []
			
			response = api_client.get("/api/v1/locales?language_code=en")
			
			mock_services["language_service"].get_locales.assert_called_once_with("en")

class TestFormattingEndpoints:
	"""Test formatting API endpoints"""
	
	def test_format_number_success(self, api_client, mock_services):
		"""Test successful number formatting"""
		with patch('integration_api_management.api.get_formatting_service', return_value=mock_services["formatting_service"]):
			mock_services["formatting_service"].format_number.return_value = "1,234.56"
			
			request_data = {
				"value": 1234.56,
				"locale": "en-US",
				"format_type": "decimal"
			}
			
			response = api_client.post("/api/v1/format", json=request_data)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["formatted_value"] == "1,234.56"
			assert data["meta"]["locale"] == "en-US"
	
	def test_format_date_success(self, api_client, mock_services):
		"""Test successful date formatting"""
		with patch('integration_api_management.api.get_formatting_service', return_value=mock_services["formatting_service"]):
			mock_services["formatting_service"].format_date.return_value = "Jan 15, 2025"
			
			request_data = {
				"value": "2025-01-15T10:30:00Z",
				"locale": "en-US",
				"format_type": "date"
			}
			
			response = api_client.post("/api/v1/format", json=request_data)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["formatted_value"] == "Jan 15, 2025"
	
	def test_format_unsupported_type(self, api_client):
		"""Test formatting unsupported value type"""
		request_data = {
			"value": {"unsupported": "object"},
			"locale": "en-US",
			"format_type": "decimal"
		}
		
		response = api_client.post("/api/v1/format", json=request_data)
		assert response.status_code == 400

class TestContentManagementEndpoints:
	"""Test content management API endpoints"""
	
	def test_create_translation_key_success(self, api_client, mock_services):
		"""Test successful translation key creation"""
		with patch('integration_api_management.api.get_content_service', return_value=mock_services["content_service"]):
			mock_services["content_service"].create_translation_key.return_value = "key_123"
			
			request_data = {
				"namespace_id": "namespace_123",
				"key": "test.new.key",
				"source_text": "New test key",
				"context": "Test context",
				"content_type": "ui_text",
				"translation_priority": 75
			}
			
			response = api_client.post("/api/v1/translation-keys", json=request_data)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["translation_key_id"] == "key_123"
	
	def test_get_translation_keys_success(self, api_client, mock_services):
		"""Test successful translation key retrieval"""
		with patch('integration_api_management.api.get_content_service', return_value=mock_services["content_service"]):
			mock_keys = [
				{"id": "key_1", "key": "test.key.1", "source_text": "Test 1"},
				{"id": "key_2", "key": "test.key.2", "source_text": "Test 2"}
			]
			mock_services["content_service"].get_translation_keys.return_value = mock_keys
			
			response = api_client.get("/api/v1/translation-keys?namespace=test&page=1&per_page=25")
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			# Check pagination structure
			assert "items" in data["data"]
			assert "total" in data["data"]
			assert "page" in data["data"]
	
	def test_extract_content_success(self, api_client, mock_services):
		"""Test successful content extraction"""
		with patch('integration_api_management.api.get_content_service', return_value=mock_services["content_service"]):
			mock_services["content_service"].extract_content_keys.return_value = [
				"Welcome to our site",
				"Click here to continue",
				"Contact us"
			]
			
			request_data = {
				"content": "<h1>Welcome to our site</h1><button>Click here to continue</button>",
				"content_type": "html"
			}
			
			response = api_client.post("/api/v1/extract-content", json=request_data)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert len(data["data"]["extracted_content"]) == 3
			assert data["meta"]["total_items"] == 3

class TestUserPreferenceEndpoints:
	"""Test user preference API endpoints"""
	
	def test_get_user_preferences_success(self, api_client, mock_services):
		"""Test successful user preference retrieval"""
		with patch('integration_api_management.api.get_user_preference_service', return_value=mock_services["user_preference_service"]):
			with patch('integration_api_management.api.get_current_user', return_value="user_123"):
				mock_preferences = {
					"primary_language_id": "lang_en",
					"timezone": "UTC",
					"auto_translate_enabled": True
				}
				mock_services["user_preference_service"].get_user_preferences.return_value = mock_preferences
				
				response = api_client.get("/api/v1/user/user_123/preferences")
				
				assert response.status_code == 200
				data = response.json()
				assert data["success"] is True
				assert data["data"]["preferences"]["timezone"] == "UTC"
	
	def test_get_user_preferences_unauthorized(self, api_client, mock_services):
		"""Test unauthorized access to user preferences"""
		with patch('integration_api_management.api.get_current_user', return_value="different_user"):
			response = api_client.get("/api/v1/user/user_123/preferences")
			
			assert response.status_code == 403
			data = response.json()
			assert data["success"] is False
			assert "Access denied" in data["message"]
	
	def test_update_user_preferences_success(self, api_client, mock_services):
		"""Test successful user preference update"""
		with patch('integration_api_management.api.get_user_preference_service', return_value=mock_services["user_preference_service"]):
			with patch('integration_api_management.api.get_current_user', return_value="user_123"):
				mock_services["user_preference_service"].set_user_preferences.return_value = True
				
				request_data = {
					"timezone": "America/New_York",
					"auto_translate_enabled": False,
					"font_size_adjustment": 1.2
				}
				
				response = api_client.put("/api/v1/user/user_123/preferences", json=request_data)
				
				assert response.status_code == 200
				data = response.json()
				assert data["success"] is True
				assert data["data"]["updated"] is True

class TestStatisticsEndpoints:
	"""Test statistics and analytics API endpoints"""
	
	def test_get_translation_stats_success(self, api_client, mock_services):
		"""Test successful statistics retrieval"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			mock_stats = Mock(model_dump=lambda: {
				"total_keys": 1000,
				"translated_keys": 850,
				"completion_percentage": 85.0,
				"languages_supported": 25,
				"quality_average": 8.2
			})
			mock_services["translation_service"].get_translation_stats.return_value = mock_stats
			
			response = api_client.get("/api/v1/stats")
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["stats"]["total_keys"] == 1000
			assert data["data"]["stats"]["completion_percentage"] == 85.0
	
	def test_get_translation_stats_filtered(self, api_client, mock_services):
		"""Test statistics retrieval with namespace filter"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			mock_services["translation_service"].get_translation_stats.return_value = Mock(model_dump=lambda: {})
			
			response = api_client.get("/api/v1/stats?namespace=ui")
			
			mock_services["translation_service"].get_translation_stats.assert_called_once_with("ui")

class TestHealthEndpoints:
	"""Test health and status API endpoints"""
	
	def test_health_check(self, api_client):
		"""Test basic health check"""
		response = api_client.get("/health")
		
		assert response.status_code == 200
		data = response.json()
		assert data["success"] is True
		assert data["data"]["status"] == "healthy"
		assert "operational" in data["message"]
	
	def test_status_check(self, api_client, mock_services):
		"""Test detailed status check"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			response = api_client.get("/status")
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert "timestamp" in data["data"]
			assert "version" in data["data"]
			assert "services" in data["data"]

@pytest.mark.performance
class TestAPIPerformance:
	"""Performance tests for API endpoints"""
	
	def test_translation_endpoint_performance(self, api_client, mock_services):
		"""Test translation endpoint response time"""
		import time
		
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			mock_services["translation_service"].get_translation.return_value = "Test translation"
			
			start_time = time.time()
			response = api_client.get("/api/v1/translate?key=test&language=en")
			end_time = time.time()
			
			assert response.status_code == 200
			response_time = end_time - start_time
			assert response_time < 0.1  # Should respond within 100ms
	
	def test_batch_translation_performance(self, api_client, mock_services):
		"""Test batch translation endpoint with large payload"""
		with patch('integration_api_management.api.get_translation_service', return_value=mock_services["translation_service"]):
			# Mock returning translations for 100 keys
			mock_translations = {f"key_{i}": f"translation_{i}" for i in range(100)}
			mock_services["translation_service"].get_translations.return_value = mock_translations
			
			request_data = {
				"keys": [f"key_{i}" for i in range(100)],
				"language": "en"
			}
			
			import time
			start_time = time.time()
			response = api_client.post("/api/v1/translate/batch", json=request_data)
			end_time = time.time()
			
			assert response.status_code == 200
			response_time = end_time - start_time
			assert response_time < 0.5  # Should handle 100 keys within 500ms

class TestLocalizationClient:
	"""Test the client SDK"""
	
	@pytest.mark.asyncio
	async def test_client_get_translation(self):
		"""Test client SDK translation retrieval"""
		client = LocalizationClient("http://localhost:8000")
		
		# Mock the HTTP session behavior
		with patch.object(client, 'session') as mock_session:
			# This would test the actual HTTP client implementation
			pass
	
	def test_client_initialization(self):
		"""Test client SDK initialization"""
		client = LocalizationClient("http://localhost:8000", "api_key_123")
		
		assert client.base_url == "http://localhost:8000"
		assert client.api_key == "api_key_123"

@pytest.mark.integration
class TestAPIIntegration:
	"""Full integration tests with real database"""
	
	# These would be comprehensive end-to-end tests
	# that use actual database and Redis instances
	pass

if __name__ == "__main__":
	pytest.main([__file__])