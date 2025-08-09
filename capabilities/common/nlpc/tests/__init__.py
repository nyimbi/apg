"""
APG Natural Language Processing Test Suite

Comprehensive test suite for enterprise NLP platform with async patterns,
integration testing, and APG ecosystem validation.

All tests follow APG patterns with modern pytest-asyncio and real objects.
"""

# Test configuration for APG NLP capability
TEST_CONFIG = {
	'test_tenant_id': 'test-tenant-nlp-12345',
	'test_user_id': 'test-user-nlp-67890',
	'test_database_url': 'sqlite:///test_nlp.db',
	'test_redis_url': 'redis://localhost:6379/15',
	'mock_external_services': True,
	'log_level': 'DEBUG',
	'test_timeout': 30,
	'enable_integration_tests': True,
	'enable_performance_tests': False  # Disabled by default
}

# Test data constants
TEST_TEXTS = {
	'positive_sentiment': "I love this product! It's absolutely amazing and works perfectly.",
	'negative_sentiment': "This is terrible. I hate it and want my money back.",
	'neutral_sentiment': "The product arrived on time. It's okay, nothing special.",
	'mixed_sentiment': "I love the design but hate the price. It's expensive but beautiful.",
	'long_text': "This is a very long text that will be used to test chunking and streaming capabilities. " * 50,
	'multilingual': "Hello world. Bonjour le monde. Hola mundo. こんにちは世界。",
	'entities_text': "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
	'technical_text': "The API endpoint returns a JSON response with status code 200 for successful requests."
}

# Mock model responses for testing
MOCK_MODEL_RESPONSES = {
	'sentiment_analysis': {
		'sentiment': 'positive',
		'confidence': 0.89,
		'scores': {'positive': 0.89, 'negative': 0.08, 'neutral': 0.03}
	},
	'entity_extraction': {
		'entities': [
			{'text': 'Apple Inc.', 'label': 'ORG', 'start': 0, 'end': 10, 'confidence': 0.95},
			{'text': 'Steve Jobs', 'label': 'PERSON', 'start': 25, 'end': 35, 'confidence': 0.98},
			{'text': 'Cupertino', 'label': 'GPE', 'start': 39, 'end': 48, 'confidence': 0.92}
		],
		'entity_count': 3
	},
	'text_classification': {
		'predicted_class': 'technology',
		'confidence': 0.85,
		'class_probabilities': {'technology': 0.85, 'business': 0.12, 'other': 0.03}
	}
}

__all__ = ['TEST_CONFIG', 'TEST_TEXTS', 'MOCK_MODEL_RESPONSES']