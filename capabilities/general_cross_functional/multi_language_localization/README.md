# APG Multi-language Localization Capability

Enterprise-grade internationalization and localization platform for the APG ecosystem, providing comprehensive multi-language support across 50+ languages with advanced cultural adaptation features.

## Overview

The Multi-language Localization capability delivers professional-grade i18n/l10n services including translation management, cultural formatting, user preference handling, and seamless APG platform integration.

### Key Features

- **50+ Language Support**: Comprehensive language coverage including RTL languages, complex scripts (CJK, Arabic, Devanagari), and regional variants
- **Advanced Translation Management**: Human and machine translation workflows with quality assurance, translation memory, and collaborative review processes
- **Cultural Formatting**: Locale-specific number, date, currency, and address formatting using industry-standard libraries
- **Real-time Translation**: Dynamic content translation with caching and fallback mechanisms
- **Translation Workbench**: Professional interface for translators with context, glossaries, and productivity tools
- **Enterprise Integration**: Seamless APG platform integration with service discovery and event streaming

## Quick Start

### Installation

```bash
# Install the capability
cd capabilities/general_cross_functional/multi_language_localization
pip install -r requirements.txt

# Setup database
alembic upgrade head

# Configure Redis
redis-server --port 6379
```

### Basic Usage

```python
from multi_language_localization.service import TranslationService, FormattingService

# Initialize services
translation_service = TranslationService(db_session, redis_client)
formatting_service = FormattingService(db_session, redis_client)

# Get translation
translation = await translation_service.get_translation(
    "welcome.message", 
    "es", 
    "app",
    variables={"username": "Juan"}
)

# Format currency
formatted = formatting_service.format_number(99.99, "es-ES", "currency")
```

## Architecture

### Components

- **Translation Service**: Core translation logic with caching and fallbacks
- **Language Management**: Language and locale configuration
- **Formatting Service**: Cultural formatting using Babel library
- **Content Management**: Translation key management and content extraction
- **User Preferences**: Personalized language and locale settings

### Database Schema

```sql
-- Languages and locales
ml_languages (id, code, name, native_name, script, direction, status)
ml_locales (id, language_id, region_code, locale_code, currency_code)

-- Translation content
ml_namespaces (id, name, description)
ml_translation_keys (id, namespace_id, key, source_text, context)
ml_translations (id, translation_key_id, language_id, content, status)

-- User preferences
ml_user_preferences (id, user_id, primary_language_id, timezone, settings)
```

## Language Support

### Supported Language Families

- **Latin Scripts**: English, Spanish, French, German, Italian, Portuguese, Dutch, etc.
- **Cyrillic Scripts**: Russian, Ukrainian, Bulgarian, Serbian, etc.
- **CJK Scripts**: Chinese (Simplified/Traditional), Japanese, Korean
- **Arabic Script**: Arabic, Persian, Urdu
- **Indic Scripts**: Hindi, Bengali, Tamil, Telugu, Gujarati, etc.
- **Other Scripts**: Hebrew, Thai, Greek, Armenian, Georgian

### Regional Variants

- English: en-US, en-GB, en-CA, en-AU
- Spanish: es-ES, es-MX, es-AR, es-CO
- French: fr-FR, fr-CA, fr-BE, fr-CH
- Portuguese: pt-BR, pt-PT
- Chinese: zh-CN, zh-TW, zh-HK

## API Reference

### Translation Endpoints

```http
GET /api/v1/translate?key=welcome.message&language=es&namespace=app
POST /api/v1/translate/batch
POST /api/v1/translations
POST /api/v1/detect-language
```

### Language Management

```http
GET /api/v1/languages
POST /api/v1/languages
GET /api/v1/locales
POST /api/v1/locales
```

### Formatting Services

```http
POST /api/v1/format
```

### Content Management

```http
GET /api/v1/translation-keys
POST /api/v1/translation-keys
POST /api/v1/extract-content
```

## Configuration

### Environment Variables

```bash
# Database
LOCALIZATION_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/localization

# Redis Cache
LOCALIZATION_REDIS_URL=redis://localhost:6379/0

# Machine Translation
GOOGLE_TRANSLATE_API_KEY=your_api_key
AZURE_TRANSLATOR_KEY=your_key

# Features
ENABLE_AUTO_TRANSLATION=true
ENABLE_TRANSLATION_MEMORY=true
DEFAULT_LOCALE=en-US
```

### Settings File

```json
{
  "localization": {
    "default_language": "en",
    "fallback_language": "en",
    "cache_ttl": 3600,
    "quality_threshold": 7.0,
    "supported_formats": ["text", "html", "markdown", "json"],
    "translation_providers": ["google", "azure", "deepl"]
  }
}
```

## Usage Examples

### Basic Translation

```python
# Simple translation
result = await translation_service.get_translation("hello", "fr")
# Returns: "Bonjour"

# Translation with variables
result = await translation_service.get_translation(
    "welcome.user",
    "es", 
    variables={"name": "María"}
)
# Returns: "Bienvenida, María"
```

### Bulk Translation

```python
keys = ["save", "cancel", "delete"]
translations = await translation_service.get_translations(keys, "de", "ui")
# Returns: {"save": "Speichern", "cancel": "Abbrechen", "delete": "Löschen"}
```

### Cultural Formatting

```python
# Currency formatting
amount = formatting_service.format_number(1234.56, "de-DE", "currency")
# Returns: "1.234,56 €"

# Date formatting  
date = formatting_service.format_date(datetime.now(), "ja-JP", "long")
# Returns: "2025年1月26日"
```

### User Preferences

```python
# Set user language preference
await user_service.set_user_preferences("user_123", {
    "primary_language_id": "lang_es",
    "timezone": "Europe/Madrid",
    "auto_translate_enabled": True
})

# Get user's preferred language
language = await user_service.get_user_language("user_123")
# Returns: "es"
```

## Translation Workflows

### Professional Translation Process

1. **Content Extraction**: Extract translatable strings from source files
2. **Key Creation**: Generate translation keys with context and metadata
3. **Translation Assignment**: Assign translation tasks to linguists
4. **Translation**: Professional translation using workbench interface
5. **Review**: Quality assurance and linguistic review
6. **Approval**: Final approval and publishing to production
7. **Deployment**: Automatic deployment via APG platform

### Machine Translation Integration

```python
# Enable machine translation
await translation_service.set_translation(
    key="auto.generated",
    language_code="fr",
    content="Machine translated content",
    translation_type=MLTranslationType.MACHINE,
    auto_approve=False  # Requires human review
)
```

## UI Components

### Translation Dashboard

- Language coverage overview
- Translation progress tracking
- Quality metrics and analytics
- Team productivity monitoring

### Translation Workbench

- Side-by-side source/target editing
- Translation memory suggestions
- Glossary integration
- Context and screenshots
- Keyboard shortcuts for productivity

### Administrative Interface

- Language and locale management
- User permission configuration
- Translation project management
- Import/export functionality

## Testing

### Running Tests

```bash
# Unit tests
pytest tests/test_models.py -v

# Service integration tests
pytest tests/test_services.py -v

# API endpoint tests
pytest tests/test_api.py -v

# Internationalization tests
pytest tests/test_i18n.py -v

# Full test suite
pytest tests/ -v --cov=multi_language_localization
```

### Test Coverage

- Unit tests for all models and validation
- Service layer integration tests
- API endpoint testing with authentication
- Internationalization tests for RTL and complex scripts
- Performance tests for large datasets
- Security tests for input validation

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: localization-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: localization-service
  template:
    metadata:
      labels:
        app: localization-service
    spec:
      containers:
      - name: localization
        image: datacraft/apg-localization:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: localization-secrets
              key: database-url
```

### Performance Considerations

- **Redis Caching**: Translation caching with TTL configuration
- **Database Indexing**: Optimized indexes for translation lookups
- **CDN Integration**: Static content delivery for UI assets
- **Load Balancing**: Horizontal scaling with session affinity
- **Monitoring**: APG platform integration for metrics and alerts

## Best Practices

### Translation Management

1. **Consistent Naming**: Use hierarchical key naming (e.g., `ui.buttons.save`)
2. **Context Information**: Provide detailed context for translators
3. **Variable Handling**: Use clear variable names and documentation
4. **Pluralization**: Implement proper plural rule handling
5. **Character Limits**: Set appropriate length constraints for UI text

### Performance Optimization

1. **Lazy Loading**: Load translations on demand
2. **Bulk Operations**: Use batch APIs for multiple translations
3. **Caching Strategy**: Implement multi-level caching
4. **CDN Usage**: Serve static translation files via CDN
5. **Database Optimization**: Use proper indexing and query optimization

### Security Considerations

1. **Input Validation**: Sanitize all translation content
2. **Access Control**: Implement role-based permissions
3. **Audit Logging**: Track all translation changes
4. **XSS Prevention**: Escape HTML content appropriately
5. **API Security**: Use authentication and rate limiting

## Troubleshooting

### Common Issues

**Translation Not Found**
```python
# Check fallback configuration
result = await translation_service.get_translation(
    "missing.key", 
    "es", 
    fallback=True
)
```

**Invalid Locale Formatting**
```python
# Verify locale code format
try:
    result = formatting_service.format_number(123.45, "invalid-locale", "decimal")
except ValueError as e:
    logger.error(f"Invalid locale: {e}")
```

**Cache Issues**
```bash
# Clear Redis cache
redis-cli FLUSHDB

# Check cache connectivity
redis-cli PING
```

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger('multi_language_localization').setLevel(logging.DEBUG)
```

Monitor performance:
```python
import time
start = time.time()
result = await translation_service.get_translation("key", "lang")
logger.info(f"Translation time: {time.time() - start:.3f}s")
```

## Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements-dev.txt`
3. Run database migrations: `alembic upgrade head`
4. Start Redis: `redis-server`
5. Run tests: `pytest tests/ -v`

### Adding New Languages

1. Update language definitions in `models.py`
2. Add locale configuration
3. Update formatting rules
4. Add test cases for new language
5. Update documentation

### Translation Provider Integration

1. Implement provider interface
2. Add configuration options
3. Update service layer
4. Add comprehensive tests
5. Document usage

## Support

- **Documentation**: [APG Platform Docs](https://docs.apg.platform)
- **API Reference**: [Localization API](https://api.apg.platform/localization)
- **Community**: [APG Community Forum](https://community.apg.platform)
- **Issues**: [GitHub Issues](https://github.com/datacraft/apg/issues)

## License

Copyright © 2025 Datacraft. All rights reserved.

---

**Company**: Datacraft  
**Website**: www.datacraft.co.ke  
**Contact**: nyimbi@gmail.com