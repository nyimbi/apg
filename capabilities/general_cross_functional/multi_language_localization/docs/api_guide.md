# Multi-language Localization API Guide

Complete API reference for the APG Multi-language Localization capability with authentication, endpoints, examples, and integration patterns.

## Table of Contents

1. [Authentication](#authentication)
2. [Translation Endpoints](#translation-endpoints)
3. [Language Management](#language-management)
4. [Formatting Services](#formatting-services)
5. [Content Management](#content-management)
6. [User Preferences](#user-preferences)
7. [Statistics & Analytics](#statistics--analytics)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [SDKs & Integration](#sdks--integration)

## Authentication

### API Key Authentication

```http
GET /api/v1/translate
Authorization: Bearer your_api_key_here
Content-Type: application/json
```

### JWT Token Authentication

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## Translation Endpoints

### Get Single Translation

Retrieve a single translation for a specific key and language.

**Endpoint**: `GET /api/v1/translate`

**Parameters**:
- `key` (required): Translation key identifier
- `language` (required): Target language code (ISO 639-1)
- `namespace` (optional): Translation namespace
- `variables` (optional): JSON object with variable substitutions
- `fallback` (optional): Enable fallback to default language (default: true)

**Example Request**:
```http
GET /api/v1/translate?key=welcome.message&language=es&namespace=app&variables={"username":"Juan"}
Authorization: Bearer your_api_key
```

**Example Response**:
```json
{
  "success": true,
  "data": {
    "translation": "¡Bienvenido, Juan!",
    "key": "welcome.message",
    "language": "es",
    "namespace": "app",
    "variables_applied": true
  },
  "meta": {
    "cached": true,
    "fallback_used": false,
    "translation_quality": 9.2,
    "last_updated": "2025-01-26T10:30:00Z"
  }
}
```

### Batch Translation

Retrieve multiple translations in a single request for improved performance.

**Endpoint**: `POST /api/v1/translate/batch`

**Request Body**:
```json
{
  "keys": ["save", "cancel", "delete", "welcome.message"],
  "language": "fr",
  "namespace": "ui",
  "variables": {
    "welcome.message": {"username": "Marie"}
  },
  "fallback": true
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "translations": {
      "save": "Enregistrer",
      "cancel": "Annuler", 
      "delete": "Supprimer",
      "welcome.message": "Bienvenue, Marie!"
    },
    "missing_keys": [],
    "fallback_used": false
  },
  "meta": {
    "total_keys": 4,
    "found_translations": 4,
    "cache_hits": 3,
    "processing_time_ms": 12
  }
}
```

### Create Translation

Add or update a translation for a specific key and language.

**Endpoint**: `POST /api/v1/translations`

**Request Body**:
```json
{
  "translation_key_id": "key_123",
  "language_id": "lang_es", 
  "content": "Nuevo contenido traducido",
  "translation_type": "human",
  "quality_score": 9.0,
  "translator_id": "translator_456",
  "context": "Updated button text",
  "auto_approve": false
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "translation_id": "trans_789",
    "status": "pending_review",
    "created_at": "2025-01-26T10:30:00Z"
  },
  "message": "Translation created successfully and queued for review"
}
```

### Language Detection

Automatically detect the language of provided text.

**Endpoint**: `POST /api/v1/detect-language`

**Request Body**:
```json
{
  "text": "Bonjour le monde, comment allez-vous?",
  "confidence_threshold": 0.8
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "detected_language": "fr",
    "confidence": 0.95,
    "alternative_languages": [
      {"language": "fr-CA", "confidence": 0.12},
      {"language": "fr-BE", "confidence": 0.08}
    ]
  },
  "meta": {
    "text_length": 42,
    "processing_time_ms": 156
  }
}
```

## Language Management

### Get Supported Languages

Retrieve all supported languages with optional filtering.

**Endpoint**: `GET /api/v1/languages`

**Parameters**:
- `include_inactive` (optional): Include deprecated languages (default: false)
- `script` (optional): Filter by script (e.g., "Latn", "Arab", "Hans")
- `direction` (optional): Filter by text direction ("ltr", "rtl")
- `status` (optional): Filter by status ("active", "deprecated", "beta")

**Example Request**:
```http
GET /api/v1/languages?include_inactive=false&script=Arab
```

**Response**:
```json
{
  "success": true,
  "data": {
    "languages": [
      {
        "id": "lang_ar",
        "code": "ar",
        "name": "Arabic",
        "native_name": "العربية",
        "script": "Arab",
        "direction": "rtl",
        "status": "active",
        "completion_percentage": 85.2,
        "priority": 75,
        "fallback_language": null
      },
      {
        "id": "lang_fa", 
        "code": "fa",
        "name": "Persian",
        "native_name": "فارسی",
        "script": "Arab",
        "direction": "rtl", 
        "status": "active",
        "completion_percentage": 72.1,
        "priority": 50,
        "fallback_language": "ar"
      }
    ]
  },
  "meta": {
    "total": 2,
    "filtered": true,
    "available_scripts": ["Latn", "Arab", "Hans", "Cyrl", "Deva"]
  }
}
```

### Create Language

Add a new language to the system.

**Endpoint**: `POST /api/v1/languages`

**Request Body**:
```json
{
  "code": "sw",
  "name": "Swahili",
  "native_name": "Kiswahili", 
  "script": "Latn",
  "direction": "ltr",
  "region_codes": ["KE", "TZ", "UG"],
  "priority": 60,
  "fallback_language_id": "lang_en"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "language": {
      "id": "lang_sw",
      "code": "sw",
      "name": "Swahili",
      "native_name": "Kiswahili",
      "script": "Latn",
      "direction": "ltr",
      "status": "active",
      "created_at": "2025-01-26T10:30:00Z"
    }
  },
  "message": "Language created successfully"
}
```

### Get Locales

Retrieve locale configurations for formatting.

**Endpoint**: `GET /api/v1/locales`

**Parameters**:
- `language_code` (optional): Filter by language code
- `region_code` (optional): Filter by region code
- `currency_code` (optional): Filter by currency

**Response**:
```json
{
  "success": true,
  "data": {
    "locales": [
      {
        "id": "locale_en_us",
        "locale_code": "en-US",
        "language_code": "en",
        "region_code": "US",
        "currency_code": "USD",
        "date_format": "MM/dd/yyyy",
        "time_format": "h:mm a",
        "number_format": {
          "decimal_separator": ".",
          "group_separator": ",",
          "group_size": 3
        },
        "first_day_of_week": 0
      }
    ]
  },
  "meta": {
    "total": 1
  }
}
```

## Formatting Services

### Format Values

Format numbers, dates, and times according to locale conventions.

**Endpoint**: `POST /api/v1/format`

**Request Body**:
```json
{
  "value": 1234.56,
  "locale": "de-DE", 
  "format_type": "currency"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "formatted_value": "1.234,56 €",
    "original_value": 1234.56,
    "format_type": "currency"
  },
  "meta": {
    "locale": "de-DE",
    "currency_code": "EUR",
    "formatting_rules_applied": ["group_separator", "decimal_separator", "currency_symbol"]
  }
}
```

**Supported Format Types**:
- `decimal`: Standard number formatting
- `currency`: Currency formatting with symbol
- `percent`: Percentage formatting  
- `date`: Date formatting
- `time`: Time formatting
- `datetime`: Combined date and time

**Date/Time Formatting**:
```json
{
  "value": "2025-01-26T15:30:00Z",
  "locale": "ja-JP",
  "format_type": "date",
  "style": "long"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "formatted_value": "2025年1月26日",
    "original_value": "2025-01-26T15:30:00Z"
  }
}
```

## Content Management

### Get Translation Keys

Retrieve translation keys with filtering and pagination.

**Endpoint**: `GET /api/v1/translation-keys`

**Parameters**:
- `namespace` (optional): Filter by namespace
- `content_type` (optional): Filter by content type
- `status` (optional): Filter by translation status
- `search` (optional): Search in key names and source text
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 25)

**Example Request**:
```http
GET /api/v1/translation-keys?namespace=ui&content_type=ui_text&page=1&per_page=50
```

**Response**:
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "key_123",
        "key": "ui.buttons.save",
        "source_text": "Save",
        "context": "Primary save button",
        "content_type": "ui_text",
        "max_length": 20,
        "word_count": 1,
        "character_count": 4,
        "translation_priority": 90,
        "is_plural": false,
        "created_at": "2025-01-26T10:30:00Z",
        "translations_count": 25,
        "completion_percentage": 89.3
      }
    ],
    "total": 1247,
    "page": 1,
    "per_page": 50,
    "pages": 25
  }
}
```

### Create Translation Key

Add a new translation key to the system.

**Endpoint**: `POST /api/v1/translation-keys`

**Request Body**:
```json
{
  "namespace_id": "namespace_ui",
  "key": "ui.dialog.confirm_delete",
  "source_text": "Are you sure you want to delete this item?",
  "context": "Confirmation dialog for delete operations",
  "content_type": "ui_text",
  "max_length": 100,
  "translation_priority": 85,
  "is_html": false,
  "is_plural": false,
  "variables": ["item_name"],
  "tags": ["ui", "dialog", "confirmation"]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "translation_key_id": "key_456",
    "key": "ui.dialog.confirm_delete",
    "word_count": 11,
    "character_count": 45
  },
  "message": "Translation key created successfully"
}
```

### Extract Content

Extract translatable strings from source content (HTML, JSON, etc.).

**Endpoint**: `POST /api/v1/extract-content`

**Request Body**:
```json
{
  "content": "<h1>Welcome to our platform</h1><p>Get started today!</p><button>Sign Up</button>",
  "content_type": "html",
  "extract_options": {
    "include_attributes": ["title", "alt", "placeholder"],
    "exclude_tags": ["script", "style"],
    "min_length": 2
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "extracted_content": [
      "Welcome to our platform",
      "Get started today!",
      "Sign Up"
    ],
    "content_map": [
      {
        "text": "Welcome to our platform",
        "location": "h1",
        "context": "Page title"
      },
      {
        "text": "Get started today!",
        "location": "p",
        "context": "Description paragraph"
      },
      {
        "text": "Sign Up", 
        "location": "button",
        "context": "Call-to-action button"
      }
    ]
  },
  "meta": {
    "total_items": 3,
    "content_type": "html",
    "processing_time_ms": 45
  }
}
```

## User Preferences

### Get User Preferences

Retrieve localization preferences for a specific user.

**Endpoint**: `GET /api/v1/user/{user_id}/preferences`

**Response**:
```json
{
  "success": true,
  "data": {
    "preferences": {
      "primary_language_id": "lang_es",
      "preferred_locale_id": "locale_es_mx",
      "timezone": "America/Mexico_City",
      "date_format_preference": "dd/MM/yyyy",
      "time_format_preference": "HH:mm",
      "number_format_preference": "european",
      "auto_translate_enabled": true,
      "font_size_adjustment": 1.1,
      "high_contrast": false,
      "notification_language": "es"
    }
  },
  "meta": {
    "user_id": "user_123",
    "last_updated": "2025-01-26T10:30:00Z"
  }
}
```

### Update User Preferences

Update localization preferences for a user.

**Endpoint**: `PUT /api/v1/user/{user_id}/preferences`

**Request Body**:
```json
{
  "primary_language_id": "lang_fr",
  "preferred_locale_id": "locale_fr_ca",
  "timezone": "America/Toronto",
  "auto_translate_enabled": false,
  "font_size_adjustment": 1.2
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "updated": true,
    "changes_applied": 5
  },
  "message": "User preferences updated successfully"
}
```

## Statistics & Analytics

### Translation Statistics

Get comprehensive translation statistics and analytics.

**Endpoint**: `GET /api/v1/stats`

**Parameters**:
- `namespace` (optional): Filter by namespace
- `language` (optional): Filter by language
- `date_from` (optional): Start date for time-based metrics
- `date_to` (optional): End date for time-based metrics

**Response**:
```json
{
  "success": true,
  "data": {
    "stats": {
      "total_keys": 2847,
      "translated_keys": 2456,
      "completion_percentage": 86.3,
      "languages_supported": 42,
      "active_translators": 18,
      "quality_average": 8.7,
      "translations_this_month": 1245,
      "machine_translation_percentage": 23.1,
      "human_translation_percentage": 76.9,
      "pending_review": 89,
      "auto_approved": 2367
    },
    "by_language": [
      {
        "language_code": "es",
        "completion_percentage": 95.2,
        "total_translations": 2712,
        "quality_average": 9.1
      },
      {
        "language_code": "fr", 
        "completion_percentage": 88.7,
        "total_translations": 2523,
        "quality_average": 8.8
      }
    ],
    "recent_activity": [
      {
        "date": "2025-01-26",
        "translations_created": 47,
        "translations_approved": 52,
        "quality_issues": 3
      }
    ]
  }
}
```

### Performance Metrics

Get API performance and usage metrics.

**Endpoint**: `GET /api/v1/metrics`

**Response**:
```json
{
  "success": true,
  "data": {
    "performance": {
      "avg_response_time_ms": 23,
      "cache_hit_rate": 87.3,
      "requests_per_minute": 1247,
      "error_rate": 0.12
    },
    "usage": {
      "api_calls_today": 89453,
      "unique_users": 342,
      "most_requested_languages": ["en", "es", "fr", "de"],
      "peak_hours": ["09:00-10:00", "14:00-15:00"]
    }
  }
}
```

## Error Handling

### Standard Error Response

```json
{
  "success": false,
  "error": {
    "code": "TRANSLATION_NOT_FOUND",
    "message": "Translation not found for key 'invalid.key' in language 'es'",
    "details": {
      "key": "invalid.key",
      "language": "es",
      "namespace": "app",
      "fallback_attempted": true
    }
  },
  "meta": {
    "request_id": "req_123456",
    "timestamp": "2025-01-26T10:30:00Z"
  }
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `TRANSLATION_NOT_FOUND` | Translation key not found | 404 |
| `LANGUAGE_NOT_SUPPORTED` | Language not supported | 400 |
| `INVALID_LOCALE` | Invalid locale format | 400 |
| `VALIDATION_ERROR` | Request validation failed | 422 |
| `AUTHENTICATION_REQUIRED` | API key missing or invalid | 401 |
| `PERMISSION_DENIED` | Insufficient permissions | 403 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `INTERNAL_ERROR` | Server error | 500 |

### Validation Errors

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field_errors": [
        {
          "field": "language",
          "message": "Language code must be 2-5 characters",
          "value": "invalid-language-code"
        },
        {
          "field": "translation_priority",
          "message": "Priority must be between 0 and 100",
          "value": 150
        }
      ]
    }
  }
}
```

## Rate Limiting

### Rate Limit Headers

All API responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 943
X-RateLimit-Reset: 1643184000
X-RateLimit-Retry-After: 60
```

### Rate Limit Response

When rate limit is exceeded:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "details": {
      "limit": 1000,
      "window": "1h",
      "retry_after": 60
    }
  }
}
```

### Rate Limit Tiers

| Tier | Requests/Hour | Burst Limit |
|------|---------------|-------------|
| Free | 1,000 | 100/min |
| Basic | 10,000 | 500/min |
| Pro | 100,000 | 2,000/min |
| Enterprise | Unlimited | Custom |

## SDKs & Integration

### Python SDK

```python
from localization_client import LocalizationClient

client = LocalizationClient(
    base_url="https://api.example.com",
    api_key="your_api_key"
)

# Get translation
translation = await client.get_translation("welcome", "es")

# Batch translations
translations = await client.get_translations(["save", "cancel"], "fr")

# Format currency
formatted = client.format_number(99.99, "en-US", "currency")
```

### JavaScript/TypeScript SDK

```typescript
import { LocalizationClient } from '@datacraft/localization-client';

const client = new LocalizationClient({
  baseUrl: 'https://api.example.com',
  apiKey: 'your_api_key'
});

// Get translation
const translation = await client.getTranslation('welcome', 'es');

// Format date
const formatted = client.formatDate(new Date(), 'fr-FR', 'long');
```

### cURL Examples

```bash
# Get translation
curl -H "Authorization: Bearer your_api_key" \
     "https://api.example.com/api/v1/translate?key=welcome&language=es"

# Create translation
curl -X POST \
     -H "Authorization: Bearer your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"translation_key_id":"key_123","language_id":"lang_es","content":"Hola"}' \
     "https://api.example.com/api/v1/translations"
```

### Webhook Integration

Configure webhooks to receive notifications about translation events:

```json
{
  "webhook_url": "https://your-app.com/webhooks/localization",
  "events": ["translation.created", "translation.approved", "translation.rejected"],
  "secret": "your_webhook_secret"
}
```

Webhook payload example:
```json
{
  "event": "translation.approved",
  "data": {
    "translation_id": "trans_123",
    "key": "welcome.message",
    "language": "es",
    "content": "¡Bienvenido!",
    "approved_by": "reviewer_456",
    "approved_at": "2025-01-26T10:30:00Z"
  },
  "timestamp": "2025-01-26T10:30:00Z",
  "signature": "sha256=..."
}
```

---

**Company**: Datacraft  
**Website**: www.datacraft.co.ke  
**Contact**: nyimbi@gmail.com