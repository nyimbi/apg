# API Reference - Deep Personalization

Complete REST API reference for the APG Deep Personalization Subcapability. All endpoints support JSON request/response format and include comprehensive OpenAPI documentation.

## 🌐 Base URL

```
https://your-domain.com/api/v1/personalization
```

## 🔐 Authentication

All endpoints require Bearer token authentication:

```http
Authorization: Bearer YOUR_JWT_TOKEN
X-Tenant-ID: your-tenant-id
X-User-ID: requesting-user-id
```

## 📚 API Namespaces

The API is organized into four main namespaces:

- **`/personalization`** - Core personalization operations
- **`/insights`** - User insights and behavioral analysis
- **`/ai`** - AI model operations and content generation
- **`/management`** - Service administration and monitoring

---

## 🎯 Core Personalization API

### POST /personalization/personalize

Personalize content for a single user using AI-powered algorithms.

**Request Body:**
```json
{
  "user_id": "user123",
  "content": {
    "subject": "Welcome to {{company_name}}!",
    "text": "Hi {{user_name}}, welcome to our platform!",
    "html": "<h1>Welcome {{user_name}}!</h1>"
  },
  "context": {
    "user_name": "John Doe",
    "company_name": "ACME Corp",
    "user_segment": "new_customer"
  },
  "strategies": ["neural_content", "behavioral_adaptive"],
  "channels": ["email", "sms"],
  "priority": "normal",
  "min_quality_score": 0.7,
  "require_real_time": false
}
```

**Response:**
```json
{
  "request_id": "req_abc123",
  "user_id": "user123",
  "personalized_content": {
    "subject": "Welcome to ACME Corp, John!",
    "text": "Hi John, we're excited to welcome you to ACME Corp! Based on your interests in technology, we've prepared some special recommendations just for you.",
    "html": "<h1>Welcome John!</h1><p>We're excited to welcome you to ACME Corp...</p>"
  },
  "original_content": {
    "subject": "Welcome to {{company_name}}!",
    "text": "Hi {{user_name}}, welcome to our platform!"
  },
  "strategies_applied": ["neural_content", "behavioral_adaptive"],
  "quality_score": 0.89,
  "confidence_score": 0.85,
  "personalization_level": "comprehensive",
  "processing_time_ms": 45,
  "recommendations": [
    "Consider adding more behavioral triggers",
    "User shows high engagement potential"
  ],
  "predicted_engagement": {
    "open_probability": 0.78,
    "click_probability": 0.12,
    "convert_probability": 0.08
  },
  "optimal_channels": ["email", "push"],
  "optimal_timing": "2025-01-30T10:00:00Z",
  "cache_hit": false
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid request data
- `401` - Authentication required
- `403` - Insufficient permissions
- `429` - Rate limit exceeded
- `500` - Internal server error

### POST /personalization/campaigns/{campaign_id}/personalize

Personalize campaign content for multiple users with intelligent batching.

**Path Parameters:**
- `campaign_id` (string) - Unique campaign identifier

**Request Body:**
```json
{
  "campaign_id": "camp_welcome_2025",
  "target_users": ["user1", "user2", "user3"],
  "content": {
    "subject": "Special offer just for you!",
    "text": "Hi {{user_name}}, check out our latest {{category}} products!"
  },
  "context": {
    "campaign_type": "promotional",
    "discount_percentage": 20
  },
  "batch_size": 100
}
```

**Response:**
```json
{
  "personalized_content": {
    "user1": {
      "request_id": "req_user1_123",
      "personalized_content": {...},
      "quality_score": 0.92
    },
    "user2": {
      "request_id": "req_user2_124", 
      "personalized_content": {...},
      "quality_score": 0.85
    }
  },
  "campaign_stats": {
    "total_users": 3,
    "successful_personalizations": 3,
    "success_rate": 1.0,
    "avg_quality_score": 0.88
  },
  "responses": [...]
}
```

---

## 🧠 User Insights API

### GET /insights/users/{user_id}

Get comprehensive personalization insights for a specific user.

**Path Parameters:**
- `user_id` (string) - User identifier

**Query Parameters:**
- `include_predictions` (boolean) - Include future predictions (default: true)

**Response:**
```json
{
  "user_id": "user123",
  "profile_summary": {
    "completeness": 0.85,
    "confidence": 0.78,
    "data_quality": 0.92,
    "last_updated": "2025-01-29T10:00:00Z"
  },
  "personalization_preferences": {
    "content_preferences": {
      "topics": ["technology", "business"],
      "tone": "professional",
      "length": "medium"
    },
    "channel_preferences": {
      "email": 0.9,
      "sms": 0.6,
      "push": 0.8
    },
    "timing_preferences": {
      "optimal_hours": [9, 10, 14, 15],
      "timezone": "America/New_York"
    },
    "frequency_preferences": {
      "max_daily": 3,
      "max_weekly": 10
    }
  },
  "behavioral_insights": {
    "engagement_patterns": {
      "pattern": "highly_engaged",
      "score": 0.89,
      "frequency": 3.2,
      "trend": "increasing"
    },
    "interaction_patterns": {
      "preferred_interactions": [
        ["click", 45],
        ["open", 89],
        ["reply", 12]
      ]
    },
    "predictions": {
      "future_engagement_score": 0.91,
      "churn_risk": 0.15,
      "optimal_strategy": "maintain_engagement"
    }
  },
  "predictive_scores": {
    "engagement_prediction": 0.89,
    "churn_risk": 0.15,
    "lifetime_value": 0.76,
    "personalization_receptivity": 0.88
  },
  "predictions": {
    "next_optimal_contact": "2025-01-30T10:00:00Z",
    "recommended_content_type": "educational",
    "predicted_interests": ["ai", "automation"]
  }
}
```

### PUT /insights/users/{user_id}/preferences

Update user personalization preferences and trigger profile recalculation.

**Path Parameters:**
- `user_id` (string) - User identifier

**Request Body:**
```json
{
  "content_preferences": {
    "topics": ["technology", "innovation"],
    "tone": "casual",
    "length": "short"
  },
  "channel_preferences": {
    "email": 0.9,
    "sms": 0.7,
    "push": 0.8
  },
  "timing_preferences": {
    "preferred_hours": [9, 10, 16, 17],
    "timezone": "America/New_York",
    "quiet_hours": {
      "start": 22,
      "end": 7
    }
  },
  "frequency_preferences": {
    "max_daily": 2,
    "max_weekly": 8
  },
  "personalization_enabled": true,
  "trigger": "user_interaction"
}
```

**Response:**
```json
{
  "message": "Preferences updated successfully",
  "updated_at": "2025-01-29T10:30:00Z",
  "profile_version": "v2.1"
}
```

---

## 🤖 AI Models API

### POST /ai/generate-content

Generate completely new personalized content using AI models.

**Request Body:**
```json
{
  "user_id": "user123",
  "content_type": "promotional",
  "tone": "friendly",
  "length": "medium",
  "language": "en",
  "context": {
    "product_category": "electronics",
    "user_segment": "tech_enthusiasts",
    "campaign_goal": "conversion"
  }
}
```

**Response:**
```json
{
  "generated_content": {
    "subject": "🔥 Amazing Tech Deals Just for You, John!",
    "body": "Hi John! We know you love cutting-edge technology, so we've handpicked some incredible electronics deals that we think you'll absolutely love. From smart home devices to the latest gadgets, these offers are available for a limited time only.",
    "tone": "friendly",
    "language": "en",
    "personalization_level": "high"
  },
  "confidence_score": 0.91,
  "generation_metadata": {
    "model_id": "content_generator_v1",
    "model_version": "1.2.0",
    "processing_time_ms": 120,
    "explanation": "Generated promotional content with friendly tone"
  }
}
```

### POST /ai/analyze-behavior

Analyze user behavior using advanced AI models for deeper insights.

**Request Body:**
```json
{
  "user_id": "user123",
  "analysis_type": "comprehensive"
}
```

**Response:**
```json
{
  "behavioral_analysis": {
    "engagement_patterns": {
      "pattern": "highly_engaged",
      "score": 0.89,
      "frequency": 3.2,
      "total_engagements": 156,
      "recent_engagements": 23,
      "trend": "increasing"
    },
    "interaction_patterns": {
      "preferred_interactions": [
        ["open", 89],
        ["click", 45],
        ["reply", 12]
      ],
      "timing_preferences": {
        "optimal_hours": [9, 10, 14, 15],
        "consistency_score": 0.78
      }
    },
    "content_preferences": {
      "top_categories": [
        ["technology", 0.89],
        ["business", 0.76],
        ["innovation", 0.71]
      ],
      "tone_preference": "professional",
      "length_preference": "medium"
    },
    "predictions": {
      "future_engagement_score": 0.91,
      "churn_risk": 0.15,
      "optimal_strategy": "maintain_engagement",
      "next_optimal_contact": "Next optimal contact: 10:00",
      "recommended_content": "Recommended content: technology"
    }
  },
  "confidence_score": 0.87,
  "analysis_metadata": {
    "model_id": "behavioral_analyzer_v1",
    "model_version": "1.1.0",
    "processing_time_ms": 89,
    "explanation": "Behavioral analysis: comprehensive"
  }
}
```

---

## ⚙️ Management API

### GET /management/stats

Get comprehensive service performance statistics.

**Response:**
```json
{
  "service_stats": {
    "total_personalizations": 15420,
    "successful_personalizations": 14891,
    "cache_hits": 8234,
    "ai_model_calls": 7186,
    "avg_quality_score": 0.87,
    "avg_response_time_ms": 67.3
  },
  "engine_stats": {
    "total_personalizations": 15420,
    "orchestrator_stats": {
      "requests_processed": 15420,
      "avg_processing_time_ms": 67.3,
      "quality_score_avg": 0.87,
      "cache_hit_rate": 0.534,
      "real_time_requests": 3420
    }
  },
  "ai_model_stats": {
    "content_generator": {
      "model_id": "content_generator_v1",
      "status": "ready",
      "version": "1.2.0",
      "performance_metrics": {
        "bleu_score": 0.68,
        "training_loss": 0.25
      }
    },
    "behavioral_analyzer": {
      "model_id": "behavioral_analyzer_v1", 
      "status": "ready",
      "version": "1.1.0",
      "performance_metrics": {
        "accuracy": 0.91,
        "f1_score": 0.89
      }
    }
  },
  "service_config": {
    "service_level": "enterprise",
    "features_enabled": {
      "real_time": true,
      "predictive": true,
      "emotional_intelligence": true,
      "cross_channel_sync": true,
      "content_generation": true,
      "behavioral_analysis": true
    }
  }
}
```

### GET /management/health

Perform comprehensive health check of all system components.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-29T10:00:00Z",
  "components": {
    "personalization_engine": {
      "status": "healthy",
      "total_personalizations": 15420
    },
    "ai_model_content_generator": {
      "status": "ready",
      "version": "1.2.0"
    },
    "ai_model_behavioral_analyzer": {
      "status": "ready", 
      "version": "1.1.0"
    },
    "redis": {
      "status": "healthy"
    }
  }
}
```

**Status Codes:**
- `200` - System healthy
- `503` - System degraded or unhealthy

### GET /management/config

Get current service configuration and feature flags.

**Response:**
```json
{
  "service_level": "enterprise",
  "features": {
    "real_time": true,
    "predictive": true,
    "emotional_intelligence": true,
    "cross_channel_sync": true,
    "content_generation": true,
    "behavioral_analysis": true
  },
  "performance": {
    "max_response_time_ms": 100,
    "min_quality_score": 0.7,
    "cache_ttl_seconds": 3600
  },
  "tenant_id": "your-tenant-id"
}
```

---

## 📊 Data Models

### PersonalizationStrategy Enum
```
neural_content
behavioral_adaptive  
emotional_resonance
contextual_intelligence
predictive_optimization
cross_channel_sync
quantum_personalization
empathy_driven
real_time_adaptation
multi_dimensional
```

### PersonalizationServiceLevel Enum
```
basic
standard
premium
enterprise
quantum
```

### DeliveryChannel Enum
```
email, sms, push, voice, webhook, slack, teams, 
whatsapp, telegram, discord, in_app, web_push,
mqtt, alexa, google_assistant, wearables,
arkit, oculus, steam, xbox, android_auto,
carplay, tesla, fax, print, digital_signage
```

### NotificationPriority Enum
```
low, normal, high, urgent, critical
```

---

## 🚀 Rate Limits

| Endpoint | Rate Limit | Burst |
|----------|------------|-------|
| `/personalization/personalize` | 1000/min | 100/sec |
| `/personalization/campaigns/*/personalize` | 100/min | 10/sec |
| `/insights/users/*` | 2000/min | 50/sec |
| `/ai/generate-content` | 500/min | 20/sec |
| `/ai/analyze-behavior` | 200/min | 10/sec |
| `/management/*` | 100/min | 10/sec |

## 🔍 Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "PERSONALIZATION_FAILED",
    "message": "Personalization failed: insufficient user data",
    "details": {
      "user_id": "user123",
      "quality_score": 0.45,
      "threshold": 0.7
    },
    "timestamp": "2025-01-29T10:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes
- `AUTHENTICATION_REQUIRED` - Missing or invalid authentication
- `INSUFFICIENT_PERMISSIONS` - User lacks required permissions
- `INVALID_REQUEST_DATA` - Request data validation failed
- `USER_NOT_FOUND` - Specified user does not exist
- `PERSONALIZATION_FAILED` - Personalization process failed
- `QUALITY_THRESHOLD_NOT_MET` - Result quality below threshold
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `SERVICE_UNAVAILABLE` - Service temporarily unavailable
- `INTERNAL_SERVER_ERROR` - Unexpected server error

---

## 🔗 OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
```
GET /api/v1/personalization/docs/swagger.json
```

Interactive API documentation:
```
https://your-domain.com/api/v1/personalization/docs/
```

---

Need more details? Check out the [Developer Guide](developer-guide.md) for integration patterns and best practices!