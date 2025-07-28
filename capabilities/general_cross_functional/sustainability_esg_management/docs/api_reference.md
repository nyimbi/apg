# APG Sustainability & ESG Management - API Reference

**Version:** 1.0.0  
**Last Updated:** January 2025  
**Base URL:** `/api/v1/esg`

---

## Table of Contents

1. [Authentication](#authentication)
2. [API Overview](#api-overview)
3. [Error Handling](#error-handling)
4. [Rate Limiting](#rate-limiting)
5. [ESG Metrics API](#esg-metrics-api)
6. [ESG Targets API](#esg-targets-api)
7. [ESG Measurements API](#esg-measurements-api)
8. [ESG Stakeholders API](#esg-stakeholders-api)
9. [ESG Suppliers API](#esg-suppliers-api)
10. [ESG Initiatives API](#esg-initiatives-api)
11. [ESG Reports API](#esg-reports-api)
12. [AI Intelligence API](#ai-intelligence-api)
13. [Real-Time API](#real-time-api)
14. [WebSocket Events](#websocket-events)
15. [SDK Examples](#sdk-examples)

---

## Authentication

The APG ESG Management API uses the APG platform's authentication system through the `auth_rbac` capability.

### Authentication Methods

#### Bearer Token Authentication
```http
Authorization: Bearer <jwt_token>
```

#### APG API Key Authentication
```http
X-API-Key: <api_key>
X-Tenant-ID: <tenant_id>
```

### Required Headers
```http
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token>
X-Tenant-ID: <tenant_id>
```

### Permission System

The API uses APG's role-based access control with the following permissions:

- **esg_metrics**: `read`, `create`, `update`, `delete`
- **esg_targets**: `read`, `create`, `update`, `delete`
- **esg_stakeholders**: `read`, `create`, `update`, `delete`
- **esg_suppliers**: `read`, `create`, `update`, `delete`
- **esg_reports**: `read`, `create`, `update`, `delete`
- **esg_ai_insights**: `read`, `create`

---

## API Overview

### Base URLs

- **Production:** `https://api.apg.platform/api/v1/esg`
- **Staging:** `https://staging-api.apg.platform/api/v1/esg`
- **Development:** `http://localhost:8000/api/v1/esg`

### API Versioning

The API uses path-based versioning:
- Current version: `v1`
- Future versions will be available at `/api/v2/esg`

### Content Types

- **Request:** `application/json`
- **Response:** `application/json`
- **File Upload:** `multipart/form-data`

### Response Format

All API responses follow this standard format:

```json
{
	"status": "success|error",
	"data": {
		// Response data
	},
	"message": "Human readable message",
	"timestamp": "2025-01-28T10:30:00Z",
	"request_id": "req_123456789"
}
```

---

## Error Handling

### HTTP Status Codes

- **200 OK:** Request successful
- **201 Created:** Resource created successfully
- **400 Bad Request:** Invalid request data
- **401 Unauthorized:** Authentication required
- **403 Forbidden:** Insufficient permissions
- **404 Not Found:** Resource not found
- **409 Conflict:** Resource already exists
- **422 Unprocessable Entity:** Validation errors
- **429 Too Many Requests:** Rate limit exceeded
- **500 Internal Server Error:** Server error

### Error Response Format

```json
{
	"status": "error",
	"error": {
		"code": "VALIDATION_ERROR",
		"message": "Validation failed",
		"details": {
			"field_name": ["Error message"]
		}
	},
	"timestamp": "2025-01-28T10:30:00Z",
	"request_id": "req_123456789"
}
```

### Common Error Codes

- `AUTHENTICATION_REQUIRED`: Missing or invalid authentication
- `PERMISSION_DENIED`: Insufficient permissions
- `VALIDATION_ERROR`: Request validation failed
- `RESOURCE_NOT_FOUND`: Requested resource not found
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `TENANT_NOT_FOUND`: Invalid tenant ID
- `AI_SERVICE_UNAVAILABLE`: AI predictions temporarily unavailable

---

## Rate Limiting

### Limits

- **Standard Users:** 1,000 requests per hour
- **Premium Users:** 5,000 requests per hour
- **Enterprise Users:** 10,000 requests per hour

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1643723400
```

---

## ESG Metrics API

### List ESG Metrics

```http
GET /metrics
```

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Number of results (default: 50, max: 200) |
| `offset` | integer | Pagination offset (default: 0) |
| `metric_type` | string | Filter by type: `environmental`, `social`, `governance` |
| `category` | string | Filter by category |
| `is_kpi` | boolean | Filter by KPI status |
| `is_public` | boolean | Filter by public visibility |
| `search` | string | Search in name, code, description |
| `sort` | string | Sort field (default: `created_at`) |
| `order` | string | Sort order: `asc`, `desc` (default: `desc`) |

#### Response

```json
{
	"status": "success",
	"data": {
		"metrics": [
			{
				"id": "metric_123",
				"name": "Carbon Emissions Scope 1",
				"code": "CARBON_SCOPE1",
				"metric_type": "environmental",
				"category": "emissions",
				"subcategory": "direct_emissions",
				"unit": "tonnes_co2",
				"current_value": 12500.75,
				"target_value": 10000.0,
				"baseline_value": 15000.0,
				"is_kpi": true,
				"is_public": false,
				"data_quality_score": 94.5,
				"ai_predictions": {
					"trend": "improving",
					"6_month_forecast": 11800.0,
					"confidence": 0.89
				},
				"created_at": "2024-01-01T00:00:00Z",
				"updated_at": "2025-01-28T10:30:00Z"
			}
		],
		"pagination": {
			"total": 156,
			"limit": 50,
			"offset": 0,
			"has_next": true,
			"has_prev": false
		}
	}
}
```

### Get ESG Metric

```http
GET /metrics/{metric_id}
```

#### Response

```json
{
	"status": "success",
	"data": {
		"metric": {
			"id": "metric_123",
			"name": "Carbon Emissions Scope 1",
			"code": "CARBON_SCOPE1",
			"description": "Direct carbon emissions from owned sources",
			"metric_type": "environmental",
			"category": "emissions",
			"subcategory": "direct_emissions",
			"unit": "tonnes_co2",
			"current_value": 12500.75,
			"target_value": 10000.0,
			"baseline_value": 15000.0,
			"calculation_method": "sum_of_facility_emissions",
			"data_sources": ["facility_management", "energy_bills"],
			"collection_frequency": "monthly",
			"is_kpi": true,
			"is_public": false,
			"is_automated": true,
			"enable_ai_predictions": true,
			"data_quality_score": 94.5,
			"validation_rules": [
				{
					"type": "range_check",
					"min_value": 0,
					"max_value": 50000
				}
			],
			"ai_predictions": {
				"trend": "improving",
				"6_month_forecast": 11800.0,
				"12_month_forecast": 10500.0,
				"confidence": 0.89,
				"factors": ["energy_efficiency", "renewable_adoption"]
			},
			"trend_analysis": {
				"direction": "decreasing",
				"rate": -2.5,
				"period": "12_months"
			},
			"created_at": "2024-01-01T00:00:00Z",
			"updated_at": "2025-01-28T10:30:00Z"
		}
	}
}
```

### Create ESG Metric

```http
POST /metrics
```

#### Request Body

```json
{
	"name": "Water Consumption",
	"code": "WATER_CONSUMPTION",
	"description": "Total water consumption across all facilities",
	"metric_type": "environmental",
	"category": "water",
	"subcategory": "consumption",
	"unit": "cubic_meters",
	"target_value": 50000.0,
	"baseline_value": 75000.0,
	"calculation_method": "sum_of_facility_consumption",
	"data_sources": ["facility_meters", "utility_bills"],
	"collection_frequency": "monthly",
	"is_kpi": true,
	"is_public": true,
	"is_automated": true,
	"enable_ai_predictions": true,
	"validation_rules": [
		{
			"type": "range_check",
			"min_value": 0,
			"max_value": 200000
		}
	]
}
```

#### Response

```json
{
	"status": "success",
	"data": {
		"metric": {
			"id": "metric_456",
			"name": "Water Consumption",
			"code": "WATER_CONSUMPTION",
			// ... full metric object
		}
	},
	"message": "ESG metric created successfully"
}
```

### Update ESG Metric

```http
PUT /metrics/{metric_id}
```

#### Request Body

Same as create, with optional fields.

### Delete ESG Metric

```http
DELETE /metrics/{metric_id}
```

#### Response

```json
{
	"status": "success",
	"message": "ESG metric deleted successfully"
}
```

---

## ESG Targets API

### List ESG Targets

```http
GET /targets
```

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric_id` | string | Filter by metric ID |
| `status` | string | Filter by status: `draft`, `active`, `on_track`, `at_risk`, `achieved`, `missed` |
| `priority` | string | Filter by priority: `low`, `medium`, `high`, `critical` |
| `owner_id` | string | Filter by owner ID |

#### Response

```json
{
	"status": "success",
	"data": {
		"targets": [
			{
				"id": "target_123",
				"name": "Reduce Carbon Emissions by 30%",
				"description": "Achieve 30% reduction in Scope 1 emissions by 2025",
				"metric_id": "metric_123",
				"target_value": 10500.0,
				"baseline_value": 15000.0,
				"current_progress": 75.5,
				"start_date": "2024-01-01",
				"target_date": "2025-12-31",
				"status": "on_track",
				"priority": "high",
				"owner_id": "user_456",
				"achievement_probability": 87.5,
				"predicted_completion_date": "2025-10-15",
				"risk_factors": [
					"regulatory_changes",
					"supply_chain_disruption"
				],
				"milestones": [
					{
						"id": "milestone_123",
						"name": "Q1 Milestone",
						"due_date": "2024-03-31",
						"target_value": 13500.0,
						"status": "completed"
					}
				],
				"created_at": "2024-01-01T00:00:00Z",
				"updated_at": "2025-01-28T10:30:00Z"
			}
		],
		"pagination": {
			"total": 45,
			"limit": 50,
			"offset": 0
		}
	}
}
```

### Create ESG Target

```http
POST /targets
```

#### Request Body

```json
{
	"name": "Achieve Zero Waste to Landfill",
	"description": "Divert 100% of waste from landfills by 2025",
	"metric_id": "metric_789",
	"target_value": 0.0,
	"baseline_value": 500.0,
	"start_date": "2024-01-01",
	"target_date": "2025-12-31",
	"priority": "high",
	"owner_id": "user_456",
	"create_milestones": true,
	"milestone_frequency": "quarterly"
}
```

---

## ESG Measurements API

### Record ESG Measurement

```http
POST /measurements
```

#### Request Body

```json
{
	"metric_id": "metric_123",
	"value": 12250.5,
	"measurement_date": "2025-01-28T00:00:00Z",
	"data_source": "facility_management",
	"collection_method": "automated",
	"quality_indicators": {
		"completeness": 98.5,
		"accuracy": 95.2,
		"timeliness": 100.0
	},
	"metadata": {
		"facility_id": "facility_456",
		"measurement_equipment": "smart_meter_789"
	}
}
```

#### Response

```json
{
	"status": "success",
	"data": {
		"measurement": {
			"id": "measurement_123",
			"metric_id": "metric_123",
			"value": 12250.5,
			"measurement_date": "2025-01-28T00:00:00Z",
			"data_source": "facility_management",
			"collection_method": "automated",
			"validation_score": 96.8,
			"anomaly_score": 2.1,
			"quality_indicators": {
				"completeness": 98.5,
				"accuracy": 95.2,
				"timeliness": 100.0
			},
			"ai_insights": {
				"anomaly_detected": false,
				"trend_contribution": "positive",
				"data_quality": "high"
			},
			"created_at": "2025-01-28T10:30:00Z"
		}
	},
	"message": "Measurement recorded successfully"
}
```

### List ESG Measurements

```http
GET /measurements
```

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric_id` | string | Filter by metric ID |
| `start_date` | string | Filter from date (ISO 8601) |
| `end_date` | string | Filter to date (ISO 8601) |
| `data_source` | string | Filter by data source |
| `collection_method` | string | Filter by collection method |

---

## ESG Stakeholders API

### List ESG Stakeholders

```http
GET /stakeholders
```

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `stakeholder_type` | string | Filter by type: `investor`, `employee`, `customer`, `community`, `regulator`, `supplier`, `ngo` |
| `country` | string | Filter by country |
| `portal_access` | boolean | Filter by portal access |
| `is_active` | boolean | Filter by active status |

#### Response

```json
{
	"status": "success",
	"data": {
		"stakeholders": [
			{
				"id": "stakeholder_123",
				"name": "Global Investment Partners",
				"organization": "GIP Holdings",
				"stakeholder_type": "institutional_investor",
				"contact_person": "Jane Smith",
				"email": "jane.smith@gip.com",
				"phone": "+1-555-0123",
				"country": "USA",
				"language_preference": "en_US",
				"esg_interests": [
					"climate_risk",
					"governance",
					"supply_chain"
				],
				"engagement_frequency": "monthly",
				"engagement_score": 87.5,
				"sentiment_score": 82.3,
				"influence_score": 94.2,
				"portal_access": true,
				"data_access_level": "confidential",
				"last_engagement": "2025-01-20T14:30:00Z",
				"is_active": true,
				"created_at": "2024-03-15T09:00:00Z",
				"updated_at": "2025-01-28T10:30:00Z"
			}
		],
		"pagination": {
			"total": 89,
			"limit": 50,
			"offset": 0
		}
	}
}
```

### Create ESG Stakeholder

```http
POST /stakeholders
```

#### Request Body

```json
{
	"name": "Community Environmental Group",
	"organization": "Green Future Initiative",
	"stakeholder_type": "ngo",
	"contact_person": "Dr. Alex Johnson",
	"email": "alex@greenfuture.org",
	"phone": "+1-555-0456",
	"country": "USA",
	"language_preference": "en_US",
	"esg_interests": [
		"biodiversity",
		"water_stewardship",
		"community_impact"
	],
	"engagement_frequency": "quarterly",
	"portal_access": true,
	"data_access_level": "public",
	"communication_preferences": {
		"email": true,
		"phone": false,
		"portal_notifications": true
	}
}
```

### Get Stakeholder Analytics

```http
GET /stakeholders/{stakeholder_id}/analytics
```

#### Response

```json
{
	"status": "success",
	"data": {
		"stakeholder_id": "stakeholder_123",
		"analytics": {
			"engagement_insights": {
				"satisfaction_level": "high",
				"response_rate": 94.2,
				"preferred_communication": "email",
				"engagement_trend": "increasing"
			},
			"influence_score": 94.2,
			"sentiment_analysis": {
				"current_sentiment": 82.3,
				"sentiment_trend": "positive",
				"key_topics": [
					"climate_action",
					"transparency",
					"social_impact"
				]
			},
			"engagement_optimization": {
				"recommended_frequency": "bi_weekly",
				"preferred_content_types": [
					"progress_reports",
					"impact_stories"
				],
				"optimal_engagement_time": "tuesday_afternoon"
			}
		}
	}
}
```

---

## ESG Suppliers API

### List ESG Suppliers

```http
GET /suppliers
```

#### Response

```json
{
	"status": "success",
	"data": {
		"suppliers": [
			{
				"id": "supplier_123",
				"name": "Sustainable Materials Co.",
				"legal_name": "Sustainable Materials Corporation",
				"country": "USA",
				"industry_sector": "manufacturing",
				"business_size": "medium",
				"relationship_start": "2023-06-01",
				"contract_value": 2500000.0,
				"criticality_level": "high",
				"overall_esg_score": 78.5,
				"environmental_score": 82.1,
				"social_score": 75.3,
				"governance_score": 78.2,
				"risk_level": "medium",
				"certifications": [
					"ISO_14001",
					"B_CORP",
					"FSC_CERTIFIED"
				],
				"improvement_areas": [
					"carbon_reporting",
					"diversity_metrics"
				],
				"last_assessment": "2025-01-15T00:00:00Z",
				"next_assessment": "2025-07-15T00:00:00Z",
				"created_at": "2023-06-01T00:00:00Z",
				"updated_at": "2025-01-28T10:30:00Z"
			}
		],
		"pagination": {
			"total": 234,
			"limit": 50,
			"offset": 0
		}
	}
}
```

---

## ESG Reports API

### List ESG Reports

```http
GET /reports
```

#### Response

```json
{
	"status": "success",
	"data": {
		"reports": [
			{
				"id": "report_123",
				"name": "Annual Sustainability Report 2024",
				"report_type": "sustainability",
				"framework": "GRI",
				"status": "published",
				"reporting_year": 2024,
				"auto_generated": true,
				"file_url": "https://storage.example.com/reports/sustainability_2024.pdf",
				"public_url": "https://portal.company.com/reports/sustainability_2024",
				"metrics_included": 45,
				"pages": 68,
				"generated_at": "2025-01-15T00:00:00Z",
				"published_at": "2025-01-20T00:00:00Z",
				"created_at": "2025-01-10T00:00:00Z",
				"updated_at": "2025-01-28T10:30:00Z"
			}
		],
		"pagination": {
			"total": 12,
			"limit": 50,
			"offset": 0
		}
	}
}
```

### Generate ESG Report

```http
POST /reports/generate
```

#### Request Body

```json
{
	"name": "Q4 2024 ESG Performance Report",
	"report_type": "quarterly",
	"framework": "SASB",
	"reporting_period": {
		"start_date": "2024-10-01",
		"end_date": "2024-12-31"
	},
	"include_sections": [
		"executive_summary",
		"environmental_metrics",
		"social_impact",
		"governance_practices",
		"stakeholder_engagement"
	],
	"format": "pdf",
	"auto_publish": false,
	"recipients": [
		"board@company.com",
		"investors@company.com"
	]
}
```

#### Response

```json
{
	"status": "success",
	"data": {
		"report_id": "report_456",
		"generation_status": "in_progress",
		"estimated_completion": "2025-01-28T11:00:00Z"
	},
	"message": "Report generation started successfully"
}
```

---

## AI Intelligence API

### Get AI Insights for Metric

```http
GET /ai/metrics/{metric_id}/insights
```

#### Response

```json
{
	"status": "success",
	"data": {
		"metric_id": "metric_123",
		"insights": {
			"predictions": {
				"3_month": 11850.0,
				"6_month": 11200.0,
				"12_month": 10500.0
			},
			"trend_analysis": {
				"direction": "decreasing",
				"rate": -3.2,
				"acceleration": 0.1,
				"seasonality": "low"
			},
			"confidence": 0.89,
			"factors": [
				{
					"name": "energy_efficiency_projects",
					"impact": 0.35,
					"confidence": 0.92
				},
				{
					"name": "renewable_energy_adoption",
					"impact": 0.28,
					"confidence": 0.87
				}
			],
			"recommendations": [
				{
					"type": "optimization",
					"title": "Accelerate Energy Efficiency Program",
					"description": "Focus on high-impact facilities to exceed targets",
					"impact": "15% additional reduction",
					"priority": "high"
				}
			],
			"anomalies": [],
			"last_updated": "2025-01-28T10:30:00Z"
		}
	}
}
```

### Get AI Predictions for Target

```http
GET /ai/targets/{target_id}/prediction
```

#### Response

```json
{
	"status": "success",
	"data": {
		"target_id": "target_123",
		"prediction": {
			"achievement_probability": 87.5,
			"predicted_completion_date": "2025-10-15",
			"confidence_level": "high",
			"risk_factors": [
				{
					"factor": "supply_chain_disruption",
					"probability": 0.15,
					"impact": "moderate"
				},
				{
					"factor": "regulatory_changes",
					"probability": 0.08,
					"impact": "high"
				}
			],
			"optimization_recommendations": [
				{
					"action": "increase_renewable_energy_pace",
					"impact": "+5% probability",
					"timeline": "3_months"
				}
			],
			"scenario_analysis": {
				"best_case": {
					"probability": 95.2,
					"completion_date": "2025-08-30"
				},
				"worst_case": {
					"probability": 72.3,
					"completion_date": "2026-02-15"
				}
			},
			"last_updated": "2025-01-28T10:30:00Z"
		}
	}
}
```

---

## Real-Time API

### Server-Sent Events (SSE)

```http
GET /realtime/events
```

#### Headers

```http
Accept: text/event-stream
Cache-Control: no-cache
```

#### Event Stream Format

```
event: metric_updated
data: {"metric_id": "metric_123", "new_value": 12250.5, "timestamp": "2025-01-28T10:30:00Z"}

event: target_progress
data: {"target_id": "target_456", "progress": 78.5, "status": "on_track", "timestamp": "2025-01-28T10:30:00Z"}

event: ai_insight
data: {"type": "optimization_opportunity", "metric_id": "metric_123", "recommendation": "...", "timestamp": "2025-01-28T10:30:00Z"}
```

### Health Check

```http
GET /health
```

#### Response

```json
{
	"status": "success",
	"data": {
		"status": "healthy",
		"timestamp": "2025-01-28T10:30:00Z",
		"services": {
			"database": "healthy",
			"ai_engine": "healthy",
			"real_time": "healthy",
			"external_apis": "healthy"
		},
		"performance": {
			"response_time_ms": 45,
			"cpu_usage": 23.5,
			"memory_usage": 67.2
		}
	}
}
```

---

## WebSocket Events

### Connection

```javascript
const ws = new WebSocket('wss://api.apg.platform/api/v1/esg/ws');
```

### Authentication

```json
{
	"type": "auth",
	"token": "jwt_token_here",
	"tenant_id": "tenant_123"
}
```

### Subscribe to Events

```json
{
	"type": "subscribe",
	"channels": [
		"metrics_updates",
		"targets_progress",
		"ai_insights",
		"stakeholder_activities"
	]
}
```

### Event Types

#### Metric Update
```json
{
	"type": "metric_updated",
	"channel": "metrics_updates",
	"data": {
		"metric_id": "metric_123",
		"old_value": 12500.0,
		"new_value": 12250.5,
		"change_percent": -2.0,
		"timestamp": "2025-01-28T10:30:00Z"
	}
}
```

#### Target Progress
```json
{
	"type": "target_progress",
	"channel": "targets_progress",
	"data": {
		"target_id": "target_456",
		"progress": 78.5,
		"status": "on_track",
		"achievement_probability": 87.5,
		"timestamp": "2025-01-28T10:30:00Z"
	}
}
```

#### AI Insight
```json
{
	"type": "ai_insight",
	"channel": "ai_insights",
	"data": {
		"insight_type": "optimization_opportunity",
		"metric_id": "metric_123",
		"title": "Energy Efficiency Opportunity",
		"description": "Potential 15% reduction through facility upgrades",
		"priority": "high",
		"timestamp": "2025-01-28T10:30:00Z"
	}
}
```

---

## SDK Examples

### Python SDK

```python
from apg_esg_client import ESGClient

# Initialize client
client = ESGClient(
    api_key="your_api_key",
    tenant_id="your_tenant_id",
    base_url="https://api.apg.platform"
)

# Get metrics
metrics = await client.metrics.list(
    metric_type="environmental",
    is_kpi=True,
    limit=10
)

# Create metric
new_metric = await client.metrics.create({
    "name": "Energy Consumption",
    "code": "ENERGY_TOTAL",
    "metric_type": "environmental",
    "category": "energy",
    "unit": "kwh",
    "is_kpi": True
})

# Record measurement
measurement = await client.measurements.create({
    "metric_id": new_metric.id,
    "value": 15000.0,
    "measurement_date": "2025-01-28T00:00:00Z"
})

# Get AI insights
insights = await client.ai.get_metric_insights(new_metric.id)
print(f"Predicted 6-month value: {insights.predictions['6_month']}")
```

### JavaScript SDK

```javascript
import { ESGClient } from '@apg/esg-client';

// Initialize client
const client = new ESGClient({
    apiKey: 'your_api_key',
    tenantId: 'your_tenant_id',
    baseUrl: 'https://api.apg.platform'
});

// Get metrics
const metrics = await client.metrics.list({
    metricType: 'environmental',
    isKpi: true,
    limit: 10
});

// Create stakeholder
const stakeholder = await client.stakeholders.create({
    name: 'Green Investment Fund',
    stakeholderType: 'institutional_investor',
    email: 'esg@greeninvest.com',
    esgInterests: ['climate_risk', 'governance']
});

// Subscribe to real-time updates
client.realtime.subscribe('metrics_updates', (event) => {
    console.log('Metric updated:', event.data);
});

// Get target predictions
const prediction = await client.ai.getTargetPrediction('target_123');
console.log(`Achievement probability: ${prediction.achievementProbability}%`);
```

### cURL Examples

#### Get Metrics
```bash
curl -X GET \
  "https://api.apg.platform/api/v1/esg/metrics?metric_type=environmental&limit=10" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json"
```

#### Create Target
```bash
curl -X POST \
  "https://api.apg.platform/api/v1/esg/targets" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Carbon Neutrality by 2030",
    "metric_id": "metric_123",
    "target_value": 0.0,
    "target_date": "2030-12-31",
    "priority": "critical"
  }'
```

#### Record Measurement
```bash
curl -X POST \
  "https://api.apg.platform/api/v1/esg/measurements" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "metric_id": "metric_123",
    "value": 12250.5,
    "measurement_date": "2025-01-28T00:00:00Z",
    "data_source": "automated_sensor"
  }'
```

---

**Copyright © 2025 Datacraft - All rights reserved.**  
**Author: Nyimbi Odero <nyimbi@gmail.com>**  
**Website: www.datacraft.co.ke**