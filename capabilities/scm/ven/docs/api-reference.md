# API Reference

Comprehensive REST API documentation for the APG Vendor Management system. All endpoints support JSON request/response formats with JWT authentication.

## 🔐 Authentication

All API endpoints require authentication via JWT Bearer token in the Authorization header.

### Authentication Headers
```http
Authorization: Bearer <jwt_token>
X-Tenant-ID: <tenant_uuid>
X-User-ID: <user_uuid>
Content-Type: application/json
```

### Authentication Endpoints

#### Get JWT Token
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "user_id": "user-uuid-here",
    "tenant_id": "tenant-uuid-here"
  }
}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

## 🏢 Vendor Management

### List Vendors

Retrieve paginated list of vendors with filtering and sorting options.

```http
GET /api/v1/vendor-management/vendors
```

#### Query Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `page` | integer | Page number | 1 |
| `page_size` | integer | Items per page (max 100) | 25 |
| `status` | string | Filter by status | all |
| `category` | string | Filter by category | all |
| `vendor_type` | string | Filter by vendor type | all |
| `strategic_importance` | string | Filter by importance | all |
| `search` | string | Text search across fields | - |
| `sort_by` | string | Sort field | name |
| `sort_order` | string | Sort direction (asc/desc) | asc |

#### Example Request
```http
GET /api/v1/vendor-management/vendors?page=1&page_size=25&status=active&category=technology&sort_by=performance_score&sort_order=desc
Authorization: Bearer <token>
X-Tenant-ID: <tenant_id>
```

#### Response
```json
{
  "success": true,
  "data": {
    "vendors": [
      {
        "id": "vendor-uuid-1",
        "vendor_code": "TECH001",
        "name": "TechCorp Solutions",
        "legal_name": "TechCorp Solutions Inc.",
        "display_name": "TechCorp",
        "vendor_type": "supplier",
        "category": "technology",
        "subcategory": "software",
        "status": "active",
        "strategic_importance": "high",
        "preferred_vendor": true,
        "strategic_partner": false,
        "performance_score": 87.5,
        "risk_score": 23.2,
        "intelligence_score": 91.8,
        "relationship_score": 85.4,
        "email": "contact@techcorp.com",
        "phone": "+1-555-0123",
        "website": "https://techcorp.com",
        "created_at": "2025-01-15T10:30:00Z",
        "updated_at": "2025-01-29T14:22:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 25,
      "total_count": 156,
      "has_next": true,
      "has_prev": false,
      "total_pages": 7
    }
  }
}
```

### Create Vendor

Create a new vendor record with comprehensive validation.

```http
POST /api/v1/vendor-management/vendors
Content-Type: application/json
```

#### Request Body
```json
{
  "vendor_code": "ACME001",
  "name": "ACME Corporation",
  "legal_name": "ACME Corporation Inc.",
  "display_name": "ACME",
  "vendor_type": "supplier",
  "category": "manufacturing",
  "subcategory": "electronics",
  "industry": "consumer_electronics",
  "size_classification": "large",
  "email": "contact@acme.com",
  "phone": "+1-555-0199",
  "website": "https://acme.com",
  "address_line1": "123 Industry Blvd",
  "address_line2": "Suite 100",
  "city": "Tech City",
  "state_province": "CA",
  "postal_code": "94105",
  "country": "USA",
  "credit_rating": "A+",
  "payment_terms": "Net 30",
  "currency": "USD",
  "tax_id": "12-3456789",
  "duns_number": "123456789",
  "strategic_importance": "standard",
  "preferred_vendor": false,
  "strategic_partner": false,
  "diversity_category": "woman_owned",
  "capabilities": [
    "electronic_manufacturing",
    "quality_assurance",
    "supply_chain_management"
  ],
  "certifications": [
    {
      "name": "ISO 9001",
      "issued_date": "2024-01-15",
      "expiry_date": "2027-01-15",
      "issuing_body": "ISO"
    }
  ],
  "geographic_coverage": ["USA", "Canada", "Mexico"]
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "vendor-uuid-new",
    "vendor_code": "ACME001",
    "name": "ACME Corporation",
    "status": "active",
    "created_at": "2025-01-29T15:30:00Z",
    "performance_score": 85.0,
    "risk_score": 25.0,
    "intelligence_score": 80.0,
    "relationship_score": 75.0
  }
}
```

### Get Vendor Details

Retrieve comprehensive information for a specific vendor.

```http
GET /api/v1/vendor-management/vendors/{vendor_id}
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "vendor-uuid-1",
    "tenant_id": "tenant-uuid",
    "vendor_code": "ACME001",
    "name": "ACME Corporation",
    "legal_name": "ACME Corporation Inc.",
    "display_name": "ACME",
    "vendor_type": "supplier",
    "category": "manufacturing",
    "subcategory": "electronics",
    "industry": "consumer_electronics",
    "size_classification": "large",
    "status": "active",
    "lifecycle_stage": "qualified",
    "strategic_importance": "standard",
    "preferred_vendor": false,
    "strategic_partner": false,
    "diversity_category": "woman_owned",
    "email": "contact@acme.com",
    "phone": "+1-555-0199",
    "website": "https://acme.com",
    "address": {
      "address_line1": "123 Industry Blvd",
      "address_line2": "Suite 100",
      "city": "Tech City",
      "state_province": "CA",
      "postal_code": "94105",
      "country": "USA"
    },
    "financial": {
      "credit_rating": "A+",
      "payment_terms": "Net 30",
      "currency": "USD",
      "tax_id": "12-3456789",
      "duns_number": "123456789"
    },
    "scores": {
      "performance_score": 87.5,
      "risk_score": 23.2,
      "intelligence_score": 91.8,
      "relationship_score": 85.4
    },
    "capabilities": [
      "electronic_manufacturing",
      "quality_assurance",
      "supply_chain_management"
    ],
    "certifications": [
      {
        "name": "ISO 9001",
        "issued_date": "2024-01-15",
        "expiry_date": "2027-01-15",
        "issuing_body": "ISO"
      }
    ],
    "geographic_coverage": ["USA", "Canada", "Mexico"],
    "predicted_performance": {
      "next_quarter": {
        "overall_score": 89.2,
        "confidence": 0.82
      }
    },
    "risk_predictions": {
      "delivery_risk": {
        "probability": 0.15,
        "impact": "medium",
        "confidence": 0.75
      }
    },
    "optimization_recommendations": [
      {
        "type": "performance_improvement",
        "description": "Increase order frequency to improve economies of scale",
        "expected_impact": 0.12,
        "priority": "high"
      }
    ],
    "ai_insights": {
      "relationship_health": "excellent",
      "performance_trend": "improving",
      "risk_level": "low",
      "optimization_potential": "medium"
    },
    "performance_summary": {
      "avg_overall_score": 87.5,
      "performance_trend": "improving",
      "measurement_count": 12,
      "last_measurement": "2025-01-25T10:00:00Z"
    },
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-29T14:22:00Z",
    "version": 3
  }
}
```

### Update Vendor

Update vendor information with partial updates supported.

```http
PUT /api/v1/vendor-management/vendors/{vendor_id}
Content-Type: application/json
```

#### Request Body
```json
{
  "name": "ACME Corporation Ltd",
  "email": "newcontact@acme.com",
  "strategic_importance": "high",
  "preferred_vendor": true,
  "phone": "+1-555-0299"
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "vendor-uuid-1",
    "name": "ACME Corporation Ltd",
    "status": "active",
    "updated_at": "2025-01-29T16:45:00Z",
    "version": 4
  }
}
```

### Delete Vendor

Soft delete (deactivate) a vendor record.

```http
DELETE /api/v1/vendor-management/vendors/{vendor_id}
```

#### Response
```json
{
  "success": true,
  "message": "Vendor deactivated successfully"
}
```

## 📊 Performance Management

### Get Vendor Performance

Retrieve performance data and summary for a specific vendor.

```http
GET /api/v1/vendor-management/vendors/{vendor_id}/performance
```

#### Response
```json
{
  "success": true,
  "data": {
    "vendor_id": "vendor-uuid-1",
    "current_performance": {
      "overall_score": 87.5,
      "quality_score": 92.0,
      "delivery_score": 85.0,
      "cost_score": 89.0,
      "service_score": 84.0,
      "innovation_score": 88.0
    },
    "performance_summary": {
      "avg_overall_score": 86.3,
      "performance_trend": "improving",
      "measurement_count": 12,
      "last_measurement": "2025-01-25T10:00:00Z",
      "improvement_rate": 0.05
    },
    "detailed_metrics": {
      "on_time_delivery_rate": 95.5,
      "quality_rejection_rate": 2.1,
      "cost_variance": -2.5,
      "service_level_achievement": 98.0,
      "order_volume": 1250000.00,
      "order_count": 45,
      "total_spend": 1185000.00,
      "average_order_value": 26333.33
    },
    "historical_performance": [
      {
        "period": "2024-Q4",
        "overall_score": 87.5,
        "measurement_date": "2025-01-25T10:00:00Z"
      },
      {
        "period": "2024-Q3",
        "overall_score": 85.2,
        "measurement_date": "2024-10-25T10:00:00Z"
      }
    ],
    "benchmark_comparison": {
      "industry_average": 78.5,
      "percentile_rank": 85,
      "peer_comparison": "above_average"
    },
    "improvement_recommendations": [
      {
        "area": "delivery_performance",
        "current_score": 85.0,
        "target_score": 90.0,
        "recommended_actions": [
          "Implement advanced delivery tracking",
          "Optimize transportation routes"
        ],
        "expected_improvement": 5.0,
        "implementation_effort": "medium"
      }
    ]
  }
}
```

### Record Performance Data

Add new performance measurement for a vendor.

```http
POST /api/v1/vendor-management/vendors/{vendor_id}/performance
Content-Type: application/json
```

#### Request Body
```json
{
  "measurement_period": "quarterly",
  "start_date": "2025-01-01T00:00:00Z",
  "end_date": "2025-03-31T23:59:59Z",
  "overall_score": 89.5,
  "quality_score": 92.0,
  "delivery_score": 87.0,
  "cost_score": 91.0,
  "service_score": 88.0,
  "innovation_score": 90.0,
  "on_time_delivery_rate": 96.5,
  "quality_rejection_rate": 1.8,
  "cost_variance": -1.2,
  "service_level_achievement": 99.0,
  "order_volume": 1500000.00,
  "order_count": 52,
  "total_spend": 1425000.00,
  "average_order_value": 27403.85,
  "performance_trends": {
    "quality_improvement": 0.08,
    "delivery_consistency": 0.95,
    "cost_efficiency": 0.12
  },
  "improvement_recommendations": [
    {
      "area": "innovation",
      "description": "Increase collaborative innovation projects",
      "priority": "medium"
    }
  ],
  "data_sources": [
    "erp_system",
    "quality_management_system",
    "transportation_management_system"
  ],
  "data_completeness": 98.5,
  "calculation_method": "weighted_average",
  "notes": "Exceptional performance this quarter with significant quality improvements"
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "performance-uuid-new",
    "vendor_id": "vendor-uuid-1",
    "overall_score": 89.5,
    "measurement_period": "quarterly",
    "created_at": "2025-01-29T16:00:00Z"
  }
}
```

## 🛡️ Risk Management

### Get Vendor Risk Profile

Retrieve comprehensive risk assessment for a vendor.

```http
GET /api/v1/vendor-management/vendors/{vendor_id}/risk
```

#### Response
```json
{
  "success": true,
  "data": {
    "vendor_id": "vendor-uuid-1",
    "risk_summary": {
      "overall_risk_score": 28.5,
      "risk_level": "low",
      "total_risks": 3,
      "active_risks": 2,
      "high_risks": 0,
      "medium_risks": 2,
      "low_risks": 1,
      "risks_pending_mitigation": 1
    },
    "active_risks": [
      {
        "id": "risk-uuid-1",
        "risk_type": "operational",
        "risk_category": "delivery",
        "severity": "medium",
        "title": "Potential delivery delays during peak season",
        "description": "Risk of delivery delays during Q4 peak season due to capacity constraints",
        "overall_risk_score": 65.0,
        "probability": 0.4,
        "financial_impact": 75000.00,
        "mitigation_status": "in_progress",
        "mitigation_strategy": "Implement capacity planning and alternative routing",
        "assigned_to": "supply_chain_manager@company.com",
        "target_completion": "2025-02-15T00:00:00Z",
        "identified_date": "2025-01-15T10:00:00Z"
      }
    ],
    "risk_trends": {
      "risk_score_trend": "decreasing",
      "new_risks_last_30_days": 1,
      "resolved_risks_last_30_days": 2,
      "risk_mitigation_effectiveness": 0.85
    },
    "predictive_risks": [
      {
        "risk_type": "financial",
        "description": "Potential credit rating downgrade",
        "predicted_probability": 0.15,
        "time_horizon": 180,
        "confidence": 0.72,
        "early_warning_indicators": [
          "Increased payment delays",
          "Financial reporting delays"
        ]
      }
    ],
    "mitigation_recommendations": [
      {
        "risk_id": "risk-uuid-1",
        "recommended_actions": [
          "Diversify supplier base for critical components",
          "Establish buffer inventory levels",
          "Implement real-time capacity monitoring"
        ],
        "expected_risk_reduction": 0.3,
        "implementation_cost": 25000.00,
        "roi_projection": 3.2
      }
    ]
  }
}
```

### Record Risk Assessment

Add new risk assessment for a vendor.

```http
POST /api/v1/vendor-management/vendors/{vendor_id}/risk
Content-Type: application/json
```

#### Request Body
```json
{
  "risk_type": "compliance",
  "risk_category": "regulatory",
  "severity": "high",
  "title": "GDPR compliance gap identified",
  "description": "Vendor's data processing practices may not fully comply with GDPR requirements",
  "root_cause": "Lack of updated privacy policies and data processing agreements",
  "potential_impact": "Regulatory fines, contract termination, reputational damage",
  "overall_risk_score": 78.0,
  "probability": 0.6,
  "financial_impact": 150000.00,
  "operational_impact": 8,
  "reputational_impact": 9,
  "mitigation_strategy": "Work with vendor to update privacy policies and implement GDPR-compliant processes",
  "mitigation_actions": [
    {
      "action": "Conduct GDPR compliance audit",
      "responsible_party": "compliance_team@company.com",
      "due_date": "2025-02-15T00:00:00Z",
      "status": "planned"
    },
    {
      "action": "Update data processing agreements",
      "responsible_party": "legal_team@company.com",
      "due_date": "2025-03-01T00:00:00Z",
      "status": "planned"
    }
  ],
  "target_residual_risk": 25.0,
  "monitoring_frequency": "monthly",
  "assigned_to": "compliance_manager@company.com",
  "ai_risk_factors": [
    "regulatory_change_monitoring",
    "vendor_communication_sentiment",
    "industry_compliance_trends"
  ]
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "risk-uuid-new",
    "vendor_id": "vendor-uuid-1",
    "risk_type": "compliance",
    "severity": "high",
    "overall_risk_score": 78.0,
    "created_at": "2025-01-29T17:00:00Z"
  }
}
```

## 🧠 AI Intelligence

### Get Vendor Intelligence

Retrieve latest AI-generated insights for a vendor.

```http
GET /api/v1/vendor-management/vendors/{vendor_id}/intelligence
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "intelligence-uuid-1",
    "vendor_id": "vendor-uuid-1",
    "intelligence_date": "2025-01-29T18:00:00Z",
    "model_version": "v1.0",
    "confidence_score": 0.87,
    "behavior_patterns": [
      {
        "pattern_type": "communication",
        "pattern_name": "highly_responsive",
        "confidence": 0.92,
        "description": "Vendor consistently responds to communications within 2 hours",
        "supporting_data": {
          "avg_response_time": 1.3,
          "response_consistency": 0.95,
          "communication_frequency": "high"
        }
      },
      {
        "pattern_type": "performance",
        "pattern_name": "consistent_high_performer",
        "confidence": 0.89,
        "description": "Vendor maintains consistently high performance scores across all dimensions",
        "supporting_data": {
          "score_variance": 0.03,
          "improvement_trend": 0.08,
          "performance_stability": 0.94
        }
      }
    ],
    "predictive_insights": [
      {
        "insight_type": "performance_forecast",
        "prediction": "performance_improvement",
        "confidence": 0.82,
        "time_horizon": 90,
        "description": "Expected 8% improvement in overall performance over next quarter",
        "predicted_values": {
          "overall_score": 94.2,
          "quality_score": 95.5,
          "delivery_score": 92.8
        },
        "contributing_factors": [
          "Recent process improvements",
          "Increased communication frequency",
          "Investment in new technology"
        ]
      },
      {
        "insight_type": "risk_forecast",
        "prediction": "low_risk_continuation",
        "confidence": 0.78,
        "time_horizon": 180,
        "description": "Risk levels expected to remain low with slight improvement",
        "predicted_risk_score": 22.1,
        "risk_factors": [
          "Strong financial position",
          "Diversified customer base",
          "Proactive risk management"
        ]
      }
    ],
    "performance_forecasts": {
      "next_quarter": {
        "overall_score": 94.2,
        "confidence": 0.82,
        "factors": ["process_improvements", "technology_investment"]
      },
      "next_six_months": {
        "overall_score": 95.8,
        "confidence": 0.75,
        "factors": ["market_expansion", "capacity_increase"]
      }
    },
    "risk_assessments": {
      "delivery_risk": {
        "probability": 0.12,
        "impact": "low",
        "confidence": 0.84,
        "mitigation_effectiveness": 0.91
      },
      "financial_risk": {
        "probability": 0.08,
        "impact": "very_low",
        "confidence": 0.79,
        "trend": "decreasing"
      }
    },
    "market_position": {
      "competitive_strength": "strong",
      "market_share_trend": "growing",
      "innovation_index": 0.87,
      "customer_satisfaction": 0.92
    },
    "improvement_opportunities": [
      {
        "area": "cost_optimization",
        "description": "Opportunity to reduce costs through process automation",
        "potential_savings": 45000.00,
        "implementation_effort": "medium",
        "risk_level": "low"
      },
      {
        "area": "service_enhancement",
        "description": "Expand service portfolio to include maintenance services",
        "revenue_potential": 125000.00,
        "strategic_value": "high",
        "implementation_timeline": "6_months"
      }
    ],
    "relationship_optimization": [
      {
        "recommendation": "Increase strategic collaboration",
        "description": "Partner on joint innovation projects",
        "expected_benefit": "mutual_innovation_acceleration",
        "implementation_steps": [
          "Establish innovation steering committee",
          "Define joint R&D projects",
          "Create shared IP framework"
        ]
      }
    ],
    "data_sources": [
      "performance_records",
      "communication_history",
      "market_data",
      "financial_reports"
    ],
    "data_quality_score": 0.94,
    "analysis_scope": {
      "historical_data_months": 24,
      "external_data_sources": 5,
      "analysis_dimensions": 12
    },
    "valid_from": "2025-01-29T18:00:00Z",
    "valid_until": "2025-02-29T18:00:00Z"
  }
}
```

### Generate Fresh Intelligence

Trigger AI analysis to generate fresh insights for a vendor.

```http
POST /api/v1/vendor-management/vendors/{vendor_id}/intelligence
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "intelligence-uuid-new",
    "vendor_id": "vendor-uuid-1",
    "confidence_score": 0.89,
    "intelligence_date": "2025-01-29T19:00:00Z",
    "behavior_patterns_count": 4,
    "predictive_insights_count": 6,
    "processing_time_seconds": 23.5,
    "data_sources_analyzed": 8
  }
}
```

### Get Optimization Plan

Generate AI-powered optimization recommendations for a vendor.

```http
POST /api/v1/vendor-management/vendors/{vendor_id}/optimization
Content-Type: application/json
```

#### Request Body
```json
{
  "objectives": [
    "performance_improvement",
    "cost_reduction",
    "risk_mitigation"
  ],
  "constraints": {
    "budget_limit": 100000.00,
    "timeline_months": 6,
    "risk_tolerance": "medium"
  },
  "priorities": {
    "performance_improvement": 0.4,
    "cost_reduction": 0.35,
    "risk_mitigation": 0.25
  }
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "optimization-plan-uuid-new",
    "vendor_id": "vendor-uuid-1",
    "optimization_objectives": [
      "performance_improvement",
      "cost_reduction",
      "risk_mitigation"
    ],
    "recommended_actions": [
      {
        "action_id": "action-1",
        "action_type": "process_improvement",
        "title": "Implement automated quality control",
        "description": "Deploy AI-powered quality control system to reduce defect rates",
        "category": "performance_improvement",
        "priority": "high",
        "estimated_cost": 45000.00,
        "implementation_effort": "medium",
        "expected_roi": 2.8,
        "timeline_months": 3,
        "success_probability": 0.85,
        "expected_outcomes": {
          "quality_score_improvement": 0.08,
          "defect_rate_reduction": 0.35,
          "cost_savings_annual": 125000.00
        },
        "implementation_steps": [
          "Vendor assessment and system design",
          "System procurement and installation",
          "Staff training and process integration",
          "Performance monitoring and optimization"
        ],
        "risk_factors": [
          "Technology adoption challenges",
          "Integration complexity"
        ],
        "mitigation_strategies": [
          "Phased implementation approach",
          "Comprehensive training program"
        ]
      },
      {
        "action_id": "action-2",
        "action_type": "contract_optimization",
        "title": "Negotiate volume-based pricing",
        "description": "Restructure pricing model to include volume discounts",
        "category": "cost_reduction",
        "priority": "high",
        "estimated_cost": 5000.00,
        "implementation_effort": "low",
        "expected_roi": 8.2,
        "timeline_months": 1,
        "success_probability": 0.92,
        "expected_outcomes": {
          "cost_savings_annual": 41000.00,
          "pricing_efficiency": 0.12
        }
      }
    ],
    "predicted_outcomes": {
      "performance_improvement": {
        "overall_score_increase": 0.12,
        "quality_improvement": 0.15,
        "efficiency_gain": 0.08
      },
      "cost_reduction": {
        "annual_savings": 166000.00,
        "cost_per_unit_reduction": 0.09,
        "total_cost_optimization": 0.14
      },
      "risk_mitigation": {
        "risk_score_reduction": 0.18,
        "compliance_improvement": 0.22,
        "operational_stability": 0.11
      }
    },
    "implementation_roadmap": {
      "phase_1": {
        "duration_months": 2,
        "actions": ["action-2"],
        "investment": 5000.00,
        "expected_benefits": 41000.00
      },
      "phase_2": {
        "duration_months": 4,
        "actions": ["action-1"],
        "investment": 45000.00,
        "expected_benefits": 125000.00
      }
    },
    "success_metrics": [
      {
        "metric": "overall_performance_score",
        "baseline": 87.5,
        "target": 98.0,
        "measurement_frequency": "monthly"
      },
      {
        "metric": "annual_cost_savings",
        "baseline": 0.00,
        "target": 166000.00,
        "measurement_frequency": "quarterly"
      }
    ],
    "risk_assessment": {
      "implementation_risks": [
        {
          "risk": "Technology adoption delays",
          "probability": 0.25,
          "impact": "medium",
          "mitigation": "Phased implementation with pilot testing"
        }
      ],
      "success_probability": 0.87,
      "confidence_interval": {
        "lower": 0.82,
        "upper": 0.92
      }
    },
    "created_at": "2025-01-29T20:00:00Z",
    "valid_until": "2025-04-29T20:00:00Z"
  }
}
```

## 📊 Analytics

### Get Vendor Analytics

Retrieve comprehensive analytics across the vendor portfolio.

```http
GET /api/v1/vendor-management/analytics
```

#### Query Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `time_period` | string | Analysis period (30d, 90d, 1y) | 90d |
| `category` | string | Filter by vendor category | all |
| `include_predictions` | boolean | Include predictive analytics | true |

#### Response
```json
{
  "success": true,
  "data": {
    "vendor_counts": {
      "total_vendors": 1247,
      "active_vendors": 1089,
      "inactive_vendors": 158,
      "preferred_vendors": 123,
      "strategic_partners": 34,
      "new_vendors_30_days": 12,
      "vendors_by_status": {
        "active": 1089,
        "pending": 23,
        "suspended": 8,
        "terminated": 127
      },
      "vendors_by_category": {
        "technology": 234,
        "manufacturing": 445,
        "services": 312,
        "consulting": 156,
        "logistics": 100
      }
    },
    "performance_metrics": {
      "avg_performance": 82.7,
      "performance_distribution": {
        "excellent_90_plus": 156,
        "good_80_89": 567,
        "satisfactory_70_79": 289,
        "needs_improvement_60_69": 67,
        "poor_below_60": 10
      },
      "performance_trends": {
        "improving": 634,
        "stable": 398,
        "declining": 57
      },
      "top_performers": [
        {
          "vendor_id": "vendor-uuid-top1",
          "name": "Elite Tech Solutions",
          "performance_score": 96.8,
          "category": "technology"
        }
      ]
    },
    "risk_metrics": {
      "avg_risk": 31.5,
      "risk_distribution": {
        "low_0_25": 456,
        "medium_26_50": 523,
        "high_51_75": 89,
        "critical_76_100": 21
      },
      "total_active_risks": 234,
      "risks_by_category": {
        "operational": 89,
        "financial": 67,
        "compliance": 45,
        "strategic": 33
      },
      "risk_trends": {
        "new_risks_30_days": 12,
        "resolved_risks_30_days": 18,
        "mitigation_success_rate": 0.87
      }
    },
    "financial_metrics": {
      "total_vendor_spend": 245600000.00,
      "avg_spend_per_vendor": 196835.00,
      "spend_by_category": {
        "manufacturing": 98240000.00,
        "technology": 67890000.00,
        "services": 45670000.00,
        "consulting": 23400000.00,
        "logistics": 10400000.00
      },
      "cost_savings_ytd": 12500000.00,
      "cost_optimization_opportunities": 8900000.00
    },
    "ai_insights": {
      "total_intelligence_records": 1089,
      "avg_confidence_score": 0.84,
      "prediction_accuracy": 0.87,
      "optimization_recommendations": 234,
      "implemented_recommendations": 167,
      "recommendation_success_rate": 0.78
    },
    "relationship_health": {
      "avg_relationship_score": 78.9,
      "strong_relationships": 567,
      "at_risk_relationships": 89,
      "relationship_trends": {
        "improving": 423,
        "stable": 578,
        "declining": 88
      }
    },
    "operational_efficiency": {
      "avg_response_time_hours": 4.2,
      "contract_compliance_rate": 0.94,
      "sla_achievement_rate": 0.91,
      "issue_resolution_time_avg": 2.8
    },
    "predictive_analytics": {
      "performance_forecasts": {
        "next_quarter_avg": 84.2,
        "improvement_probability": 0.73,
        "decline_risk": 0.12
      },
      "risk_predictions": {
        "emerging_risks_90_days": 23,
        "risk_escalation_probability": 0.15,
        "mitigation_success_forecast": 0.89
      },
      "optimization_potential": {
        "performance_improvement": 0.08,
        "cost_reduction": 0.12,
        "risk_reduction": 0.22
      }
    },
    "recent_activities": [
      {
        "activity_type": "performance_update",
        "vendor_name": "TechCorp Solutions",
        "description": "Q4 2024 performance data recorded",
        "timestamp": "2025-01-29T10:30:00Z",
        "impact": "positive"
      },
      {
        "activity_type": "risk_mitigation",
        "vendor_name": "Global Manufacturing Inc",
        "description": "Supply chain risk successfully mitigated",
        "timestamp": "2025-01-29T09:15:00Z",
        "impact": "positive"
      }
    ],
    "benchmark_data": {
      "industry_average_performance": 76.5,
      "peer_average_risk": 38.2,
      "best_in_class_performance": 92.1,
      "performance_percentile": 78
    },
    "generation_timestamp": "2025-01-29T21:00:00Z",
    "data_freshness": {
      "performance_data": "2025-01-29T18:00:00Z",
      "risk_data": "2025-01-29T20:00:00Z",
      "intelligence_data": "2025-01-29T19:30:00Z"
    }
  }
}
```

## 📄 Pagination

All list endpoints support pagination with consistent parameters:

### Pagination Parameters
| Parameter | Type | Description | Default | Max |
|-----------|------|-------------|---------|-----|
| `page` | integer | Page number (1-based) | 1 | - |
| `page_size` | integer | Items per page | 25 | 100 |

### Pagination Response Format
```json
{
  "pagination": {
    "page": 1,
    "page_size": 25,
    "total_count": 156,
    "has_next": true,
    "has_prev": false,
    "total_pages": 7,
    "next_page": 2,
    "prev_page": null
  }
}
```

## ❌ Error Handling

### Standard Error Response Format
```json
{
  "success": false,
  "error": "Error description",
  "error_code": "VENDOR_NOT_FOUND",
  "details": {
    "field": "vendor_id",
    "message": "Vendor with ID 'invalid-uuid' not found"
  },
  "timestamp": "2025-01-29T22:00:00Z",
  "request_id": "req-uuid-here"
}
```

### HTTP Status Codes
| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (e.g., duplicate) |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Common Error Codes
| Error Code | Description |
|------------|-------------|
| `VENDOR_NOT_FOUND` | Vendor with specified ID not found |
| `DUPLICATE_VENDOR_CODE` | Vendor code already exists |
| `INVALID_PERFORMANCE_SCORE` | Performance score outside valid range |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| `VALIDATION_ERROR` | Request validation failed |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `AI_SERVICE_UNAVAILABLE` | AI intelligence service temporarily unavailable |

## 🔒 Security

### API Security Features
- **JWT Authentication**: Bearer token authentication
- **Rate Limiting**: Configurable rate limits per endpoint
- **Input Validation**: Comprehensive request validation
- **SQL Injection Protection**: Parameterized queries
- **XSS Protection**: Input sanitization and output encoding
- **CORS Support**: Configurable CORS policies

### Rate Limits
| Endpoint Category | Limit | Window |
|------------------|--------|---------|
| Authentication | 10 requests | 1 minute |
| Vendor CRUD | 1000 requests | 1 hour |
| Analytics | 100 requests | 1 hour |
| AI Intelligence | 50 requests | 1 hour |

---

## 📞 Support

For API support and questions:
- **Email**: nyimbi@gmail.com  
- **Documentation**: Complete API guides and examples
- **Status Page**: Real-time API status and maintenance updates

*This completes the comprehensive API reference documentation.* 🚀