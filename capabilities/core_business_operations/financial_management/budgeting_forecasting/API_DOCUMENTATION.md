# APG Budgeting & Forecasting - API Documentation

## Overview

The APG Budgeting & Forecasting capability provides a comprehensive API for enterprise-grade budget management, forecasting, analytics, and AI-powered automation. This document covers all available endpoints, request/response formats, and usage examples.

**Base URL**: `/core_financials/budgeting_forecasting`
**API Version**: 2.0.0
**Authentication**: APG Auth/RBAC Integration Required

---

## Table of Contents

1. [Core Budget Management](#core-budget-management)
2. [Real-Time Collaboration](#real-time-collaboration)
3. [Approval Workflows](#approval-workflows)
4. [Advanced Analytics](#advanced-analytics)
5. [Interactive Dashboards](#interactive-dashboards)
6. [Report Builder](#report-builder)
7. [ML Forecasting](#ml-forecasting)
8. [AI Recommendations](#ai-recommendations)
9. [Automated Monitoring](#automated-monitoring)
10. [Error Handling](#error-handling)
11. [Rate Limiting](#rate-limiting)

---

## Core Budget Management

### Create Budget

Creates a new budget with comprehensive validation and APG integration.

**Endpoint**: `POST /api/budgets`

**Request Body**:
```json
{
  "budget_name": "2025 Annual Budget",
  "budget_type": "annual",
  "fiscal_year": "2025",
  "total_amount": 1500000.00,
  "base_currency": "USD",
  "department_id": "dept_001",
  "budget_lines": [
    {
      "line_name": "Personnel Costs",
      "category": "SALARIES",
      "amount": 800000.00,
      "line_type": "expense"
    }
  ]
}
```

**Response**:
```json
{
  "success": true,
  "message": "Budget created successfully",
  "data": {
    "budget_id": "bf_budget_12345",
    "budget_name": "2025 Annual Budget",
    "status": "draft",
    "created_date": "2025-01-26T10:00:00Z",
    "total_amount": 1500000.00,
    "currency": "USD"
  }
}
```

### Create Budget from Template

Creates a budget using an existing template with customization options.

**Endpoint**: `POST /api/budgets/from-template/{template_id}`

**Request Body**:
```json
{
  "budget_name": "Q1 2025 Marketing Budget",
  "fiscal_year": "2025",
  "customizations": {
    "scale_factor": 1.1,
    "department_overrides": {
      "MARKETING": 150000.00
    }
  }
}
```

### Update Budget

Updates an existing budget with version control and audit tracking.

**Endpoint**: `PUT /api/budgets/{budget_id}`

**Request Body**:
```json
{
  "budget_name": "Updated 2025 Annual Budget",
  "total_amount": 1600000.00,
  "notes": "Increased due to market expansion"
}
```

### Get Budget

Retrieves budget details with optional line items.

**Endpoint**: `GET /api/budgets/{budget_id}?include_lines=true`

**Response**:
```json
{
  "success": true,
  "data": {
    "budget_id": "bf_budget_12345",
    "budget_name": "2025 Annual Budget",
    "status": "active",
    "total_amount": 1500000.00,
    "budget_lines": [
      {
        "line_id": "bf_line_001",
        "line_name": "Personnel Costs",
        "amount": 800000.00,
        "category": "SALARIES"
      }
    ]
  }
}
```

### Delete Budget

Soft deletes a budget with audit trail.

**Endpoint**: `DELETE /api/budgets/{budget_id}?soft_delete=true`

---

## Real-Time Collaboration

### Create Collaboration Session

Starts a real-time collaborative editing session for a budget.

**Endpoint**: `POST /api/collaboration/sessions`

**Request Body**:
```json
{
  "session_name": "Q1 Budget Review",
  "budget_id": "bf_budget_12345",
  "max_participants": 5,
  "session_type": "budget_editing",
  "permissions": {
    "can_edit": true,
    "can_comment": true,
    "can_approve": false
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "session_id": "collab_session_001",
    "session_name": "Q1 Budget Review",
    "status": "active",
    "join_url": "/collaboration/join/collab_session_001",
    "created_date": "2025-01-26T10:00:00Z"
  }
}
```

### Join Collaboration Session

Joins an existing collaboration session.

**Endpoint**: `POST /api/collaboration/sessions/{session_id}/join`

**Request Body**:
```json
{
  "user_name": "John Doe",
  "role": "editor",
  "permissions": ["edit", "comment"]
}
```

### Send Collaboration Event

Sends real-time events during collaboration (edits, comments, presence).

**Endpoint**: `POST /api/collaboration/sessions/{session_id}/events`

**Request Body**:
```json
{
  "event_type": "budget_line_edit",
  "target_id": "bf_line_001",
  "changes": {
    "amount": 850000.00,
    "previous_amount": 800000.00
  },
  "user_context": {
    "cursor_position": "amount_field"
  }
}
```

---

## Approval Workflows

### Submit Budget for Approval

Submits a budget through the approval workflow system.

**Endpoint**: `POST /api/workflows/approvals/{budget_id}/submit`

**Request Body**:
```json
{
  "workflow_template": "department_approval",
  "priority": "high",
  "notes": "Ready for Q1 review",
  "attachments": ["budget_summary.pdf"],
  "deadline": "2025-02-15T17:00:00Z"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "workflow_instance_id": "wf_inst_001",
    "status": "pending_approval",
    "current_step": "department_manager_review",
    "estimated_completion": "2025-02-10T17:00:00Z"
  }
}
```

### Process Approval Action

Processes an approval action (approve, reject, delegate).

**Endpoint**: `POST /api/workflows/instances/{workflow_instance_id}/actions`

**Request Body**:
```json
{
  "action_type": "approve",
  "decision_reason": "Budget aligns with strategic goals",
  "conditions_or_requirements": [],
  "delegate_to": null,
  "digital_signature": "signature_hash_here"
}
```

### Get Workflow Status

Retrieves current workflow status and history.

**Endpoint**: `GET /api/workflows/instances/{workflow_instance_id}`

---

## Advanced Analytics

### Generate Analytics Dashboard

Creates a comprehensive analytics dashboard with ML insights.

**Endpoint**: `POST /api/analytics/dashboards/{budget_id}`

**Request Body**:
```json
{
  "dashboard_name": "Executive Budget Analytics",
  "period": "monthly",
  "granularity": "detailed",
  "include_predictions": true,
  "metrics": [
    "variance_analysis",
    "trend_analysis",
    "performance_indicators"
  ]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "dashboard_id": "analytics_dash_001",
    "kpi_metrics": [
      {
        "metric_name": "Budget Utilization",
        "current_value": 75.5,
        "target_value": 80.0,
        "variance_percent": -5.625,
        "confidence_level": "high"
      }
    ],
    "budget_summary": {
      "total_budget": 1500000,
      "total_actual": 1487500,
      "variance": -12500
    },
    "generated_date": "2025-01-26T10:00:00Z"
  }
}
```

### Perform Variance Analysis

Executes advanced variance analysis with ML-powered insights.

**Endpoint**: `POST /api/analytics/variance/{budget_id}`

**Request Body**:
```json
{
  "analysis_period": "monthly",
  "include_root_cause": true,
  "ml_insights": true,
  "comparison_baseline": "previous_year"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "report_id": "variance_rpt_001",
    "total_variance": -12500.00,
    "significant_variances": [
      {
        "category": "IT Operations",
        "variance_amount": 7500.00,
        "variance_percent": 7.5,
        "significance": "moderate"
      }
    ],
    "ml_insights": {
      "anomalies_detected": 1,
      "predictive_warnings": [
        "Projected budget overrun in Q4"
      ]
    }
  }
}
```

---

## Interactive Dashboards

### Create Interactive Dashboard

Creates a new interactive dashboard with drill-down capabilities.

**Endpoint**: `POST /api/dashboards/interactive`

**Request Body**:
```json
{
  "dashboard_name": "Executive Budget Dashboard",
  "dashboard_type": "executive",
  "budget_ids": ["bf_budget_12345"],
  "widgets": [
    {
      "widget_name": "Budget Overview",
      "widget_type": "kpi_card",
      "position_x": 0,
      "position_y": 0,
      "width": 4,
      "height": 2,
      "data_source": "budget_summary",
      "metrics": ["total_budget", "total_actual", "variance"]
    }
  ]
}
```

### Perform Dashboard Drill-Down

Executes drill-down operation on dashboard data.

**Endpoint**: `POST /api/dashboards/{dashboard_id}/drill-down`

**Request Body**:
```json
{
  "target_level": "department",
  "context": {
    "widget_id": "widget_001",
    "filter_criteria": {
      "department": "Sales"
    }
  }
}
```

---

## Report Builder

### Create Report Template

Creates a custom report template with flexible configuration.

**Endpoint**: `POST /api/reports/templates`

**Request Body**:
```json
{
  "template_name": "Monthly Budget Report",
  "report_type": "budget_summary",
  "description": "Comprehensive monthly budget analysis",
  "sections": [
    {
      "section_name": "Executive Summary",
      "section_type": "data",
      "title": "Budget Performance Summary",
      "data_source": "budget_data",
      "fields": [
        {
          "field_name": "department",
          "display_name": "Department",
          "data_type": "string"
        },
        {
          "field_name": "budget_amount",
          "display_name": "Budget Amount",
          "data_type": "currency",
          "number_format": "$#,##0.00"
        }
      ]
    }
  ]
}
```

### Generate Report

Generates a report from an existing template.

**Endpoint**: `POST /api/reports/generate/{template_id}`

**Request Body**:
```json
{
  "report_name": "January 2025 Budget Report",
  "output_format": "pdf",
  "parameters": {
    "period": "2025-01",
    "include_forecasts": true
  },
  "delivery": {
    "method": "email",
    "recipients": ["manager@company.com"]
  }
}
```

### Create Report Schedule

Sets up automated report generation and delivery.

**Endpoint**: `POST /api/reports/schedules`

**Request Body**:
```json
{
  "schedule_name": "Monthly Budget Reports",
  "report_template_id": "template_001",
  "frequency": "monthly",
  "run_time": "09:00:00",
  "output_formats": ["pdf", "excel"],
  "delivery_method": "email",
  "recipients": ["team@company.com"]
}
```

---

## ML Forecasting

### Create ML Forecasting Model

Creates a new machine learning forecasting model.

**Endpoint**: `POST /api/ml/models`

**Request Body**:
```json
{
  "model_name": "Budget Forecasting Model v1",
  "algorithm": "random_forest",
  "target_variable": "budget_amount",
  "horizon": "medium_term",
  "frequency": "monthly",
  "training_window": 24,
  "features": [
    {
      "feature_name": "historical_budget",
      "feature_type": "historical_values",
      "source_column": "budget_amount",
      "lag_periods": 1
    }
  ]
}
```

### Train ML Model

Trains an existing ML forecasting model.

**Endpoint**: `POST /api/ml/models/{model_id}/train`

**Request Body**:
```json
{
  "training_config": {
    "validation_split": 0.2,
    "test_split": 0.1,
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 10
    }
  }
}
```

### Generate ML Forecast

Generates forecasts using a trained ML model.

**Endpoint**: `POST /api/ml/forecasts/{model_id}`

**Request Body**:
```json
{
  "scenario_name": "Q2 2025 Forecast",
  "start_date": "2025-04-01",
  "end_date": "2025-06-30",
  "assumptions": {
    "growth_rate": 0.05,
    "inflation_adjustment": 0.02
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "scenario_id": "forecast_scenario_001",
    "predictions": [
      {
        "forecast_date": "2025-04-01",
        "predicted_value": 125000.00,
        "confidence_level": 0.95,
        "lower_bound": 120000.00,
        "upper_bound": 130000.00
      }
    ],
    "total_forecast": 375000.00,
    "confidence_score": 0.87
  }
}
```

---

## AI Recommendations

### Generate Budget Recommendations

Creates AI-powered budget recommendations with industry benchmarks.

**Endpoint**: `POST /api/ai/recommendations`

**Request Body**:
```json
{
  "budget_id": "bf_budget_12345",
  "analysis_period": "last_12_months",
  "industry": "Technology",
  "company_size": "medium",
  "strategic_goals": ["cost_optimization", "revenue_growth"],
  "risk_tolerance": "medium"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "bundle_id": "recommendations_001",
    "recommendations": [
      {
        "recommendation_id": "rec_001",
        "title": "Optimize Cost per Employee",
        "type": "cost_optimization",
        "category": "operational_efficiency",
        "estimated_impact": -135000.00,
        "confidence_level": "high",
        "implementation_effort": "medium",
        "rationale": "Current cost per employee ($45,000) exceeds industry median ($42,000)",
        "required_actions": [
          "Conduct departmental efficiency audit",
          "Implement process automation"
        ]
      }
    ],
    "total_estimated_impact": -135000.00,
    "average_confidence": 0.78
  }
}
```

### Implement Recommendation

Implements a specific AI recommendation.

**Endpoint**: `POST /api/ai/recommendations/{recommendation_id}/implement`

**Request Body**:
```json
{
  "implementation_plan": "automated",
  "approval_required": false,
  "target_date": "2025-03-01",
  "notes": "Implementing cost optimization measures"
}
```

### Track Recommendation Performance

Tracks the performance of implemented recommendations.

**Endpoint**: `GET /api/ai/recommendations/{recommendation_id}/performance`

**Response**:
```json
{
  "success": true,
  "data": {
    "recommendation_id": "rec_001",
    "implementation_status": "completed",
    "actual_impact": -120000.00,
    "predicted_impact": -135000.00,
    "accuracy": 0.89,
    "performance_rating": "high"
  }
}
```

---

## Automated Monitoring

### Create Monitoring Rule

Creates an automated monitoring rule for budget alerts.

**Endpoint**: `POST /api/monitoring/rules`

**Request Body**:
```json
{
  "rule_name": "Budget Variance Alert",
  "alert_type": "variance_threshold",
  "description": "Alert when budget variance exceeds threshold",
  "scope": "budget",
  "target_entities": ["bf_budget_12345"],
  "metric_name": "variance_amount",
  "trigger_condition": "greater_than",
  "threshold_value": 10000.00,
  "severity": "warning",
  "frequency": "daily",
  "notification_channels": ["email", "in_app"],
  "recipients": ["budget.manager@company.com"]
}
```

### Start Automated Monitoring

Activates automated monitoring processes.

**Endpoint**: `POST /api/monitoring/start`

**Response**:
```json
{
  "success": true,
  "data": {
    "monitoring_active": true,
    "active_tasks": 3,
    "start_time": "2025-01-26T10:00:00Z"
  }
}
```

### Get Active Alerts

Retrieves current active monitoring alerts.

**Endpoint**: `GET /api/monitoring/alerts?severity=warning&status=active`

**Response**:
```json
{
  "success": true,
  "data": {
    "alerts": [
      {
        "alert_id": "alert_001",
        "title": "Budget Variance Alert",
        "severity": "warning",
        "status": "active",
        "current_value": 12500.00,
        "threshold_value": 10000.00,
        "triggered_date": "2025-01-26T10:00:00Z"
      }
    ],
    "total_count": 1,
    "statistics": {
      "by_severity": {"warning": 1},
      "by_status": {"active": 1}
    }
  }
}
```

### Perform Anomaly Detection

Executes anomaly detection on budget data.

**Endpoint**: `POST /api/monitoring/anomaly-detection`

**Request Body**:
```json
{
  "detection_name": "Budget Anomaly Detection",
  "metric_name": "budget_variance",
  "detection_method": "statistical",
  "sensitivity": 0.8,
  "analysis_start": "2025-01-01",
  "analysis_end": "2025-01-26"
}
```

---

## Error Handling

All API endpoints follow a consistent error response format:

```json
{
  "success": false,
  "message": "Error description",
  "errors": ["Specific error details"],
  "error_code": "BF_ERROR_001",
  "timestamp": "2025-01-26T10:00:00Z",
  "request_id": "req_12345"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `BF_AUTH_001` | Authentication required |
| `BF_AUTH_002` | Insufficient permissions |
| `BF_VALID_001` | Validation error |
| `BF_NOT_FOUND_001` | Resource not found |
| `BF_CONFLICT_001` | Resource conflict |
| `BF_LIMIT_001` | Rate limit exceeded |
| `BF_SERVER_001` | Internal server error |

---

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Standard Rate Limit**: 1000 requests per hour per tenant
- **ML Operations**: 100 requests per hour per tenant
- **Real-time Collaboration**: 500 events per minute per session

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1643212800
```

---

## Authentication & Authorization

All API endpoints require valid APG authentication tokens and appropriate permissions:

**Headers Required**:
```
Authorization: Bearer <jwt_token>
X-Tenant-ID: <tenant_identifier>
Content-Type: application/json
```

**Permission Levels**:
- `can_view_budgets`: Read access to budgets
- `can_create_budgets`: Create new budgets
- `can_edit_budgets`: Modify existing budgets
- `can_approve_budgets`: Approve budgets in workflows
- `can_use_ai_features`: Access AI-powered features
- `can_admin_budgets`: Full administrative access

---

## WebSocket Connections

Real-time features use WebSocket connections for live updates:

**Connection Endpoint**: `wss://api.domain.com/ws/budgeting-forecasting`

**Message Format**:
```json
{
  "type": "collaboration_event",
  "session_id": "collab_session_001",
  "event": {
    "type": "budget_line_edit",
    "user_id": "user_123",
    "timestamp": "2025-01-26T10:00:00Z",
    "data": {
      "line_id": "bf_line_001",
      "field": "amount",
      "new_value": 85000.00,
      "old_value": 80000.00
    }
  }
}
```

---

## SDK and Libraries

Official SDKs are available for:

- **Python**: `apg-budgeting-forecasting-sdk`
- **JavaScript/Node.js**: `@apg/budgeting-forecasting`
- **Java**: `com.apg.budgeting-forecasting-sdk`

**Python Example**:
```python
from apg.budgeting_forecasting import BudgetingForecastingClient

client = BudgetingForecastingClient(
    api_key="your_api_key",
    tenant_id="your_tenant_id"
)

# Create a budget
budget = client.budgets.create({
    "budget_name": "2025 Annual Budget",
    "total_amount": 1500000.00
})

# Generate AI recommendations
recommendations = client.ai.generate_recommendations({
    "budget_id": budget.budget_id,
    "industry": "Technology"
})
```

---

## Changelog

### Version 2.0.0 (Current)
- Added AI-powered recommendations
- Implemented ML forecasting engine
- Enhanced real-time collaboration
- Added automated monitoring and alerts
- Introduced interactive dashboards

### Version 1.x.x
- Basic budget management
- Simple reporting
- Approval workflows
- Legacy dashboard

---

*© 2025 Datacraft. All rights reserved.*
*For support: nyimbi@gmail.com | www.datacraft.co.ke*