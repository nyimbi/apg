# APG Employee Data Management - API Reference

## 📋 Table of Contents

- [Authentication](#authentication)
- [Base URL & Versioning](#base-url--versioning)
- [Request/Response Format](#requestresponse-format)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Core Employee Operations](#core-employee-operations)
- [AI & Analytics](#ai--analytics)
- [Global Workforce](#global-workforce)
- [Workflow Automation](#workflow-automation)
- [System Administration](#system-administration)

## 🔐 Authentication

All API endpoints require authentication using JWT Bearer tokens.

### Getting Access Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "user@company.com",
  "password": "secure_password",
  "tenant_id": "your_tenant_id"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here",
  "scope": "employee:read employee:write analytics:read"
}
```

### Using Bearer Token

Include the token in the Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Token Refresh

```http
POST /api/v1/auth/refresh
Content-Type: application/json
Authorization: Bearer <refresh_token>

{
  "refresh_token": "refresh_token_here"
}
```

## 🌐 Base URL & Versioning

**Base URL:** `https://api.datacraft.co.ke/employee-management`  
**Current Version:** `v1`  
**Full API Base:** `https://api.datacraft.co.ke/employee-management/api/v1`

### API Versioning

- **Current:** `v1` (Stable)
- **Beta:** `v2` (Preview features)

Access beta features:
```http
GET /api/v2/employees
Accept: application/vnd.apg.v2+json
```

## 📄 Request/Response Format

### Content Types

- **Request:** `application/json`
- **Response:** `application/json`
- **File Upload:** `multipart/form-data`

### Standard Response Structure

```json
{
  "data": {},
  "meta": {
    "timestamp": "2025-01-27T10:00:00Z",
    "request_id": "req_123456789",
    "version": "v1.0.0"
  },
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 250,
    "total_pages": 5
  }
}
```

### Headers

#### Request Headers
```http
Content-Type: application/json
Authorization: Bearer <jwt_token>
X-Tenant-ID: your_tenant_id
X-Request-ID: req_123456789
Accept: application/json
```

#### Response Headers
```http
Content-Type: application/json
X-Rate-Limit-Remaining: 95
X-Rate-Limit-Reset: 1640995200
X-Request-ID: req_123456789
X-Response-Time: 150ms
```

## ❌ Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "work_email",
        "code": "INVALID_FORMAT",
        "message": "Invalid email format"
      }
    ],
    "request_id": "req_123456789",
    "timestamp": "2025-01-27T10:00:00Z"
  }
}
```

### HTTP Status Codes

| Code | Description | Usage |
|------|-------------|--------|
| 200 | OK | Successful GET, PUT requests |
| 201 | Created | Successful POST requests |
| 204 | No Content | Successful DELETE requests |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource already exists |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Input validation failed |
| `AUTHENTICATION_REQUIRED` | Valid authentication required |
| `PERMISSION_DENIED` | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `RESOURCE_CONFLICT` | Resource already exists |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `AI_SERVICE_UNAVAILABLE` | AI analysis service unavailable |
| `EXTERNAL_SERVICE_ERROR` | External integration error |

## 🚦 Rate Limiting

Rate limits are applied per tenant and endpoint:

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Authentication | 10 requests | 1 minute |
| Read Operations | 1000 requests | 1 hour |
| Write Operations | 100 requests | 1 hour |
| AI Analysis | 50 requests | 1 hour |
| Bulk Operations | 10 requests | 1 hour |

### Rate Limit Headers

```http
X-Rate-Limit-Limit: 1000
X-Rate-Limit-Remaining: 945
X-Rate-Limit-Reset: 1640998800
X-Rate-Limit-Window: 3600
```

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 1000,
      "remaining": 0,
      "reset_at": "2025-01-27T11:00:00Z"
    }
  }
}
```

## 👥 Core Employee Operations

### Employee Data Model

```json
{
  "id": "emp_7a8b9c1d2e3f4",
  "employee_number": "EMP001234",
  "first_name": "John",
  "last_name": "Doe",
  "middle_name": "Michael",
  "work_email": "john.doe@company.com",
  "personal_email": "john.doe@gmail.com",
  "phone_number": "+1-555-0123",
  "hire_date": "2024-01-15",
  "birth_date": "1990-05-20",
  "department_id": "dept_123",
  "department_name": "Engineering",
  "position_id": "pos_456",
  "position_title": "Senior Software Engineer",
  "manager_id": "emp_manager123",
  "employment_status": "Active",
  "employment_type": "Full-Time",
  "work_location": "Remote",
  "base_salary": 75000.00,
  "currency_code": "USD",
  "salary_frequency": "Annual",
  "start_date": "2024-01-15",
  "end_date": null,
  "profile_image_url": "https://storage.apg.com/profiles/emp_123.jpg",
  "skills": ["Python", "JavaScript", "React", "PostgreSQL"],
  "certifications": ["AWS Certified", "Scrum Master"],
  "languages": ["English", "Spanish"],
  "emergency_contact": {
    "name": "Jane Doe",
    "relationship": "Spouse",
    "phone": "+1-555-0456"
  },
  "address": {
    "street": "123 Main St",
    "city": "New York",
    "state": "NY",
    "zip_code": "10001",
    "country": "US"
  },
  "created_at": "2024-01-15T09:00:00Z",
  "updated_at": "2024-01-20T14:30:00Z",
  "created_by": "user_admin123",
  "updated_by": "user_hr456"
}
```

### List Employees

```http
GET /api/v1/employees
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `page` | integer | Page number (1-based) | 1 |
| `limit` | integer | Items per page (1-100) | 50 |
| `search` | string | Search term for name/email | - |
| `department` | string | Filter by department ID | - |
| `position` | string | Filter by position ID | - |
| `status` | string | Filter by employment status | - |
| `location` | string | Filter by work location | - |
| `manager` | string | Filter by manager ID | - |
| `hire_date_from` | date | Filter by hire date (from) | - |
| `hire_date_to` | date | Filter by hire date (to) | - |
| `sort` | string | Sort field (name, email, hire_date) | name |
| `order` | string | Sort order (asc, desc) | asc |
| `include` | string | Include related data (department,position,manager) | - |

**Example Request:**

```http
GET /api/v1/employees?page=1&limit=25&department=dept_123&status=Active&include=department,position
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response:**

```json
{
  "data": [
    {
      "id": "emp_7a8b9c1d2e3f4",
      "employee_number": "EMP001234",
      "first_name": "John",
      "last_name": "Doe",
      "work_email": "john.doe@company.com",
      "employment_status": "Active",
      "department": {
        "id": "dept_123",
        "name": "Engineering",
        "code": "ENG"
      },
      "position": {
        "id": "pos_456",
        "title": "Senior Software Engineer",
        "level": "Senior"
      }
    }
  ],
  "meta": {
    "timestamp": "2025-01-27T10:00:00Z",
    "request_id": "req_123456789"
  },
  "pagination": {
    "page": 1,
    "limit": 25,
    "total": 150,
    "total_pages": 6,
    "has_next": true,
    "has_prev": false
  }
}
```

### Get Employee

```http
GET /api/v1/employees/{employee_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `employee_id` | string | Unique employee identifier |

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `include` | string | Include related data (department,position,manager,performance,ai_profile) |

**Example Request:**

```http
GET /api/v1/employees/emp_7a8b9c1d2e3f4?include=department,position,ai_profile
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response:**

```json
{
  "data": {
    "id": "emp_7a8b9c1d2e3f4",
    "employee_number": "EMP001234",
    "first_name": "John",
    "last_name": "Doe",
    "work_email": "john.doe@company.com",
    "hire_date": "2024-01-15",
    "department": {
      "id": "dept_123",
      "name": "Engineering"
    },
    "position": {
      "id": "pos_456",
      "title": "Senior Software Engineer"
    },
    "ai_profile": {
      "retention_risk_score": 0.15,
      "performance_prediction": 0.87,
      "last_analysis_date": "2025-01-25T14:30:00Z"
    }
  }
}
```

### Create Employee

```http
POST /api/v1/employees
```

**Request Body:**

```json
{
  "first_name": "Jane",
  "last_name": "Smith",
  "work_email": "jane.smith@company.com",
  "personal_email": "jane.smith@gmail.com",
  "phone_number": "+1-555-0789",
  "hire_date": "2025-02-01",
  "birth_date": "1992-08-15",
  "department_id": "dept_123",
  "position_id": "pos_789",
  "manager_id": "emp_manager456",
  "employment_type": "Full-Time",
  "work_location": "New York Office",
  "base_salary": 80000.00,
  "currency_code": "USD",
  "emergency_contact": {
    "name": "John Smith",
    "relationship": "Spouse",
    "phone": "+1-555-0987"
  },
  "address": {
    "street": "456 Oak Ave",
    "city": "New York",
    "state": "NY",
    "zip_code": "10002",
    "country": "US"
  },
  "skills": ["Python", "Django", "PostgreSQL"],
  "enable_ai_enhancement": true
}
```

**Response:**

```json
{
  "data": {
    "id": "emp_8b9c1d2e3f4a5",
    "employee_number": "EMP001235",
    "first_name": "Jane",
    "last_name": "Smith",
    "work_email": "jane.smith@company.com",
    "employment_status": "Active",
    "created_at": "2025-01-27T10:15:00Z",
    "ai_enhancement": {
      "profile_created": true,
      "initial_analysis_scheduled": true,
      "data_quality_score": 0.95
    }
  }
}
```

### Update Employee

```http
PUT /api/v1/employees/{employee_id}
```

**Request Body:**

```json
{
  "position_id": "pos_senior_789",
  "base_salary": 90000.00,
  "manager_id": "emp_manager789",
  "skills": ["Python", "Django", "PostgreSQL", "Machine Learning"],
  "work_location": "Remote"
}
```

**Response:**

```json
{
  "data": {
    "id": "emp_8b9c1d2e3f4a5",
    "position_id": "pos_senior_789",
    "base_salary": 90000.00,
    "updated_at": "2025-01-27T10:30:00Z",
    "updated_by": "user_hr456",
    "changes_summary": {
      "position_changed": true,
      "salary_increased": 10000.00,
      "skills_updated": true
    }
  }
}
```

### Delete Employee

```http
DELETE /api/v1/employees/{employee_id}
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `soft_delete` | boolean | Soft delete (retain data) vs hard delete | true |
| `reason` | string | Reason for deletion | - |

**Response:**

```json
{
  "data": {
    "id": "emp_8b9c1d2e3f4a5",
    "deleted_at": "2025-01-27T10:45:00Z",
    "deleted_by": "user_admin123",
    "deletion_type": "soft",
    "reason": "Employee resignation"
  }
}
```

### Search Employees (Advanced)

```http
POST /api/v1/employees/search
```

**Request Body:**

```json
{
  "query": {
    "search_text": "senior engineer python",
    "filters": {
      "departments": ["dept_123", "dept_456"],
      "positions": ["pos_senior"],
      "employment_status": ["Active"],
      "hire_date_range": {
        "from": "2023-01-01",
        "to": "2024-12-31"
      },
      "salary_range": {
        "min": 70000,
        "max": 120000,
        "currency": "USD"
      },
      "skills": {
        "required": ["Python"],
        "preferred": ["Machine Learning", "PostgreSQL"]
      }
    },
    "ai_search": {
      "enabled": true,
      "semantic_search": true,
      "similarity_threshold": 0.7
    }
  },
  "sort": [
    {"field": "hire_date", "order": "desc"},
    {"field": "last_name", "order": "asc"}
  ],
  "pagination": {
    "page": 1,
    "limit": 20
  },
  "include": ["department", "position", "ai_profile"]
}
```

**Response:**

```json
{
  "data": [
    {
      "id": "emp_match1",
      "relevance_score": 0.95,
      "match_reasons": ["skills match", "position match", "experience level"],
      "employee_data": {
        "first_name": "Alice",
        "last_name": "Johnson",
        "position": {
          "title": "Senior Software Engineer"
        },
        "skills": ["Python", "Machine Learning", "PostgreSQL"]
      }
    }
  ],
  "meta": {
    "search_type": "ai_enhanced",
    "total_matches": 12,
    "search_time_ms": 150
  }
}
```

### Bulk Operations

#### Bulk Create

```http
POST /api/v1/employees/bulk
```

**Request Body:**

```json
{
  "employees": [
    {
      "first_name": "Bob",
      "last_name": "Wilson",
      "work_email": "bob.wilson@company.com",
      "hire_date": "2025-02-01",
      "department_id": "dept_123"
    },
    {
      "first_name": "Carol",
      "last_name": "Brown",
      "work_email": "carol.brown@company.com",
      "hire_date": "2025-02-01",
      "department_id": "dept_456"
    }
  ],
  "options": {
    "validate_emails": true,
    "auto_assign_numbers": true,
    "enable_ai_enhancement": true,
    "send_welcome_emails": false
  }
}
```

**Response:**

```json
{
  "data": {
    "total_processed": 2,
    "successful": 2,
    "failed": 0,
    "results": [
      {
        "status": "success",
        "employee_id": "emp_bulk1",
        "employee_number": "EMP001236"
      },
      {
        "status": "success", 
        "employee_id": "emp_bulk2",
        "employee_number": "EMP001237"
      }
    ],
    "processing_time_ms": 850
  }
}
```

#### Bulk Update

```http
PUT /api/v1/employees/bulk
```

**Request Body:**

```json
{
  "employee_ids": ["emp_1", "emp_2", "emp_3"],
  "updates": {
    "manager_id": "emp_new_manager",
    "work_location": "Remote"
  },
  "options": {
    "validate_changes": true,
    "notify_affected_employees": true,
    "reason": "Organization restructure"
  }
}
```

## 🤖 AI & Analytics

### Employee AI Analysis

```http
POST /api/v1/employees/{employee_id}/analyze
```

**Request Body:**

```json
{
  "analysis_type": "comprehensive",
  "include_predictions": true,
  "include_recommendations": true,
  "model_config": {
    "temperature": 0.1,
    "use_latest_model": true
  }
}
```

**Response:**

```json
{
  "data": {
    "employee_id": "emp_7a8b9c1d2e3f4",
    "analysis_timestamp": "2025-01-27T10:00:00Z",
    "retention_analysis": {
      "risk_score": 0.15,
      "risk_level": "Low",
      "contributing_factors": [
        {"factor": "high_performance", "impact": -0.3},
        {"factor": "recent_promotion", "impact": -0.2},
        {"factor": "competitive_salary", "impact": -0.1}
      ],
      "recommendations": [
        "Continue current engagement strategies",
        "Consider for leadership development program"
      ]
    },
    "performance_prediction": {
      "predicted_rating": 4.2,
      "confidence": 0.87,
      "improvement_areas": ["public speaking", "project management"],
      "strength_areas": ["technical skills", "teamwork", "innovation"]
    },
    "career_analysis": {
      "promotion_readiness": 0.75,
      "recommended_next_roles": [
        "Technical Lead",
        "Senior Software Engineer",
        "Engineering Manager"
      ],
      "skill_gaps": [
        {"skill": "leadership", "gap_level": 0.4},
        {"skill": "project_management", "gap_level": 0.3}
      ],
      "development_plan": [
        "Enroll in leadership training program",
        "Lead a small project team",
        "Complete project management certification"
      ]
    },
    "skills_analysis": {
      "current_skills": {
        "Python": {"level": 0.9, "last_assessed": "2024-12-15"},
        "JavaScript": {"level": 0.8, "last_assessed": "2024-12-15"},
        "Leadership": {"level": 0.6, "last_assessed": "2024-11-20"}
      },
      "market_demand": {
        "Python": {"demand_level": "Very High", "salary_impact": 1.2},
        "Machine Learning": {"demand_level": "High", "salary_impact": 1.4}
      },
      "learning_recommendations": [
        "Advanced Python patterns",
        "Machine Learning fundamentals",
        "Cloud architecture"
      ]
    }
  }
}
```

### Skills Gap Analysis

```http
GET /api/v1/analytics/skills-gaps
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `department` | string | Filter by department |
| `position_level` | string | Filter by position level |
| `time_horizon` | string | Analysis timeframe (3m, 6m, 1y) |

**Response:**

```json
{
  "data": {
    "overall_gaps": [
      {
        "skill": "Machine Learning",
        "gap_percentage": 0.65,
        "affected_employees": 45,
        "priority": "Critical",
        "market_demand": "Very High"
      },
      {
        "skill": "Cloud Architecture", 
        "gap_percentage": 0.52,
        "affected_employees": 38,
        "priority": "High",
        "market_demand": "High"
      }
    ],
    "department_breakdown": {
      "Engineering": {
        "critical_gaps": ["Machine Learning", "DevOps"],
        "gap_score": 0.42
      },
      "Data": {
        "critical_gaps": ["Advanced Statistics", "MLOps"],
        "gap_score": 0.38
      }
    },
    "recommendations": [
      {
        "type": "training_program",
        "title": "Machine Learning Bootcamp",
        "target_employees": 45,
        "estimated_cost": 50000,
        "expected_impact": 0.7
      }
    ]
  }
}
```

### Predictive Analytics

```http
POST /api/v1/analytics/predictions
```

**Request Body:**

```json
{
  "prediction_type": "workforce_trends",
  "timeframe": "12_months",
  "include_scenarios": true,
  "departments": ["dept_123", "dept_456"],
  "confidence_threshold": 0.8
}
```

**Response:**

```json
{
  "data": {
    "turnover_predictions": {
      "overall_rate": 0.12,
      "high_risk_employees": [
        {
          "employee_id": "emp_risk1",
          "risk_score": 0.85,
          "likely_departure_month": "2025-04"
        }
      ],
      "department_rates": {
        "Engineering": 0.08,
        "Sales": 0.18,
        "Marketing": 0.15
      }
    },
    "hiring_needs": {
      "predicted_openings": 25,
      "replacement_hires": 15,
      "growth_hires": 10,
      "timeline": [
        {"month": "2025-02", "openings": 3},
        {"month": "2025-03", "openings": 5}
      ]
    },
    "performance_trends": {
      "overall_trajectory": "improving",
      "average_rating_change": 0.15,
      "top_performers": 12,
      "improvement_needed": 8
    }
  }
}
```

### Analytics Dashboard

```http
GET /api/v1/analytics/dashboard
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dashboard_id` | string | Specific dashboard ID |
| `timeframe` | string | Data timeframe (1w, 1m, 3m, 6m, 1y) |
| `refresh` | boolean | Force data refresh |

**Response:**

```json
{
  "data": {
    "dashboard_id": "executive_summary",
    "last_updated": "2025-01-27T10:00:00Z",
    "widgets": [
      {
        "id": "headcount_widget",
        "type": "metric_card",
        "title": "Total Workforce",
        "value": 1250,
        "change": {
          "value": 45,
          "percentage": 3.7,
          "direction": "up",
          "period": "month"
        }
      },
      {
        "id": "retention_widget",
        "type": "gauge",
        "title": "Retention Rate",
        "value": 0.88,
        "target": 0.85,
        "status": "above_target"
      },
      {
        "id": "performance_chart",
        "type": "line_chart",
        "title": "Performance Trends",
        "data": [
          {"period": "2024-Q1", "value": 3.8},
          {"period": "2024-Q2", "value": 3.9},
          {"period": "2024-Q3", "value": 4.0},
          {"period": "2024-Q4", "value": 4.1}
        ]
      }
    ],
    "ai_insights": [
      {
        "type": "opportunity",
        "title": "Engineering Retention Improvement",
        "description": "Engineering retention improved 15% after implementing flexible work policy",
        "confidence": 0.92
      }
    ]
  }
}
```

## 🌍 Global Workforce

### Supported Countries

```http
GET /api/v1/global/countries
```

**Response:**

```json
{
  "data": [
    {
      "country_code": "US",
      "country_name": "United States",
      "currency": "USD",
      "timezone": "America/New_York",
      "compliance_regions": ["CCPA"],
      "supported_features": [
        "payroll", "benefits", "compliance", "tax_reporting"
      ],
      "minimum_wage": 7.25,
      "working_hours": {
        "standard_per_week": 40,
        "overtime_threshold": 40
      }
    },
    {
      "country_code": "GB",
      "country_name": "United Kingdom", 
      "currency": "GBP",
      "timezone": "Europe/London",
      "compliance_regions": ["GDPR"],
      "minimum_wage": 10.42,
      "working_hours": {
        "standard_per_week": 37.5,
        "overtime_threshold": 48
      }
    }
  ]
}
```

### Localize Employee Data

```http
POST /api/v1/global/localize
```

**Request Body:**

```json
{
  "employee_id": "emp_7a8b9c1d2e3f4",
  "target_country": "GB",
  "include_compliance_check": true,
  "convert_compensation": true
}
```

**Response:**

```json
{
  "data": {
    "employee_id": "emp_7a8b9c1d2e3f4",
    "localized_data": {
      "local_employee_number": "UK000123",
      "compensation": {
        "amount": 54750.00,
        "currency": "GBP",
        "frequency": "monthly",
        "last_conversion_rate": 0.73
      },
      "working_hours": {
        "standard_per_week": 37.5,
        "standard_per_day": 7.5,
        "overtime_multiplier": 1.5
      },
      "benefits": {
        "mandatory": {
          "national_insurance": true,
          "pension_auto_enrollment": true,
          "statutory_sick_pay": true
        },
        "optional": ["private_health", "dental", "life_insurance"]
      },
      "tax_information": {
        "forms_required": ["P45", "P46"],
        "tax_id_type": "NINO",
        "withholding_table": "HMRC_2023"
      },
      "compliance_status": {
        "gdpr_compliant": true,
        "data_consent_obtained": true,
        "right_to_deletion_implemented": true
      },
      "local_holidays": [
        {"date": "2025-01-01", "name": "New Year's Day"},
        {"date": "2025-12-25", "name": "Christmas Day"}
      ]
    }
  }
}
```

### Compliance Check

```http
GET /api/v1/global/compliance/{employee_id}
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `regions` | string | Specific compliance regions to check |
| `detailed` | boolean | Include detailed compliance breakdown |

**Response:**

```json
{
  "data": {
    "employee_id": "emp_7a8b9c1d2e3f4",
    "overall_compliance": true,
    "compliance_score": 0.95,
    "regions_checked": ["GDPR", "CCPA"],
    "detailed_results": {
      "GDPR": {
        "compliant": true,
        "requirements_checked": [
          {
            "requirement": "Data Consent",
            "status": "compliant",
            "details": "Explicit consent obtained on 2024-01-15"
          },
          {
            "requirement": "Data Retention",
            "status": "compliant",
            "details": "Data retention period: 7 years"
          }
        ]
      }
    },
    "violations": [],
    "recommendations": [
      "Consider obtaining additional consent for marketing communications"
    ],
    "next_review_date": "2025-07-27"
  }
}
```

### Currency Conversion

```http
POST /api/v1/global/currency/convert
```

**Request Body:**

```json
{
  "amount": 75000.00,
  "from_currency": "USD",
  "to_currency": "EUR",
  "conversion_date": "2025-01-27"
}
```

**Response:**

```json
{
  "data": {
    "original_amount": 75000.00,
    "original_currency": "USD",
    "converted_amount": 63750.00,
    "target_currency": "EUR",
    "exchange_rate": 0.85,
    "conversion_date": "2025-01-27",
    "rate_source": "ECB",
    "rate_timestamp": "2025-01-27T08:00:00Z"
  }
}
```

## ⚡ Workflow Automation

### List Workflows

```http
GET /api/v1/workflows
```

**Response:**

```json
{
  "data": [
    {
      "workflow_id": "employee_onboarding",
      "workflow_name": "Employee Onboarding Process",
      "description": "Comprehensive employee onboarding with AI validation",
      "status": "active",
      "version": "1.2.0",
      "triggers": ["api_call", "integration_event"],
      "estimated_duration": "2-3 hours",
      "tasks_count": 6,
      "success_rate": 0.98
    }
  ]
}
```

### Execute Workflow

```http
POST /api/v1/workflows/{workflow_id}/execute
```

**Request Body:**

```json
{
  "input_data": {
    "employee_data": {
      "first_name": "New",
      "last_name": "Employee",
      "work_email": "new.employee@company.com",
      "department_id": "dept_123"
    },
    "role_assignments": ["employee", "team_member"],
    "compliance_regions": ["GDPR", "CCPA"]
  },
  "options": {
    "notify_stakeholders": true,
    "auto_approve_standard_tasks": false,
    "priority": "normal"
  }
}
```

**Response:**

```json
{
  "data": {
    "execution_id": "exec_789abc123def",
    "workflow_id": "employee_onboarding",
    "status": "running",
    "started_at": "2025-01-27T10:15:00Z",
    "estimated_completion": "2025-01-27T13:15:00Z",
    "current_task": "validate_employee_data",
    "progress": {
      "completed_tasks": 0,
      "total_tasks": 6,
      "percentage": 0
    }
  }
}
```

### Workflow Execution Status

```http
GET /api/v1/workflows/executions/{execution_id}
```

**Response:**

```json
{
  "data": {
    "execution_id": "exec_789abc123def",
    "workflow_id": "employee_onboarding",
    "status": "completed",
    "started_at": "2025-01-27T10:15:00Z",
    "completed_at": "2025-01-27T12:45:00Z",
    "total_duration_minutes": 150,
    "task_executions": [
      {
        "task_id": "validate_employee_data",
        "task_name": "Validate Employee Data",
        "status": "completed",
        "execution_time_ms": 1200,
        "result": "validation_passed"
      },
      {
        "task_id": "create_employee_record",
        "task_name": "Create Employee Record",
        "status": "completed",
        "execution_time_ms": 850,
        "result": {"employee_id": "emp_new123"}
      }
    ],
    "output_data": {
      "employee_id": "emp_new123",
      "user_account_created": true,
      "welcome_email_sent": true
    }
  }
}
```

## ⚙️ System Administration

### Health Check

```http
GET /api/v1/health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T10:00:00Z",
  "version": "v1.0.0",
  "services": {
    "database": {
      "status": "healthy",
      "response_time_ms": 15,
      "connections": {
        "active": 8,
        "max": 20
      }
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 5,
      "memory_usage": "45MB"
    },
    "ai_orchestration": {
      "status": "healthy",
      "response_time_ms": 150,
      "models_loaded": 3
    }
  },
  "metrics": {
    "requests_per_minute": 245,
    "average_response_time_ms": 120,
    "error_rate_percent": 0.02
  }
}
```

### API Statistics

```http
GET /api/v1/stats
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `period` | string | Statistics period (hour, day, week, month) |
| `detailed` | boolean | Include detailed breakdown |

**Response:**

```json
{
  "data": {
    "period": "day",
    "period_start": "2025-01-27T00:00:00Z",
    "period_end": "2025-01-27T23:59:59Z",
    "total_requests": 12450,
    "successful_requests": 12425,
    "failed_requests": 25,
    "average_response_time_ms": 125,
    "endpoints": {
      "/api/v1/employees": {
        "requests": 8500,
        "avg_response_time_ms": 95,
        "error_rate": 0.01
      },
      "/api/v1/employees/{id}/analyze": {
        "requests": 450,
        "avg_response_time_ms": 750,
        "error_rate": 0.02
      }
    },
    "error_breakdown": {
      "400": 15,
      "401": 5,
      "404": 3,
      "500": 2
    },
    "peak_hour": {
      "hour": "14:00-15:00",
      "requests": 850
    }
  }
}
```

### System Configuration

```http
GET /api/v1/config
```

**Response:**

```json
{
  "data": {
    "features": {
      "ai_analysis": true,
      "global_workforce": true,
      "workflow_automation": true,
      "advanced_analytics": true
    },
    "limits": {
      "max_employees_per_tenant": 10000,
      "max_file_upload_size_mb": 50,
      "api_rate_limit_per_hour": 1000
    },
    "supported_integrations": [
      "workday", "bamboohr", "adp", "slack", "teams"
    ],
    "ai_models": {
      "primary": "gpt-4",
      "fallback": "claude-3",
      "embedding": "text-embedding-ada-002"
    }
  }
}
```

---

## 📚 Examples & SDKs

### Python SDK Example

```python
import apg_employee_sdk

# Initialize client
client = apg_employee_sdk.Client(
    api_key="your_api_key",
    base_url="https://api.datacraft.co.ke/employee-management"
)

# Create employee
employee = client.employees.create({
    "first_name": "John",
    "last_name": "Doe",
    "work_email": "john.doe@company.com",
    "hire_date": "2025-02-01",
    "department_id": "dept_123"
})

# AI analysis
analysis = client.ai.analyze_employee(employee.id)
print(f"Retention risk: {analysis.retention_risk_score}")

# Search employees
results = client.employees.search(
    query="senior python developer",
    ai_search=True
)
```

### JavaScript/Node.js SDK Example

```javascript
const { APGEmployeeClient } = require('@datacraft/apg-employee-sdk');

const client = new APGEmployeeClient({
  apiKey: 'your_api_key',
  baseURL: 'https://api.datacraft.co.ke/employee-management'
});

// Create employee
const employee = await client.employees.create({
  firstName: 'Jane',
  lastName: 'Smith',
  workEmail: 'jane.smith@company.com',
  hireDate: '2025-02-01',
  departmentId: 'dept_123'
});

// Get AI insights
const insights = await client.ai.analyzeEmployee(employee.id);
console.log('Performance prediction:', insights.performancePrediction);
```

### cURL Examples

#### Create Employee
```bash
curl -X POST "https://api.datacraft.co.ke/employee-management/api/v1/employees" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Test",
    "last_name": "Employee", 
    "work_email": "test@company.com",
    "hire_date": "2025-02-01",
    "department_id": "dept_123"
  }'
```

#### AI Analysis
```bash
curl -X POST "https://api.datacraft.co.ke/employee-management/api/v1/employees/emp_123/analyze" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "comprehensive",
    "include_predictions": true
  }'
```

---

© 2025 Datacraft. All rights reserved.  
API Reference Version 1.0 | Last Updated: January 2025