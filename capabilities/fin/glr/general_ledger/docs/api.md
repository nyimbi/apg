# APG General Ledger API Documentation

**Complete API Reference for the Revolutionary General Ledger System**

*Version 1.0 | Last Updated: January 2025*

---

## 🚀 **Quick Start**

### **Base URL**
```
https://api.your-domain.com/v1/financial/general-ledger
```

### **Authentication**
```http
Authorization: Bearer your-jwt-token
X-Tenant-ID: your-tenant-id
```

### **Content Type**
```http
Content-Type: application/json
```

---

## 📋 **Core GL Operations**

### **Journal Entries**

#### **Create Journal Entry**
```http
POST /journal-entries
```

**Request Body:**
```json
{
  "description": "Monthly office rent payment",
  "entry_date": "2025-01-31",
  "source": "MANUAL",
  "reference_number": "JE-2025-001",
  "lines": [
    {
      "account_id": "6000",
      "debit_amount": "5000.00",
      "credit_amount": "0.00", 
      "description": "Office rent expense",
      "department_id": "ADMIN",
      "cost_center": "CC001"
    },
    {
      "account_id": "1000",
      "debit_amount": "0.00",
      "credit_amount": "5000.00",
      "description": "Cash payment",
      "department_id": "ADMIN",
      "cost_center": "CC001"
    }
  ],
  "attachments": [
    {
      "file_name": "rent_invoice.pdf",
      "file_content": "base64_encoded_content"
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "journal_entry_id": "je_2025_001_abc123",
  "entry_number": "JE-2025-001",
  "total_debits": "5000.00",
  "total_credits": "5000.00",
  "is_balanced": true,
  "created_at": "2025-01-31T10:30:00Z",
  "created_by": "user123",
  "workflow_status": "POSTED",
  "approval_required": false
}
```

#### **Get Journal Entry**
```http
GET /journal-entries/{journal_entry_id}
```

**Response:**
```json
{
  "journal_entry_id": "je_2025_001_abc123",
  "entry_number": "JE-2025-001", 
  "description": "Monthly office rent payment",
  "entry_date": "2025-01-31",
  "source": "MANUAL",
  "status": "POSTED",
  "total_debits": "5000.00",
  "total_credits": "5000.00",
  "lines": [...],
  "audit_trail": {...},
  "attachments": [...]
}
```

#### **List Journal Entries**
```http
GET /journal-entries?page=1&limit=50&date_from=2025-01-01&date_to=2025-01-31
```

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 50, max: 1000)
- `date_from` (date): Start date filter
- `date_to` (date): End date filter  
- `account_id` (string): Filter by account
- `source` (string): Filter by source (MANUAL, IMPORT, API, etc.)
- `status` (string): Filter by status (DRAFT, POSTED, REVERSED)

### **Accounts**

#### **Create Account**
```http
POST /accounts
```

**Request Body:**
```json
{
  "account_code": "6500",
  "account_name": "Travel Expenses",
  "account_type": "EXPENSE",
  "parent_account_id": "6000",
  "description": "Business travel and transportation costs",
  "is_active": true,
  "currency": "USD",
  "tax_code": "EXPENSE_TAX",
  "department_restriction": ["SALES", "MARKETING"]
}
```

#### **Get Chart of Accounts**
```http
GET /accounts?include_inactive=false&account_type=EXPENSE
```

**Response:**
```json
{
  "accounts": [
    {
      "account_id": "acc_6500_xyz789",
      "account_code": "6500",
      "account_name": "Travel Expenses", 
      "account_type": "EXPENSE",
      "parent_account_id": "6000",
      "balance": "15750.00",
      "is_active": true,
      "created_at": "2025-01-01T00:00:00Z",
      "children": []
    }
  ],
  "total_count": 125,
  "page": 1,
  "limit": 50
}
```

### **Trial Balance**

#### **Generate Trial Balance**
```http
GET /trial-balance?as_of_date=2025-01-31&include_zero_balances=false
```

**Response:**
```json
{
  "as_of_date": "2025-01-31",
  "generated_at": "2025-01-31T15:30:00Z",
  "accounts": [
    {
      "account_id": "1000",
      "account_code": "1000",
      "account_name": "Cash",
      "account_type": "ASSET",
      "debit_balance": "150000.00",
      "credit_balance": "0.00"
    },
    {
      "account_id": "4000", 
      "account_code": "4000",
      "account_name": "Revenue",
      "account_type": "REVENUE",
      "debit_balance": "0.00", 
      "credit_balance": "200000.00"
    }
  ],
  "totals": {
    "total_debits": "500000.00",
    "total_credits": "500000.00",
    "is_balanced": true
  },
  "summary_by_type": {
    "ASSET": {"debits": "300000.00", "credits": "0.00"},
    "LIABILITY": {"debits": "0.00", "credits": "100000.00"},
    "EQUITY": {"debits": "0.00", "credits": "100000.00"},
    "REVENUE": {"debits": "0.00", "credits": "200000.00"},
    "EXPENSE": {"debits": "200000.00", "credits": "0.00"}
  }
}
```

---

## 🧠 **AI-Powered Features**

### **Natural Language Processing**

#### **Process Natural Language Entry**
```http
POST /ai/process-natural-language
```

**Request Body:**
```json
{
  "description": "Paid $5,000 office rent for January with check #1234",
  "amount": "5000.00",
  "context": {
    "user_role": "accountant",
    "current_period": "2025-01",
    "department": "ADMIN"
  }
}
```

**Response:**
```json
{
  "transaction_intent": {
    "type": "payment",
    "confidence": "HIGH",
    "entities": {
      "amount": "5000.00",
      "vendor": "office rent",
      "payment_method": "check",
      "check_number": "1234"
    }
  },
  "suggested_lines": [
    {
      "account_id": "6100",
      "account_name": "Rent Expense",
      "debit_amount": "5000.00",
      "credit_amount": "0.00",
      "description": "Office rent - January",
      "confidence": 0.95
    },
    {
      "account_id": "1000",
      "account_name": "Cash",
      "debit_amount": "0.00", 
      "credit_amount": "5000.00",
      "description": "Check #1234 payment",
      "confidence": 0.95
    }
  ],
  "overall_confidence": "HIGH",
  "reasoning": "Identified expense payment transaction with clear account mappings",
  "alternative_treatments": [
    {
      "description": "If prepaid rent for multiple months",
      "accounts": ["1300", "6100"],
      "confidence": 0.3
    }
  ],
  "compliance_check": {
    "sox_compliant": true,
    "approval_required": false,
    "documentation_needed": ["Receipt", "Invoice"]
  },
  "estimated_time_saved": 3.5
}
```

#### **Get Smart Account Suggestions**
```http
GET /ai/account-suggestions?context=travel+expense&amount=2500&entry_side=debit
```

**Response:**
```json
{
  "suggestions": [
    {
      "account_id": "6500",
      "account_code": "6500", 
      "account_name": "Travel Expenses",
      "confidence": "VERY_HIGH",
      "reasoning": "Perfect match for travel expense context",
      "historical_usage": 45,
      "similar_transactions": ["txn_001", "txn_002"],
      "compliance_notes": "Ensure receipt attachment for amounts >$75"
    },
    {
      "account_id": "6510",
      "account_code": "6510",
      "account_name": "Airfare",
      "confidence": "HIGH", 
      "reasoning": "Common subcategory for travel expenses",
      "historical_usage": 12,
      "similar_transactions": ["txn_003"]
    }
  ],
  "context_analysis": {
    "detected_keywords": ["travel", "expense"],
    "amount_analysis": "Typical range for travel expenses",
    "seasonal_factors": "Q1 typically lower travel"
  }
}
```

### **Error Detection and Prevention**

#### **Validate Journal Entry**
```http
POST /ai/validate-entry
```

**Request Body:**
```json
{
  "lines": [
    {
      "account_id": "1000",
      "debit_amount": "1000.00",
      "credit_amount": "0.00"
    },
    {
      "account_id": "4000", 
      "debit_amount": "0.00",
      "credit_amount": "900.00"
    }
  ]
}
```

**Response:**
```json
{
  "validation_result": {
    "is_valid": false,
    "overall_score": 65,
    "estimated_review_time": "5 minutes"
  },
  "errors": [
    {
      "type": "UNBALANCED_ENTRY",
      "severity": "ERROR",
      "message": "Entry is out of balance by $100.00",
      "auto_fix": true,
      "suggested_fix": "Adjust credit amount to $1000.00"
    }
  ],
  "warnings": [
    {
      "type": "UNUSUAL_AMOUNT",
      "severity": "WARNING",
      "message": "Amount is 50% higher than typical for this account",
      "explanation": "Recent transactions average $650"
    }
  ],
  "suggestions": [
    {
      "type": "ACCOUNT_SUGGESTION",
      "message": "Consider using more specific revenue account 4100",
      "confidence": 0.7
    }
  ]
}
```

---

## 🤝 **Collaborative Features**

### **Real-Time Workspace**

#### **Join Collaborative Workspace**
```http
POST /collaboration/workspaces/join
```

**Request Body:**
```json
{
  "entity_type": "journal_entry",
  "entity_id": "je_2025_001_abc123",
  "user_info": {
    "user_id": "user123",
    "user_name": "John Doe", 
    "avatar_url": "/avatars/user123.jpg"
  }
}
```

**Response:**
```json
{
  "workspace_id": "journal_entry:je_2025_001_abc123",
  "session_token": "ws_token_xyz789",
  "active_users": [
    {
      "user_id": "user123",
      "user_name": "John Doe",
      "activity": "VIEWING",
      "last_seen": "2025-01-31T15:30:00Z",
      "cursor_position": {"field": "description", "line": 1}
    }
  ],
  "workflow_state": {
    "current_step": "DRAFT",
    "pending_approvals": [],
    "next_actions": ["complete_entry", "request_approval"]
  }
}
```

#### **Update User Activity**
```http
PUT /collaboration/workspaces/{workspace_id}/activity
```

**Request Body:**
```json
{
  "activity": "EDITING",
  "location": "/journal-entries/je_2025_001_abc123/line/1",
  "cursor_position": {
    "field": "debit_amount",
    "line": 1,
    "character_position": 5
  },
  "typing_indicator": true
}
```

#### **Propose Change**
```http
POST /collaboration/workspaces/{workspace_id}/changes
```

**Request Body:**
```json
{
  "change_type": "field_update",
  "target": {
    "line_number": 1,
    "field": "debit_amount"
  },
  "old_value": "5000.00",
  "new_value": "5500.00",
  "reason": "Amount correction per updated invoice"
}
```

**Response:**
```json
{
  "change_id": "change_abc123",
  "status": "accepted",
  "conflicts_detected": false,
  "requires_approval": false,
  "applied_at": "2025-01-31T15:35:00Z"
}
```

---

## 🔍 **Advanced Search**

### **Contextual Search**

#### **Natural Language Search**
```http
GET /search?q=unusual+office+expenses+last+quarter&context=variance_analysis
```

**Response:**
```json
{
  "query": "unusual office expenses last quarter",
  "query_type": "ANOMALY",
  "total_results": 15,
  "processing_time_ms": 245,
  "results": [
    {
      "result_id": "search_001",
      "result_type": "TRANSACTION",
      "title": "Office Equipment Purchase - $15,000",
      "description": "Unusual large office equipment purchase",
      "relevance_score": 0.95,
      "confidence": 0.88,
      "data": {
        "transaction_id": "je_2024_q4_055",
        "amount": "15000.00",
        "date": "2024-12-15",
        "account": "Office Equipment"
      },
      "highlights": ["unusual", "office", "$15,000"],
      "business_impact": "400% above normal quarterly spending",
      "action_suggestions": [
        "Review purchase approval documentation",
        "Verify asset tag and location"
      ]
    }
  ],
  "suggestions": [
    "Show variance analysis for office expenses",
    "Compare to previous quarters",
    "Show approval trail for large purchases"
  ],
  "insights": [
    {
      "type": "amount_distribution",
      "title": "Amount Distribution",
      "description": "Office expenses ranged from $500 to $15,000",
      "data": {
        "min_amount": "500.00",
        "max_amount": "15000.00", 
        "avg_amount": "3200.00"
      }
    }
  ]
}
```

#### **Find Similar Transactions**
```http
GET /search/similar/{transaction_id}
```

**Response:**
```json
{
  "reference_transaction": {
    "transaction_id": "je_2025_001",
    "description": "Office rent payment",
    "amount": "5000.00"
  },
  "similar_transactions": [
    {
      "transaction_id": "je_2024_012_345",
      "similarity_score": 0.92,
      "similarity_factors": {
        "amount": 0.98,
        "description": 0.95,
        "account": 1.0,
        "timing": 0.85
      },
      "explanation": "Very similar monthly rent payment"
    }
  ]
}
```

---

## 🏢 **Multi-Entity Operations**

### **Multi-Entity Transactions**

#### **Create Inter-Entity Sale**
```http
POST /multi-entity/inter-entity-sale
```

**Request Body:**
```json
{
  "seller_entity_id": "entity_us_001",
  "buyer_entity_id": "entity_uk_002", 
  "amount": "100000.00",
  "currency": "USD",
  "product_description": "Software license transfer",
  "sale_date": "2025-01-31",
  "transfer_pricing": {
    "method": "market_rate",
    "documentation_level": "detailed"
  }
}
```

**Response:**
```json
{
  "transaction_id": "me_sale_abc123",
  "status": "completed",
  "entity_entries": {
    "entity_us_001": {
      "journal_entry_id": "je_us_001",
      "currency": "USD",
      "lines": [
        {
          "account": "Intercompany Receivable",
          "debit_amount": "100000.00"
        },
        {
          "account": "Intercompany Revenue", 
          "credit_amount": "100000.00"
        }
      ]
    },
    "entity_uk_002": {
      "journal_entry_id": "je_uk_002",
      "currency": "GBP",
      "exchange_rate": "0.7500",
      "lines": [
        {
          "account": "Software Assets",
          "debit_amount": "75000.00"
        },
        {
          "account": "Intercompany Payable",
          "credit_amount": "75000.00"
        }
      ]
    }
  },
  "consolidation_entries": [
    {
      "entry_type": "elimination",
      "description": "Eliminate intercompany sale",
      "amount": "100000.00"
    }
  ]
}
```

#### **Generate Consolidation Package**
```http
POST /multi-entity/consolidation
```

**Request Body:**
```json
{
  "consolidation_date": "2025-01-31",
  "consolidation_level": "group",
  "entities": ["entity_us_001", "entity_uk_002", "entity_de_003"]
}
```

---

## 🛡️ **Compliance & Audit**

### **Real-Time Compliance Monitoring**

#### **Monitor Transaction Compliance**
```http
POST /compliance/monitor-transaction
```

**Request Body:**
```json
{
  "transaction_data": {
    "id": "txn_001",
    "amount": "25000.00",
    "created_by": "user123",
    "approvals": []
  }
}
```

**Response:**
```json
{
  "compliance_status": "non_compliant",
  "violations_detected": [
    {
      "rule_id": "SOX_APPROVAL_001",
      "severity": "HIGH",
      "description": "Transactions over $10,000 require manager approval",
      "remediation_required": true,
      "resolution_deadline": "2025-02-01T17:00:00Z"
    }
  ],
  "risk_score": 75,
  "recommendations": [
    "Obtain manager approval immediately",
    "Document business justification",
    "Attach supporting documentation"
  ]
}
```

#### **Generate Audit Package**
```http
POST /compliance/audit-package
```

**Request Body:**
```json
{
  "audit_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "audit_scope": ["journal_entries", "account_balances", "controls"],
  "frameworks": ["SOX", "GAAP"]
}
```

**Response:**
```json
{
  "package_id": "audit_2024_annual_xyz789",
  "generation_status": "completed",
  "package_integrity_hash": "sha256_hash_value",
  "sections": {
    "executive_summary": {
      "overall_compliance_rate": "98.5%",
      "material_weaknesses": 0,
      "significant_deficiencies": 2
    },
    "audit_trails": {
      "total_events": 125000,
      "integrity_verified": true
    },
    "control_testing": {
      "controls_tested": 45,
      "effectiveness_rate": "96.7%"
    }
  },
  "download_url": "/downloads/audit-packages/audit_2024_annual_xyz789.zip"
}
```

---

## 🎨 **Visual Flow Designer**

### **Transaction Flow Management**

#### **Create Visual Flow**
```http
POST /flows
```

**Request Body:**
```json
{
  "flow_name": "Monthly Accrual Process",
  "description": "Automated monthly accrual calculation and posting",
  "nodes": [
    {
      "id": "start_1",
      "type": "START",
      "name": "Begin Accrual Process",
      "position": {"x": 100, "y": 100}
    },
    {
      "id": "calc_1", 
      "type": "CALCULATION",
      "name": "Calculate Payroll Accrual",
      "position": {"x": 250, "y": 100},
      "properties": {
        "formula": "$payroll_base * $accrual_rate",
        "variable_name": "payroll_accrual"
      }
    },
    {
      "id": "debit_1",
      "type": "ACCOUNT_DEBIT", 
      "name": "Debit Payroll Expense",
      "position": {"x": 400, "y": 100},
      "properties": {
        "account_id": "7200",
        "amount": "$payroll_accrual"
      }
    }
  ],
  "connections": [
    {
      "from": "start_1",
      "to": "calc_1",
      "type": "success"
    },
    {
      "from": "calc_1", 
      "to": "debit_1",
      "type": "success"
    }
  ],
  "variables": {
    "payroll_base": "50000.00",
    "accrual_rate": "0.15"
  }
}
```

#### **Execute Flow**
```http
POST /flows/{flow_id}/execute
```

**Request Body:**
```json
{
  "input_data": {
    "payroll_base": "52000.00",
    "accrual_rate": "0.15",
    "period": "2025-01"
  },
  "user_id": "user123"
}
```

**Response:**
```json
{
  "execution_id": "exec_abc123",
  "status": "completed",
  "execution_time_ms": 1250,
  "steps_executed": [
    {
      "node_id": "calc_1",
      "result": {
        "payroll_accrual": "7800.00"
      }
    }
  ],
  "generated_journal_entries": [
    {
      "entry_id": "je_flow_generated_001",
      "description": "Monthly payroll accrual - generated by flow",
      "lines": [...]
    }
  ]
}
```

---

## 📅 **Smart Period Close**

### **Period Close Automation**

#### **Initiate Smart Close**
```http
POST /period-close/initiate
```

**Request Body:**
```json
{
  "close_type": "MONTHLY",
  "period_end": "2025-01-31",
  "entity_id": "entity_001",
  "automation_level": "FULL"
}
```

**Response:**
```json
{
  "session_id": "close_monthly_2025_01_abc123",
  "status": "IN_PROGRESS",
  "target_completion_date": "2025-02-03T17:00:00Z",
  "tasks": [
    {
      "task_id": "task_001",
      "task_name": "Validate Trial Balance",
      "status": "COMPLETED",
      "automation_level": 0.9
    },
    {
      "task_id": "task_002", 
      "task_name": "Calculate Payroll Accrual",
      "status": "IN_PROGRESS",
      "automation_level": 0.85
    }
  ],
  "progress_percentage": 25.5,
  "automation_percentage": 87.3
}
```

#### **Execute Automated Tasks**
```http
POST /period-close/{session_id}/execute-automated
```

**Response:**
```json
{
  "tasks_executed": 5,
  "tasks_completed": 4,
  "tasks_failed": 1,
  "automation_savings": "4.5 hours",
  "exceptions_identified": [
    {
      "exception_type": "variance_threshold",
      "description": "Travel expenses 25% above budget", 
      "requires_review": true
    }
  ],
  "next_actions": [
    "Review variance exceptions",
    "Approve accrual calculations",
    "Generate financial statements"
  ]
}
```

#### **Monitor Close Progress**
```http
GET /period-close/{session_id}/progress
```

**Response:**
```json
{
  "overall_progress": 72.5,
  "timeline_analysis": {
    "time_elapsed": "2 days 4 hours",
    "time_remaining": "6 hours",
    "on_track": true
  },
  "bottlenecks": [
    {
      "type": "dependency_block",
      "task_name": "Bank Reconciliation",
      "severity": "medium",
      "description": "Waiting for bank statement import"
    }
  ],
  "recommendations": [
    "Import bank statement to unblock reconciliation",
    "Consider parallel processing of independent tasks"
  ]
}
```

---

## 📊 **Financial Health Monitoring**

### **Health Assessment**

#### **Assess Financial Health**
```http
GET /health/assessment/{entity_id}
```

**Response:**
```json
{
  "assessment_id": "health_assessment_abc123",
  "entity_id": "entity_001",
  "assessment_date": "2025-01-31T15:30:00Z",
  "overall_score": 82.5,
  "overall_grade": "GOOD",
  "dimension_scores": {
    "LIQUIDITY": 85.2,
    "PROFITABILITY": 78.9,
    "EFFICIENCY": 82.1,
    "LEVERAGE": 75.6,
    "GROWTH": 88.3,
    "CASH_FLOW": 91.2
  },
  "key_metrics": [
    {
      "metric_name": "Current Ratio",
      "current_value": "2.35",
      "benchmark_value": "2.00",
      "trend_direction": "improving",
      "score": 88.0
    }
  ],
  "active_alerts": [
    {
      "alert_type": "liquidity_warning",
      "severity": "MEDIUM",
      "title": "Cash Flow Projection Alert",
      "recommended_actions": ["Accelerate receivables collection"]
    }
  ],
  "recommendations": [
    {
      "type": "operational",
      "priority": "medium",
      "title": "Optimize Working Capital",
      "estimated_impact": "5-10% improvement in cash flow"
    }
  ]
}
```

#### **Predict Financial Trends**
```http
POST /health/predict-trends
```

**Request Body:**
```json
{
  "entity_id": "entity_001",
  "prediction_horizon": "90 days",
  "scenarios": ["base_case", "optimistic", "pessimistic"]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "insight_type": "cash_flow_trend",
      "prediction_horizon": "90 days",
      "confidence_level": 0.85,
      "predicted_outcome": {
        "trend": "stable_with_slight_growth",
        "predicted_cash_flow": "580000.00",
        "confidence_interval": {
          "lower": "520000.00",
          "upper": "640000.00"
        }
      },
      "risk_probability": 0.15,
      "contributing_factors": [
        "Seasonal revenue increase",
        "Controlled expense growth"
      ],
      "recommended_mitigations": [
        "Maintain current cash management",
        "Monitor seasonal variations"
      ]
    }
  ]
}
```

---

## 📈 **Reporting & Analytics**

### **Financial Reports**

#### **Generate Income Statement**
```http
GET /reports/income-statement?period_start=2025-01-01&period_end=2025-01-31&format=json
```

**Response:**
```json
{
  "report_type": "INCOME_STATEMENT",
  "period": {
    "start_date": "2025-01-01",
    "end_date": "2025-01-31"
  },
  "sections": {
    "revenue": {
      "total": "500000.00",
      "accounts": [
        {
          "account_name": "Product Sales",
          "amount": "350000.00"
        },
        {
          "account_name": "Service Revenue", 
          "amount": "150000.00"
        }
      ]
    },
    "expenses": {
      "total": "320000.00",
      "accounts": [...]
    },
    "net_income": "180000.00"
  },
  "variance_analysis": {
    "budget_variance": {
      "revenue": "25000.00",
      "expenses": "-15000.00",
      "net_income": "40000.00"
    },
    "prior_period_variance": {
      "revenue": "50000.00",
      "expenses": "20000.00", 
      "net_income": "30000.00"
    }
  }
}
```

#### **Custom Report Builder**
```http
POST /reports/custom
```

**Request Body:**
```json
{
  "report_name": "Department Expense Analysis",
  "date_range": {
    "start_date": "2025-01-01",
    "end_date": "2025-01-31"
  },
  "dimensions": ["department", "expense_type"],
  "metrics": ["total_amount", "transaction_count", "variance_vs_budget"],
  "filters": {
    "account_type": "EXPENSE",
    "amount_range": {"min": "1000.00"}
  },
  "format": "json"
}
```

---

## 🔐 **Security & Permissions**

### **Access Control**

#### **Check User Permissions**
```http
GET /security/permissions?resource=journal_entries&action=create
```

**Response:**
```json
{
  "user_id": "user123",
  "permissions": {
    "journal_entries": {
      "create": true,
      "read": true,
      "update": false,
      "delete": false,
      "approve": false
    },
    "constraints": {
      "amount_limit": "10000.00",
      "department_restriction": ["SALES", "MARKETING"],
      "approval_required_above": "5000.00"
    }
  }
}
```

### **Audit Trail**

#### **Get Audit Trail**
```http
GET /audit/trail/{entity_id}?entity_type=journal_entry&date_from=2025-01-01
```

**Response:**
```json
{
  "audit_events": [
    {
      "event_id": "audit_001",
      "event_type": "TRANSACTION_CREATED",
      "timestamp": "2025-01-31T10:30:00Z",
      "user_id": "user123",
      "entity_id": "je_2025_001",
      "changes": {
        "action": "CREATE",
        "data": {...}
      },
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0...",
      "integrity_hash": "sha256_hash"
    }
  ],
  "integrity_verified": true,
  "total_events": 1250
}
```

---

## ⚡ **Performance & Monitoring**

### **System Health**

#### **Get System Health**
```http
GET /health/system
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-31T15:30:00Z",
  "services": {
    "database": {
      "status": "healthy",
      "response_time_ms": 15,
      "connections": 25
    },
    "cache": {
      "status": "healthy", 
      "hit_rate": 0.95,
      "memory_usage": "65%"
    },
    "ai_services": {
      "status": "healthy",
      "response_time_ms": 250,
      "queue_depth": 5
    }
  },
  "metrics": {
    "transactions_per_second": 150,
    "active_users": 45,
    "error_rate": 0.001
  }
}
```

---

## 🚨 **Error Handling**

### **Error Response Format**

All API endpoints return errors in a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Journal entry validation failed",
    "details": [
      {
        "field": "lines[0].debit_amount",
        "code": "REQUIRED_FIELD",
        "message": "Debit amount is required"
      }
    ],
    "request_id": "req_abc123",
    "timestamp": "2025-01-31T15:30:00Z"
  }
}
```

### **HTTP Status Codes**

- `200` - Success
- `201` - Created
- `400` - Bad Request (validation errors)
- `401` - Unauthorized
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `409` - Conflict (business rule violation)
- `422` - Unprocessable Entity (semantic errors)
- `429` - Too Many Requests (rate limiting)
- `500` - Internal Server Error

### **Common Error Codes**

- `VALIDATION_ERROR` - Request validation failed
- `BUSINESS_RULE_VIOLATION` - Business logic constraint violated
- `INSUFFICIENT_PERMISSIONS` - User lacks required permissions
- `RESOURCE_NOT_FOUND` - Requested resource does not exist
- `DUPLICATE_RESOURCE` - Resource already exists
- `SYSTEM_ERROR` - Internal system error
- `RATE_LIMIT_EXCEEDED` - API rate limit exceeded

---

## 📱 **Webhooks**

### **Webhook Events**

Register webhooks to receive real-time notifications:

```http
POST /webhooks
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhooks/gl-events",
  "events": [
    "journal_entry.created",
    "journal_entry.posted", 
    "compliance.violation_detected",
    "period_close.completed"
  ],
  "secret": "your-webhook-secret"
}
```

### **Event Payload Example**

```json
{
  "event_type": "journal_entry.created",
  "timestamp": "2025-01-31T15:30:00Z",
  "data": {
    "journal_entry_id": "je_2025_001_abc123",
    "description": "Monthly rent payment",
    "amount": "5000.00",
    "created_by": "user123"
  },
  "webhook_id": "webhook_123",
  "signature": "sha256=signature_hash"
}
```

---

## 🔧 **SDK Examples**

### **Python SDK**

```python
from apg_gl_sdk import GeneralLedgerClient

# Initialize client
gl = GeneralLedgerClient(
    api_key="your-api-key",
    tenant_id="your-tenant"
)

# Create journal entry
entry = await gl.journal_entries.create({
    "description": "Office rent payment",
    "entry_date": "2025-01-31",
    "lines": [
        {"account_id": "6100", "debit_amount": "5000.00"},
        {"account_id": "1000", "credit_amount": "5000.00"}
    ]
})

# Process with AI
suggestion = await gl.ai.process_natural_language(
    "Paid $2,500 for marketing consulting"
)

# Search transactions
results = await gl.search.query(
    "unusual travel expenses last month"
)
```

### **JavaScript SDK**

```javascript
import { GeneralLedgerClient } from '@apg/gl-sdk';

const gl = new GeneralLedgerClient({
  apiKey: 'your-api-key',
  tenantId: 'your-tenant'
});

// Create journal entry
const entry = await gl.journalEntries.create({
  description: 'Office rent payment',
  entryDate: '2025-01-31',
  lines: [
    { accountId: '6100', debitAmount: '5000.00' },
    { accountId: '1000', creditAmount: '5000.00' }
  ]
});

// AI processing
const suggestion = await gl.ai.processNaturalLanguage(
  'Received $10,000 payment from ABC Corp'
);
```

---

## 📋 **Rate Limits**

- **Standard Tier**: 1,000 requests/hour
- **Professional Tier**: 10,000 requests/hour  
- **Enterprise Tier**: Unlimited

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1643723400
```

---

## 🆘 **Support**

- **API Documentation**: [https://docs.datacraft.co.ke/api](https://docs.datacraft.co.ke/api)
- **Support Email**: api-support@datacraft.co.ke
- **Status Page**: [https://status.datacraft.co.ke](https://status.datacraft.co.ke)
- **GitHub Issues**: [https://github.com/datacraft/apg-gl/issues](https://github.com/datacraft/apg-gl/issues)

---

*API Documentation Version 1.0 | © 2025 Datacraft. All rights reserved.*