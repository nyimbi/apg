# Enterprise Asset Management API Reference

## Overview

The EAM API provides comprehensive REST endpoints for enterprise asset management with full APG platform integration. All endpoints support async operations, multi-tenant security, and real-time collaboration.

**Base URL**: `https://eam.apg.datacraft.co.ke/api/v1`

**Authentication**: APG OAuth2 JWT tokens required for all endpoints

**Rate Limiting**: 1000 requests per hour per user

## Authentication

### Bearer Token Authentication

```http
Authorization: Bearer <jwt_token>
X-Tenant-ID: <tenant_id>
Content-Type: application/json
```

### Example Authentication

```python
import httpx

headers = {
    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "X-Tenant-ID": "tenant_12345",
    "Content-Type": "application/json"
}

async with httpx.AsyncClient() as client:
    response = await client.get(
        "https://eam.apg.datacraft.co.ke/api/v1/assets",
        headers=headers
    )
```

## Asset Management

### List Assets

```http
GET /assets
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `page` | integer | Page number | 1 |
| `limit` | integer | Items per page (max 100) | 25 |
| `asset_type` | string | Filter by asset type | - |
| `status` | string | Filter by status | - |
| `location_id` | string | Filter by location | - |
| `criticality_level` | string | Filter by criticality | - |
| `health_score_min` | number | Minimum health score | - |
| `health_score_max` | number | Maximum health score | - |
| `search` | string | Text search in name/description | - |
| `sort_by` | string | Sort field | `asset_number` |
| `sort_order` | string | `asc` or `desc` | `asc` |

**Response:**

```json
{
    "data": [
        {
            "asset_id": "01933b2e-1234-7890-abcd-ef1234567890",
            "asset_number": "ASSET-001",
            "asset_name": "CNC Machine #1",
            "description": "High-precision CNC machining center",
            "asset_type": "equipment",
            "asset_category": "production",
            "asset_class": "rotating",
            "criticality_level": "high",
            "status": "active",
            "operational_status": "operational",
            "health_score": 95.5,
            "condition_status": "excellent",
            "manufacturer": "ACME Manufacturing",
            "model_number": "CNC-5000",
            "serial_number": "SN123456789",
            "year_manufactured": 2022,
            "location": {
                "location_id": "01933b2e-5678-7890-abcd-ef1234567890",
                "location_name": "Production Floor A",
                "location_code": "PFA-001"
            },
            "purchase_cost": 250000.00,
            "replacement_cost": 300000.00,
            "current_book_value": 225000.00,
            "installation_date": "2023-01-15",
            "commissioning_date": "2023-02-01",
            "last_maintenance_date": "2024-01-15",
            "next_maintenance_due": "2024-02-15",
            "maintenance_strategy": "predictive",
            "maintenance_frequency_days": 30,
            "has_digital_twin": true,
            "iot_enabled": true,
            "created_on": "2023-01-01T00:00:00Z",
            "changed_on": "2024-01-01T12:30:00Z"
        }
    ],
    "pagination": {
        "page": 1,
        "limit": 25,
        "total": 156,
        "pages": 7,
        "has_next": true,
        "has_prev": false
    },
    "filters_applied": {
        "asset_type": "equipment",
        "criticality_level": "high"
    }
}
```

### Create Asset

```http
POST /assets
```

**Request Body:**

```json
{
    "asset_number": "ASSET-002",
    "asset_name": "Conveyor Belt System",
    "description": "Main production line conveyor system",
    "asset_type": "equipment",
    "asset_category": "production",
    "asset_class": "static",
    "criticality_level": "medium",
    "manufacturer": "ConveyTech Inc",
    "model_number": "CBT-2000",
    "serial_number": "CBT987654321",
    "year_manufactured": 2023,
    "location_id": "01933b2e-5678-7890-abcd-ef1234567890",
    "parent_asset_id": null,
    "custodian_employee_id": "emp_001",
    "department": "Production",
    "cost_center": "CC-PROD-001",
    "purchase_cost": 85000.00,
    "replacement_cost": 95000.00,
    "is_capitalized": true,
    "installation_date": "2023-06-01",
    "commissioning_date": "2023-06-15",
    "expected_retirement_date": "2033-06-01",
    "maintenance_strategy": "preventive",
    "maintenance_frequency_days": 60,
    "has_digital_twin": false,
    "iot_enabled": true
}
```

**Response:**

```json
{
    "message": "Asset created successfully",
    "data": {
        "asset_id": "01933b2e-9876-7890-abcd-ef1234567890",
        "asset_number": "ASSET-002",
        "status": "active",
        "created_on": "2024-01-01T15:30:00Z"
    }
}
```

### Get Asset Details

```http
GET /assets/{asset_id}
```

**Path Parameters:**
- `asset_id` (string): Unique asset identifier

**Response:**

```json
{
    "data": {
        "asset_id": "01933b2e-1234-7890-abcd-ef1234567890",
        "asset_number": "ASSET-001",
        "asset_name": "CNC Machine #1",
        "description": "High-precision CNC machining center",
        "asset_type": "equipment",
        "asset_category": "production",
        "criticality_level": "high",
        "status": "active",
        "operational_status": "operational",
        "health_score": 95.5,
        "condition_status": "excellent",
        "location": {
            "location_id": "01933b2e-5678-7890-abcd-ef1234567890",
            "location_name": "Production Floor A",
            "location_code": "PFA-001",
            "hierarchy_path": "Site > Building A > Floor 1 > Production Area"
        },
        "parent_asset": {
            "asset_id": "01933b2e-1111-7890-abcd-ef1234567890",
            "asset_name": "Production Line #1",
            "asset_number": "LINE-001"
        },
        "child_assets": [
            {
                "asset_id": "01933b2e-2222-7890-abcd-ef1234567890",
                "asset_name": "CNC Spindle Motor",
                "asset_number": "MOTOR-001"
            }
        ],
        "maintenance_history": [
            {
                "maintenance_id": "01933b2e-3333-7890-abcd-ef1234567890",
                "maintenance_date": "2024-01-15",
                "maintenance_type": "preventive",
                "outcome": "completed",
                "health_score_after": 95.5
            }
        ],
        "performance_metrics": {
            "availability_percentage": 98.5,
            "oee_overall": 85.2,
            "mtbf_hours": 2160,
            "mttr_hours": 4.5
        },
        "digital_twin": {
            "enabled": true,
            "last_sync": "2024-01-01T15:25:00Z",
            "real_time_data": {
                "temperature": 42.5,
                "vibration": 0.15,
                "power_consumption": 15.2
            }
        }
    }
}
```

### Update Asset

```http
PUT /assets/{asset_id}
```

**Request Body:**

```json
{
    "asset_name": "CNC Machine #1 - Updated",
    "description": "Updated description",
    "status": "active",
    "operational_status": "operational",
    "health_score": 92.0,
    "condition_status": "good",
    "change_reason": "Routine inspection update"
}
```

**Response:**

```json
{
    "message": "Asset updated successfully",
    "data": {
        "asset_id": "01933b2e-1234-7890-abcd-ef1234567890",
        "changes_applied": [
            "asset_name",
            "health_score",
            "condition_status"
        ],
        "changed_on": "2024-01-01T16:00:00Z"
    }
}
```

### Delete Asset

```http
DELETE /assets/{asset_id}
```

**Query Parameters:**
- `soft_delete` (boolean): Use soft delete (default: true)
- `reason` (string): Deletion reason (required)

**Response:**

```json
{
    "message": "Asset deleted successfully",
    "data": {
        "asset_id": "01933b2e-1234-7890-abcd-ef1234567890",
        "deletion_type": "soft",
        "deleted_on": "2024-01-01T16:30:00Z"
    }
}
```

### Asset Hierarchy

```http
GET /assets/{asset_id}/hierarchy
```

**Response:**

```json
{
    "data": {
        "root_asset": {
            "asset_id": "01933b2e-1111-7890-abcd-ef1234567890",
            "asset_name": "Production Line #1",
            "level": 0
        },
        "hierarchy": [
            {
                "asset_id": "01933b2e-1234-7890-abcd-ef1234567890",
                "asset_name": "CNC Machine #1",
                "level": 1,
                "children": [
                    {
                        "asset_id": "01933b2e-2222-7890-abcd-ef1234567890",
                        "asset_name": "CNC Spindle Motor",
                        "level": 2,
                        "children": []
                    }
                ]
            }
        ]
    }
}
```

## Work Order Management

### List Work Orders

```http
GET /work-orders
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `page` | integer | Page number | 1 |
| `limit` | integer | Items per page (max 100) | 25 |
| `status` | string | Filter by status | - |
| `priority` | string | Filter by priority | - |
| `work_type` | string | Filter by work type | - |
| `assigned_to` | string | Filter by assignee | - |
| `asset_id` | string | Filter by asset | - |
| `scheduled_date_from` | date | Scheduled from date | - |
| `scheduled_date_to` | date | Scheduled to date | - |
| `overdue_only` | boolean | Show only overdue orders | false |

**Response:**

```json
{
    "data": [
        {
            "work_order_id": "01933b2e-4444-7890-abcd-ef1234567890",
            "work_order_number": "WO-001234",
            "title": "Preventive Maintenance - CNC Machine",
            "description": "Scheduled PM including lubrication and calibration",
            "work_type": "maintenance",
            "priority": "medium",
            "status": "scheduled",
            "asset": {
                "asset_id": "01933b2e-1234-7890-abcd-ef1234567890",
                "asset_number": "ASSET-001",
                "asset_name": "CNC Machine #1"
            },
            "location": {
                "location_id": "01933b2e-5678-7890-abcd-ef1234567890",
                "location_name": "Production Floor A"
            },
            "assigned_to": "tech_001",
            "assigned_team": "Maintenance Team A",
            "estimated_hours": 4.0,
            "estimated_cost": 500.00,
            "scheduled_start": "2024-01-02T08:00:00Z",
            "scheduled_end": "2024-01-02T12:00:00Z",
            "created_on": "2024-01-01T10:00:00Z",
            "requested_by": "supervisor_001"
        }
    ],
    "pagination": {
        "page": 1,
        "limit": 25,
        "total": 89,
        "pages": 4
    }
}
```

### Create Work Order

```http
POST /work-orders
```

**Request Body:**

```json
{
    "title": "Emergency Repair - Conveyor Belt",
    "description": "Conveyor belt motor failure requires immediate attention",
    "work_type": "repair",
    "priority": "high",
    "asset_id": "01933b2e-9876-7890-abcd-ef1234567890",
    "location_id": "01933b2e-5678-7890-abcd-ef1234567890",
    "work_category": "mechanical",
    "maintenance_type": "corrective",
    "safety_category": "medium_risk",
    "estimated_hours": 6.0,
    "estimated_cost": 1200.00,
    "assigned_to": "tech_002",
    "required_crew_size": 2,
    "scheduled_start": "2024-01-02T14:00:00Z",
    "scheduled_end": "2024-01-02T20:00:00Z",
    "special_instructions": "Ensure production line is stopped before starting work"
}
```

**Response:**

```json
{
    "message": "Work order created successfully",
    "data": {
        "work_order_id": "01933b2e-5555-7890-abcd-ef1234567890",
        "work_order_number": "WO-001235",
        "status": "draft",
        "created_on": "2024-01-01T16:45:00Z"
    }
}
```

### Update Work Order Status

```http
PATCH /work-orders/{work_order_id}/status
```

**Request Body:**

```json
{
    "status": "in_progress",
    "status_notes": "Work started on schedule",
    "actual_start": "2024-01-02T14:00:00Z"
}
```

**Response:**

```json
{
    "message": "Work order status updated",
    "data": {
        "work_order_id": "01933b2e-5555-7890-abcd-ef1234567890",
        "previous_status": "scheduled",
        "current_status": "in_progress",
        "updated_on": "2024-01-02T14:00:00Z"
    }
}
```

### Complete Work Order

```http
POST /work-orders/{work_order_id}/complete
```

**Request Body:**

```json
{
    "work_performed": "Replaced conveyor motor and aligned belt. Tested operation for 30 minutes.",
    "completion_notes": "All systems operating normally. Recommend monitoring for first 24 hours.",
    "actual_hours": 5.5,
    "actual_cost": 1150.00,
    "quality_rating": 5,
    "health_score_before": 45.0,
    "health_score_after": 95.0,
    "parts_used": [
        {
            "inventory_id": "01933b2e-6666-7890-abcd-ef1234567890",
            "part_number": "MOTOR-CBT-001",
            "quantity": 1,
            "unit_cost": 800.00
        },
        {
            "inventory_id": "01933b2e-7777-7890-abcd-ef1234567890",
            "part_number": "BELT-CBT-001",
            "quantity": 1,
            "unit_cost": 150.00
        }
    ],
    "follow_up_required": true,
    "follow_up_date": "2024-01-09T08:00:00Z",
    "recommendations": "Schedule alignment check in 1 week"
}
```

**Response:**

```json
{
    "message": "Work order completed successfully",
    "data": {
        "work_order_id": "01933b2e-5555-7890-abcd-ef1234567890",
        "completion_time": "2024-01-02T19:30:00Z",
        "efficiency_rating": 0.92,
        "cost_variance": -50.00,
        "quality_score": 5.0,
        "follow_up_work_order": "01933b2e-8888-7890-abcd-ef1234567890"
    }
}
```

## Inventory Management

### List Inventory Items

```http
GET /inventory
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `page` | integer | Page number | 1 |
| `limit` | integer | Items per page (max 100) | 25 |
| `item_type` | string | Filter by item type | - |
| `category` | string | Filter by category | - |
| `criticality` | string | Filter by criticality | - |
| `low_stock_only` | boolean | Show only low stock items | false |
| `reorder_needed` | boolean | Show items needing reorder | false |
| `location_id` | string | Filter by storage location | - |

**Response:**

```json
{
    "data": [
        {
            "inventory_id": "01933b2e-6666-7890-abcd-ef1234567890",
            "part_number": "MOTOR-CBT-001",
            "description": "Conveyor Belt Motor 5HP",
            "item_type": "spare_part",
            "category": "motors",
            "manufacturer": "MotorTech Corp",
            "manufacturer_part_number": "MT-5HP-001",
            "model_number": "MT-5000",
            "current_stock": 3,
            "minimum_stock": 2,
            "maximum_stock": 10,
            "reorder_point": 3,
            "unit_cost": 800.00,
            "average_cost": 785.50,
            "total_value": 2356.50,
            "location": {
                "location_id": "01933b2e-9999-7890-abcd-ef1234567890",
                "location_name": "Main Warehouse",
                "storage_location": "Aisle 5, Shelf B3"
            },
            "primary_vendor": {
                "vendor_id": "vendor_001",
                "vendor_name": "Industrial Supply Co"
            },
            "lead_time_days": 7,
            "criticality": "high",
            "auto_reorder": true,
            "annual_usage": 12,
            "last_movement_date": "2024-01-02T19:30:00Z",
            "stock_status": "reorder_needed"
        }
    ],
    "pagination": {
        "page": 1,
        "limit": 25,
        "total": 234,
        "pages": 10
    },
    "summary": {
        "total_value": 456789.50,
        "low_stock_items": 12,
        "reorder_needed": 8,
        "stockout_items": 2
    }
}
```

### Create Inventory Item

```http
POST /inventory
```

**Request Body:**

```json
{
    "part_number": "BEARING-001",
    "description": "Deep Groove Ball Bearing 6205",
    "item_type": "spare_part",
    "category": "bearings",
    "manufacturer": "BearingTech Inc",
    "manufacturer_part_number": "BT-6205",
    "model_number": "BT-Standard",
    "current_stock": 25,
    "minimum_stock": 5,
    "maximum_stock": 50,
    "reorder_point": 10,
    "unit_cost": 15.50,
    "location_id": "01933b2e-9999-7890-abcd-ef1234567890",
    "storage_location": "Aisle 2, Shelf A1",
    "primary_vendor_id": "vendor_002",
    "lead_time_days": 3,
    "criticality": "medium",
    "auto_reorder": true,
    "safety_stock": 5
}
```

**Response:**

```json
{
    "message": "Inventory item created successfully",
    "data": {
        "inventory_id": "01933b2e-aaaa-7890-abcd-ef1234567890",
        "part_number": "BEARING-001",
        "current_stock": 25,
        "total_value": 387.50,
        "created_on": "2024-01-01T17:00:00Z"
    }
}
```

### Adjust Stock Levels

```http
POST /inventory/{inventory_id}/adjust
```

**Request Body:**

```json
{
    "movement_type": "adjustment",
    "quantity": -5,
    "reason": "Physical count correction",
    "reference_id": "PHY-COUNT-001",
    "cost_per_unit": 15.50,
    "notes": "Annual physical inventory count discrepancy"
}
```

**Response:**

```json
{
    "message": "Stock adjustment completed",
    "data": {
        "inventory_id": "01933b2e-aaaa-7890-abcd-ef1234567890",
        "previous_stock": 25,
        "adjustment": -5,
        "current_stock": 20,
        "movement_id": "01933b2e-bbbb-7890-abcd-ef1234567890",
        "adjusted_on": "2024-01-01T17:15:00Z"
    }
}
```

### Stock Movement History

```http
GET /inventory/{inventory_id}/movements
```

**Query Parameters:**
- `from_date` (date): Start date for movement history
- `to_date` (date): End date for movement history
- `movement_type` (string): Filter by movement type

**Response:**

```json
{
    "data": [
        {
            "movement_id": "01933b2e-bbbb-7890-abcd-ef1234567890",
            "movement_type": "adjustment",
            "quantity": -5,
            "previous_stock": 25,
            "new_stock": 20,
            "unit_cost": 15.50,
            "total_cost": -77.50,
            "reason": "Physical count correction",
            "reference_id": "PHY-COUNT-001",
            "moved_by": "user_001",
            "movement_date": "2024-01-01T17:15:00Z"
        },
        {
            "movement_id": "01933b2e-cccc-7890-abcd-ef1234567890",
            "movement_type": "issue",
            "quantity": -2,
            "previous_stock": 20,
            "new_stock": 18,
            "unit_cost": 15.50,
            "total_cost": -31.00,
            "reason": "Work order consumption",
            "reference_id": "WO-001235",
            "moved_by": "tech_002",
            "movement_date": "2024-01-02T19:30:00Z"
        }
    ]
}
```

## Analytics and Reporting

### Asset Performance Analytics

```http
GET /analytics/asset-performance
```

**Query Parameters:**
- `time_period` (string): `last_7_days`, `last_30_days`, `last_quarter`, `last_year`
- `asset_type` (string): Filter by asset type
- `location_id` (string): Filter by location
- `criticality_level` (string): Filter by criticality

**Response:**

```json
{
    "data": {
        "summary": {
            "total_assets": 156,
            "average_health_score": 87.5,
            "assets_critical_health": 8,
            "assets_excellent_health": 89,
            "overall_availability": 96.8,
            "overall_oee": 82.4
        },
        "trends": {
            "health_score_trend": [
                {"date": "2024-01-01", "value": 88.2},
                {"date": "2024-01-02", "value": 87.8},
                {"date": "2024-01-03", "value": 87.5}
            ],
            "availability_trend": [
                {"date": "2024-01-01", "value": 97.1},
                {"date": "2024-01-02", "value": 96.5},
                {"date": "2024-01-03", "value": 96.8}
            ]
        },
        "by_asset_type": [
            {
                "asset_type": "equipment",
                "count": 89,
                "avg_health_score": 89.2,
                "avg_availability": 97.5
            },
            {
                "asset_type": "vehicles",
                "count": 34,
                "avg_health_score": 84.1,
                "avg_availability": 95.2
            }
        ],
        "critical_assets": [
            {
                "asset_id": "01933b2e-dddd-7890-abcd-ef1234567890",
                "asset_name": "Hydraulic Press #3",
                "health_score": 45.2,
                "risk_level": "high",
                "recommended_action": "immediate_maintenance"
            }
        ]
    }
}
```

### Maintenance Effectiveness

```http
GET /analytics/maintenance-effectiveness
```

**Response:**

```json
{
    "data": {
        "summary": {
            "preventive_ratio": 0.78,
            "mean_time_between_failures": 2160.5,
            "mean_time_to_repair": 4.2,
            "maintenance_cost_per_asset": 2450.00,
            "maintenance_effectiveness_score": 85.6
        },
        "work_order_metrics": {
            "total_completed": 234,
            "on_time_completion": 0.89,
            "average_completion_time": 4.8,
            "cost_variance": -0.05
        },
        "by_maintenance_type": [
            {
                "maintenance_type": "preventive",
                "count": 182,
                "percentage": 78,
                "avg_cost": 450.00,
                "avg_duration": 3.2
            },
            {
                "maintenance_type": "corrective",
                "count": 52,
                "percentage": 22,
                "avg_cost": 1250.00,
                "avg_duration": 8.5
            }
        ],
        "recommendations": [
            {
                "type": "increase_preventive",
                "target_assets": ["equipment"],
                "expected_savings": 45000.00,
                "priority": "high"
            }
        ]
    }
}
```

### Cost Analysis

```http
GET /analytics/cost-analysis
```

**Query Parameters:**
- `time_period` (string): Analysis time period
- `cost_category` (string): Filter by cost category
- `breakdown_by` (string): Group results by field

**Response:**

```json
{
    "data": {
        "total_costs": {
            "maintenance": 234500.00,
            "parts": 89500.00,
            "labor": 145000.00,
            "downtime": 67500.00,
            "total": 536500.00
        },
        "cost_trends": [
            {
                "month": "2024-01",
                "maintenance": 18500.00,
                "parts": 7200.00,
                "labor": 11300.00,
                "downtime": 5500.00
            }
        ],
        "cost_by_asset_type": [
            {
                "asset_type": "equipment",
                "total_cost": 345600.00,
                "cost_per_asset": 3885.39,
                "percentage": 64.4
            }
        ],
        "high_cost_assets": [
            {
                "asset_id": "01933b2e-eeee-7890-abcd-ef1234567890",
                "asset_name": "Stamping Press #2",
                "total_cost": 15600.00,
                "cost_breakdown": {
                    "maintenance": 8900.00,
                    "parts": 4200.00,
                    "downtime": 2500.00
                }
            }
        ],
        "optimization_opportunities": [
            {
                "opportunity": "Predictive maintenance implementation",
                "potential_savings": 67500.00,
                "investment_required": 25000.00,
                "roi": 2.7,
                "payback_months": 4
            }
        ]
    }
}
```

## Real-time Features

### WebSocket Connection

```javascript
// Connect to EAM WebSocket
const ws = new WebSocket('wss://eam.apg.datacraft.co.ke/ws');

// Authentication
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your_jwt_token',
        tenant_id: 'your_tenant_id'
    }));
};

// Subscribe to asset updates
ws.send(JSON.stringify({
    type: 'subscribe',
    channels: [
        'asset.health.01933b2e-1234-7890-abcd-ef1234567890',
        'workorder.status.tech_001',
        'inventory.reorder.your_tenant_id'
    ]
}));

// Handle messages
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'asset.health.update':
            console.log('Asset health updated:', data.payload);
            break;
        case 'workorder.assigned':
            console.log('New work order assigned:', data.payload);
            break;
        case 'inventory.reorder.alert':
            console.log('Reorder alert:', data.payload);
            break;
    }
};
```

### Real-time Events

#### Asset Health Updates

```json
{
    "type": "asset.health.update",
    "timestamp": "2024-01-01T18:00:00Z",
    "payload": {
        "asset_id": "01933b2e-1234-7890-abcd-ef1234567890",
        "health_score": 87.5,
        "previous_score": 89.2,
        "trend": "declining",
        "alerts": [
            {
                "type": "vibration_high",
                "severity": "warning",
                "message": "Vibration levels above normal threshold"
            }
        ]
    }
}
```

#### Work Order Notifications

```json
{
    "type": "workorder.assigned",
    "timestamp": "2024-01-01T18:05:00Z",
    "payload": {
        "work_order_id": "01933b2e-5555-7890-abcd-ef1234567890",
        "work_order_number": "WO-001235",
        "title": "Emergency Repair - Conveyor Belt",
        "priority": "high",
        "assigned_to": "tech_002",
        "due_date": "2024-01-02T20:00:00Z"
    }
}
```

## Error Handling

### Standard Error Response

```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input data provided",
        "details": [
            {
                "field": "asset_name",
                "message": "Asset name is required"
            },
            {
                "field": "purchase_cost",
                "message": "Purchase cost must be a positive number"
            }
        ],
        "timestamp": "2024-01-01T18:10:00Z",
        "request_id": "req_01933b2e-ffff-7890-abcd-ef1234567890"
    }
}
```

### HTTP Status Codes

| Code | Description | Use Case |
|------|-------------|----------|
| 200 | OK | Successful GET, PUT, PATCH |
| 201 | Created | Successful POST |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Invalid input data |
| 401 | Unauthorized | Missing or invalid token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Duplicate resource |
| 422 | Unprocessable Entity | Business logic error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Maintenance mode |

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Input validation failed |
| `PERMISSION_DENIED` | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `DUPLICATE_RESOURCE` | Resource already exists |
| `BUSINESS_RULE_VIOLATION` | Business logic constraint violated |
| `EXTERNAL_SERVICE_ERROR` | APG service dependency error |
| `RATE_LIMIT_EXCEEDED` | API rate limit reached |

## SDK Examples

### Python SDK

```python
from apg_eam_client import EAMClient

# Initialize client
client = EAMClient(
    base_url="https://eam.apg.datacraft.co.ke/api/v1",
    token="your_jwt_token",
    tenant_id="your_tenant_id"
)

# Create asset
asset_data = {
    "asset_name": "New CNC Machine",
    "asset_type": "equipment",
    "asset_category": "production"
}
asset = await client.assets.create(asset_data)

# List work orders
work_orders = await client.work_orders.list(
    status="scheduled",
    priority="high",
    limit=50
)

# Update asset health
await client.assets.update_health(
    asset_id=asset.asset_id,
    health_score=92.5,
    condition_status="good"
)
```

### JavaScript SDK

```javascript
import { EAMClient } from '@apg/eam-client';

// Initialize client
const client = new EAMClient({
    baseURL: 'https://eam.apg.datacraft.co.ke/api/v1',
    token: 'your_jwt_token',
    tenantId: 'your_tenant_id'
});

// Create work order
const workOrder = await client.workOrders.create({
    title: 'Maintenance Task',
    description: 'Routine maintenance',
    assetId: 'asset_123',
    workType: 'maintenance'
});

// Get analytics
const analytics = await client.analytics.getAssetPerformance({
    timePeriod: 'last_30_days',
    assetType: 'equipment'
});
```

## Rate Limiting

### Limits

- **Standard Users**: 1,000 requests per hour
- **Premium Users**: 5,000 requests per hour
- **Enterprise Users**: 10,000 requests per hour

### Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641024000
```

### Rate Limit Exceeded Response

```json
{
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "API rate limit exceeded",
        "details": {
            "limit": 1000,
            "reset_time": "2024-01-01T19:00:00Z"
        }
    }
}
```

---

*API Reference version 1.0.0 - Last updated: 2024-01-01*