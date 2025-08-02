# APG Workflow Orchestration - API Reference

**Complete API reference for developers and integrators**

© 2025 Datacraft. All rights reserved.

## Table of Contents

1. [Authentication](#authentication)
2. [REST API Endpoints](#rest-api-endpoints)
3. [GraphQL API](#graphql-api)
4. [WebSocket API](#websocket-api)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Webhooks](#webhooks)
9. [Bulk Operations](#bulk-operations)
10. [SDK Examples](#sdk-examples)

## Authentication

### Authentication Methods

The API supports multiple authentication methods:

**1. JWT Token Authentication (Recommended)**
```http
Authorization: Bearer <jwt_token>
```

**2. API Key Authentication**
```http
X-API-Key: <api_key>
```

**3. OAuth 2.0**
```http
Authorization: Bearer <oauth_token>
```

### Getting Authentication Tokens

**JWT Token via Login:**
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "expires_in": 3600,
    "token_type": "Bearer"
  }
}
```

**API Key Generation:**
```http
POST /api/v1/auth/api-keys
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "name": "My Integration",
  "description": "API key for workflow automation",
  "expires_at": "2025-12-31T23:59:59Z",
  "permissions": ["workflow.read", "workflow.execute"]
}
```

## REST API Endpoints

### Base URL
```
https://your-domain.com/api/v1/workflow_orchestration
```

### Response Format

All API responses follow this structure:
```json
{
  "success": boolean,
  "data": object | array | null,
  "message": string,
  "errors": array,
  "metadata": {
    "timestamp": "2025-01-01T00:00:00Z",
    "request_id": "uuid",
    "version": "1.0.0"
  }
}
```

### Workflows

#### List Workflows
```http
GET /workflows
```

**Query Parameters:**
- `limit` (integer, optional): Maximum number of results (default: 50, max: 1000)
- `offset` (integer, optional): Number of results to skip (default: 0)
- `category` (string, optional): Filter by category
- `status` (string, optional): Filter by status (`draft`, `active`, `inactive`, `archived`)
- `search` (string, optional): Search in name and description
- `tags` (array, optional): Filter by tags
- `created_by` (string, optional): Filter by creator user ID
- `sort` (string, optional): Sort field (`name`, `created_at`, `updated_at`)
- `order` (string, optional): Sort order (`asc`, `desc`)

**Example Request:**
```http
GET /workflows?limit=20&category=data_processing&status=active&sort=updated_at&order=desc
Authorization: Bearer <token>
```

**Example Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W",
      "name": "Data Processing Pipeline",
      "description": "ETL pipeline for customer data",
      "status": "active",
      "category": "data_processing",
      "tags": ["etl", "customer_data"],
      "version": 2,
      "component_count": 5,
      "execution_count": 142,
      "success_rate": 0.95,
      "last_execution_at": "2025-01-01T12:30:00Z",
      "created_by": "01HQNZ2G5XKJ8P7M9N6V3R4T5X",
      "created_at": "2024-12-01T10:00:00Z",
      "updated_at": "2025-01-01T09:15:00Z"
    }
  ],
  "metadata": {
    "total_count": 1,
    "limit": 20,
    "offset": 0,
    "has_more": false
  }
}
```

#### Get Workflow
```http
GET /workflows/{workflow_id}
```

**Path Parameters:**
- `workflow_id` (string, required): Unique workflow identifier

**Query Parameters:**
- `include_definition` (boolean, optional): Include full workflow definition (default: true)
- `include_stats` (boolean, optional): Include execution statistics (default: false)

**Example Response:**
```json
{
  "success": true,
  "data": {
    "id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W",
    "name": "Data Processing Pipeline",
    "description": "ETL pipeline for customer data",
    "definition": {
      "components": [
        {
          "id": "start",
          "type": "start",
          "config": {
            "trigger_type": "manual"
          },
          "position": {"x": 100, "y": 100}
        },
        {
          "id": "extract",
          "type": "http_request",
          "config": {
            "url": "https://api.example.com/customers",
            "method": "GET",
            "headers": {
              "Authorization": "Bearer {{api_token}}"
            }
          },
          "position": {"x": 300, "y": 100}
        }
      ],
      "connections": [
        {
          "source": "start",
          "target": "extract",
          "type": "success"
        }
      ],
      "parameters": [
        {
          "name": "api_token",
          "type": "string",
          "required": true,
          "description": "API token for data source"
        }
      ]
    },
    "status": "active",
    "category": "data_processing",
    "tags": ["etl", "customer_data"],
    "permissions": {
      "can_edit": true,
      "can_execute": true,
      "can_delete": true,
      "can_share": true
    },
    "created_by": "01HQNZ2G5XKJ8P7M9N6V3R4T5X",
    "created_at": "2024-12-01T10:00:00Z",
    "updated_at": "2025-01-01T09:15:00Z"
  }
}
```

#### Create Workflow
```http
POST /workflows
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "New Workflow",
  "description": "Description of the workflow",
  "definition": {
    "components": [
      {
        "id": "start",
        "type": "start",
        "config": {"trigger_type": "manual"},
        "position": {"x": 100, "y": 100}
      }
    ],
    "connections": [],
    "parameters": []
  },
  "category": "automation",
  "tags": ["example"],
  "priority": 5
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W",
    "name": "New Workflow",
    "status": "draft",
    "created_at": "2025-01-01T12:00:00Z"
  },
  "message": "Workflow created successfully"
}
```

#### Update Workflow
```http
PUT /workflows/{workflow_id}
Content-Type: application/json
```

**Request Body:** Same as Create Workflow

#### Delete Workflow
```http
DELETE /workflows/{workflow_id}
```

**Query Parameters:**
- `force` (boolean, optional): Force delete even if there are running executions

**Response:**
```json
{
  "success": true,
  "message": "Workflow deleted successfully"
}
```

#### Execute Workflow
```http
POST /workflows/{workflow_id}/execute
Content-Type: application/json
```

**Request Body:**
```json
{
  "parameters": {
    "api_token": "your-api-token",
    "batch_size": 1000
  },
  "priority": 7,
  "scheduled_at": "2025-01-01T15:00:00Z",
  "tags": ["manual_execution"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "execution_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5Y",
    "status": "queued",
    "workflow_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W",
    "priority": 7,
    "estimated_duration": 300,
    "queue_position": 2
  },
  "message": "Workflow execution started"
}
```

#### Validate Workflow
```http
POST /workflows/{workflow_id}/validate
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_valid": true,
    "errors": [],
    "warnings": [
      {
        "component_id": "transform",
        "message": "Component has no error handling configured",
        "severity": "warning"
      }
    ],
    "info": [
      {
        "message": "Workflow contains 5 components and 4 connections",
        "severity": "info"
      }
    ]
  }
}
```

### Executions

#### List Executions
```http
GET /executions
```

**Query Parameters:**
- `workflow_id` (string, optional): Filter by workflow ID
- `status` (string, optional): Filter by status
- `user_id` (string, optional): Filter by user who started execution
- `start_date` (string, optional): Filter executions started after this date (ISO format)
- `end_date` (string, optional): Filter executions started before this date (ISO format)
- `limit` (integer, optional): Maximum results (default: 50)
- `offset` (integer, optional): Results to skip (default: 0)

#### Get Execution
```http
GET /executions/{execution_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "01HQNZ2G5XKJ8P7M9N6V3R4T5Y",
    "workflow_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W",
    "workflow_name": "Data Processing Pipeline",
    "status": "running",
    "progress": {
      "current_component": "transform",
      "completed_components": 2,
      "total_components": 5,
      "percentage": 40
    },
    "parameters": {
      "api_token": "[REDACTED]",
      "batch_size": 1000
    },
    "results": null,
    "error_message": null,
    "started_at": "2025-01-01T12:00:00Z",
    "completed_at": null,
    "duration_seconds": 120,
    "resource_usage": {
      "cpu_seconds": 15.5,
      "memory_mb": 256,
      "network_bytes": 1048576
    },
    "created_by": "01HQNZ2G5XKJ8P7M9N6V3R4T5X"
  }
}
```

#### Cancel Execution
```http
POST /executions/{execution_id}/cancel
```

#### Retry Execution
```http
POST /executions/{execution_id}/retry
Content-Type: application/json
```

**Request Body:**
```json
{
  "from_component": "transform",
  "parameters": {
    "batch_size": 500
  }
}
```

#### Get Execution Logs
```http
GET /executions/{execution_id}/logs
```

**Query Parameters:**
- `level` (string, optional): Filter by log level (`debug`, `info`, `warning`, `error`)
- `component_id` (string, optional): Filter by component
- `limit` (integer, optional): Maximum log entries
- `offset` (integer, optional): Log entries to skip

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2025-01-01T12:00:15Z",
      "level": "info",
      "component_id": "extract",
      "message": "Starting data extraction from API",
      "metadata": {
        "url": "https://api.example.com/customers",
        "method": "GET"
      }
    },
    {
      "timestamp": "2025-01-01T12:00:18Z",
      "level": "info",
      "component_id": "extract",
      "message": "Successfully extracted 1500 records",
      "metadata": {
        "record_count": 1500,
        "duration_ms": 2800
      }
    }
  ]
}
```

### Templates

#### List Templates
```http
GET /templates
```

**Query Parameters:**
- `category` (string, optional): Filter by category
- `complexity` (string, optional): Filter by complexity level
- `search` (string, optional): Search in name and description
- `sort` (string, optional): Sort by (`name`, `rating`, `usage_count`, `created_at`)

#### Get Template
```http
GET /templates/{template_id}
```

#### Create Workflow from Template
```http
POST /templates/{template_id}/create-workflow
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "My Data Pipeline",
  "description": "Created from template",
  "parameters": {
    "source_url": "https://my-api.com/data",
    "output_format": "json",
    "batch_size": 1000
  }
}
```

### Components

#### List Available Components
```http
GET /components
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "type": "http_request",
      "name": "HTTP Request",
      "description": "Make HTTP requests to APIs and web services",
      "category": "integration",
      "config_schema": {
        "type": "object",
        "properties": {
          "url": {"type": "string", "format": "uri"},
          "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
          "headers": {"type": "object"},
          "timeout": {"type": "integer", "minimum": 1, "maximum": 3600}
        },
        "required": ["url", "method"]
      },
      "input_schema": {"type": "object"},
      "output_schema": {"type": "object"}
    }
  ]
}
```

#### Get Component Schema
```http
GET /components/{component_type}/schema
```

### System

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 86400,
    "components": {
      "database": {"status": "healthy", "response_time_ms": 5},
      "redis": {"status": "healthy", "response_time_ms": 2},
      "workflow_engine": {"status": "healthy"},
      "scheduler": {"status": "healthy", "queue_depth": 12}
    }
  }
}
```

#### System Statistics
```http
GET /stats
```

**Response:**
```json
{
  "success": true,
  "data": {
    "workflows": {
      "total": 1250,
      "active": 180,
      "draft": 45
    },
    "executions": {
      "today": 340,
      "this_week": 2100,
      "this_month": 8500,
      "success_rate": 0.94
    },
    "system": {
      "cpu_usage": 0.35,
      "memory_usage": 0.62,
      "disk_usage": 0.28
    }
  }
}
```

## GraphQL API

### Endpoint
```
https://your-domain.com/api/beta/graphql
```

### Authentication
Include JWT token in Authorization header:
```http
Authorization: Bearer <jwt_token>
```

### Schema

#### Types

```graphql
type Workflow {
  id: ID!
  name: String!
  description: String
  definition: JSON!
  status: WorkflowStatus!
  category: String
  tags: [String!]!
  version: Int!
  componentCount: Int!
  executionCount: Int!
  successRate: Float!
  lastExecutionAt: DateTime
  createdBy: ID!
  createdAt: DateTime!
  updatedAt: DateTime!
  
  # Relations
  executions(limit: Int, offset: Int): [WorkflowExecution!]!
  recentExecutions(limit: Int = 10): [WorkflowExecution!]!
}

type WorkflowExecution {
  id: ID!
  workflowId: ID!
  workflow: Workflow!
  status: ExecutionStatus!
  priority: Int!
  progress: ExecutionProgress!
  parameters: JSON
  results: JSON
  errorMessage: String
  startedAt: DateTime
  completedAt: DateTime
  duration: Int
  resourceUsage: ResourceUsage!
  createdBy: ID!
  createdAt: DateTime!
  
  # Relations
  logs(level: LogLevel, componentId: String): [ExecutionLog!]!
  componentExecutions: [ComponentExecution!]!
}

type ExecutionProgress {
  currentComponent: String
  completedComponents: Int!
  totalComponents: Int!
  percentage: Float!
}

type ResourceUsage {
  cpuSeconds: Float!
  memoryMb: Int!
  networkBytes: Int!
}

type ExecutionLog {
  timestamp: DateTime!
  level: LogLevel!
  componentId: String
  message: String!
  metadata: JSON
}

type ComponentExecution {
  id: ID!
  executionId: ID!
  componentId: String!
  componentType: String!
  status: ExecutionStatus!
  inputData: JSON
  outputData: JSON
  errorMessage: String
  duration: Int
  startedAt: DateTime
  completedAt: DateTime
}

enum WorkflowStatus {
  DRAFT
  ACTIVE
  INACTIVE
  ARCHIVED
}

enum ExecutionStatus {
  PENDING
  QUEUED
  RUNNING
  COMPLETED
  FAILED
  CANCELLED
}

enum LogLevel {
  DEBUG
  INFO
  WARNING
  ERROR
}
```

#### Queries

```graphql
type Query {
  # Workflows
  workflows(
    limit: Int = 50
    offset: Int = 0
    category: String
    status: WorkflowStatus
    search: String
    tags: [String!]
    createdBy: ID
  ): [Workflow!]!
  
  workflow(id: ID!): Workflow
  
  # Executions
  executions(
    limit: Int = 50
    offset: Int = 0
    workflowId: ID
    status: ExecutionStatus
    startDate: DateTime
    endDate: DateTime
  ): [WorkflowExecution!]!
  
  execution(id: ID!): WorkflowExecution
  
  # Templates
  templates(
    category: String
    complexity: String
    search: String
  ): [WorkflowTemplate!]!
  
  template(id: ID!): WorkflowTemplate
  
  # System
  systemStats: SystemStats!
  systemHealth: SystemHealth!
}
```

#### Mutations

```graphql
type Mutation {
  # Workflow mutations
  createWorkflow(input: CreateWorkflowInput!): CreateWorkflowResult!
  updateWorkflow(id: ID!, input: UpdateWorkflowInput!): UpdateWorkflowResult!
  deleteWorkflow(id: ID!, force: Boolean = false): DeleteWorkflowResult!
  executeWorkflow(workflowId: ID!, input: ExecuteWorkflowInput!): ExecuteWorkflowResult!
  
  # Execution mutations
  cancelExecution(id: ID!): CancelExecutionResult!
  retryExecution(id: ID!, input: RetryExecutionInput!): RetryExecutionResult!
  
  # Template mutations
  createWorkflowFromTemplate(templateId: ID!, input: CreateFromTemplateInput!): CreateWorkflowResult!
}

input CreateWorkflowInput {
  name: String!
  description: String
  definition: JSON!
  category: String
  tags: [String!] = []
  priority: Int = 5
}

input ExecuteWorkflowInput {
  parameters: JSON = {}
  priority: Int = 5
  scheduledAt: DateTime
  tags: [String!] = []
}

type CreateWorkflowResult {
  success: Boolean!
  workflow: Workflow
  errors: [String!]!
}

type ExecuteWorkflowResult {
  success: Boolean!
  execution: WorkflowExecution
  errors: [String!]!
}
```

#### Subscriptions (Real-time Updates)

```graphql
type Subscription {
  # Workflow execution updates
  executionUpdates(executionId: ID!): WorkflowExecution!
  
  # Workflow updates  
  workflowUpdates(workflowId: ID!): Workflow!
  
  # System notifications
  systemNotifications: SystemNotification!
}

type SystemNotification {
  type: NotificationType!
  title: String!
  message: String!
  data: JSON
  timestamp: DateTime!
}

enum NotificationType {
  EXECUTION_COMPLETED
  EXECUTION_FAILED
  SYSTEM_ALERT
  MAINTENANCE_SCHEDULED
}
```

### Example Queries

**Get Workflows with Executions:**
```graphql
query GetWorkflows {
  workflows(limit: 10, status: ACTIVE) {
    id
    name
    description
    status
    executionCount
    successRate
    recentExecutions(limit: 3) {
      id
      status
      startedAt
      duration
    }
  }
}
```

**Execute Workflow:**
```graphql
mutation ExecuteWorkflow {
  executeWorkflow(
    workflowId: "01HQNZ2G5XKJ8P7M9N6V3R4T5W"
    input: {
      parameters: {
        api_token: "your-token"
        batch_size: 1000
      }
      priority: 7
    }
  ) {
    success
    execution {
      id
      status
      progress {
        percentage
      }
    }
    errors
  }
}
```

**Subscribe to Execution Updates:**
```graphql
subscription ExecutionUpdates {
  executionUpdates(executionId: "01HQNZ2G5XKJ8P7M9N6V3R4T5Y") {
    id
    status
    progress {
      currentComponent
      percentage
    }
    duration
  }
}
```

## WebSocket API

### Connection

Connect to WebSocket endpoint:
```
wss://your-domain.com/api/v1/ws
```

Include authentication in query parameters:
```
wss://your-domain.com/api/v1/ws?token=<jwt_token>
```

### Message Format

All WebSocket messages use this format:
```json
{
  "type": "message_type",
  "id": "unique_message_id",
  "timestamp": "2025-01-01T12:00:00Z",
  "data": {}
}
```

### Client Messages

**Subscribe to Workflow Updates:**
```json
{
  "type": "subscribe",
  "id": "msg_001",
  "data": {
    "channel": "workflow",
    "workflow_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W"
  }
}
```

**Subscribe to Execution Updates:**
```json
{
  "type": "subscribe",
  "id": "msg_002", 
  "data": {
    "channel": "execution",
    "execution_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5Y"
  }
}
```

**Unsubscribe:**
```json
{
  "type": "unsubscribe",
  "id": "msg_003",
  "data": {
    "channel": "execution",
    "execution_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5Y"
  }
}
```

### Server Messages

**Execution Status Update:**
```json
{
  "type": "execution_update",
  "id": "srv_001",
  "timestamp": "2025-01-01T12:05:00Z",
  "data": {
    "execution_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5Y",
    "status": "running",
    "progress": {
      "current_component": "transform",
      "percentage": 60
    },
    "duration": 300
  }
}
```

**Component Completion:**
```json
{
  "type": "component_completed",
  "id": "srv_002",
  "timestamp": "2025-01-01T12:05:30Z",
  "data": {
    "execution_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5Y",
    "component_id": "transform",
    "status": "completed",
    "duration": 180,
    "output_data": {"processed_records": 1500}
  }
}
```

**Error Notification:**
```json
{
  "type": "error",
  "id": "srv_003", 
  "timestamp": "2025-01-01T12:06:00Z",
  "data": {
    "execution_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5Y",
    "component_id": "load",
    "error": "Connection timeout to database",
    "retry_count": 2
  }
}
```

## Data Models

### Workflow Model

```json
{
  "id": "string (UUID)",
  "name": "string (1-255 chars)",
  "description": "string (optional, max 1000 chars)",
  "definition": {
    "components": [
      {
        "id": "string",
        "type": "string",
        "config": "object",
        "position": {"x": "number", "y": "number"}
      }
    ],
    "connections": [
      {
        "source": "string",
        "target": "string", 
        "type": "string (success|error|conditional)"
      }
    ],
    "parameters": [
      {
        "name": "string",
        "type": "string",
        "required": "boolean",
        "default_value": "any"
      }
    ]
  },
  "status": "string (draft|active|inactive|archived)",
  "category": "string (optional)",
  "tags": ["string"],
  "version": "integer",
  "priority": "integer (1-10)",
  "created_by": "string (UUID)",
  "created_at": "string (ISO datetime)",
  "updated_at": "string (ISO datetime)"
}
```

### Execution Model

```json
{
  "id": "string (UUID)",
  "workflow_id": "string (UUID)",
  "status": "string (pending|queued|running|completed|failed|cancelled)",
  "priority": "integer (1-10)",
  "parameters": "object",
  "results": "object (nullable)",
  "error_message": "string (nullable)",
  "progress": {
    "current_component": "string",
    "completed_components": "integer",
    "total_components": "integer", 
    "percentage": "number (0-100)"
  },
  "resource_usage": {
    "cpu_seconds": "number",
    "memory_mb": "integer",
    "network_bytes": "integer"
  },
  "started_at": "string (ISO datetime, nullable)",
  "completed_at": "string (ISO datetime, nullable)",
  "duration_seconds": "integer (nullable)",
  "created_by": "string (UUID)",
  "created_at": "string (ISO datetime)"
}
```

### Template Model

```json
{
  "id": "string (UUID)",
  "name": "string",
  "description": "string",
  "category": "string",
  "tags": ["string"],
  "workflow_definition": "object",
  "parameters": [
    {
      "name": "string",
      "display_name": "string",
      "description": "string",
      "parameter_type": "string",
      "required": "boolean",
      "default_value": "any",
      "validation_rules": "object",
      "input_type": "string",
      "options": ["string"],
      "placeholder": "string",
      "help_text": "string"
    }
  ],
  "author": "string",
  "version": "string",
  "complexity_level": "string (beginner|intermediate|advanced|expert)",
  "estimated_setup_time_minutes": "integer",
  "usage_count": "integer",
  "rating": "number (0-5)",
  "created_at": "string (ISO datetime)",
  "updated_at": "string (ISO datetime)"
}
```

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "data": null,
  "message": "Error description",
  "errors": [
    {
      "code": "VALIDATION_ERROR",
      "message": "Invalid workflow definition",
      "field": "definition.components[0].config.url",
      "details": {
        "expected": "valid URL",
        "received": "invalid-url"
      }
    }
  ],
  "metadata": {
    "timestamp": "2025-01-01T12:00:00Z",
    "request_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5Z",
    "version": "1.0.0"
  }
}
```

### HTTP Status Codes

- `200 OK` - Successful request
- `201 Created` - Resource created successfully
- `204 No Content` - Successful request with no response body
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict (e.g., duplicate name)
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request data validation failed |
| `AUTHENTICATION_ERROR` | Authentication failed |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `RESOURCE_CONFLICT` | Resource already exists |
| `WORKFLOW_INVALID` | Workflow definition is invalid |
| `EXECUTION_FAILED` | Workflow execution failed |
| `COMPONENT_ERROR` | Component execution error |
| `TIMEOUT_ERROR` | Operation timeout |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `SYSTEM_ERROR` | Internal system error |

## Rate Limiting

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999  
X-RateLimit-Reset: 1640995200
X-RateLimit-RetryAfter: 60
```

### Rate Limit Tiers

| Tier | Requests/Hour | Concurrent Executions |
|------|---------------|----------------------|
| Free | 1,000 | 5 |
| Pro | 10,000 | 25 |
| Enterprise | 100,000 | 100 |
| Custom | Negotiable | Negotiable |

### Rate Limit Exceeded Response

```json
{
  "success": false,
  "message": "Rate limit exceeded",
  "errors": [
    {
      "code": "RATE_LIMIT_EXCEEDED",
      "message": "Too many requests. Try again in 60 seconds.",
      "retry_after": 60
    }
  ]
}
```

## Webhooks

### Webhook Configuration

**Create Webhook:**
```http
POST /webhooks
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/workflow",
  "events": [
    "workflow.created",
    "workflow.updated",
    "execution.completed",
    "execution.failed"
  ],
  "secret": "your-webhook-secret",
  "active": true,
  "retry_policy": {
    "max_attempts": 3,
    "initial_delay_seconds": 60,
    "max_delay_seconds": 3600,
    "backoff_multiplier": 2
  }
}
```

### Webhook Events

| Event | Description |
|-------|-------------|
| `workflow.created` | New workflow created |
| `workflow.updated` | Workflow modified |
| `workflow.deleted` | Workflow deleted |
| `workflow.activated` | Workflow activated |
| `workflow.deactivated` | Workflow deactivated |
| `execution.started` | Workflow execution started |
| `execution.completed` | Workflow execution completed successfully |
| `execution.failed` | Workflow execution failed |
| `execution.cancelled` | Workflow execution cancelled |
| `component.completed` | Component execution completed |
| `component.failed` | Component execution failed |

### Webhook Payload

```json
{
  "event": "execution.completed",
  "timestamp": "2025-01-01T12:00:00Z",
  "webhook_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5A",
  "data": {
    "execution": {
      "id": "01HQNZ2G5XKJ8P7M9N6V3R4T5Y",
      "workflow_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W",
      "status": "completed",
      "duration_seconds": 300,
      "results": {
        "processed_records": 1500,
        "output_file": "/tmp/processed_data.json"
      }
    },
    "workflow": {
      "id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W",
      "name": "Data Processing Pipeline"
    }
  }
}
```

### Webhook Security

**Signature Verification:**
```python
import hmac
import hashlib

def verify_webhook_signature(payload, signature, secret):
    expected_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)
```

## Bulk Operations

### Bulk Workflow Operations

**Bulk Create:**
```http
POST /v2/workflows/bulk
Content-Type: application/json

{
  "operations": [
    {
      "operation": "create",
      "data": {
        "name": "Workflow 1",
        "definition": {...}
      }
    },
    {
      "operation": "create", 
      "data": {
        "name": "Workflow 2",
        "definition": {...}
      }
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "success": true,
        "data": {"id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W", "name": "Workflow 1"}
      },
      {
        "success": false,
        "errors": [{"code": "VALIDATION_ERROR", "message": "Invalid definition"}]
      }
    ],
    "summary": {
      "total": 2,
      "successful": 1,
      "failed": 1
    }
  }
}
```

### Batch Execution

**Execute Multiple Workflows:**
```http
POST /v2/workflows/execute-batch
Content-Type: application/json

{
  "executions": [
    {
      "workflow_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5W",
      "parameters": {"param1": "value1"},
      "priority": 5
    },
    {
      "workflow_id": "01HQNZ2G5XKJ8P7M9N6V3R4T5X", 
      "parameters": {"param2": "value2"},
      "priority": 7
    }
  ]
}
```

## SDK Examples

### Python SDK

**Installation:**
```bash
pip install apg-workflow-client
```

**Usage:**
```python
from apg_workflow import WorkflowClient

# Initialize client
client = WorkflowClient(
    base_url="https://your-domain.com/api/v1",
    api_key="your-api-key"
)

# Create workflow
workflow = await client.workflows.create({
    "name": "My Workflow",
    "description": "Test workflow",
    "definition": {
        "components": [
            {
                "id": "start",
                "type": "start",
                "config": {"trigger_type": "manual"}
            }
        ],
        "connections": []
    }
})

# Execute workflow
execution = await client.workflows.execute(
    workflow.id,
    parameters={"param1": "value1"}
)

# Wait for completion
result = await client.executions.wait_for_completion(
    execution.id,
    timeout=300
)

print(f"Execution completed with status: {result.status}")
```

### JavaScript SDK

**Installation:**
```bash
npm install @apg/workflow-client
```

**Usage:**
```javascript
import { WorkflowClient } from '@apg/workflow-client';

// Initialize client
const client = new WorkflowClient({
  baseUrl: 'https://your-domain.com/api/v1',
  apiKey: 'your-api-key'
});

// Create workflow
const workflow = await client.workflows.create({
  name: 'My Workflow',
  description: 'Test workflow',
  definition: {
    components: [
      {
        id: 'start',
        type: 'start',
        config: { trigger_type: 'manual' }
      }
    ],
    connections: []
  }
});

// Execute workflow
const execution = await client.workflows.execute(workflow.id, {
  parameters: { param1: 'value1' }
});

// Subscribe to execution updates
client.executions.subscribe(execution.id, (update) => {
  console.log('Execution update:', update);
});
```

### cURL Examples

**Create Workflow:**
```bash
curl -X POST "https://your-domain.com/api/v1/workflows" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Workflow",
    "description": "Simple test workflow",
    "definition": {
      "components": [
        {
          "id": "start",
          "type": "start", 
          "config": {"trigger_type": "manual"}
        }
      ],
      "connections": []
    }
  }'
```

**Execute Workflow:**
```bash
curl -X POST "https://your-domain.com/api/v1/workflows/{workflow_id}/execute" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "param1": "value1"
    },
    "priority": 5
  }'
```

**Get Execution Status:**
```bash
curl -X GET "https://your-domain.com/api/v1/executions/{execution_id}" \
  -H "Authorization: Bearer <token>"
```

---

This API reference provides comprehensive documentation for integrating with the APG Workflow Orchestration platform. For additional support or questions, please contact our development team.

**© 2025 Datacraft. All rights reserved.**