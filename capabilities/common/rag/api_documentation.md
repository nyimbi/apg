# APG RAG API Documentation

> **Complete REST API Reference for Enterprise RAG Platform**

## Table of Contents

- [Authentication](#authentication)
- [API Overview](#api-overview)
- [Knowledge Base API](#knowledge-base-api)
- [Document API](#document-api)
- [Query & Retrieval API](#query--retrieval-api)
- [Generation API](#generation-api)
- [Conversation API](#conversation-api)
- [Chat API](#chat-api)
- [Health & Monitoring API](#health--monitoring-api)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Webhooks](#webhooks)

## Authentication

The APG RAG API uses Bearer token authentication. Include your token in the Authorization header:

```http
Authorization: Bearer YOUR_API_TOKEN
```

### Obtaining API Tokens

API tokens are issued by your system administrator and can be managed through the admin interface or by contacting support.

### Token Scopes

Tokens can have different scopes:
- `rag:read` - Read access to knowledge bases and documents
- `rag:write` - Create and modify knowledge bases and documents
- `rag:query` - Query knowledge bases and generate responses
- `rag:chat` - Access conversation features
- `rag:admin` - Administrative access

## API Overview

### Base URL
```
https://your-apg-server.com/api/v1/rag
```

### Content Types
- **Request**: `application/json` (except file uploads)
- **Response**: `application/json`
- **File Upload**: `multipart/form-data`

### Common Headers
```http
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json
Accept: application/json
X-Tenant-ID: your-tenant-id  # Multi-tenant deployments
```

### Response Format
All API responses follow this standard format:

```json
{
  "success": true|false,
  "data": { ... },           // Response data (on success)
  "error": "Error message",  // Error description (on failure)
  "error_code": "ERROR_CODE", // Machine-readable error code
  "timestamp": "2025-01-29T10:30:00Z",
  "request_id": "req_abc123" // Unique request identifier
}
```

## Knowledge Base API

### List Knowledge Bases

Get a list of all knowledge bases accessible to the current user.

```http
GET /knowledge-bases
```

#### Query Parameters
- `user_id` (optional): Filter by user ID
- `limit` (optional): Number of results to return (default: 50, max: 100)
- `offset` (optional): Number of results to skip (default: 0)

#### Response
```json
{
  "success": true,
  "data": [
    {
      "id": "kb_abc123",
      "tenant_id": "tenant_123",
      "name": "Company Policies",
      "description": "All company policies and procedures",
      "embedding_model": "bge-m3",
      "generation_model": "qwen3",
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "similarity_threshold": 0.7,
      "max_retrievals": 10,
      "status": "active",
      "document_count": 25,
      "total_chunks": 1250,
      "user_id": "user_456",
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-29T08:45:00Z",
      "created_by": "user_456",
      "updated_by": "user_456"
    }
  ],
  "count": 1
}
```

### Create Knowledge Base

Create a new knowledge base.

```http
POST /knowledge-bases
```

#### Request Body
```json
{
  "name": "Company Policies",
  "description": "All company policies and procedures",
  "embedding_model": "bge-m3",        // Optional, defaults to "bge-m3"
  "generation_model": "qwen3",        // Optional, defaults to "qwen3"
  "chunk_size": 1000,                 // Optional, defaults to 1000
  "chunk_overlap": 200,               // Optional, defaults to 200
  "similarity_threshold": 0.7,        // Optional, defaults to 0.7
  "max_retrievals": 10                // Optional, defaults to 10
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "kb_abc123",
    "tenant_id": "tenant_123",
    "name": "Company Policies",
    "description": "All company policies and procedures",
    "embedding_model": "bge-m3",
    "generation_model": "qwen3",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "similarity_threshold": 0.7,
    "max_retrievals": 10,
    "status": "active",
    "document_count": 0,
    "total_chunks": 0,
    "user_id": "user_456",
    "created_at": "2025-01-29T10:30:00Z",
    "updated_at": "2025-01-29T10:30:00Z",
    "created_by": "user_456",
    "updated_by": "user_456"
  },
  "message": "Knowledge base created successfully"
}
```

### Get Knowledge Base

Retrieve details of a specific knowledge base.

```http
GET /knowledge-bases/{kb_id}
```

#### Path Parameters
- `kb_id` (required): Knowledge base ID

#### Response
```json
{
  "success": true,
  "data": {
    "id": "kb_abc123",
    "tenant_id": "tenant_123",
    "name": "Company Policies",
    "description": "All company policies and procedures",
    "embedding_model": "bge-m3",
    "generation_model": "qwen3",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "similarity_threshold": 0.7,
    "max_retrievals": 10,
    "status": "active",
    "document_count": 25,
    "total_chunks": 1250,
    "user_id": "user_456",
    "created_at": "2025-01-15T10:30:00Z",
    "updated_at": "2025-01-29T08:45:00Z",
    "created_by": "user_456",
    "updated_by": "user_456"
  }
}
```

### Update Knowledge Base

Update an existing knowledge base.

```http
PUT /knowledge-bases/{kb_id}
```

#### Path Parameters
- `kb_id` (required): Knowledge base ID

#### Request Body
```json
{
  "name": "Updated Company Policies",
  "description": "Updated description",
  "similarity_threshold": 0.8,
  "max_retrievals": 15
}
```

#### Response
```json
{
  "success": true,
  "message": "Knowledge base updated successfully"
}
```

### Delete Knowledge Base

Delete a knowledge base and all its associated documents.

```http
DELETE /knowledge-bases/{kb_id}
```

#### Path Parameters
- `kb_id` (required): Knowledge base ID

#### Response
```json
{
  "success": true,
  "message": "Knowledge base deleted successfully"
}
```

## Document API

### Upload Document

Upload a document to a knowledge base for processing.

```http
POST /documents/{kb_id}
```

#### Path Parameters
- `kb_id` (required): Knowledge base ID

#### Request Body (multipart/form-data)
- `file` (required): Document file
- `title` (optional): Custom document title
- `metadata` (optional): JSON string with additional metadata

#### Example
```bash
curl -X POST http://your-server/api/v1/rag/documents/kb_abc123 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@policy_document.pdf" \
  -F "title=Employee Handbook" \
  -F 'metadata={"department": "HR", "version": "2.1", "classification": "internal"}'
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "doc_xyz789",
    "tenant_id": "tenant_123",
    "knowledge_base_id": "kb_abc123",
    "title": "Employee Handbook",
    "filename": "policy_document.pdf",
    "file_type": "application/pdf",
    "file_size": 2048576,
    "content_hash": "sha256:abc123...",
    "chunk_count": 45,
    "processing_status": "completed",
    "metadata": {
      "department": "HR",
      "version": "2.1",
      "classification": "internal"
    },
    "user_id": "user_456",
    "created_at": "2025-01-29T10:30:00Z",
    "updated_at": "2025-01-29T10:32:00Z",
    "created_by": "user_456",
    "updated_by": "user_456"
  },
  "message": "Document uploaded and processed successfully"
}
```

### Get Document

Retrieve details of a specific document.

```http
GET /documents/{document_id}
```

#### Path Parameters
- `document_id` (required): Document ID

#### Response
```json
{
  "success": true,
  "data": {
    "id": "doc_xyz789",
    "tenant_id": "tenant_123",
    "knowledge_base_id": "kb_abc123",
    "title": "Employee Handbook",
    "filename": "policy_document.pdf",
    "file_type": "application/pdf",
    "file_size": 2048576,
    "content_hash": "sha256:abc123...",
    "chunk_count": 45,
    "processing_status": "completed",
    "metadata": {
      "department": "HR",
      "version": "2.1",
      "classification": "internal"
    },
    "user_id": "user_456",
    "created_at": "2025-01-29T10:30:00Z",
    "updated_at": "2025-01-29T10:32:00Z",
    "created_by": "user_456",
    "updated_by": "user_456"
  }
}
```

### List Documents

Get a list of documents in a knowledge base.

```http
GET /documents
```

#### Query Parameters
- `knowledge_base_id` (optional): Filter by knowledge base
- `user_id` (optional): Filter by user
- `status` (optional): Filter by processing status
- `limit` (optional): Number of results (default: 50, max: 100)
- `offset` (optional): Number of results to skip (default: 0)

#### Response
```json
{
  "success": true,
  "data": [
    {
      "id": "doc_xyz789",
      "knowledge_base_id": "kb_abc123",
      "title": "Employee Handbook",
      "filename": "policy_document.pdf",
      "file_type": "application/pdf",
      "file_size": 2048576,
      "chunk_count": 45,
      "processing_status": "completed",
      "created_at": "2025-01-29T10:30:00Z",
      "updated_at": "2025-01-29T10:32:00Z"
    }
  ],
  "count": 1
}
```

### Delete Document

Delete a document and all its associated chunks.

```http
DELETE /documents/{document_id}
```

#### Path Parameters
- `document_id` (required): Document ID

#### Response
```json
{
  "success": true,
  "message": "Document deleted successfully"
}
```

## Query & Retrieval API

### Query Knowledge Base

Search for relevant content in a knowledge base.

```http
POST /query/{kb_id}
```

#### Path Parameters
- `kb_id` (required): Knowledge base ID

#### Request Body
```json
{
  "query_text": "What is the remote work policy?",
  "k": 10,                            // Optional, number of results (default: 10)
  "similarity_threshold": 0.7,        // Optional, minimum relevance (default: 0.7)
  "retrieval_method": "hybrid_search" // Optional: "vector_search", "hybrid_search", "semantic_search"
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "query_text": "What is the remote work policy?",
    "retrieved_chunks": [
      "chunk_abc123",
      "chunk_def456",
      "chunk_ghi789"
    ],
    "similarity_scores": [0.95, 0.87, 0.82],
    "processing_time_ms": 125.5,
    "total_chunks_found": 3,
    "chunks_details": [
      {
        "chunk_id": "chunk_abc123",
        "document_id": "doc_xyz789",
        "content": "Remote work is permitted for all employees with manager approval...",
        "document_title": "Employee Handbook",
        "document_filename": "handbook.pdf",
        "similarity_score": 0.95,
        "section_title": "Remote Work Policy",
        "section_level": 2
      }
    ]
  }
}
```

### Advanced Query

Perform an advanced query with custom parameters.

```http
POST /query/{kb_id}/advanced
```

#### Request Body
```json
{
  "query_text": "What are the safety procedures?",
  "retrieval_config": {
    "k": 15,
    "similarity_threshold": 0.8,
    "retrieval_method": "hybrid_search",
    "enable_reranking": true,
    "rerank_top_k": 20,
    "vector_weight": 0.7,
    "text_weight": 0.3
  },
  "filters": {
    "document_type": "policy",
    "department": "safety",
    "classification": "public"
  }
}
```

#### Response
Same format as basic query with additional metadata about the advanced processing.

## Generation API

### Generate RAG Response

Generate an AI response based on knowledge base content.

```http
POST /generate/{kb_id}
```

#### Path Parameters
- `kb_id` (required): Knowledge base ID

#### Query Parameters
- `conversation_id` (optional): Conversation ID for context
- `generation_model` (optional): Override default generation model

#### Request Body
```json
{
  "query_text": "What is our data retention policy?",
  "k": 10,                     // Optional, number of chunks to retrieve
  "similarity_threshold": 0.7  // Optional, minimum relevance threshold
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "query_text": "What is our data retention policy?",
    "response_text": "According to our data retention policy outlined in the Employee Handbook (Section 5.3), personal data must be retained for a maximum of 7 years after the last customer interaction, with certain exceptions for legal and regulatory requirements. Financial records are kept for 10 years as required by law, while marketing data is typically purged after 3 years unless explicit consent is maintained.",
    "sources_used": [
      {
        "document_title": "Employee Handbook",
        "document_filename": "handbook.pdf",
        "section_title": "Data Retention Policy",
        "chunk_content": "Personal data must be retained for a maximum of 7 years...",
        "similarity_score": 0.94,
        "citation_id": "[1]"
      },
      {
        "document_title": "Legal Compliance Guide",
        "document_filename": "compliance.pdf",
        "section_title": "Financial Record Keeping",
        "chunk_content": "Financial records are kept for 10 years...",
        "similarity_score": 0.89,
        "citation_id": "[2]"
      }
    ],
    "generation_model": "qwen3",
    "token_count": 156,
    "generation_time_ms": 1250.3,
    "confidence_score": 0.92,
    "factual_accuracy_score": 0.95,
    "citation_coverage": 0.88
  }
}
```

### Streaming Generation

Generate a streaming RAG response for real-time display.

```http
POST /generate/{kb_id}/stream
```

#### Request Body
Same as standard generation request.

#### Response
Server-sent events (SSE) stream:

```
data: {"type": "start", "request_id": "req_abc123"}

data: {"type": "sources", "sources": [...]}

data: {"type": "token", "token": "According", "position": 0}

data: {"type": "token", "token": " to", "position": 1}

data: {"type": "complete", "response": {...}}
```

## Conversation API

### Create Conversation

Create a new conversation for a knowledge base.

```http
POST /conversations/{kb_id}
```

#### Path Parameters
- `kb_id` (required): Knowledge base ID

#### Request Body
```json
{
  "title": "HR Policy Discussion",
  "description": "Questions about company policies",
  "generation_model": "qwen3",         // Optional, defaults to KB setting
  "max_context_tokens": 8000,         // Optional, defaults to 8000
  "temperature": 0.7,                 // Optional, defaults to 0.7
  "session_id": "session_123"         // Optional, for session grouping
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "id": "conv_abc123",
    "tenant_id": "tenant_123",
    "knowledge_base_id": "kb_abc123",
    "title": "HR Policy Discussion",
    "description": "Questions about company policies",
    "generation_model": "qwen3",
    "max_context_tokens": 8000,
    "temperature": 0.7,
    "status": "active",
    "turn_count": 0,
    "total_tokens_used": 0,
    "user_id": "user_456",
    "session_id": "session_123",
    "created_at": "2025-01-29T10:30:00Z",
    "updated_at": "2025-01-29T10:30:00Z",
    "created_by": "user_456",
    "updated_by": "user_456"
  },
  "message": "Conversation created successfully"
}
```

### Get Conversation

Retrieve conversation details and history.

```http
GET /conversations/{conversation_id}
```

#### Path Parameters
- `conversation_id` (required): Conversation ID

#### Query Parameters
- `include_turns` (optional): Include conversation turns (default: true)
- `limit_turns` (optional): Maximum number of turns to include (default: 50)

#### Response
```json
{
  "success": true,
  "data": {
    "id": "conv_abc123",
    "tenant_id": "tenant_123",
    "knowledge_base_id": "kb_abc123",
    "title": "HR Policy Discussion",
    "description": "Questions about company policies",
    "generation_model": "qwen3",
    "max_context_tokens": 8000,
    "temperature": 0.7,
    "status": "active",
    "turn_count": 4,
    "total_tokens_used": 1250,
    "user_id": "user_456",
    "session_id": "session_123",
    "created_at": "2025-01-29T10:30:00Z",
    "updated_at": "2025-01-29T10:45:00Z",
    "turns": [
      {
        "id": "turn_1",
        "turn_number": 1,
        "turn_type": "user",
        "content": "What is our vacation policy?",
        "created_at": "2025-01-29T10:31:00Z"
      },
      {
        "id": "turn_2",
        "turn_number": 2,
        "turn_type": "assistant",
        "content": "Our vacation policy allows for 20 days...",
        "model_used": "qwen3",
        "confidence_score": 0.92,
        "generation_time_ms": 1100,
        "created_at": "2025-01-29T10:31:05Z"
      }
    ]
  }
}
```

### List Conversations

Get a list of conversations.

```http
GET /conversations
```

#### Query Parameters
- `user_id` (optional): Filter by user
- `session_id` (optional): Filter by session
- `status` (optional): Filter by status (active, archived, deleted)
- `knowledge_base_id` (optional): Filter by knowledge base
- `limit` (optional): Number of results (default: 50)
- `offset` (optional): Number of results to skip (default: 0)

#### Response
```json
{
  "success": true,
  "data": [
    {
      "id": "conv_abc123",
      "knowledge_base_id": "kb_abc123",
      "title": "HR Policy Discussion",
      "status": "active",
      "turn_count": 4,
      "total_tokens_used": 1250,
      "created_at": "2025-01-29T10:30:00Z",
      "updated_at": "2025-01-29T10:45:00Z"
    }
  ],
  "count": 1
}
```

### Update Conversation

Update conversation metadata.

```http
PUT /conversations/{conversation_id}
```

#### Request Body
```json
{
  "title": "Updated HR Discussion",
  "description": "Updated description",
  "status": "archived"
}
```

#### Response
```json
{
  "success": true,
  "message": "Conversation updated successfully"
}
```

### Delete Conversation

Delete a conversation and all its turns.

```http
DELETE /conversations/{conversation_id}
```

#### Response
```json
{
  "success": true,
  "message": "Conversation deleted successfully"
}
```

## Chat API

### Send Chat Message

Send a message in a conversation and receive an AI response.

```http
POST /chat/{conversation_id}
```

#### Path Parameters
- `conversation_id` (required): Conversation ID

#### Request Body
```json
{
  "message": "What about remote work options?",
  "user_context": {
    "department": "Engineering",
    "role": "Senior Developer",
    "location": "California"
  }
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "conversation_id": "conv_abc123",
    "user_turn": {
      "id": "turn_5",
      "content": "What about remote work options?",
      "turn_number": 5,
      "created_at": "2025-01-29T10:46:00Z"
    },
    "assistant_turn": {
      "id": "turn_6",
      "content": "Regarding remote work options, our company policy allows full-time remote work for engineering roles with manager approval. As a Senior Developer in California, you would be eligible under our distributed team program...",
      "turn_number": 6,
      "model_used": "qwen3",
      "confidence_score": 0.89,
      "generation_time_ms": 1350,
      "created_at": "2025-01-29T10:46:03Z",
      "sources_used": [
        {
          "document_title": "Remote Work Policy",
          "similarity_score": 0.93,
          "citation_id": "[1]"
        }
      ]
    }
  }
}
```

### Streaming Chat

Send a chat message with streaming response.

```http
POST /chat/{conversation_id}/stream
```

#### Request Body
Same as regular chat message.

#### Response
Server-sent events stream with real-time message generation.

## Health & Monitoring API

### Health Check

Get system health status.

```http
GET /health
```

#### Response
```json
{
  "success": true,
  "data": {
    "service_status": "running",
    "uptime_seconds": 86400,
    "database_connection": true,
    "components_healthy": true,
    "active_operations": 12,
    "timestamp": "2025-01-29T10:30:00Z",
    "components": {
      "vector_service": {
        "service_status": "healthy",
        "database_connection": true,
        "ollama_integration": true,
        "indexes_healthy": true
      },
      "retrieval_engine": {
        "service_status": "healthy",
        "cache_hit_rate": 0.85,
        "average_query_time_ms": 45.2
      },
      "generation_engine": {
        "service_status": "healthy",
        "model_availability": {
          "qwen3": true,
          "deepseek-r1": true
        },
        "average_generation_time_ms": 1200.5
      }
    }
  }
}
```

### Service Statistics

Get detailed service statistics.

```http
GET /health/stats
```

#### Response
```json
{
  "success": true,
  "data": {
    "service_metrics": {
      "status": "running",
      "start_time": "2025-01-28T10:00:00Z",
      "uptime_seconds": 86400,
      "total_operations": 15420,
      "successful_operations": 15250,
      "failed_operations": 170,
      "active_operations": 12
    },
    "performance_metrics": {
      "documents_processed": 1250,
      "chunks_indexed": 62500,
      "queries_executed": 8400,
      "conversations_active": 45,
      "average_processing_time_ms": 850.2,
      "average_query_time_ms": 125.5,
      "average_generation_time_ms": 1200.3
    },
    "component_stats": {
      "vector_service": {
        "embeddings_generated": 62500,
        "chunks_indexed": 62500,
        "cache_operations": 25600,
        "query_operations": 8400,
        "cache_stats": {
          "hits": 21760,
          "misses": 3840,
          "hit_rate": 0.85,
          "size": 15000
        }
      }
    },
    "active_operations": [
      "generate_response",
      "index_chunks",
      "process_document"
    ],
    "health_history_count": 1440
  }
}
```

### Performance Metrics

Get detailed performance metrics.

```http
GET /health/metrics
```

#### Query Parameters
- `start_time` (optional): Start time for metrics (ISO 8601)
- `end_time` (optional): End time for metrics (ISO 8601)
- `granularity` (optional): Time granularity (minute, hour, day)

#### Response
```json
{
  "success": true,
  "data": {
    "time_range": {
      "start": "2025-01-29T09:30:00Z",
      "end": "2025-01-29T10:30:00Z",
      "granularity": "minute"
    },
    "metrics": [
      {
        "timestamp": "2025-01-29T10:30:00Z",
        "cpu_percent": 45.2,
        "memory_mb": 1250.5,
        "active_connections": 25,
        "queries_per_minute": 15,
        "average_response_time_ms": 125.5
      }
    ]
  }
}
```

## Error Handling

### Error Response Format

All errors follow this standard format:

```json
{
  "success": false,
  "error": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "timestamp": "2025-01-29T10:30:00Z",
  "request_id": "req_abc123",
  "details": {
    "field": "Additional error details"
  }
}
```

### Common Error Codes

#### Authentication Errors (401)
- `INVALID_TOKEN`: API token is invalid or expired
- `TOKEN_MISSING`: Authorization header missing
- `TOKEN_EXPIRED`: API token has expired

#### Authorization Errors (403)
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `TENANT_ACCESS_DENIED`: Access denied for tenant
- `RESOURCE_FORBIDDEN`: Access to specific resource denied

#### Validation Errors (400)
- `INVALID_REQUEST`: Request format is invalid
- `MISSING_REQUIRED_FIELD`: Required field is missing
- `INVALID_FIELD_VALUE`: Field value is invalid
- `FILE_TOO_LARGE`: Uploaded file exceeds size limit
- `UNSUPPORTED_FILE_TYPE`: File type not supported

#### Resource Errors (404)
- `KNOWLEDGE_BASE_NOT_FOUND`: Knowledge base does not exist
- `DOCUMENT_NOT_FOUND`: Document does not exist
- `CONVERSATION_NOT_FOUND`: Conversation does not exist

#### Processing Errors (422)
- `DOCUMENT_PROCESSING_FAILED`: Document could not be processed
- `EMBEDDING_GENERATION_FAILED`: Embedding generation failed
- `QUERY_PROCESSING_FAILED`: Query could not be processed

#### System Errors (500)
- `INTERNAL_SERVER_ERROR`: Generic server error
- `DATABASE_ERROR`: Database operation failed
- `EXTERNAL_SERVICE_ERROR`: External service (Ollama) error
- `SYSTEM_OVERLOADED`: System is temporarily overloaded

#### Service Errors (503)
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable
- `MAINTENANCE_MODE`: System in maintenance mode
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded

## Rate Limiting

The API implements rate limiting to ensure fair usage and system stability.

### Rate Limit Headers

Response headers include rate limit information:

```http
X-RateLimit-Limit: 1000          # Requests per hour
X-RateLimit-Remaining: 995       # Remaining requests
X-RateLimit-Reset: 1706523600    # Reset time (Unix timestamp)
X-RateLimit-Window: 3600         # Window duration in seconds
```

### Rate Limit Policies

Default rate limits by endpoint category:

- **Authentication**: 10 requests/minute
- **Knowledge Base Management**: 100 requests/hour
- **Document Upload**: 50 requests/hour
- **Query/Generation**: 1000 requests/hour
- **Chat**: 500 requests/hour
- **Health/Monitoring**: 200 requests/hour

### Rate Limit Exceeded Response

```json
{
  "success": false,
  "error": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 300,
  "timestamp": "2025-01-29T10:30:00Z"
}
```

## Webhooks

The API supports webhooks for real-time notifications about events.

### Webhook Events

Available webhook events:

- `document.processing.completed`
- `document.processing.failed`
- `knowledge_base.created`
- `knowledge_base.updated`
- `conversation.created`
- `system.alert.triggered`
- `system.maintenance.scheduled`

### Webhook Configuration

Configure webhooks through the admin interface or API:

```http
POST /webhooks
```

```json
{
  "url": "https://your-app.com/webhooks/rag",
  "events": ["document.processing.completed", "system.alert.triggered"],
  "secret": "your-webhook-secret",
  "active": true
}
```

### Webhook Payload

Example webhook payload:

```json
{
  "event": "document.processing.completed",
  "timestamp": "2025-01-29T10:30:00Z",
  "data": {
    "document_id": "doc_xyz789",
    "knowledge_base_id": "kb_abc123",
    "status": "completed",
    "chunk_count": 45,
    "processing_time_ms": 5500
  },
  "signature": "sha256=abc123..."
}
```

### Webhook Verification

Verify webhook authenticity using the signature header:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

---

**Questions?** Contact your system administrator or check our support documentation for additional help.