# APG GraphRAG API Reference

Complete reference for the APG GraphRAG REST API with 40+ endpoints for comprehensive knowledge graph operations.

## 📋 Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Knowledge Graph Management](#knowledge-graph-management)
4. [Entity Operations](#entity-operations)
5. [Relationship Operations](#relationship-operations)
6. [Document Processing](#document-processing)
7. [Query Processing](#query-processing)
8. [Analytics & Monitoring](#analytics--monitoring)
9. [Visualization](#visualization)
10. [Administrative Operations](#administrative-operations)
11. [Error Handling](#error-handling)
12. [Rate Limits](#rate-limits)

## 🌐 API Overview

### Base URL
```
https://your-domain.com/api/v1/graphrag
```

### Content Type
All requests and responses use `application/json` unless otherwise specified.

### Common Headers
```http
Content-Type: application/json
X-Tenant-ID: your_tenant_id
X-User-ID: user_identifier
X-Session-ID: session_identifier
Authorization: Bearer <token>
```

### API Versioning
The API uses URL versioning (`/api/v1/`). Breaking changes will increment the version number.

## 🔐 Authentication

APG GraphRAG API uses Bearer token authentication:

```http
Authorization: Bearer <your_api_token>
```

### Getting an API Token

```bash
curl -X POST https://your-domain.com/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## 📊 Knowledge Graph Management

### List Knowledge Graphs

Get all knowledge graphs for a tenant.

```http
GET /graphs/
```

**Query Parameters:**
- `limit` (integer, optional): Maximum number of results (default: 50)
- `offset` (integer, optional): Number of results to skip (default: 0)
- `domain` (string, optional): Filter by domain
- `status` (string, optional): Filter by status (active, inactive, archived)

**Example Request:**
```bash
curl -X GET "https://your-domain.com/api/v1/graphrag/graphs/?limit=10&domain=business" \
  -H "Authorization: Bearer <token>" \
  -H "X-Tenant-ID: your_tenant"
```

**Response:**
```json
[
  {
    "knowledge_graph_id": "kg_123456789",
    "name": "Business Intelligence Graph",
    "description": "Knowledge graph for business entities",
    "domain": "business",
    "entity_count": 1250,
    "relationship_count": 2100,
    "document_count": 45,
    "avg_entity_confidence": 0.87,
    "status": "active",
    "created_at": "2024-01-15T10:30:00Z",
    "last_updated": "2024-01-20T14:15:00Z",
    "metadata": {
      "created_by": "data_team",
      "purpose": "market_analysis"
    }
  }
]
```

### Create Knowledge Graph

Create a new knowledge graph.

```http
POST /graphs/
```

**Request Body:**
```json
{
  "name": "Technology Innovation Graph",
  "description": "Knowledge graph for technology and innovation tracking",
  "domain": "technology",
  "metadata": {
    "created_by": "innovation_team",
    "data_sources": ["patents", "research_papers", "startup_databases"]
  }
}
```

**Response (201 Created):**
```json
{
  "knowledge_graph_id": "kg_987654321",
  "name": "Technology Innovation Graph",
  "description": "Knowledge graph for technology and innovation tracking",
  "domain": "technology",
  "entity_count": 0,
  "relationship_count": 0,
  "document_count": 0,
  "avg_entity_confidence": 0.0,
  "status": "active",
  "created_at": "2024-01-21T09:00:00Z",
  "last_updated": "2024-01-21T09:00:00Z",
  "metadata": {
    "created_by": "innovation_team",
    "data_sources": ["patents", "research_papers", "startup_databases"]
  }
}
```

### Get Knowledge Graph

Retrieve a specific knowledge graph.

```http
GET /graphs/{graph_id}
```

**Path Parameters:**
- `graph_id` (string, required): Knowledge graph identifier

**Example Request:**
```bash
curl -X GET "https://your-domain.com/api/v1/graphrag/graphs/kg_123456789" \
  -H "Authorization: Bearer <token>" \
  -H "X-Tenant-ID: your_tenant"
```

### Update Knowledge Graph

Update knowledge graph metadata.

```http
PUT /graphs/{graph_id}
```

**Request Body:**
```json
{
  "name": "Updated Graph Name",
  "description": "Updated description",
  "metadata": {
    "updated_by": "admin_user",
    "version": "2.0"
  }
}
```

### Delete Knowledge Graph

Delete a knowledge graph and all associated data.

```http
DELETE /graphs/{graph_id}
```

**Response (200 OK):**
```json
{
  "message": "Knowledge graph kg_123456789 deleted successfully"
}
```

### Get Knowledge Graph Statistics

Get detailed statistics for a knowledge graph.

```http
GET /graphs/{graph_id}/statistics
```

**Response:**
```json
{
  "basic_stats": {
    "entity_count": 1250,
    "relationship_count": 2100,
    "document_count": 45
  },
  "quality_metrics": {
    "avg_entity_confidence": 0.87,
    "avg_relationship_strength": 0.83,
    "data_completeness": 0.92
  },
  "graph_metrics": {
    "density": 0.0024,
    "avg_degree": 3.7,
    "clustering_coefficient": 0.45,
    "diameter": 8,
    "connected_components": 1
  },
  "temporal_analysis": {
    "growth_rate_entities": 0.15,
    "growth_rate_relationships": 0.22,
    "recent_activity_score": 0.68,
    "last_major_update": "2024-01-20T14:15:00Z"
  },
  "distribution": {
    "entity_types": {
      "person": 450,
      "organization": 380,
      "location": 220,
      "product": 120,
      "concept": 80
    },
    "relationship_types": {
      "works_for": 520,
      "located_in": 340,
      "partner_of": 280,
      "competitor_of": 180,
      "acquired_by": 95
    }
  }
}
```

## 🔗 Entity Operations

### List Entities

Get entities from a knowledge graph.

```http
GET /graphs/{graph_id}/entities
```

**Query Parameters:**
- `limit` (integer, optional): Maximum results (default: 100)
- `offset` (integer, optional): Results to skip (default: 0)
- `entity_type` (string, optional): Filter by entity type
- `search` (string, optional): Search in entity names and aliases
- `min_confidence` (float, optional): Minimum confidence score
- `sort_by` (string, optional): Sort field (name, confidence, created_at)
- `sort_order` (string, optional): Sort order (asc, desc)

**Example Request:**
```bash
curl -X GET "https://your-domain.com/api/v1/graphrag/graphs/kg_123456789/entities?entity_type=person&limit=20" \
  -H "Authorization: Bearer <token>" \
  -H "X-Tenant-ID: your_tenant"
```

**Response:**
```json
[
  {
    "canonical_entity_id": "ent_abc123",
    "canonical_name": "John Doe",
    "entity_type": "person",
    "aliases": ["John D.", "J. Doe", "Johnny"],
    "properties": {
      "occupation": "CEO",
      "company": "Acme Corp",
      "location": "San Francisco",
      "education": "Stanford MBA"
    },
    "confidence_score": 0.92,
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-18T16:45:00Z"
  }
]
```

### Create Entity

Create a new entity in the knowledge graph.

```http
POST /graphs/{graph_id}/entities
```

**Request Body:**
```json
{
  "canonical_name": "Jane Smith",
  "entity_type": "person",
  "aliases": ["J. Smith", "Jane"],
  "properties": {
    "role": "CTO",
    "company": "TechStart Inc",
    "experience": "15 years",
    "specialization": "AI/ML"
  },
  "confidence_score": 0.95
}
```

**Response (201 Created):**
```json
{
  "canonical_entity_id": "ent_def456",
  "canonical_name": "Jane Smith",
  "entity_type": "person",
  "aliases": ["J. Smith", "Jane"],
  "properties": {
    "role": "CTO",
    "company": "TechStart Inc",
    "experience": "15 years",
    "specialization": "AI/ML"
  },
  "confidence_score": 0.95,
  "created_at": "2024-01-21T09:15:00Z",
  "updated_at": "2024-01-21T09:15:00Z"
}
```

### Get Entity

Retrieve a specific entity with its relationships.

```http
GET /graphs/{graph_id}/entities/{entity_id}
```

**Query Parameters:**
- `include_relationships` (boolean, optional): Include related entities (default: false)
- `relationship_hops` (integer, optional): Number of relationship hops to include (default: 1)

**Response:**
```json
{
  "canonical_entity_id": "ent_abc123",
  "canonical_name": "John Doe",
  "entity_type": "person",
  "aliases": ["John D.", "J. Doe"],
  "properties": {
    "occupation": "CEO",
    "company": "Acme Corp"
  },
  "confidence_score": 0.92,
  "relationships": [
    {
      "relationship_id": "rel_123",
      "relationship_type": "works_for",
      "target_entity": {
        "entity_id": "ent_xyz789",
        "canonical_name": "Acme Corp",
        "entity_type": "organization"
      },
      "strength": 0.95,
      "confidence_score": 0.88
    }
  ],
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-18T16:45:00Z"
}
```

### Update Entity

Update entity properties and metadata.

```http
PUT /graphs/{graph_id}/entities/{entity_id}
```

**Request Body:**
```json
{
  "canonical_name": "John Doe Jr.",
  "aliases": ["John D.", "J. Doe", "Johnny", "John Jr."],
  "properties": {
    "occupation": "CEO",
    "company": "Acme Corp",
    "title": "Chief Executive Officer",
    "tenure": "5 years"
  }
}
```

### Delete Entity

Remove an entity and its relationships.

```http
DELETE /graphs/{graph_id}/entities/{entity_id}
```

**Query Parameters:**
- `cascade` (boolean, optional): Delete dependent relationships (default: true)

### Search Entities

Semantic search for entities using vector embeddings.

```http
POST /graphs/{graph_id}/entities/search
```

**Request Body:**
```json
{
  "query_text": "AI startup founders in Silicon Valley",
  "limit": 10,
  "similarity_threshold": 0.75,
  "entity_types": ["person"],
  "filters": {
    "location": "Silicon Valley",
    "industry": "AI"
  }
}
```

**Response:**
```json
[
  {
    "canonical_entity_id": "ent_ai_001",
    "canonical_name": "Sarah Johnson",
    "entity_type": "person",
    "similarity_score": 0.89,
    "properties": {
      "company": "AI Innovations Inc",
      "role": "Founder & CEO",
      "location": "Palo Alto"
    },
    "match_explanation": "Founded AI company in Silicon Valley, high semantic similarity to query"
  }
]
```

## 🔀 Relationship Operations

### List Relationships

Get relationships from a knowledge graph.

```http
GET /graphs/{graph_id}/relationships
```

**Query Parameters:**
- `limit` (integer, optional): Maximum results
- `offset` (integer, optional): Results to skip
- `relationship_type` (string, optional): Filter by relationship type
- `entity_id` (string, optional): Filter by connected entity
- `min_strength` (float, optional): Minimum relationship strength
- `min_confidence` (float, optional): Minimum confidence score

**Response:**
```json
[
  {
    "canonical_relationship_id": "rel_123456",
    "source_entity_id": "ent_abc123",
    "target_entity_id": "ent_xyz789",
    "relationship_type": "works_for",
    "strength": 0.95,
    "properties": {
      "start_date": "2020-01-15",
      "position": "CEO",
      "department": "Executive"
    },
    "confidence_score": 0.88,
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-18T16:45:00Z"
  }
]
```

### Create Relationship

Create a relationship between two entities.

```http
POST /graphs/{graph_id}/relationships
```

**Request Body:**
```json
{
  "source_entity_id": "ent_abc123",
  "target_entity_id": "ent_def456",
  "relationship_type": "collaborates_with",
  "strength": 0.85,
  "properties": {
    "project": "AI Research Initiative",
    "start_date": "2024-01-01",
    "collaboration_type": "technical"
  },
  "confidence_score": 0.90
}
```

### Get Relationship

Retrieve a specific relationship with context.

```http
GET /graphs/{graph_id}/relationships/{relationship_id}
```

### Update Relationship

Update relationship properties and strength.

```http
PUT /graphs/{graph_id}/relationships/{relationship_id}
```

### Delete Relationship

Remove a relationship from the graph.

```http
DELETE /graphs/{graph_id}/relationships/{relationship_id}
```

## 📄 Document Processing

### Process Document

Extract entities and relationships from a document.

```http
POST /graphs/{graph_id}/process
```

**Request Body:**
```json
{
  "title": "Company Annual Report 2024",
  "content": "TechCorp, founded by Dr. Sarah Johnson in 2021, has grown rapidly...",
  "source_url": "https://example.com/annual-report.pdf",
  "source_type": "pdf",
  "processing_options": {
    "extract_entities": true,
    "extract_relationships": true,
    "entity_types": ["person", "organization", "location", "product"],
    "relationship_types": ["works_for", "founded", "located_in", "produces"],
    "confidence_threshold": 0.8,
    "enable_coreference_resolution": true,
    "enable_temporal_extraction": true,
    "chunk_size": 512,
    "chunk_overlap": 50
  }
}
```

**Response (201 Created):**
```json
{
  "document_id": "doc_789123",
  "processing_status": "completed",
  "entities_extracted": 15,
  "relationships_extracted": 23,
  "processing_time_ms": 2500.0,
  "confidence_score": 0.87,
  "metadata": {
    "text_length": 5240,
    "chunks_processed": 12,
    "entities_by_type": {
      "person": 6,
      "organization": 4,
      "location": 3,
      "product": 2
    },
    "relationships_by_type": {
      "works_for": 8,
      "founded": 2,
      "located_in": 7,
      "produces": 6
    }
  },
  "extracted_entities": [
    {
      "canonical_name": "Dr. Sarah Johnson",
      "entity_type": "person",
      "confidence_score": 0.95,
      "mentions": [
        {
          "text": "Dr. Sarah Johnson",
          "start_position": 25,
          "end_position": 42,
          "context": "TechCorp, founded by Dr. Sarah Johnson in 2021"
        }
      ]
    }
  ],
  "extracted_relationships": [
    {
      "source_entity": "Dr. Sarah Johnson",
      "target_entity": "TechCorp",
      "relationship_type": "founded",
      "strength": 0.92,
      "confidence_score": 0.88,
      "evidence": "TechCorp, founded by Dr. Sarah Johnson in 2021"
    }
  ]
}
```

### List Documents

Get processed documents for a knowledge graph.

```http
GET /graphs/{graph_id}/documents
```

**Query Parameters:**
- `limit` (integer, optional): Maximum results
- `offset` (integer, optional): Results to skip
- `status` (string, optional): Filter by processing status
- `source_type` (string, optional): Filter by document type

**Response:**
```json
[
  {
    "document_id": "doc_789123",
    "title": "Company Annual Report 2024",
    "source_type": "pdf",
    "source_url": "https://example.com/annual-report.pdf",
    "processing_status": "completed",
    "entity_count": 15,
    "relationship_count": 23,
    "processing_metadata": {
      "processing_time_ms": 2500.0,
      "confidence_score": 0.87
    },
    "processed_at": "2024-01-21T10:15:00Z",
    "created_at": "2024-01-21T10:12:30Z"
  }
]
```

### Get Document

Retrieve details of a processed document.

```http
GET /graphs/{graph_id}/documents/{document_id}
```

### Reprocess Document

Reprocess a document with updated parameters.

```http
POST /graphs/{graph_id}/documents/{document_id}/reprocess
```

## 🔍 Query Processing

### Process GraphRAG Query

Execute a natural language query against the knowledge graph.

```http
POST /graphs/{graph_id}/query
```

**Request Body:**
```json
{
  "query_text": "What companies has John Doe worked for and who are his current business partners?",
  "query_type": "factual",
  "context": {
    "user_id": "analyst_001",
    "session_id": "session_123",
    "conversation_history": [
      {
        "role": "user",
        "content": "Tell me about John Doe"
      },
      {
        "role": "assistant", 
        "content": "John Doe is the CEO of Acme Corp..."
      }
    ],
    "domain_context": {
      "domain": "business",
      "focus": "professional_relationships"
    },
    "temporal_context": {
      "timeframe": "current",
      "reference_date": "2024-01-21"
    }
  },
  "retrieval_config": {
    "max_entities": 50,
    "similarity_threshold": 0.75,
    "enable_vector_search": true,
    "enable_graph_traversal": true,
    "traversal_depth": 3
  },
  "reasoning_config": {
    "reasoning_type": "multi_hop",
    "max_reasoning_steps": 6,
    "enable_hypothesis_generation": true,
    "confidence_threshold": 0.7
  },
  "explanation_level": "detailed",
  "max_hops": 3
}
```

**Response:**
```json
{
  "query_id": "query_456789",
  "answer": "Based on the knowledge graph analysis, John Doe has worked for three companies: StartupX (2018-2020) as VP Engineering, TechCorp (2020-2022) as CTO, and currently serves as CEO of Acme Corp since 2022. His current business partners include Jane Smith (co-founder of Acme Corp), Mike Johnson (Strategic Partnership at Acme), and the company has partnerships with Google Cloud and AWS.",
  "confidence_score": 0.89,
  "processing_time_ms": 1250.0,
  "entities_used": [
    {
      "entity_id": "ent_john_doe",
      "canonical_name": "John Doe",
      "entity_type": "person",
      "relevance_score": 1.0
    },
    {
      "entity_id": "ent_acme_corp",
      "canonical_name": "Acme Corp",
      "entity_type": "organization",
      "relevance_score": 0.95
    }
  ],
  "relationships_used": [
    {
      "relationship_id": "rel_works_for_001",
      "source_entity": "John Doe",
      "target_entity": "Acme Corp",
      "relationship_type": "works_for",
      "strength": 0.95
    }
  ],
  "reasoning_chain": {
    "steps": [
      {
        "step_number": 1,
        "description": "Identified John Doe entity in knowledge graph",
        "entities_involved": ["ent_john_doe"],
        "confidence": 0.98,
        "evidence_count": 5
      },
      {
        "step_number": 2,
        "description": "Found employment relationships for John Doe",
        "entities_involved": ["ent_john_doe", "ent_startupx", "ent_techcorp", "ent_acme_corp"],
        "relationships_involved": ["rel_worked_for_001", "rel_worked_for_002", "rel_works_for_001"],
        "confidence": 0.92
      },
      {
        "step_number": 3,
        "description": "Identified current business partnerships through Acme Corp",
        "entities_involved": ["ent_acme_corp", "ent_jane_smith", "ent_mike_johnson"],
        "confidence": 0.85
      }
    ],
    "overall_confidence": 0.89,
    "reasoning_type": "multi_hop",
    "total_steps": 3
  },
  "evidence": [
    {
      "source_id": "doc_company_profile",
      "content": "John Doe joined Acme Corp as CEO in January 2022, bringing extensive experience from his previous roles at TechCorp and StartupX.",
      "relevance_score": 0.94,
      "confidence": 0.91,
      "source_type": "document"
    }
  ],
  "entity_mentions": [
    {
      "entity_id": "ent_john_doe",
      "mention_text": "John Doe",
      "position_start": 89,
      "position_end": 97,
      "confidence": 0.95
    }
  ],
  "source_attribution": [
    {
      "source_id": "doc_company_profile",
      "source_type": "document",
      "contribution_weight": 0.45,
      "citation_text": "[1] Company Profile Document",
      "confidence": 0.91
    }
  ],
  "quality_indicators": {
    "factual_accuracy": 0.92,
    "completeness": 0.87,
    "relevance": 0.94,
    "coherence": 0.89,
    "clarity": 0.91,
    "confidence": 0.89,
    "source_reliability": 0.88
  },
  "metadata": {
    "model_used": "qwen3",
    "retrieval_method": "hybrid",
    "reasoning_hops": 3,
    "total_entities_considered": 127,
    "total_relationships_considered": 284
  },
  "status": "completed"
}
```

### Get Query History

Retrieve query history for analysis.

```http
GET /queries/history
```

**Query Parameters:**
- `limit` (integer, optional): Maximum results
- `offset` (integer, optional): Results to skip
- `user_id` (string, optional): Filter by user
- `graph_id` (string, optional): Filter by knowledge graph
- `start_date` (string, optional): Filter by start date (ISO 8601)
- `end_date` (string, optional): Filter by end date (ISO 8601)

**Response:**
```json
[
  {
    "query_id": "query_456789",
    "query_text": "What companies has John Doe worked for?",
    "query_type": "factual",
    "processing_time_ms": 1250.0,
    "confidence_score": 0.89,
    "user_id": "analyst_001",
    "session_id": "session_123",
    "knowledge_graph_id": "kg_123456789",
    "created_at": "2024-01-21T14:30:00Z"
  }
]
```

## 📊 Analytics & Monitoring

### Analytics Overview

Get comprehensive system analytics.

```http
GET /analytics/overview
```

**Response:**
```json
{
  "knowledge_graphs": {
    "total": 15,
    "active": 12,
    "archived": 3,
    "total_entities": 125000,
    "total_relationships": 340000
  },
  "entities": {
    "total": 125000,
    "avg_confidence": 0.87,
    "by_type": {
      "person": 45000,
      "organization": 35000,
      "location": 25000,
      "product": 15000,
      "concept": 5000
    }
  },
  "relationships": {
    "total": 340000,
    "avg_strength": 0.83,
    "by_type": {
      "works_for": 85000,
      "located_in": 65000,
      "partner_of": 55000,
      "competitor_of": 45000,
      "other": 90000
    }
  },
  "queries": {
    "today": 245,
    "this_week": 1680,
    "this_month": 7250,
    "avg_response_time_ms": 1150,
    "avg_confidence": 0.86,
    "by_type": {
      "factual": 0.45,
      "analytical": 0.35,
      "exploratory": 0.20
    }
  },
  "documents": {
    "processed": 2840,
    "pending": 12,
    "failed": 8,
    "total_size_mb": 15600,
    "avg_processing_time_ms": 2400
  },
  "system_health": {
    "database_status": "healthy",
    "ollama_status": "healthy",
    "api_status": "healthy",
    "last_health_check": "2024-01-21T15:00:00Z"
  }
}
```

### Performance Analytics

Get detailed performance metrics for a knowledge graph.

```http
GET /graphs/{graph_id}/performance
```

**Query Parameters:**
- `days` (integer, optional): Number of days to analyze (default: 7)
- `metrics` (string, optional): Comma-separated list of metrics to include

**Response:**
```json
{
  "time_period": {
    "start_date": "2024-01-14T00:00:00Z",
    "end_date": "2024-01-21T00:00:00Z",
    "days": 7
  },
  "query_performance": {
    "total_queries": 1250,
    "avg_response_time_ms": 1150,
    "p95_response_time_ms": 2800,
    "p99_response_time_ms": 4500,
    "throughput_qps": 8.5,
    "success_rate": 0.984,
    "by_query_type": {
      "factual": {
        "count": 562,
        "avg_response_time_ms": 850,
        "success_rate": 0.992
      },
      "analytical": {
        "count": 438,
        "avg_response_time_ms": 1450,
        "success_rate": 0.976
      },
      "exploratory": {
        "count": 250,
        "avg_response_time_ms": 1680,
        "success_rate": 0.980
      }
    }
  },
  "system_performance": {
    "database": {
      "avg_query_time_ms": 145,
      "connection_pool_utilization": 0.65,
      "cache_hit_rate": 0.78
    },
    "ollama": {
      "embedding_avg_time_ms": 120,
      "generation_avg_time_ms": 680,
      "model_health": {
        "bge-m3": "healthy",
        "qwen3": "healthy",
        "deepseek-r1": "healthy"
      }
    },
    "memory": {
      "usage_percent": 68,
      "cache_size_mb": 2048,
      "gc_frequency": 0.15
    }
  },
  "data_quality": {
    "avg_entity_confidence": 0.87,
    "avg_relationship_strength": 0.83,
    "data_completeness": 0.92,
    "recent_updates": 145,
    "quality_score": 0.89
  },
  "usage_patterns": {
    "peak_hours": ["09:00-10:00", "14:00-15:00"],
    "most_active_users": ["analyst_001", "researcher_042"],
    "popular_query_types": ["factual", "analytical"],
    "frequent_entities": ["John Doe", "Acme Corp", "Silicon Valley"]
  }
}
```

### Usage Analytics

Get detailed usage analytics and trends.

```http
GET /analytics/usage
```

**Query Parameters:**
- `start_date` (string, optional): Start date for analysis
- `end_date` (string, optional): End date for analysis
- `granularity` (string, optional): Data granularity (hour, day, week, month)

## 🎨 Visualization

### Generate Visualization

Create a visualization of a knowledge graph.

```http
POST /graphs/{graph_id}/visualize
```

**Request Body:**
```json
{
  "config": {
    "width": 1200,
    "height": 800,
    "layout_algorithm": "spring",
    "enable_3d": false,
    "max_nodes": 100,
    "max_edges": 200,
    "node_size_range": [8, 40],
    "edge_width_range": [1, 8],
    "confidence_threshold": 0.7,
    "enable_clustering": true,
    "cluster_threshold": 0.8,
    "enable_animations": true,
    "background_color": "#ffffff"
  },
  "filters": {
    "entity_types": ["person", "organization"],
    "relationship_types": ["works_for", "partner_of"],
    "min_confidence": 0.8
  }
}
```

**Response:**
```json
{
  "visualization_id": "viz_789456",
  "nodes": [
    {
      "id": "ent_john_doe",
      "label": "John Doe",
      "type": "person",
      "position": {"x": 150.5, "y": 200.3, "z": 0},
      "style": {
        "size": 25,
        "color": "#3498db",
        "border_color": "#2980b9",
        "shape": "circle"
      },
      "data": {
        "canonical_name": "John Doe",
        "entity_type": "person",
        "confidence_score": 0.92
      }
    }
  ],
  "edges": [
    {
      "id": "rel_works_for_001",
      "source": "ent_john_doe",
      "target": "ent_acme_corp",
      "label": "works_for",
      "type": "works_for",
      "style": {
        "width": 3,
        "color": "#7f8c8d",
        "opacity": 0.8
      },
      "data": {
        "strength": 0.95,
        "confidence_score": 0.88
      }
    }
  ],
  "clusters": [
    {
      "id": "cluster_tech_companies",
      "label": "Technology Companies",
      "entity_ids": ["ent_acme_corp", "ent_techstart_inc"],
      "color": "#e74c3c",
      "size": 2
    }
  ],
  "statistics": {
    "node_count": 45,
    "edge_count": 78,
    "cluster_count": 5,
    "density": 0.038,
    "avg_degree": 3.4
  },
  "metadata": {
    "graph_id": "kg_123456789",
    "generated_at": "2024-01-21T15:30:00Z",
    "processing_time_ms": 850,
    "layout_algorithm": "spring",
    "total_iterations": 100
  }
}
```

### Export Visualization

Export visualization in various formats.

```http
POST /visualizations/{visualization_id}/export
```

**Request Body:**
```json
{
  "format": "svg",
  "options": {
    "width": 1200,
    "height": 800,
    "include_labels": true,
    "background_color": "#ffffff"
  }
}
```

**Response:**
```json
{
  "export_id": "export_456123",
  "format": "svg",
  "data": "<svg width=\"1200\" height=\"800\">...</svg>",
  "size_bytes": 45280,
  "export_timestamp": "2024-01-21T15:35:00Z"
}
```

### Supported Export Formats

- `json` - JSON data structure
- `svg` - Scalable Vector Graphics
- `png` - Portable Network Graphics (requires rendering service)
- `pdf` - Portable Document Format (requires rendering service)
- `graphml` - GraphML format for graph analysis tools  
- `cytoscape` - Cytoscape.js format
- `d3` - D3.js format
- `gephi` - Gephi format

## ⚙️ Administrative Operations

### Health Check

Check system health and status.

```http
GET /admin/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-21T15:40:00Z",
  "services": {
    "database": {
      "status": "healthy",
      "response_time_ms": 45,
      "connection_pool": {
        "active": 8,
        "idle": 12,
        "max": 20
      }
    },
    "ollama": {
      "status": "healthy",
      "response_time_ms": 120,
      "models": {
        "bge-m3": "loaded",
        "qwen3": "loaded",
        "deepseek-r1": "loaded"
      }
    },
    "cache": {
      "status": "healthy",
      "memory_usage_mb": 1024,
      "hit_rate": 0.78
    }
  },
  "version": "1.0.0",
  "uptime_seconds": 3600000
}
```

### System Metrics

Get detailed system metrics.

```http
GET /admin/metrics
```

**Response:**
```json
{
  "timestamp": "2024-01-21T15:40:00Z",
  "system": {
    "cpu_usage_percent": 45.2,
    "memory_usage_percent": 68.5,
    "disk_usage_percent": 23.8,
    "network_io": {
      "bytes_sent": 15680000000,
      "bytes_received": 8940000000
    }
  },
  "application": {
    "requests_total": 125000,
    "requests_per_second": 12.5,
    "avg_response_time_ms": 850,
    "error_rate": 0.002,
    "active_connections": 45
  },
  "database": {
    "total_queries": 450000,
    "avg_query_time_ms": 25,
    "slow_queries": 12,
    "connections_active": 15,
    "cache_hit_rate": 0.82
  },
  "ollama": {
    "embedding_requests": 25000,
    "generation_requests": 8500,
    "avg_embedding_time_ms": 120,
    "avg_generation_time_ms": 680,
    "model_memory_usage_mb": {
      "bge-m3": 2048,
      "qwen3": 4096,
      "deepseek-r1": 6144
    }
  }
}
```

### Configuration

Get and update system configuration.

```http
GET /admin/config
```

```http
PUT /admin/config
```

### Backup Operations

Create and manage backups.

```http
POST /admin/backup
```

```http
GET /admin/backup/status
```

### Maintenance Mode

Enable/disable maintenance mode.

```http
POST /admin/maintenance
```

**Request Body:**
```json
{
  "enabled": true,
  "message": "System maintenance in progress. Estimated completion: 30 minutes.",
  "allowed_operations": ["read"]
}
```

## ❌ Error Handling

### Error Response Format

All API errors follow this format:

```json
{
  "error": "Error Type",
  "message": "Detailed error description",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-21T15:40:00Z",
  "details": {
    "field": "Additional error details",
    "suggestion": "How to fix the issue"
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Internal Server Error |
| 502 | Bad Gateway |
| 503 | Service Unavailable |

### Common Error Codes

- `INVALID_GRAPH_ID` - Knowledge graph not found
- `INVALID_ENTITY_ID` - Entity not found
- `INVALID_QUERY_FORMAT` - Malformed query
- `PROCESSING_FAILED` - Document processing failed
- `OLLAMA_UNAVAILABLE` - Ollama service unavailable
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INSUFFICIENT_PERMISSIONS` - Access denied
- `VALIDATION_ERROR` - Request validation failed

### Example Error Responses

**Validation Error (422):**
```json
{
  "error": "Validation Error",
  "message": "Request validation failed",
  "code": "VALIDATION_ERROR",
  "timestamp": "2024-01-21T15:40:00Z",
  "details": {
    "query_text": "Query text is required",
    "confidence_threshold": "Must be between 0.0 and 1.0"
  }
}
```

**Rate Limit Error (429):**
```json
{
  "error": "Rate Limit Exceeded",
  "message": "Too many requests. Rate limit: 100 requests per minute.",
  "code": "RATE_LIMIT_EXCEEDED",
  "timestamp": "2024-01-21T15:40:00Z",
  "details": {
    "limit": 100,
    "window": "60s",
    "retry_after": 45
  }
}
```

## ⏱️ Rate Limits

### Default Limits

| Endpoint Category | Requests per Minute | Burst Limit |
|------------------|-------------------|-------------|
| Query Processing | 100 | 20 |
| Document Processing | 20 | 5 |
| Entity/Relationship CRUD | 200 | 50 |
| Analytics | 60 | 15 |
| Visualization | 30 | 10 |
| Admin Operations | 10 | 2 |

### Rate Limit Headers

All responses include rate limit headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642788000
X-RateLimit-Retry-After: 60
```

### Handling Rate Limits

When rate limited, implement exponential backoff:

```python
import time
import requests

def api_request_with_retry(url, headers, data=None, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code != 429:
            return response
            
        retry_after = int(response.headers.get('X-RateLimit-Retry-After', 60))
        wait_time = min(retry_after, 2 ** attempt * 10)  # Exponential backoff
        time.sleep(wait_time)
    
    return response  # Return last response if all retries failed
```

## 🔧 SDK Examples

### Python SDK

```python
from apg_graphrag import GraphRAGClient

# Initialize client
client = GraphRAGClient(
    base_url="https://your-domain.com/api/v1/graphrag",
    api_token="your_api_token",
    tenant_id="your_tenant"
)

# Create knowledge graph
graph = await client.create_knowledge_graph(
    name="Business Intelligence",
    description="Graph for business analysis",
    domain="business"
)

# Process document
result = await client.process_document(
    graph_id=graph.id,
    title="Company Report",
    content="Document content...",
    source_type="text"
)

# Query the graph
response = await client.query(
    graph_id=graph.id,
    query_text="What companies has John Doe worked for?",
    query_type="factual"
)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence_score}")
```

### JavaScript SDK

```javascript
import { GraphRAGClient } from '@apg/graphrag-client';

// Initialize client
const client = new GraphRAGClient({
  baseUrl: 'https://your-domain.com/api/v1/graphrag',
  apiToken: 'your_api_token',
  tenantId: 'your_tenant'
});

// Create knowledge graph
const graph = await client.createKnowledgeGraph({
  name: 'Business Intelligence',
  description: 'Graph for business analysis',
  domain: 'business'
});

// Process document
const result = await client.processDocument({
  graphId: graph.id,
  title: 'Company Report',
  content: 'Document content...',
  sourceType: 'text'
});

// Query the graph
const response = await client.query({
  graphId: graph.id,
  queryText: 'What companies has John Doe worked for?',
  queryType: 'factual'
});

console.log(`Answer: ${response.answer}`);
console.log(`Confidence: ${response.confidenceScore}`);
```

---

For more examples and detailed implementation guides, see the [Examples Directory](./examples/) and [Developer Guide](./developer_guide.md).