# APG Document Management - API Reference

Complete REST API documentation for all 10 revolutionary document management capabilities.

## Base URL

```
https://api.apg.datacraft.co.ke/v1/document-management
```

## Authentication

All API endpoints require authentication using Bearer tokens:

```http
Authorization: Bearer <jwt_token>
```

## Core Document Management

### Create Document

Create a new document with optional AI processing.

```http
POST /documents
Content-Type: application/json

{
  "name": "Contract Agreement",
  "title": "Service Contract 2025", 
  "description": "Annual service contract",
  "document_type": "contract",
  "content_format": "pdf",
  "file_path": "/uploads/contract.pdf",
  "keywords": ["contract", "service", "2025"],
  "categories": ["legal", "agreements"],
  "process_ai": true
}
```

**Response:**
```json
{
  "message": "Document created successfully",
  "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
  "document": {
    "id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "name": "Contract Agreement",
    "title": "Service Contract 2025",
    "document_type": "contract",
    "content_format": "pdf",
    "created_at": "2025-01-29T10:30:00Z",
    "ai_tags": ["legal-document", "service-agreement", "annual-contract"],
    "content_summary": "Service contract outlining terms and conditions...",
    "sentiment_score": 0.1
  }
}
```

### Get Document

Retrieve document details with metadata.

```http
GET /documents/{document_id}
```

**Response:**
```json
{
  "document": {
    "id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "name": "Contract Agreement",
    "title": "Service Contract 2025",
    "description": "Annual service contract",
    "document_type": "contract",
    "content_format": "pdf",
    "file_size": 245760,
    "created_at": "2025-01-29T10:30:00Z",
    "updated_at": "2025-01-29T10:30:00Z",
    "keywords": ["contract", "service", "2025"],
    "ai_tags": ["legal-document", "service-agreement"],
    "content_summary": "Service contract outlining terms...",
    "view_count": 15,
    "download_count": 3
  },
  "metadata": {
    "permissions": [],
    "audit_log_count": 5,
    "comment_count": 2,
    "workflow_instances": []
  }
}
```

## Semantic Search

### Search Documents

Perform contextual and semantic document search.

```http
POST /search
Content-Type: application/json

{
  "query": "legal contracts and agreements from 2025",
  "options": {
    "semantic_search": true,
    "include_content": false,
    "limit": 50,
    "filters": {
      "document_types": ["contract", "agreement"],
      "date_from": "2025-01-01",
      "categories": ["legal"]
    }
  }
}
```

**Response:**
```json
{
  "search_result": {
    "id": "search_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "search_query": "legal contracts and agreements from 2025",
    "matching_documents": [
      "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
      "doc_01HK2N0XVYQ8R5N3JCPF7G4H6X"
    ],
    "semantic_similarity_scores": [0.95, 0.87],
    "intent_classification": {
      "intent": "find_legal_documents", 
      "confidence": 0.92
    },
    "search_time_ms": 245,
    "confidence_score": 0.88
  },
  "query": "legal contracts and agreements from 2025",
  "total_results": 2
}
```

## Generative AI Interaction

### Interact with Content

Use generative AI to interact with document content.

```http
POST /documents/{document_id}/interact
Content-Type: application/json

{
  "user_prompt": "Summarize the key terms and conditions of this contract",
  "interaction_type": "summarize",
  "context_documents": [],
  "options": {
    "max_length": 300,
    "summary_type": "executive"
  }
}
```

**Response:**
```json
{
  "interaction_result": {
    "id": "genai_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "interaction_type": "summarize",
    "user_prompt": "Summarize the key terms and conditions",
    "genai_response": "This service contract establishes a 12-month agreement between the parties with key terms including: monthly service fee of $5,000, 99.9% uptime guarantee, and termination clause with 30-day notice. The contract includes liability limitations and intellectual property protections.",
    "confidence_score": 0.91,
    "processing_time_ms": 1245,
    "model_version": "apg-genai-v1",
    "response_sources": [
      {
        "type": "document",
        "id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
        "relevance": 1.0
      }
    ]
  },
  "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
  "interaction_type": "summarize"
}
```

### Supported Interaction Types

- `summarize` - Generate document summaries
- `qa` - Answer questions about content
- `translate` - Translate content to different languages
- `enhance` - Improve content quality
- `generate` - Generate new content based on templates
- `extract` - Extract specific information
- `compare` - Compare multiple documents
- `analyze` - Analyze content for insights

## Document Classification

### Classify Document

Trigger AI-driven document classification and metadata extraction.

```http
POST /documents/{document_id}/classify
Content-Type: application/json

{
  "content_text": "This Service Agreement is entered into...",
  "extracted_data": {
    "form_fields": {},
    "metadata": {}
  }
}
```

**Response:**
```json
{
  "classification_result": {
    "id": "intel_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "analysis_type": "comprehensive_classification",
    "ai_classification": {
      "document_type": {
        "primary_type": "contract",
        "confidence": 0.95,
        "alternatives": [
          {"type": "agreement", "confidence": 0.87}
        ]
      },
      "content_category": {
        "primary_category": "legal",
        "confidence": 0.92
      },
      "industry_context": {
        "primary_industry": "technology",
        "confidence": 0.78
      }
    },
    "entity_extraction": [
      {
        "text": "$5,000",
        "label": "monetary_amount",
        "confidence": 0.98
      },
      {
        "text": "12-month",
        "label": "duration",
        "confidence": 0.95
      }
    ],
    "content_summary": "Service contract with defined terms, pricing, and obligations",
    "related_concepts": ["contract", "service-agreement", "legal-document"],
    "compliance_flags": ["SOX_APPLICABLE", "GDPR_APPLICABLE"],
    "sensitive_data_detected": false,
    "content_quality_score": 0.89,
    "readability_score": 0.75
  }
}
```

## Retention Management

### Analyze Document Retention

Analyze document retention requirements using smart policies.

```http
POST /documents/{document_id}/retention-analysis
Content-Type: application/json

{
  "content_intelligence": {
    "compliance_flags": ["SOX_APPLICABLE"],
    "sensitive_data_detected": false,
    "document_type": "contract"
  }
}
```

**Response:**
```json
{
  "retention_analysis": {
    "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "content_analysis": {
      "document_type": "contract",
      "business_records": true,
      "legal_content": true,
      "sensitive_data_present": false
    },
    "regulatory_requirements": {
      "applicable_regulations": [
        {
          "regulation": "SOX",
          "retention_days": 2555,
          "requirements": {
            "audit_trail_required": true,
            "immutable_storage": true
          }
        }
      ],
      "minimum_retention_days": 2555
    },
    "retention_recommendation": {
      "action": "archive",
      "retention_days": 2555,
      "disposition_date": "2032-01-29",
      "confidence": 0.9,
      "factors_considered": [
        "regulatory_minimum",
        "document_type_contract",
        "high_legal_significance"
      ]
    }
  }
}
```

### Apply Retention Policy

Apply retention policy to selected documents.

```http
POST /retention-policies/{policy_id}/apply
Content-Type: application/json

{
  "document_ids": [
    "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "doc_01HK2N0XVYQ8R5N3JCPF7G4H6X"
  ]
}
```

**Response:**
```json
{
  "policy_result": {
    "policy_id": "policy_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "documents_processed": 2,
    "actions_taken": [
      {
        "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
        "action": "archive",
        "status": "completed",
        "timestamp": "2025-01-29T10:45:00Z"
      },
      {
        "document_id": "doc_01HK2N0XVYQ8R5N3JCPF7G4H6X", 
        "action": "archive",
        "status": "completed",
        "timestamp": "2025-01-29T10:45:01Z"
      }
    ],
    "errors": [],
    "summary": {
      "success_rate": 1.0,
      "total_documents": 2,
      "successful_actions": 2,
      "failed_actions": 0
    }
  }
}
```

## Blockchain Provenance

### Verify Document Provenance

Verify document integrity using blockchain.

```http
GET /documents/{document_id}/provenance
```

**Response:**
```json
{
  "provenance_verification": {
    "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "verified": true,
    "transaction_hash": "0x1234567890abcdef...",
    "block_number": 12345678,
    "timestamp": "2025-01-29T10:30:00Z",
    "integrity_status": "verified",
    "provenance_chain": [
      {
        "action": "created",
        "timestamp": "2025-01-29T10:30:00Z",
        "user": "user_123",
        "hash": "0xabcdef..."
      },
      {
        "action": "classified",
        "timestamp": "2025-01-29T10:30:15Z",
        "system": "ai_classification",
        "hash": "0xbcdef1..."
      }
    ]
  }
}
```

## Predictive Analytics

### Generate Predictions

Generate predictive analytics for document value and risk.

```http
POST /documents/{document_id}/predictions
Content-Type: application/json

{
  "content_intelligence": {
    "ai_classification": {...},
    "compliance_flags": ["SOX_APPLICABLE"]
  },
  "prediction_types": ["value", "risk", "usage", "lifecycle"]
}
```

**Response:**
```json
{
  "prediction_result": {
    "id": "pred_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "prediction_type": "comprehensive",
    "model_version": "apg-predictive-v1",
    "content_value_score": 0.85,
    "business_impact_score": 0.78,
    "risk_probability": {
      "compliance_violation": 0.12,
      "security_breach": 0.08,
      "obsolescence": 0.25,
      "legal_exposure": 0.15
    },
    "compliance_risk_score": 0.12,
    "obsolescence_probability": 0.25,
    "expected_lifespan_days": 2555,
    "next_review_prediction": "2026-01-29",
    "archival_recommendation": "2032-01-29",
    "prediction_confidence": 0.82,
    "validation_status": "pending"
  }
}
```

## OCR (Optical Character Recognition)

### Process Document OCR

Process a document using OCR capabilities to extract text.

```http
POST /documents/{document_id}/ocr
Content-Type: application/json

{
  "options": {
    "language": "eng",
    "preprocessing": {
      "grayscale": true,
      "denoise": true,
      "enhance_contrast": true
    },
    "ai_enhancement": true
  }
}
```

**Response:**
```json
{
  "message": "OCR processing completed successfully",
  "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
  "ocr_result": {
    "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "total_pages": 3,
    "combined_text": "This is the extracted text content from the document...",
    "combined_confidence": 0.95,
    "processing_summary": {
      "total_words": 1247,
      "total_characters": 7832,
      "total_lines": 156,
      "average_confidence": 0.95,
      "languages_detected": ["eng"],
      "total_processing_time_ms": 4580
    },
    "ai_enhancements": {
      "corrected_text": "This is the AI-corrected text...",
      "entities": [
        {
          "text": "John Doe",
          "label": "person",
          "confidence": 0.98
        }
      ],
      "classification": {
        "document_type": "contract",
        "confidence": 0.92
      }
    }
  }
}
```

### Batch OCR Processing

Process multiple documents with OCR in batch.

```http
POST /ocr/batch
Content-Type: application/json

{
  "document_ids": [
    "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "doc_01HK2N0XVYQ8R5N3JCPF7G4H6X",
    "doc_01HK2N1XVYQ8R5N3JCPF7G4H6Y"
  ],
  "batch_name": "Legal Documents Batch",
  "options": {
    "language": "eng",
    "ai_enhancement": true,
    "parallel_processing": true
  }
}
```

**Response:**
```json
{
  "message": "Batch OCR processing initiated",
  "batch_result": {
    "batch_name": "Legal Documents Batch",
    "total_documents": 3,
    "processed_documents": 3,
    "successful_documents": 3,
    "failed_documents": 0,
    "results": {
      "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W": {
        "combined_text": "Contract text content...",
        "combined_confidence": 0.96
      }
    },
    "errors": {}
  }
}
```

### Get OCR Results

Retrieve OCR results for a processed document.

```http
GET /documents/{document_id}/ocr
```

**Response:**
```json
{
  "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
  "ocr_result": {
    "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "status": "completed",
    "text_content": "Extracted text content...",
    "confidence_score": 0.95,
    "processing_time_ms": 2340,
    "pages_processed": 1,
    "language_detected": "eng"
  }
}
```

### Get Supported OCR Languages

Get list of supported OCR languages.

```http
GET /ocr/languages
```

**Response:**
```json
{
  "supported_languages": [
    "eng", "fra", "deu", "spa", "ita", "por", "rus", "chi_sim", "jpn", "ara"
  ],
  "language_codes": {
    "eng": "English",
    "fra": "French",
    "deu": "German",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "rus": "Russian",
    "chi_sim": "Chinese (Simplified)",
    "jpn": "Japanese",
    "ara": "Arabic"
  }
}
```

### Update OCR Configuration

Update OCR engine configuration settings.

```http
PUT /ocr/configuration/{config_name}
Content-Type: application/json

{
  "tesseract_cmd": "tesseract",
  "languages": ["eng", "fra", "deu"],
  "dpi": 300,
  "enable_preprocessing": true,
  "max_image_size": 4096,
  "confidence_threshold": 0.5,
  "parallel_processing": true,
  "max_workers": 4
}
```

**Response:**
```json
{
  "message": "OCR configuration updated successfully",
  "configuration": {
    "config_name": "default_ocr_config",
    "status": "updated",
    "updated_at": "2025-01-29T11:30:00Z"
  }
}
```

### Get OCR Analytics

Retrieve OCR processing analytics and metrics.

```http
GET /ocr/analytics
```

**Response:**
```json
{
  "ocr_analytics": {
    "ocr_operations_performed": 15847,
    "average_processing_time_ms": 2340,
    "average_confidence_score": 0.92,
    "most_common_language": "eng",
    "success_rate": 0.98
  },
  "generated_at": "2025-01-29T11:30:00Z"
}
```

## Analytics & Monitoring

### Get Comprehensive Analytics

Retrieve analytics across all capabilities including OCR.

```http
GET /analytics/comprehensive
```

**Response:**
```json
{
  "comprehensive_analytics": {
    "service_statistics": {
      "documents_processed": 15847,
      "ai_operations_performed": 12456,
      "search_queries_handled": 35621,
      "retention_policies_applied": 2341
    },
    "idp_analytics": {
      "documents_processed": 15847,
      "accuracy_rate": 0.987,
      "processing_time_avg_ms": 2340
    },
    "search_analytics": {
      "searches_performed": 35621,
      "average_response_time_ms": 245,
      "success_rate": 0.94
    },
    "classification_analytics": {
      "classifications_performed": 15234,
      "average_confidence": 0.89,
      "accuracy_score": 0.92
    },
    "genai_analytics": {
      "interactions_performed": 8765,
      "average_response_time_ms": 1850,
      "user_satisfaction_avg": 4.2
    }
  },
  "generated_at": "2025-01-29T11:00:00Z"
}
```

### Health Check

Check system health across all services.

```http
GET /health
```

**Response:**
```json
{
  "overall_status": "healthy",
  "services": {
    "idp_processor": "healthy",
    "search_engine": "healthy", 
    "classification_engine": "healthy",
    "retention_engine": "healthy",
    "genai_engine": "healthy",
    "predictive_engine": "healthy"
  },
  "apg_integrations": {
    "ai_client": "connected",
    "rag_client": "connected",
    "genai_client": "connected",
    "ml_client": "connected",
    "blockchain_client": "connected"
  },
  "statistics": {
    "documents_processed": 15847,
    "ai_operations_performed": 12456,
    "uptime_hours": 720
  },
  "timestamp": "2025-01-29T11:00:00Z"
}
```

## Error Responses

All endpoints may return these error responses:

### 400 Bad Request
```json
{
  "error": "Missing required fields",
  "missing_fields": ["name", "title"]
}
```

### 401 Unauthorized
```json
{
  "error": "Authentication required"
}
```

### 403 Forbidden
```json
{
  "error": "Insufficient permissions"
}
```

### 404 Not Found
```json
{
  "error": "Document not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "request_id": "req_01HK2MZXVYQ8R5N3JCPF7G4H6W"
}
```

## Rate Limits

- **Standard**: 1000 requests/hour per user
- **Search**: 500 requests/hour per user  
- **AI Operations**: 100 requests/hour per user
- **Bulk Operations**: 50 requests/hour per user

Rate limit headers are included in all responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1643723400
```

## SDK Examples

### Python
```python
import apg_docmgmt

client = apg_docmgmt.Client(
    api_key="your_api_key",
    base_url="https://api.apg.datacraft.co.ke/v1"
)

# Create document
document = await client.documents.create(
    name="Contract Agreement",
    file_path="/path/to/contract.pdf",
    process_ai=True
)

# Search documents
results = await client.search.documents(
    query="legal contracts 2025",
    semantic_search=True
)

# GenAI interaction
summary = await client.genai.interact(
    document_id=document.id,
    prompt="Summarize key terms",
    interaction_type="summarize"
)
```

### JavaScript
```javascript
import APGDocMgmt from '@apg/document-management';

const client = new APGDocMgmt({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.apg.datacraft.co.ke/v1'
});

// Create document
const document = await client.documents.create({
  name: 'Contract Agreement',
  filePath: '/path/to/contract.pdf',
  processAI: true
});

// Search documents
const results = await client.search.documents({
  query: 'legal contracts 2025',
  semanticSearch: true
});

// GenAI interaction
const summary = await client.genai.interact({
  documentId: document.id,
  prompt: 'Summarize key terms',
  interactionType: 'summarize'
});
```

## Webhooks

Configure webhooks to receive real-time notifications:

```http
POST /webhooks
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/docmgmt",
  "events": [
    "document.created",
    "document.classified", 
    "retention.policy_applied",
    "dlp.alert_triggered"
  ],
  "secret": "your_webhook_secret"
}
```

Webhook payload example:
```json
{
  "event": "document.created",
  "timestamp": "2025-01-29T10:30:00Z",
  "data": {
    "document_id": "doc_01HK2MZXVYQ8R5N3JCPF7G4H6W",
    "name": "Contract Agreement",
    "tenant_id": "tenant_123",
    "created_by": "user_123"
  }
}
```

---

For more detailed information, see the complete [API Documentation](https://docs.apg.datacraft.co.ke/document-management/api).