# Computer Vision & Visual Intelligence - API Reference

**Version:** 1.0.0  
**Base URL:** `https://your-apg-instance.com/api/v1`  
**Authentication:** Bearer Token (JWT)  
**Content-Type:** `application/json` for requests, `multipart/form-data` for file uploads  

---

## Table of Contents

1. [Authentication](#authentication)
2. [Common Parameters](#common-parameters)
3. [Error Handling](#error-handling)
4. [Rate Limiting](#rate-limiting)
5. [Document Processing Endpoints](#document-processing-endpoints)
6. [Image Analysis Endpoints](#image-analysis-endpoints)
7. [Quality Control Endpoints](#quality-control-endpoints)
8. [Video Processing Endpoints](#video-processing-endpoints)
9. [Job Management Endpoints](#job-management-endpoints)
10. [Model Management Endpoints](#model-management-endpoints)
11. [Analytics Endpoints](#analytics-endpoints)
12. [Webhook Configuration](#webhook-configuration)
13. [SDK Examples](#sdk-examples)

---

## Authentication

All API requests require authentication using a Bearer token in the Authorization header.

```http
GET /api/v1/documents/ocr
Authorization: Bearer YOUR_JWT_TOKEN
X-Tenant-ID: your-tenant-id
```

### Obtaining Access Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
    "username": "your-username",
    "password": "your-password",
    "tenant_id": "your-tenant-id"
}
```

**Response:**
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 3600,
    "refresh_token": "refresh_token_here"
}
```

---

## Common Parameters

### Headers
- `Authorization: Bearer {token}` - Required for all endpoints
- `X-Tenant-ID: {tenant_id}` - Required for multi-tenant deployments
- `Content-Type: application/json` - For JSON payloads
- `Content-Type: multipart/form-data` - For file uploads

### Query Parameters
- `page: integer` - Page number for paginated results (default: 1)
- `limit: integer` - Items per page (default: 20, max: 100)
- `sort: string` - Sort field (e.g., "created_at", "-name" for descending)
- `filter: string` - Filter expression for results

### Response Format
All successful responses follow this structure:
```json
{
    "success": true,
    "data": { ... },
    "metadata": {
        "timestamp": "2025-01-27T12:00:00Z",
        "version": "1.0.0",
        "request_id": "req_123456789"
    }
}
```

---

## Error Handling

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `413` - Payload Too Large
- `422` - Validation Error
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

### Error Response Format
```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "File size exceeds maximum allowed limit",
        "details": {
            "field": "file",
            "max_size": "50MB",
            "received_size": "75MB"
        }
    },
    "metadata": {
        "timestamp": "2025-01-27T12:00:00Z",
        "request_id": "req_123456789"
    }
}
```

### Common Error Codes
- `AUTHENTICATION_FAILED` - Invalid or expired token
- `INSUFFICIENT_PERMISSIONS` - Missing required permissions
- `VALIDATION_ERROR` - Request validation failed
- `FILE_TOO_LARGE` - File exceeds size limits
- `UNSUPPORTED_FORMAT` - File format not supported
- `PROCESSING_FAILED` - AI processing error
- `QUOTA_EXCEEDED` - Usage quota exceeded
- `RATE_LIMIT_EXCEEDED` - Too many requests

---

## Rate Limiting

**Default Limits:**
- 1000 requests per minute per tenant
- 100 burst requests allowed
- File upload endpoints: 10 requests per minute

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1640995200
```

---

## Document Processing Endpoints

### Extract Text from Document (OCR)

Extract text content from documents and images using OCR.

```http
POST /api/v1/documents/ocr
```

**Request:**
```http
Content-Type: multipart/form-data

file: [binary file data]
language: "eng" (optional, default: "auto")
ocr_engine: "tesseract" (optional, default: "tesseract")
enhance_image: true (optional, default: false)
extract_tables: true (optional, default: false)
extract_forms: true (optional, default: false)
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "extracted_text": "This is the extracted text content...",
        "language_detected": "eng",
        "confidence_score": 0.95,
        "word_count": 245,
        "processing_time_ms": 1200,
        "pages": [
            {
                "page_number": 1,
                "text": "Page 1 content...",
                "confidence": 0.96
            }
        ],
        "tables": [
            {
                "table_id": "table_1",
                "rows": 5,
                "columns": 3,
                "data": [...]
            }
        ],
        "form_fields": [
            {
                "field_name": "customer_name",
                "field_value": "John Doe",
                "confidence": 0.92
            }
        ]
    }
}
```

### Comprehensive Document Analysis

Perform comprehensive analysis including layout, classification, and entity extraction.

```http
POST /api/v1/documents/analyze
```

**Request:**
```http
Content-Type: multipart/form-data

file: [binary file data]
analysis_level: "detailed" (basic|standard|detailed)
include_layout: true
include_classification: true
include_entities: true
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "extracted_text": "Document text content...",
        "document_classification": {
            "type": "invoice",
            "confidence": 0.88,
            "subtype": "commercial_invoice"
        },
        "layout_analysis": {
            "page_count": 2,
            "text_blocks": 15,
            "images": 2,
            "tables": 1,
            "reading_order": [...]
        },
        "entities": [
            {
                "type": "date",
                "value": "2025-01-27",
                "confidence": 0.95,
                "position": {"x": 100, "y": 50}
            },
            {
                "type": "currency",
                "value": "$1,234.56",
                "confidence": 0.98,
                "position": {"x": 200, "y": 150}
            }
        ],
        "key_value_pairs": [
            {
                "key": "Invoice Number",
                "value": "INV-2025-001",
                "confidence": 0.92
            }
        ]
    }
}
```

---

## Image Analysis Endpoints

### Object Detection

Detect and classify objects in images using YOLO models.

```http
POST /api/v1/images/detect-objects
```

**Request:**
```http
Content-Type: multipart/form-data

file: [binary file data]
model: "yolov8n" (optional, default: "yolov8n")
confidence_threshold: 0.5 (optional, default: 0.5)
max_detections: 100 (optional, default: 100)
classes: ["person", "car"] (optional, detect specific classes)
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "image_info": {
            "width": 1920,
            "height": 1080,
            "format": "JPEG",
            "size_bytes": 245760
        },
        "detected_objects": [
            {
                "object_id": "obj_1",
                "class_name": "person",
                "class_id": 0,
                "confidence": 0.95,
                "bounding_box": {
                    "x": 100,
                    "y": 200,
                    "width": 150,
                    "height": 300
                },
                "area_pixels": 45000,
                "center_point": {"x": 175, "y": 350}
            },
            {
                "object_id": "obj_2",
                "class_name": "car",
                "class_id": 2,
                "confidence": 0.87,
                "bounding_box": {
                    "x": 500,
                    "y": 400,
                    "width": 300,
                    "height": 150
                },
                "area_pixels": 45000,
                "center_point": {"x": 650, "y": 475}
            }
        ],
        "total_objects": 2,
        "objects_by_class": {
            "person": 1,
            "car": 1
        },
        "detection_confidence": 0.91,
        "processing_time_ms": 340,
        "model_used": "yolov8n",
        "model_version": "8.0.0"
    }
}
```

### Image Classification

Classify images using Vision Transformer or CNN models.

```http
POST /api/v1/images/classify
```

**Request:**
```http
Content-Type: multipart/form-data

file: [binary file data]
model: "vit_base_patch16_224" (optional)
top_k: 5 (optional, default: 5)
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "predictions": [
            {
                "class_name": "golden_retriever",
                "class_id": 207,
                "confidence": 0.96,
                "probability": 0.96
            },
            {
                "class_name": "labrador_retriever",
                "class_id": 208,
                "confidence": 0.02,
                "probability": 0.02
            }
        ],
        "top_prediction": {
            "class_name": "golden_retriever",
            "confidence": 0.96
        },
        "processing_time_ms": 180,
        "model_used": "vit_base_patch16_224",
        "image_info": {
            "width": 224,
            "height": 224,
            "format": "JPEG"
        }
    }
}
```

### Visual Similarity Search

Find visually similar images in your database.

```http
POST /api/v1/images/similarity-search
```

**Request:**
```http
Content-Type: multipart/form-data

file: [binary file data]
similarity_threshold: 0.7 (optional, default: 0.8)
max_results: 10 (optional, default: 10)
search_database: "product_images" (optional)
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "query_image": {
            "width": 800,
            "height": 600,
            "format": "JPEG"
        },
        "similar_images": [
            {
                "image_id": "img_456",
                "similarity_score": 0.94,
                "file_path": "/storage/similar_1.jpg",
                "metadata": {
                    "uploaded_at": "2025-01-25T10:30:00Z",
                    "tags": ["product", "electronics"]
                }
            },
            {
                "image_id": "img_789",
                "similarity_score": 0.87,
                "file_path": "/storage/similar_2.jpg",
                "metadata": {
                    "uploaded_at": "2025-01-24T15:45:00Z",
                    "tags": ["product", "gadget"]
                }
            }
        ],
        "total_matches": 2,
        "processing_time_ms": 250
    }
}
```

---

## Quality Control Endpoints

### Quality Inspection

Perform quality control inspection on product images.

```http
POST /api/v1/quality/inspect
```

**Request:**
```http
Content-Type: multipart/form-data

file: [binary file data]
inspection_type: "defect_detection" (defect_detection|surface_inspection|component_check)
product_type: "electronics" (optional)
sensitivity: "medium" (low|medium|high)
pass_threshold: 0.8 (optional, default: 0.8)
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "inspection_result": {
            "pass_fail_status": "PASS",
            "overall_score": 0.92,
            "inspection_confidence": 0.95
        },
        "defects_detected": [
            {
                "defect_id": "def_1",
                "defect_type": "scratch",
                "severity": "MINOR",
                "confidence": 0.78,
                "location": {
                    "x": 150,
                    "y": 200,
                    "width": 5,
                    "height": 2
                },
                "description": "Minor surface scratch detected",
                "recommended_action": "acceptable_for_shipping"
            }
        ],
        "defect_summary": {
            "total_defects": 1,
            "critical_defects": 0,
            "major_defects": 0,
            "minor_defects": 1
        },
        "quality_metrics": {
            "surface_quality": 0.95,
            "dimensional_accuracy": 0.98,
            "color_consistency": 0.89
        },
        "processing_time_ms": 890,
        "model_used": "defect_detector_v1",
        "inspection_timestamp": "2025-01-27T12:00:00Z"
    }
}
```

### Batch Quality Inspection

Process multiple images for quality control.

```http
POST /api/v1/quality/batch-inspect
```

**Request:**
```http
Content-Type: multipart/form-data

files: [multiple binary files]
inspection_type: "defect_detection"
batch_name: "production_lot_2025_001"
auto_sort: true (optional, default: false)
```

**Response:**
```json
{
    "success": true,
    "data": {
        "batch_id": "batch_123456789",
        "batch_summary": {
            "total_items": 50,
            "passed_items": 47,
            "failed_items": 3,
            "pass_rate": 0.94,
            "average_score": 0.91
        },
        "individual_results": [
            {
                "file_name": "item_001.jpg",
                "status": "PASS",
                "score": 0.95,
                "defects": 0
            },
            {
                "file_name": "item_002.jpg",
                "status": "FAIL",
                "score": 0.65,
                "defects": 2
            }
        ],
        "failed_items": [
            {
                "file_name": "item_002.jpg",
                "defects": [
                    {
                        "type": "crack",
                        "severity": "MAJOR"
                    }
                ]
            }
        ],
        "processing_time_ms": 15000
    }
}
```

---

## Video Processing Endpoints

### Video Analysis

Analyze video content for objects, actions, and events.

```http
POST /api/v1/video/analyze
```

**Request:**
```http
Content-Type: multipart/form-data

file: [binary video file]
analysis_type: "object_tracking" (object_tracking|action_recognition|motion_detection)
frame_rate: 1 (optional, frames per second to analyze)
start_time: 0 (optional, start time in seconds)
duration: 60 (optional, duration to analyze in seconds)
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "video_info": {
            "duration_seconds": 120,
            "fps": 30,
            "resolution": "1920x1080",
            "format": "mp4",
            "size_bytes": 15728640
        },
        "analysis_results": {
            "frames_analyzed": 120,
            "objects_tracked": [
                {
                    "object_id": "track_1",
                    "class_name": "person",
                    "first_appearance": 2.5,
                    "last_appearance": 45.2,
                    "confidence": 0.92,
                    "trajectory": [
                        {"time": 2.5, "x": 100, "y": 200},
                        {"time": 3.0, "x": 120, "y": 210}
                    ]
                }
            ],
            "actions_detected": [
                {
                    "action": "walking",
                    "start_time": 2.5,
                    "end_time": 15.8,
                    "confidence": 0.89,
                    "person_id": "track_1"
                }
            ],
            "events": [
                {
                    "event_type": "person_entered_frame",
                    "timestamp": 2.5,
                    "confidence": 0.95
                }
            ]
        },
        "processing_time_ms": 45000,
        "model_used": "yolov8_tracking"
    }
}
```

### Extract Video Frames

Extract specific frames or frame sequences from video.

```http
POST /api/v1/video/extract-frames
```

**Request:**
```http
Content-Type: multipart/form-data

file: [binary video file]
extraction_method: "interval" (interval|timestamps|scenes)
interval_seconds: 5 (optional, for interval method)
timestamps: [10, 20, 30] (optional, for timestamp method)
max_frames: 50 (optional, default: 100)
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "extracted_frames": [
            {
                "frame_number": 150,
                "timestamp": 5.0,
                "file_path": "/storage/frames/frame_150.jpg",
                "width": 1920,
                "height": 1080
            },
            {
                "frame_number": 300,
                "timestamp": 10.0,
                "file_path": "/storage/frames/frame_300.jpg",
                "width": 1920,
                "height": 1080
            }
        ],
        "total_frames": 2,
        "processing_time_ms": 3500
    }
}
```

---

## Job Management Endpoints

### Get Job Status

Check the status of a processing job.

```http
GET /api/v1/jobs/{job_id}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "job_name": "Document OCR Processing",
        "status": "COMPLETED",
        "progress_percentage": 100,
        "created_at": "2025-01-27T12:00:00Z",
        "started_at": "2025-01-27T12:00:05Z",
        "completed_at": "2025-01-27T12:00:07Z",
        "processing_time_ms": 1200,
        "processing_type": "OCR",
        "content_type": "DOCUMENT",
        "input_file_path": "/uploads/document.pdf",
        "results": {
            "extracted_text": "Document content...",
            "confidence_score": 0.95
        },
        "error_message": null,
        "retry_count": 0
    }
}
```

### List Jobs

Get a list of processing jobs with filtering and pagination.

```http
GET /api/v1/jobs?status=COMPLETED&limit=20&page=1
```

**Query Parameters:**
- `status` - Filter by job status (PENDING|PROCESSING|COMPLETED|FAILED|CANCELLED)
- `processing_type` - Filter by processing type (OCR|OBJECT_DETECTION|etc.)
- `created_after` - ISO timestamp to filter jobs created after
- `created_before` - ISO timestamp to filter jobs created before

**Response:**
```json
{
    "success": true,
    "data": {
        "jobs": [
            {
                "job_id": "job_123456789",
                "job_name": "Document OCR",
                "status": "COMPLETED",
                "processing_type": "OCR",
                "created_at": "2025-01-27T12:00:00Z",
                "completed_at": "2025-01-27T12:00:07Z"
            }
        ],
        "pagination": {
            "current_page": 1,
            "total_pages": 5,
            "total_items": 98,
            "items_per_page": 20
        }
    }
}
```

### Cancel Job

Cancel a pending or processing job.

```http
DELETE /api/v1/jobs/{job_id}/cancel
```

**Response:**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456789",
        "status": "CANCELLED",
        "cancelled_at": "2025-01-27T12:00:00Z",
        "message": "Job cancelled successfully"
    }
}
```

---

## Model Management Endpoints

### List Available Models

Get information about available AI models.

```http
GET /api/v1/models
```

**Response:**
```json
{
    "success": true,
    "data": {
        "models": [
            {
                "model_id": "yolov8n",
                "model_name": "YOLOv8 Nano",
                "model_type": "object_detection",
                "version": "8.0.0",
                "description": "Fast, lightweight object detection",
                "input_size": [640, 640],
                "classes": 80,
                "performance": {
                    "speed_ms": 340,
                    "accuracy_map": 0.53
                },
                "status": "active"
            },
            {
                "model_id": "vit_base_patch16_224",
                "model_name": "Vision Transformer Base",
                "model_type": "image_classification",
                "version": "1.0.0",
                "description": "State-of-the-art image classification",
                "input_size": [224, 224],
                "classes": 1000,
                "performance": {
                    "speed_ms": 180,
                    "accuracy_top1": 0.81
                },
                "status": "active"
            }
        ]
    }
}
```

### Get Model Details

Get detailed information about a specific model.

```http
GET /api/v1/models/{model_id}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "model_id": "yolov8n",
        "model_name": "YOLOv8 Nano",
        "model_type": "object_detection",
        "version": "8.0.0",
        "description": "Fast, lightweight object detection model optimized for real-time inference",
        "architecture": "YOLO",
        "framework": "PyTorch",
        "input_requirements": {
            "size": [640, 640],
            "format": "RGB",
            "normalization": "0-1"
        },
        "output_format": {
            "boxes": "xyxy",
            "confidence": "float",
            "classes": "int"
        },
        "performance_metrics": {
            "speed_ms": 340,
            "accuracy_map": 0.53,
            "memory_usage_mb": 6.2
        },
        "supported_classes": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "bicycle"},
            {"id": 2, "name": "car"}
        ],
        "deployment_info": {
            "status": "active",
            "deployed_at": "2025-01-20T10:00:00Z",
            "replicas": 3,
            "resource_usage": {
                "cpu": "500m",
                "memory": "1Gi",
                "gpu": false
            }
        }
    }
}
```

---

## Analytics Endpoints

### Processing Statistics

Get processing statistics and analytics.

```http
GET /api/v1/analytics/processing-stats
```

**Query Parameters:**
- `start_date` - Start date for statistics (ISO format)
- `end_date` - End date for statistics (ISO format)
- `granularity` - Data granularity (hour|day|week|month)

**Response:**
```json
{
    "success": true,
    "data": {
        "summary": {
            "total_jobs": 1250,
            "successful_jobs": 1198,
            "failed_jobs": 52,
            "success_rate": 0.958,
            "average_processing_time_ms": 1450
        },
        "by_type": {
            "OCR": {
                "count": 750,
                "success_rate": 0.96,
                "avg_time_ms": 1200
            },
            "OBJECT_DETECTION": {
                "count": 300,
                "success_rate": 0.95,
                "avg_time_ms": 340
            },
            "IMAGE_CLASSIFICATION": {
                "count": 200,
                "success_rate": 0.97,
                "avg_time_ms": 180
            }
        },
        "time_series": [
            {
                "timestamp": "2025-01-27T00:00:00Z",
                "jobs": 45,
                "success_rate": 0.96
            },
            {
                "timestamp": "2025-01-27T01:00:00Z",
                "jobs": 52,
                "success_rate": 0.94
            }
        ]
    }
}
```

### Usage Analytics

Get usage analytics for billing and monitoring.

```http
GET /api/v1/analytics/usage
```

**Response:**
```json
{
    "success": true,
    "data": {
        "current_period": {
            "start_date": "2025-01-01T00:00:00Z",
            "end_date": "2025-01-31T23:59:59Z",
            "processing_jobs": 1250,
            "api_requests": 15600,
            "data_processed_gb": 45.2,
            "storage_used_gb": 125.8
        },
        "quotas": {
            "monthly_processing_limit": 10000,
            "monthly_api_limit": 100000,
            "storage_limit_gb": 500
        },
        "usage_by_feature": {
            "ocr": {
                "jobs": 750,
                "data_gb": 25.1
            },
            "object_detection": {
                "jobs": 300,
                "data_gb": 12.5
            },
            "quality_control": {
                "jobs": 200,
                "data_gb": 7.6
            }
        }
    }
}
```

---

## Webhook Configuration

### Register Webhook

Register a webhook endpoint to receive processing notifications.

```http
POST /api/v1/webhooks
```

**Request:**
```json
{
    "url": "https://your-app.com/webhooks/computer-vision",
    "events": [
        "job.completed",
        "job.failed",
        "batch.completed"
    ],
    "secret": "your-webhook-secret",
    "active": true
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "webhook_id": "webhook_123456789",
        "url": "https://your-app.com/webhooks/computer-vision",
        "events": [
            "job.completed",
            "job.failed",
            "batch.completed"
        ],
        "created_at": "2025-01-27T12:00:00Z",
        "active": true
    }
}
```

### Webhook Events

**Available Events:**
- `job.created` - New processing job created
- `job.started` - Job processing started
- `job.completed` - Job completed successfully
- `job.failed` - Job processing failed
- `job.cancelled` - Job was cancelled
- `batch.completed` - Batch processing completed
- `model.deployed` - New model deployed
- `quota.warning` - Usage quota warning (80%)
- `quota.exceeded` - Usage quota exceeded

**Webhook Payload Example:**
```json
{
    "event": "job.completed",
    "timestamp": "2025-01-27T12:00:00Z",
    "data": {
        "job_id": "job_123456789",
        "job_name": "Document OCR Processing",
        "status": "COMPLETED",
        "processing_type": "OCR",
        "results": {
            "extracted_text": "Document content...",
            "confidence_score": 0.95
        }
    },
    "webhook_id": "webhook_123456789"
}
```

---

## SDK Examples

### Python SDK

```python
import asyncio
from apg_computer_vision import ComputerVisionClient

# Initialize client
client = ComputerVisionClient(
    api_key="your-api-key",
    base_url="https://your-apg-instance.com",
    tenant_id="your-tenant-id"
)

async def main():
    # OCR processing
    ocr_result = await client.documents.extract_text(
        file_path="document.pdf",
        language="eng",
        enhance_image=True
    )
    print(f"Extracted: {ocr_result.extracted_text}")
    
    # Object detection
    detection_result = await client.images.detect_objects(
        file_path="image.jpg",
        model="yolov8n",
        confidence_threshold=0.5
    )
    print(f"Found {len(detection_result.detected_objects)} objects")
    
    # Quality control
    qc_result = await client.quality.inspect(
        file_path="product.jpg",
        inspection_type="defect_detection"
    )
    print(f"Quality status: {qc_result.pass_fail_status}")

asyncio.run(main())
```

### JavaScript/Node.js SDK

```javascript
const { ComputerVisionClient } = require('@datacraft/apg-computer-vision');

const client = new ComputerVisionClient({
    apiKey: 'your-api-key',
    baseUrl: 'https://your-apg-instance.com',
    tenantId: 'your-tenant-id'
});

async function main() {
    // OCR processing
    const ocrResult = await client.documents.extractText({
        filePath: 'document.pdf',
        language: 'eng',
        enhanceImage: true
    });
    console.log(`Extracted: ${ocrResult.extractedText}`);
    
    // Object detection
    const detectionResult = await client.images.detectObjects({
        filePath: 'image.jpg',
        model: 'yolov8n',
        confidenceThreshold: 0.5
    });
    console.log(`Found ${detectionResult.detectedObjects.length} objects`);
}

main().catch(console.error);
```

### cURL Examples

**OCR Processing:**
```bash
curl -X POST \
  https://your-apg-instance.com/api/v1/documents/ocr \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "X-Tenant-ID: your-tenant-id" \
  -F "file=@document.pdf" \
  -F "language=eng" \
  -F "enhance_image=true"
```

**Object Detection:**
```bash
curl -X POST \
  https://your-apg-instance.com/api/v1/images/detect-objects \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "X-Tenant-ID: your-tenant-id" \
  -F "file=@image.jpg" \
  -F "model=yolov8n" \
  -F "confidence_threshold=0.5"
```

**Check Job Status:**
```bash
curl -X GET \
  https://your-apg-instance.com/api/v1/jobs/job_123456789 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "X-Tenant-ID: your-tenant-id"
```

---

## Support

For API support and questions:

**Documentation:** https://docs.datacraft.co.ke/computer-vision/api  
**Support Email:** api-support@datacraft.co.ke  
**Community Forum:** https://community.datacraft.co.ke/api  
**GitHub Issues:** https://github.com/datacraft/apg-computer-vision-sdk  

---

*This API reference is regularly updated. For the latest version, check the documentation portal.*

**Last Updated:** January 27, 2025  
**Version:** 1.0.0  
**© 2025 Datacraft. All rights reserved.**