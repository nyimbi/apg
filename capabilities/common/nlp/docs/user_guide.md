# APG Natural Language Processing (NLP) Capability - User Guide

**Version**: 1.0.0  
**Last Updated**: January 29, 2025  
**Copyright**: © 2025 Datacraft  
**Author**: Nyimbi Odero  

## Overview

The APG NLP capability provides enterprise-grade natural language processing using on-device models (Ollama, Transformers, spaCy) for maximum security and privacy. This capability includes 11 specialized corporate NLP elements designed to be 10x better than market leaders.

## Key Features

### 🧠 **11 Corporate NLP Elements**
1. **Sentiment Analysis** - Advanced emotion and opinion detection
2. **Intent Classification** - Customer service and automation intent detection
3. **Named Entity Recognition (NER)** - Extract people, organizations, locations
4. **Text Classification** - Categorize documents and content
5. **Entity Recognition and Linking** - Link entities to knowledge bases
6. **Topic Modeling** - Discover themes in document collections
7. **Keyword Extraction** - Extract important terms and phrases
8. **Text Summarization** - Generate concise summaries
9. **Document Clustering** - Group similar documents
10. **Language Detection** - Identify text languages
11. **Content Generation** - Generate content from prompts

### 🚀 **Enterprise Features**
- **On-Device Processing** - No external API dependencies
- **Real-Time Streaming** - WebSocket-based live processing
- **Multi-Tenant Support** - Complete tenant isolation
- **APG Integration** - Seamless composition with other capabilities
- **Advanced Analytics** - Comprehensive performance monitoring
- **Collaborative Annotation** - Team-based data labeling

## Getting Started

### Prerequisites

1. **APG Platform** - Must be deployed within APG ecosystem
2. **Python Dependencies** - Automatically managed by APG
3. **Optional: Ollama** - For enhanced content generation
4. **Optional: CUDA** - For GPU acceleration

### Basic Usage

#### 1. Access the NLP Dashboard
Navigate to: `/nlp/dashboard` in your APG application

#### 2. Process Text via API

```bash
curl -X POST http://your-apg-domain/api/nlp/process \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your-tenant-id" \
  -d '{
    "text_content": "I love this product! It works perfectly.",
    "task_type": "sentiment_analysis",
    "quality_level": "balanced"
  }'
```

#### 3. Use Specialized Endpoints

Each NLP element has a dedicated endpoint:

```bash
# Sentiment Analysis
curl -X POST /api/nlp/sentiment -d '{"text": "Great product!"}'

# Intent Classification  
curl -X POST /api/nlp/intent -d '{"text": "I need help with my order"}'

# Named Entity Recognition
curl -X POST /api/nlp/entities -d '{"text": "Apple Inc. was founded by Steve Jobs"}'

# Text Classification
curl -X POST /api/nlp/classify -d '{"text": "This is about AI technology"}'

# Text Summarization
curl -X POST /api/nlp/summarize -d '{"text": "Long document content..."}'

# Content Generation
curl -X POST /api/nlp/generate -d '{"prompt": "Write about AI benefits"}'
```

## Detailed Feature Guide

### Sentiment Analysis

**Purpose**: Analyze emotional tone and opinion in text  
**Use Cases**: Customer feedback, social media monitoring, review analysis

**Example Request**:
```json
{
  "text": "I absolutely love this new feature! It makes my work so much easier.",
  "language": "en"
}
```

**Example Response**:
```json
{
  "sentiment": "positive",
  "confidence": 0.89,
  "scores": {
    "positive": 0.89,
    "negative": 0.08,
    "neutral": 0.03
  },
  "model_used": "transformers_roberta_sentiment",
  "processing_method": "transformer_roberta"
}
```

### Intent Classification

**Purpose**: Determine user intent for automation and routing  
**Use Cases**: Customer service, chatbots, email routing

**Custom Intents**: You can specify your own intent categories:
```json
{
  "text": "I want to cancel my subscription",
  "possible_intents": ["cancel", "upgrade", "support", "billing"]
}
```

### Named Entity Recognition (NER)

**Purpose**: Extract structured information from unstructured text  
**Use Cases**: Information extraction, data mining, content analysis

**Supported Entity Types**:
- **PERSON** - People names
- **ORG** - Organizations, companies
- **GPE** - Geopolitical entities (countries, cities)
- **MONEY** - Monetary values
- **DATE** - Dates and times
- **EMAIL** - Email addresses (fallback mode)

### Text Classification

**Purpose**: Categorize documents into predefined categories  
**Use Cases**: Document management, content organization, compliance

**Default Categories**: business, technology, finance, legal, marketing, operations, hr

**Custom Categories**:
```json
{
  "text": "Contract terms and conditions...",
  "categories": ["legal", "contracts", "terms", "compliance"]
}
```

### Entity Recognition and Linking

**Purpose**: Connect entities to external knowledge bases  
**Use Cases**: Knowledge management, fact-checking, content enrichment

**Features**:
- Wikipedia linking for major entities
- Knowledge base IDs where available
- Confidence scoring for links

### Topic Modeling

**Purpose**: Discover hidden themes in document collections  
**Use Cases**: Content analysis, research, document organization

**Requirements**: Minimum 2 documents, optimal 10+ documents

```json
{
  "texts": [
    "Document 1 about AI technology...",
    "Document 2 about machine learning...",
    "Document 3 about software development..."
  ],
  "num_topics": 3
}
```

### Keyword Extraction

**Purpose**: Identify important terms and phrases  
**Use Cases**: SEO, content tagging, summarization

**Methods Used**:
- **TF-IDF** - Statistical importance
- **Named Entity** - Entities as keywords
- **Noun Phrases** - Multi-word expressions

### Text Summarization

**Purpose**: Generate concise summaries of long content  
**Use Cases**: Document processing, content curation, executive summaries

**Methods**:
- **Extractive** - Select important sentences (default)
- **Abstractive** - Generate new summary text (requires transformers)

```json
{
  "text": "Long article content...",
  "max_length": 150,
  "method": "extractive"
}
```

### Document Clustering

**Purpose**: Group similar documents together  
**Use Cases**: Content organization, duplicate detection, research

**Requirements**: Minimum documents = number of clusters

```json
{
  "documents": [
    "Tech document 1...",
    "Business document 1...",
    "Tech document 2...", 
    "Business document 2..."
  ],
  "num_clusters": 2
}
```

### Language Detection

**Purpose**: Identify the language of text content  
**Use Cases**: Internationalization, content routing, translation preparation

**Supported Languages**: en, es, fr, de, and others via langdetect library

### Content Generation

**Purpose**: Generate new content from prompts  
**Use Cases**: Content creation, email templates, report generation

**Models**: Prefers Ollama for on-device generation, falls back to transformers

```json
{
  "prompt": "Write a professional email about project completion",
  "max_length": 200,
  "task_type": "email"
}
```

## Real-Time Streaming

### WebSocket Processing

For real-time text processing, use WebSocket streaming:

1. **Start Session**:
```bash
curl -X POST /api/nlp/stream/start \
  -d '{"task_type": "sentiment_analysis", "chunk_size": 1000}'
```

2. **Connect WebSocket**:
```javascript
const socket = io('/nlp');
socket.emit('join_stream', {session_id: 'your-session-id'});
```

3. **Process Chunks**:
```javascript
socket.emit('process_chunk', {
  session_id: 'your-session-id',
  text_content: 'Text to process...',
  sequence_number: 1
});
```

4. **Receive Results**:
```javascript
socket.on('chunk_processed', (result) => {
  console.log('Processing result:', result);
});
```

## Model Management

### Available Models

The NLP capability automatically detects and uses the best available models:

- **Ollama Models** - For text generation and analysis
- **Transformers Models** - For classification and NER
- **spaCy Models** - For fast linguistic processing

### Model Health Check

Monitor model health at: `/nlp/models/health`

### Performance Optimization

**Quality Levels**:
- **FAST** - Prefer speed (spaCy, DistilBERT)
- **BALANCED** - Balance speed and accuracy (default)
- **BEST** - Maximum accuracy (large transformer models)

**GPU Acceleration**: Automatically detected and used when available

## Analytics and Monitoring

### Dashboard

Access comprehensive analytics at: `/nlp/analytics/dashboard`

**Metrics Include**:
- Request volume and success rates
- Model performance comparisons
- Processing time distributions
- Error analysis and trends
- Resource utilization

### Performance Monitoring

**Key Metrics**:
- **Processing Time** - Sub-100ms for most tasks
- **Confidence Scores** - Model certainty ratings
- **Success Rates** - Error tracking and analysis
- **Throughput** - Requests per minute

## Security and Privacy

### On-Device Processing
- No external API calls for core NLP functions
- All processing happens within your infrastructure
- Complete data privacy and security

### Multi-Tenant Isolation
- Complete tenant data separation
- Isolated model instances per tenant
- Secure permission management

### GDPR/CCPA Compliance
- No data retention beyond processing
- Audit logging for compliance
- Data processing transparency

## Integration with APG Ecosystem

### Capability Composition

The NLP capability integrates with other APG capabilities:

**Provides**:
- `text_processing` - Core NLP functions
- `sentiment_analysis` - Emotion detection
- `entity_extraction` - Information extraction

**Requires**:
- `ai_orchestration` - Model management
- `auth_rbac` - Authentication and authorization
- `audit_compliance` - Logging and compliance

### API Integration

All endpoints follow APG standards:
- Consistent authentication via headers
- Standardized error responses
- Comprehensive audit logging
- OpenAPI documentation

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```
Solution: Ensure required models are installed
- spaCy: python -m spacy download en_core_web_sm
- Transformers: Models download automatically
- Ollama: Install via ollama.ai
```

**2. Performance Issues**
```
Solutions:
- Reduce quality_level to "fast"
- Enable GPU acceleration if available
- Use smaller model variants
- Implement request batching
```

**3. Memory Issues**
```
Solutions:
- Adjust max_memory_gb in configuration
- Use model unloading between requests
- Implement model rotation for high load
```

### Error Codes

- **400** - Validation errors (missing fields, invalid formats)
- **404** - Model or session not found
- **429** - Rate limiting (too many requests)
- **500** - Internal processing errors
- **503** - Service unavailable (model loading)

### Support

For technical support:
- **Email**: nyimbi@gmail.com
- **Documentation**: This guide and API docs
- **Health Check**: `/nlp/health` endpoint
- **System Diagnostics**: `/nlp/diagnostics`

## Advanced Usage

### Batch Processing

Process multiple texts efficiently:

```bash
curl -X POST /api/nlp/process/batch \
  -d '{
    "texts": ["Text 1", "Text 2", "Text 3"],
    "task_type": "sentiment_analysis"
  }'
```

### Custom Models

To integrate custom models, extend the service class:

```python
from capabilities.common.nlp.service import NLPService

class CustomNLPService(NLPService):
    async def load_custom_model(self, model_path: str):
        # Custom model loading logic
        pass
```

### Performance Tuning

**Configuration Options**:
```python
config = ModelConfig(
    max_memory_gb=16.0,
    enable_gpu=True,
    model_timeout_seconds=600
)
```

**Optimization Tips**:
- Use GPU acceleration for large models
- Implement model caching for repeated use
- Batch similar requests together
- Monitor memory usage and adjust limits

## Conclusion

The APG NLP capability provides enterprise-grade natural language processing with complete on-device control, advanced analytics, and seamless APG ecosystem integration. With 11 specialized corporate NLP elements, it delivers performance that exceeds market leaders while maintaining maximum security and privacy.

For the latest updates and advanced features, refer to the [Developer Guide](developer_guide.md) and [API Documentation](/nlp/api/docs/).