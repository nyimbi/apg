# APG RAG User Guide

> **Complete Guide to Enterprise Retrieval-Augmented Generation**

## Table of Contents

- [Getting Started](#getting-started)
- [Knowledge Base Management](#knowledge-base-management)
- [Document Management](#document-management)
- [Querying and Search](#querying-and-search)
- [Conversation Management](#conversation-management)
- [Advanced Features](#advanced-features)
- [Administration](#administration)
- [Troubleshooting](#troubleshooting)

## Getting Started

### What is APG RAG?

APG RAG (Retrieval-Augmented Generation) is an enterprise-grade AI system that allows you to:

- **Upload and organize documents** into searchable knowledge bases
- **Ask natural language questions** and get accurate, sourced answers
- **Have intelligent conversations** with your documents and data
- **Generate insights** from your organizational knowledge
- **Maintain security and compliance** with enterprise-grade controls

### Key Concepts

#### Knowledge Bases
Containers that organize related documents and provide context for AI responses. Think of them as intelligent libraries for specific topics or domains.

#### Documents
Individual files (PDFs, Word docs, text files, etc.) that contain the knowledge you want to query. Documents are automatically processed and made searchable.

#### Conversations
Interactive chat sessions where you can ask questions and get AI-generated responses based on your documents. Conversations maintain context across multiple exchanges.

#### Embeddings
Mathematical representations of your text content that enable semantic search and similarity matching. Generated automatically using the bge-m3 model.

## Knowledge Base Management

### Creating a Knowledge Base

#### Via Web Interface
1. Navigate to **RAG Management > Knowledge Bases**
2. Click **Add Knowledge Base**
3. Fill in the required information:
   - **Name**: Descriptive name for your knowledge base
   - **Description**: Optional detailed description
   - **Chunk Size**: How large text segments should be (default: 1000 characters)
   - **Chunk Overlap**: How much segments should overlap (default: 200 characters)
   - **Similarity Threshold**: Minimum relevance score for search results (default: 0.7)

4. Click **Save**

#### Via API
```bash
curl -X POST http://your-server/api/v1/rag/knowledge-bases \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "name": "Company Policies",
    "description": "All company policies and procedures",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "similarity_threshold": 0.7,
    "max_retrievals": 10
  }'
```

### Configuring Knowledge Bases

#### Chunk Size Settings
- **Small chunks (500-800 chars)**: Better for precise, specific questions
- **Medium chunks (1000-1500 chars)**: Good general-purpose setting
- **Large chunks (2000+ chars)**: Better for complex, contextual queries

#### Similarity Threshold
- **0.5-0.6**: More permissive, includes somewhat related content
- **0.7-0.8**: Balanced setting, good for most use cases
- **0.9+**: Very strict, only highly relevant content

### Best Practices

#### Organizing Knowledge Bases
- **By Topic**: Create separate knowledge bases for different subjects (HR, Finance, Engineering)
- **By Audience**: Different knowledge bases for different user groups
- **By Sensitivity**: Separate bases for different security classifications

#### Naming Conventions
- Use clear, descriptive names
- Include version numbers if applicable
- Consider using prefixes for organization (e.g., "HR_", "ENG_", "FIN_")

## Document Management

### Supported File Formats

The APG RAG system supports a wide variety of document formats:

#### Primary Formats
- **PDF** (.pdf) - Portable Document Format
- **Microsoft Word** (.docx, .doc) - Word documents
- **Plain Text** (.txt) - Simple text files
- **HTML** (.html, .htm) - Web pages and formatted text
- **Markdown** (.md) - Markdown formatted text

#### Additional Formats
- **JSON** (.json) - Structured data files
- **CSV** (.csv) - Comma-separated values
- **XML** (.xml) - Structured markup documents
- **RTF** (.rtf) - Rich Text Format
- **ODT** (.odt) - OpenDocument Text

### Uploading Documents

#### Single Document Upload
1. Navigate to your knowledge base
2. Click **Add Document**
3. Choose your file
4. Optionally add:
   - **Custom Title**: Override the filename
   - **Metadata**: Additional tags or information
   - **Processing Options**: Special handling instructions

4. Click **Upload and Process**

#### Batch Upload
For multiple documents:
1. Select **Batch Upload**
2. Choose multiple files or drag and drop
3. Configure common settings for all files
4. Start batch processing

#### API Upload
```bash
curl -X POST http://your-server/api/v1/rag/documents/KB_ID \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf" \
  -F "title=Important Policy Document" \
  -F 'metadata={"department": "HR", "version": "2.1"}'
```

### Document Processing

#### Processing Pipeline
1. **Content Extraction**: Text is extracted from the document
2. **Chunking**: Content is split into manageable segments
3. **Embedding Generation**: AI creates vector representations
4. **Indexing**: Chunks are stored and indexed for fast retrieval
5. **Quality Analysis**: Content quality is assessed and scored

#### Processing Status
- **Pending**: Document uploaded, waiting for processing
- **Processing**: Currently being analyzed and indexed
- **Completed**: Successfully processed and searchable
- **Failed**: Processing encountered an error

#### Monitoring Processing
Track processing progress in the document list or via API:
```bash
curl -X GET http://your-server/api/v1/rag/documents/DOCUMENT_ID \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Document Metadata

#### Automatic Metadata
The system automatically extracts:
- File size and type
- Creation and modification dates
- Character and word counts
- Language detection
- Content quality scores

#### Custom Metadata
You can add custom metadata tags:
- **Department**: Source department
- **Version**: Document version
- **Author**: Document author
- **Classification**: Security classification
- **Tags**: Searchable keywords

### Managing Documents

#### Updating Documents
1. Upload a new version with the same name
2. The system will detect and update the existing document
3. Old versions are maintained for audit purposes

#### Deleting Documents
1. Select the document(s) to delete
2. Click **Delete**
3. Confirm the action
4. All associated chunks and embeddings are removed

**Note**: Deleted documents cannot be recovered. Ensure you have backups if needed.

## Querying and Search

### Basic Querying

#### Simple Questions
Ask natural language questions:
- "What is our remote work policy?"
- "How do I request vacation time?"
- "What are the safety procedures for the lab?"

#### Question Types
- **Factual**: Direct questions with specific answers
- **Procedural**: How-to questions about processes
- **Comparative**: Questions comparing different options
- **Analytical**: Questions requiring analysis or interpretation

### Advanced Search Features

#### Search Methods
1. **Vector Search**: Semantic similarity matching
2. **Hybrid Search**: Combines vector and text search
3. **Text Search**: Traditional keyword matching

#### Search Parameters
- **k**: Number of results to return (default: 10)
- **Similarity Threshold**: Minimum relevance score (default: 0.7)
- **Filters**: Restrict search to specific documents or metadata

#### API Query Example
```bash
curl -X POST http://your-server/api/v1/rag/query/KB_ID \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query_text": "What is the expense reimbursement policy?",
    "k": 10,
    "similarity_threshold": 0.7,
    "retrieval_method": "hybrid_search"
  }'
```

### Understanding Results

#### Result Components
Each search result includes:
- **Content**: The relevant text chunk
- **Source**: Document name and location
- **Similarity Score**: Relevance rating (0.0-1.0)
- **Context**: Surrounding content for better understanding

#### Interpreting Scores
- **0.9-1.0**: Highly relevant, exact matches
- **0.7-0.9**: Good relevance, likely useful
- **0.5-0.7**: Moderate relevance, may be helpful
- **<0.5**: Low relevance, likely not useful

### RAG Response Generation

#### How RAG Works
1. Your question is converted to a vector representation
2. The system finds the most relevant document chunks
3. These chunks are combined with your question
4. An AI model generates a comprehensive answer
5. Sources are cited for verification

#### Response Quality Factors
- **Source Quality**: Better documents produce better answers
- **Question Clarity**: Clear questions get better responses
- **Context Availability**: More relevant context improves answers
- **Model Performance**: Advanced models provide better generation

#### Example RAG Flow
```
Query: "What is our data retention policy?"
  ↓
Search: Find relevant chunks about data retention
  ↓
Context: Combine chunks with query
  ↓
Generate: AI creates comprehensive answer
  ↓
Response: "According to our data retention policy (Policy-2024-DR.pdf, Section 3.2), 
         personal data must be retained for a maximum of 7 years after the last 
         interaction, with certain exceptions for legal requirements..."
```

## Conversation Management

### Creating Conversations

#### Starting a New Conversation
1. Navigate to **RAG Management > Conversations**
2. Click **New Conversation**
3. Configure settings:
   - **Title**: Descriptive name for the conversation
   - **Knowledge Base**: Select the knowledge base to query
   - **AI Model**: Choose generation model (qwen3 or deepseek-r1)
   - **Temperature**: Creativity level (0.0-1.0, default: 0.7)
   - **Max Context**: Maximum tokens for context (default: 8000)

4. Start chatting!

#### API Conversation Creation
```bash
curl -X POST http://your-server/api/v1/rag/conversations/KB_ID \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "title": "HR Policy Discussion",
    "description": "Questions about company policies",
    "generation_model": "qwen3",
    "temperature": 0.7,
    "max_context_tokens": 8000
  }'
```

### Chat Features

#### Context Awareness
Conversations maintain context across multiple messages:
- Previous questions and answers are remembered
- Follow-up questions reference earlier content
- Context is intelligently summarized to stay within limits

#### Message Types
- **User Messages**: Your questions and inputs
- **Assistant Messages**: AI-generated responses
- **System Messages**: Processing status and notifications

#### Conversation Memory
The system intelligently manages conversation memory:
- **Key Facts**: Important information is retained
- **Entity Tracking**: People, places, and concepts are tracked
- **Topic Flow**: Conversation topics are mapped and connected

### Advanced Conversation Features

#### Multi-Turn Reasoning
The AI can handle complex, multi-step reasoning:
```
User: "What is our vacation policy?"
AI: "Our vacation policy allows 20 days per year..."

User: "How does that compare to industry standards?"
AI: "Compared to industry standards, our 20-day policy..."

User: "Can I carry over unused days?"
AI: "Regarding carryover, according to the same policy document..."
```

#### Context Injection
Provide additional context for better responses:
```bash
curl -X POST http://your-server/api/v1/rag/chat/CONVERSATION_ID \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "message": "What about remote work?",
    "user_context": {
      "department": "Engineering",
      "role": "Senior Developer",
      "location": "California"
    }
  }'
```

### Managing Conversations

#### Conversation History
View complete conversation history:
- All messages with timestamps
- Source citations for each response
- Performance metrics (response times, confidence scores)

#### Sharing Conversations
- **Export**: Download conversation as text or JSON
- **Share Links**: Generate shareable links (with appropriate permissions)
- **Collaboration**: Multiple users can participate in group conversations

#### Archiving and Cleanup
- **Auto-Archive**: Conversations older than 30 days are automatically archived
- **Manual Archive**: Archive conversations you no longer need
- **Permanent Delete**: Remove conversations and all associated data

## Advanced Features

### Custom Configuration

#### Per-Query Configuration
Customize behavior for specific queries:
```json
{
  "query_text": "Explain our security policies",
  "retrieval_config": {
    "k": 15,
    "similarity_threshold": 0.8,
    "enable_reranking": true
  },
  "generation_config": {
    "model": "deepseek-r1",
    "temperature": 0.5,
    "max_tokens": 1024
  }
}
```

#### Knowledge Base Templates
Create reusable configurations:
- **HR Template**: Optimized for policy documents
- **Technical Template**: Best for technical documentation
- **Legal Template**: Configured for legal document analysis

### Integration Features

#### API Webhooks
Receive notifications for events:
- Document processing completion
- System alerts and errors
- Usage threshold warnings

#### External System Integration
- **LDAP/Active Directory**: User authentication
- **SharePoint**: Document synchronization
- **Slack/Teams**: Chat bot integration
- **Salesforce**: CRM data integration

### Analytics and Insights

#### Usage Analytics
Track system usage:
- Query volume and patterns
- Popular documents and topics
- User engagement metrics
- Response quality trends

#### Content Analytics
Understand your knowledge base:
- Document coverage and gaps
- Content quality scores
- Duplicate content detection
- Update recommendations

#### Performance Analytics
Monitor system performance:
- Response time trends
- Error rates and types
- Resource utilization
- Scalability metrics

## Administration

### User Management

#### Role-Based Access Control
Define user roles and permissions:
- **Admin**: Full system access
- **Manager**: Knowledge base management
- **Editor**: Document upload and editing
- **User**: Query and conversation access
- **Viewer**: Read-only access

#### Tenant Management
For multi-tenant deployments:
- Complete data isolation between tenants
- Separate configurations and settings
- Independent user management
- Usage and billing tracking

### Security Administration

#### Access Control
- Multi-factor authentication support
- IP address restrictions
- Session timeout configuration
- API key management

#### Data Protection
- Encryption at rest and in transit
- Regular security audits
- Compliance reporting (GDPR, HIPAA, etc.)
- Data retention policies

#### Audit Logging
Comprehensive audit trails:
- All user actions logged
- Document access tracking
- System changes recorded
- Compliance report generation

### System Configuration

#### Performance Tuning
Optimize system performance:
- Vector index optimization
- Database query tuning
- Cache configuration
- Resource allocation

#### Monitoring and Alerting
Set up monitoring:
- System health checks
- Performance threshold alerts
- Error rate monitoring
- Capacity planning metrics

#### Backup and Recovery
Ensure data protection:
- Automated daily backups
- Point-in-time recovery
- Disaster recovery procedures
- Data migration tools

## Troubleshooting

### Common Issues

#### Document Processing Problems

**Issue**: Documents fail to process
**Causes**:
- Unsupported file format
- Corrupted file
- File too large
- Insufficient system resources

**Solutions**:
1. Verify file format is supported
2. Try re-uploading the document
3. Check file integrity
4. Contact administrator if resources are insufficient

#### Search Result Issues

**Issue**: No or poor search results
**Causes**:
- Similarity threshold too high
- Documents not fully processed
- Query too vague or specific
- Insufficient content in knowledge base

**Solutions**:
1. Lower similarity threshold (try 0.5-0.6)
2. Wait for document processing to complete
3. Rephrase query with different keywords
4. Add more relevant documents

#### Performance Issues

**Issue**: Slow response times
**Causes**:
- Large knowledge base
- Complex queries
- System resource constraints
- Network latency

**Solutions**:
1. Use more specific queries
2. Reduce number of results requested
3. Check system resources
4. Contact administrator for optimization

### Error Messages

#### Common Error Codes
- **400 Bad Request**: Invalid query format or parameters
- **401 Unauthorized**: Invalid or missing authentication
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Knowledge base or document not found
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: System error, contact administrator

#### API Error Responses
```json
{
  "success": false,
  "error": "Knowledge base not found",
  "error_code": "KB_NOT_FOUND",
  "timestamp": "2025-01-29T10:30:00Z"
}
```

### Getting Help

#### Self-Service Resources
- **System Health Dashboard**: Monitor system status
- **Usage Analytics**: Understand your usage patterns
- **Documentation**: Complete API and user documentation
- **FAQ**: Frequently asked questions

#### Support Channels
- **In-App Help**: Built-in help system
- **Documentation Portal**: Comprehensive guides and tutorials
- **Community Forum**: User community and discussions
- **Technical Support**: Direct support for issues

### Best Practices Summary

#### For Best Results
1. **Organize content logically** in knowledge bases
2. **Use clear, specific questions** for better answers
3. **Keep documents up-to-date** and well-organized
4. **Review and refine** search thresholds based on results
5. **Monitor usage patterns** to optimize performance

#### Security Best Practices
1. **Use strong authentication** and regular password updates
2. **Apply principle of least privilege** for user access
3. **Regularly review audit logs** for unusual activity
4. **Keep sensitive data** in appropriately secured knowledge bases
5. **Follow data retention policies** for compliance

#### Performance Best Practices
1. **Structure knowledge bases** by topic and size
2. **Use appropriate chunk sizes** for your content type
3. **Monitor system resources** and scale as needed
4. **Optimize queries** for better response times
5. **Regular maintenance** including index optimization

---

**Need more help?** Contact your system administrator or check the API documentation for technical details.