# APG Crawler Capability - User Guide

**Version:** 2.0.0  
**Author:** Datacraft  
**Copyright:** © 2025 Datacraft  
**Email:** nyimbi@gmail.com  

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Crawl Target Management](#crawl-target-management)
4. [RAG Integration](#rag-integration)
5. [GraphRAG Knowledge Graphs](#graphrag-knowledge-graphs)
6. [Collaborative Validation](#collaborative-validation)
7. [Analytics and Monitoring](#analytics-and-monitoring)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

The APG Crawler Capability is a revolutionary enterprise web intelligence platform that combines advanced web crawling with RAG (Retrieval-Augmented Generation) and GraphRAG capabilities. It provides:

- **10x Performance**: Faster than industry-standard crawling solutions
- **AI-Powered Intelligence**: Automatic content understanding and entity extraction
- **RAG Integration**: Semantic search with vector embeddings
- **GraphRAG**: Knowledge graph construction for entity relationships
- **Collaborative Validation**: Team-based data quality assurance
- **Multi-tenant Architecture**: Enterprise-grade security and isolation

### Key Features

✅ **Multi-Source Orchestration** - Unified data extraction across 20+ sources  
✅ **Content Cleaning & Fingerprinting** - Automatic content processing and duplicate detection  
✅ **RAG Processing** - Semantic chunking with vector embeddings  
✅ **GraphRAG Integration** - Entity extraction and knowledge graph construction  
✅ **Collaborative Validation** - Team-based quality assurance workflows  
✅ **Real-time Analytics** - Live dashboards and business intelligence  

## Getting Started

### Prerequisites

- Access to APG platform with crawler capability enabled
- Valid tenant credentials
- Basic understanding of web crawling concepts
- (Optional) Knowledge of RAG/GraphRAG for advanced features

### Initial Setup

1. **Access the Crawler Dashboard**
   ```
   Navigate to: /crawler/dashboard
   ```

2. **Verify Tenant Access**
   - Confirm your tenant ID appears in the dashboard
   - Check that you have appropriate permissions

3. **Review Available Features**
   - Crawl target management
   - RAG processing capabilities
   - GraphRAG knowledge graphs
   - Validation workflows

## Crawl Target Management

### Creating a Crawl Target

A crawl target defines what you want to crawl and how the system should process the data.

#### Step 1: Basic Configuration

1. Navigate to **Crawler Management > Crawl Targets**
2. Click **Add New Target**
3. Fill in the basic information:
   ```
   Name: "Company News Articles"
   Description: "Daily news articles from tech companies"
   Target Type: "web_crawl"
   Status: "active"
   ```

#### Step 2: URL Configuration

Add the URLs you want to crawl:
```
Target URLs:
- https://techcrunch.com/category/startups/
- https://www.reuters.com/technology/
- https://venturebeat.com/category/ai/
```

#### Step 3: Business Context

Define the business context to help the AI understand your goals:
```json
{
  "domain": "Technology News",
  "industry": "Technology",
  "use_case": "Market Intelligence",
  "priority_entities": ["company", "product", "funding", "executive"],
  "quality_criteria": {
    "min_article_length": 100,
    "require_publish_date": true,
    "exclude_advertisements": true
  }
}
```

#### Step 4: RAG/GraphRAG Integration

Enable advanced processing capabilities:
- ✅ **RAG Integration Enabled**: Process content for semantic search
- ✅ **GraphRAG Integration Enabled**: Extract entities and build knowledge graphs
- ✅ **Content Fingerprinting**: Detect and handle duplicates
- ✅ **Markdown Storage**: Store cleaned content as markdown

### Managing Crawl Targets

#### Monitoring Target Status

- **Active**: Target is running and processing data
- **Paused**: Target is temporarily stopped
- **Completed**: Target has finished processing
- **Draft**: Target is being configured

#### Bulk Operations

Use bulk actions to manage multiple targets:
- **Enable RAG**: Turn on RAG processing for selected targets
- **Disable RAG**: Turn off RAG processing
- **Update Status**: Change status of multiple targets

## RAG Integration

### Understanding RAG Processing

RAG (Retrieval-Augmented Generation) processing converts your crawled content into searchable, semantic chunks with vector embeddings.

#### The RAG Pipeline

1. **Content Cleaning**: Remove ads, navigation, preserve structure
2. **Markdown Conversion**: Convert to clean, formatted markdown
3. **Fingerprinting**: Generate SHA-256 hashes for duplicate detection
4. **Chunking**: Split content into semantic chunks with overlap
5. **Embedding**: Generate vector embeddings for similarity search
6. **Indexing**: Store in vector database for fast retrieval

### Configuring RAG Processing

#### Basic RAG Configuration

```json
{
  "chunk_size": 1000,
  "overlap_size": 200,
  "vector_dimensions": 1536,
  "embedding_model": "text-embedding-ada-002",
  "indexing_strategy": "semantic_chunks"
}
```

#### Advanced Settings

- **Chunk Size**: Text chunk size (100-8000 characters)
- **Overlap Size**: Overlap between chunks (0-1000 characters)
- **Vector Dimensions**: Embedding dimensions (512-4096)
- **Embedding Model**: Choose from available models
- **Entity Resolution Threshold**: Confidence threshold for entities (0.0-1.0)

### Using RAG Search

#### Semantic Search Interface

1. Navigate to **RAG Management > RAG Overview**
2. Use the search interface to find relevant content
3. Adjust similarity threshold for precision vs. recall

#### API Search Example

```bash
curl -X GET "/api/crawler/rag/search" \
  -H "X-Tenant-ID: your-tenant-id" \
  -G -d "query=artificial intelligence startups" \
  -G -d "limit=10" \
  -G -d "similarity_threshold=0.8"
```

## GraphRAG Knowledge Graphs

### Understanding GraphRAG

GraphRAG extends RAG by creating knowledge graphs that capture entity relationships and semantic connections in your data.

#### The GraphRAG Pipeline

1. **Entity Extraction**: Identify people, organizations, locations, products
2. **Relation Detection**: Find relationships between entities
3. **Node Creation**: Create graph nodes for each entity
4. **Relation Mapping**: Create edges between related entities
5. **Graph Integration**: Add to knowledge graph with statistics

### Creating Knowledge Graphs

#### Step 1: Initialize Knowledge Graph

1. Navigate to **GraphRAG > Knowledge Graphs**
2. Click **Add New Graph**
3. Configure basic settings:
   ```
   Graph Name: "Technology Companies Knowledge Graph"
   Description: "Entities and relationships in the tech industry"
   Domain: "Technology"
   ```

#### Step 2: Process RAG Chunks

After RAG processing is complete, process chunks for GraphRAG:

```bash
curl -X POST "/api/crawler/graphrag/process" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your-tenant-id" \
  -d '{
    "rag_chunk_ids": ["chunk-id-1", "chunk-id-2"],
    "knowledge_graph_id": "graph-id",
    "merge_similar_entities": true
  }'
```

### Exploring Knowledge Graphs

#### Graph Statistics

Monitor your knowledge graph growth:
- **Node Count**: Total entities extracted
- **Relation Count**: Total relationships found
- **Entity Types**: Categories of entities (Person, Organization, etc.)
- **Relation Types**: Types of relationships (works_at, produces, etc.)
- **Graph Density**: How connected your entities are

#### Entity Management

View and manage extracted entities:
- **High Confidence Entities**: Entities with confidence > 0.8
- **Entity Merging**: Combine similar entities
- **Relationship Verification**: Validate extracted relationships

## Collaborative Validation

### Setting Up Validation Sessions

Collaborative validation allows teams to review and improve data quality.

#### Creating a Validation Session

1. Navigate to **Quality Management > Validation Sessions**
2. Click **Create Session**
3. Configure the session:
   ```
   Session Name: "Q1 2025 News Validation"
   Dataset: Select the dataset to validate
   Consensus Threshold: 0.8 (80% agreement required)
   Quality Threshold: 0.7 (70% minimum quality)
   ```

#### Adding Validators

Invite team members to participate:
- **Validator Role**: Domain expert, data analyst, etc.
- **Expertise Areas**: Technology, finance, healthcare, etc.
- **Permissions**: What they can validate

### Validation Workflow

#### For Validators

1. **Access Validation Session**: Click on assigned session
2. **Review Data Records**: Examine extracted data
3. **Provide Feedback**:
   - Quality Rating (1-5 stars)
   - Accuracy Rating (1-5 stars)
   - Completeness Rating (1-5 stars)
   - Comments and suggestions
4. **Submit Validation**: Save your feedback

#### For Session Managers

Monitor validation progress:
- **Completion Percentage**: How much has been validated
- **Consensus Metrics**: Agreement between validators
- **Quality Trends**: Improving or declining quality
- **Conflict Resolution**: Handle disagreements

### Quality Assurance

#### Quality Metrics

The system tracks comprehensive quality metrics:
- **Completeness Score**: How complete is the data
- **Accuracy Score**: How accurate is the extraction
- **Consistency Score**: How consistent across records
- **Freshness Score**: How recent is the data
- **Overall Quality Score**: Combined metric (0.0-1.0)

#### Quality Grades

- **A Grade**: Overall quality ≥ 0.9 (Excellent)
- **B Grade**: Overall quality ≥ 0.8 (Good)
- **C Grade**: Overall quality ≥ 0.7 (Acceptable)
- **D Grade**: Overall quality ≥ 0.6 (Needs improvement)
- **F Grade**: Overall quality < 0.6 (Poor)

## Analytics and Monitoring

### Dashboard Overview

The main dashboard provides key metrics:
- **Total Targets**: Number of active crawl targets
- **Active Crawls**: Currently running operations
- **Total Records**: Data records extracted
- **RAG Chunks**: Processed text chunks
- **GraphRAG Nodes**: Extracted entities
- **Validation Sessions**: Quality assurance activities

### Performance Analytics

#### Crawl Performance

Monitor crawling effectiveness:
- **Success Rate**: Percentage of successful crawls
- **Average Quality**: Mean quality score across records
- **Processing Speed**: Records processed per hour
- **Error Rate**: Percentage of failed operations

#### RAG Metrics

Track RAG processing performance:
- **Chunk Count**: Total RAG chunks created
- **Embedding Coverage**: Percentage with embeddings
- **Search Accuracy**: Semantic search effectiveness
- **Vector Index Size**: Storage usage

#### GraphRAG Metrics

Monitor knowledge graph construction:
- **Entity Count**: Total entities extracted
- **Relation Count**: Total relationships found
- **Graph Density**: Connectedness of entities
- **Extraction Accuracy**: Quality of entity extraction

### Real-time Monitoring

#### Health Checks

Monitor system health:
- **Database Status**: Connection and performance
- **Processing Queue**: Pending operations
- **Vector Index Status**: Search capability
- **API Response Time**: System responsiveness

#### Alerts and Notifications

Configure alerts for important events:
- **Quality Degradation**: When quality drops below threshold
- **Processing Failures**: When operations fail repeatedly
- **Capacity Limits**: When approaching resource limits
- **Validation Conflicts**: When validators disagree significantly

## Best Practices

### Crawl Target Configuration

#### URL Selection
- ✅ **Use specific URLs**: Target exactly what you need
- ✅ **Test URLs manually**: Verify they return expected content
- ✅ **Monitor for changes**: Websites change their structure
- ❌ **Avoid overly broad targets**: This creates noise in your data

#### Business Context
- ✅ **Be specific**: Clear business context improves AI understanding
- ✅ **Define entities**: List the entities you care about
- ✅ **Set quality criteria**: Define what constitutes good data
- ✅ **Update regularly**: Business needs evolve over time

### RAG Processing

#### Chunk Configuration
- ✅ **Optimal chunk size**: 500-2000 characters works best for most content
- ✅ **Use overlap**: 100-300 character overlap improves context
- ✅ **Match your use case**: Shorter chunks for search, longer for analysis
- ✅ **Test different settings**: Measure search quality with different configurations

#### Content Quality
- ✅ **Clean content**: Remove navigation, ads, and noise
- ✅ **Preserve structure**: Keep headers, lists, and formatting
- ✅ **Handle duplicates**: Use fingerprinting to detect copies
- ✅ **Monitor quality**: Regular quality checks prevent degradation

### GraphRAG Usage

#### Entity Extraction
- ✅ **Domain-specific entities**: Configure for your industry
- ✅ **Confidence thresholds**: Balance precision vs. recall
- ✅ **Regular validation**: Review extracted entities for accuracy
- ✅ **Entity merging**: Combine similar entities to reduce duplication

#### Knowledge Graph Maintenance
- ✅ **Regular updates**: Keep graphs current with new data
- ✅ **Quality control**: Validate relationships periodically
- ✅ **Performance monitoring**: Track graph query performance
- ✅ **Backup graphs**: Protect valuable knowledge assets

### Validation Workflows

#### Team Setup
- ✅ **Diverse expertise**: Include different domain experts
- ✅ **Clear guidelines**: Provide validation criteria and examples
- ✅ **Regular training**: Keep validators updated on best practices
- ✅ **Balanced workload**: Distribute validation tasks fairly

#### Quality Management
- ✅ **Set clear thresholds**: Define minimum quality requirements
- ✅ **Monitor consensus**: Track agreement between validators
- ✅ **Address conflicts**: Have procedures for handling disagreements
- ✅ **Continuous improvement**: Use feedback to improve extraction

## Troubleshooting

### Common Issues

#### Crawl Target Problems

**Issue**: Crawl target shows no data extracted
- **Cause**: Website blocking, incorrect URLs, or content changes
- **Solution**: 
  1. Test URLs manually in browser
  2. Check if website requires authentication
  3. Verify target configuration
  4. Review crawl logs for errors

**Issue**: Poor quality scores
- **Cause**: Noisy content, incorrect extraction, or wrong configuration
- **Solution**:
  1. Review business context configuration
  2. Adjust content cleaning settings
  3. Update quality criteria
  4. Run validation session

#### RAG Processing Issues

**Issue**: RAG chunks not being created
- **Cause**: Content not in markdown format or processing failures
- **Solution**:
  1. Check content processing stage
  2. Verify markdown conversion settings
  3. Review content cleaning configuration
  4. Check processing logs

**Issue**: Poor search results
- **Cause**: Wrong embedding model, chunk size, or similarity threshold
- **Solution**:
  1. Adjust similarity threshold (try 0.7-0.9)
  2. Experiment with chunk sizes
  3. Try different embedding models
  4. Review search query formulation

#### GraphRAG Problems

**Issue**: No entities extracted
- **Cause**: Content doesn't contain recognizable entities or confidence too high
- **Solution**:
  1. Lower confidence thresholds
  2. Review content for entity types
  3. Check entity extraction configuration
  4. Try different content domains

**Issue**: Knowledge graph not updating
- **Cause**: Processing failures or configuration issues
- **Solution**:
  1. Check GraphRAG processing status
  2. Verify knowledge graph configuration
  3. Review processing logs
  4. Restart processing if needed

### Performance Issues

#### Slow Processing
- **Cause**: Large datasets, complex processing, or resource constraints
- **Solution**:
  1. Process in smaller batches
  2. Optimize chunk sizes
  3. Check system resources
  4. Consider parallel processing

#### High Memory Usage
- **Cause**: Large embeddings, many chunks, or memory leaks
- **Solution**:
  1. Reduce vector dimensions
  2. Clean up old data
  3. Monitor memory usage
  4. Restart services if needed

### Getting Help

#### Support Channels

1. **Documentation**: Check this guide and API reference
2. **System Logs**: Review application logs for errors
3. **Health Checks**: Monitor system health dashboard
4. **Technical Support**: Contact support team with:
   - Tenant ID
   - Error messages
   - Steps to reproduce
   - Expected vs. actual behavior

#### Diagnostic Information

When reporting issues, include:
- **Tenant ID**: Your organization identifier
- **Target/Session ID**: Specific resource having issues
- **Timestamp**: When the issue occurred
- **Error Messages**: Exact error text
- **Configuration Details**: Relevant settings
- **System Metrics**: Performance data if available

---

**Need More Help?**

- 📧 Email: nyimbi@gmail.com
- 🌐 Website: www.datacraft.co.ke
- 📖 API Reference: See developer guide
- 🚀 Advanced Features: Contact for enterprise support

*This guide covers the essential features of the APG Crawler Capability. For advanced configuration and development, see the Developer Guide and API Reference.*