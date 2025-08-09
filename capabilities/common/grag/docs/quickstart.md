# APG GraphRAG Quick Start Guide

Get up and running with APG GraphRAG in 5 minutes! This guide will walk you through creating your first knowledge graph, processing documents, and querying with AI-powered reasoning.

## 🚀 Prerequisites

Before you begin, ensure you have:

- Python 3.9+
- PostgreSQL 12+ with Apache AGE extension
- Ollama running locally with required models
- APG framework installed

## 📦 Installation

### 1. Install APG GraphRAG

```bash
# Install the capability
pip install -e capabilities/common/graphrag

# Install dependencies
pip install -r capabilities/common/graphrag/requirements.txt
```

### 2. Setup Database

```bash
# Create database and install Apache AGE
createdb graphrag_db
psql -d graphrag_db -c "CREATE EXTENSION age;"

# Initialize schema
psql -d graphrag_db -f capabilities/common/graphrag/database_schema.sql
```

### 3. Setup Ollama Models

```bash
# Install required models
ollama pull bge-m3        # Embedding model (8k context)
ollama pull qwen3         # Generation model
ollama pull deepseek-r1   # Advanced reasoning model
```

## 🏃‍♂️ Quick Start

### Step 1: Initialize GraphRAG Service

```python
import asyncio
from capabilities.common.graphrag.service import GraphRAGService
from capabilities.common.graphrag.database import GraphRAGDatabaseService
from capabilities.common.graphrag.ollama_integration import OllamaClient, OllamaConfig

async def main():
    # Initialize database service
    db_service = GraphRAGDatabaseService()
    await db_service.initialize()
    
    # Initialize Ollama client
    ollama_config = OllamaConfig(
        base_url="http://localhost:11434",
        embedding_model="bge-m3",
        generation_models=["qwen3", "deepseek-r1"]
    )
    ollama_client = OllamaClient(ollama_config)
    await ollama_client.initialize()
    
    # Create GraphRAG service
    service = await GraphRAGService.create(
        db_service=db_service,
        ollama_client=ollama_client
    )
    
    return service

# Run initialization
service = asyncio.run(main())
```

### Step 2: Create Your First Knowledge Graph

```python
from capabilities.common.graphrag.views import KnowledgeGraphRequest

async def create_knowledge_graph():
    # Define graph request
    graph_request = KnowledgeGraphRequest(
        tenant_id="quickstart_tenant",
        name="My First Knowledge Graph",
        description="A revolutionary knowledge graph for quick start demo",
        domain="general",
        metadata={"created_by": "quickstart_guide"}
    )
    
    # Create the graph
    graph = await service.create_knowledge_graph(graph_request)
    
    print(f"✅ Created knowledge graph: {graph.knowledge_graph_id}")
    print(f"   Name: {graph.name}")
    print(f"   Domain: {graph.domain}")
    
    return graph

# Create knowledge graph
graph = asyncio.run(create_knowledge_graph())
```

### Step 3: Process Your First Document

```python
from capabilities.common.graphrag.views import DocumentProcessingRequest

async def process_document(knowledge_graph_id):
    # Sample document content
    document_content = """
    John Doe is the CEO of Acme Corporation, a technology company based in San Francisco.
    He founded the company in 2020 with his business partner Jane Smith, who serves as CTO.
    Acme Corporation specializes in artificial intelligence and machine learning solutions.
    The company has raised $50 million in Series A funding from Venture Capital Partners.
    John previously worked at Google as a Senior Software Engineer for 8 years.
    """
    
    # Create processing request
    doc_request = DocumentProcessingRequest(
        tenant_id="quickstart_tenant",
        knowledge_graph_id=knowledge_graph_id,
        title="Acme Corporation Profile",
        content=document_content,
        source_type="text",
        processing_options={
            "extract_entities": True,
            "extract_relationships": True,
            "generate_embeddings": True
        }
    )
    
    # Process the document
    result = await service.process_document(doc_request)
    
    print(f"✅ Document processed successfully!")
    print(f"   Entities extracted: {result.entities_extracted}")
    print(f"   Relationships extracted: {result.relationships_extracted}")
    print(f"   Processing time: {result.processing_time_ms:.1f}ms")
    
    return result

# Process document
doc_result = asyncio.run(process_document(graph.knowledge_graph_id))
```

### Step 4: Query with AI-Powered Reasoning

```python
from capabilities.common.graphrag.views import GraphRAGQuery, QueryContext

async def query_knowledge_graph(knowledge_graph_id):
    # Create query
    query = GraphRAGQuery(
        tenant_id="quickstart_tenant",
        knowledge_graph_id=knowledge_graph_id,
        query_text="Who is the CEO of Acme Corporation and what is their background?",
        query_type="factual",
        context=QueryContext(
            user_id="quickstart_user",
            session_id="quickstart_session",
            conversation_history=[],
            domain_context={"domain": "business"},
            temporal_context={"timeframe": "current"}
        ),
        max_hops=3,
        explanation_level="detailed"
    )
    
    # Process query
    response = await service.process_query(query)
    
    print(f"✅ Query processed successfully!")
    print(f"   Question: {query.query_text}")
    print(f"   Answer: {response.answer}")
    print(f"   Confidence: {response.confidence_score:.2f}")
    print(f"   Processing time: {response.processing_time_ms:.1f}ms")
    
    # Show reasoning chain
    if response.reasoning_chain:
        print(f"   Reasoning steps: {len(response.reasoning_chain.steps)}")
    
    return response

# Query the knowledge graph
response = asyncio.run(query_knowledge_graph(graph.knowledge_graph_id))
```

### Step 5: Visualize Your Knowledge Graph

```python
from capabilities.common.graphrag.visualization import GraphVisualizationEngine, VisualizationConfig

async def visualize_graph(knowledge_graph_id):
    # Create visualization engine
    viz_engine = GraphVisualizationEngine(service.db_service)
    
    # Configure visualization
    config = VisualizationConfig(
        width=1200,
        height=800,
        enable_3d=False,
        layout_algorithm="spring",
        max_nodes=100
    )
    
    # Generate visualization
    viz_data = await viz_engine.generate_graph_visualization(
        tenant_id="quickstart_tenant",
        knowledge_graph_id=knowledge_graph_id,
        config=config
    )
    
    print(f"✅ Visualization generated!")
    print(f"   Nodes: {len(viz_data.nodes)}")
    print(f"   Edges: {len(viz_data.edges)}")
    print(f"   Clusters: {len(viz_data.clusters)}")
    
    # Export as JSON for web visualization
    export_result = await viz_engine.export_visualization(
        viz_data, 
        format="json"
    )
    
    print(f"   Exported as JSON: {len(export_result['data'])} bytes")
    
    return viz_data

# Visualize the graph
viz_data = asyncio.run(visualize_graph(graph.knowledge_graph_id))
```

## 🎯 Complete Quick Start Example

Here's a complete example that ties everything together:

```python
import asyncio
from capabilities.common.graphrag.service import GraphRAGService
from capabilities.common.graphrag.database import GraphRAGDatabaseService
from capabilities.common.graphrag.ollama_integration import OllamaClient, OllamaConfig
from capabilities.common.graphrag.views import (
    KnowledgeGraphRequest, DocumentProcessingRequest, 
    GraphRAGQuery, QueryContext
)

async def complete_quickstart_demo():
    """Complete GraphRAG demonstration"""
    
    print("🚀 APG GraphRAG Quick Start Demo")
    print("=" * 50)
    
    # Step 1: Initialize services
    print("\n1️⃣ Initializing services...")
    
    db_service = GraphRAGDatabaseService()
    await db_service.initialize()
    
    ollama_config = OllamaConfig(
        base_url="http://localhost:11434",
        embedding_model="bge-m3",
        generation_models=["qwen3", "deepseek-r1"]
    )
    ollama_client = OllamaClient(ollama_config)
    await ollama_client.initialize()
    
    service = await GraphRAGService.create(
        db_service=db_service,
        ollama_client=ollama_client
    )
    
    print("✅ Services initialized!")
    
    # Step 2: Create knowledge graph
    print("\n2️⃣ Creating knowledge graph...")
    
    graph_request = KnowledgeGraphRequest(
        tenant_id="demo_tenant",
        name="Demo Knowledge Graph",
        description="Demonstration of APG GraphRAG capabilities",
        domain="technology"
    )
    
    graph = await service.create_knowledge_graph(graph_request)
    print(f"✅ Created graph: {graph.name}")
    
    # Step 3: Process documents
    print("\n3️⃣ Processing documents...")
    
    documents = [
        {
            "title": "Company Overview",
            "content": """
            TechCorp is a leading AI company founded by Dr. Sarah Johnson in 2021.
            The company is headquartered in Seattle and focuses on natural language processing.
            TechCorp has 150 employees and recently secured $25M in Series B funding.
            """
        },
        {
            "title": "Product Information", 
            "content": """
            TechCorp's flagship product is NLP-Pro, an advanced language model platform.
            The platform is used by Fortune 500 companies for customer service automation.
            NLP-Pro processes over 1 million queries daily with 98% accuracy.
            """
        }
    ]
    
    for doc in documents:
        doc_request = DocumentProcessingRequest(
            tenant_id="demo_tenant",
            knowledge_graph_id=graph.knowledge_graph_id,
            title=doc["title"],
            content=doc["content"],
            source_type="text"
        )
        
        result = await service.process_document(doc_request)
        print(f"   📄 {doc['title']}: {result.entities_extracted} entities, {result.relationships_extracted} relationships")
    
    # Step 4: Query the knowledge graph  
    print("\n4️⃣ Querying knowledge graph...")
    
    queries = [
        "Who founded TechCorp and when?",
        "What is TechCorp's main product and how is it performing?",
        "Where is TechCorp located and how many employees do they have?"
    ]
    
    for query_text in queries:
        query = GraphRAGQuery(
            tenant_id="demo_tenant",
            knowledge_graph_id=graph.knowledge_graph_id,
            query_text=query_text,
            query_type="factual",
            context=QueryContext(
                user_id="demo_user",
                session_id="demo_session",
                conversation_history=[],
                domain_context={"domain": "technology"}
            )
        )
        
        response = await service.process_query(query)
        print(f"   ❓ {query_text}")
        print(f"   💡 {response.answer}")
        print(f"   📊 Confidence: {response.confidence_score:.2f}")
        print()
    
    # Step 5: Get analytics
    print("5️⃣ Analytics overview...")
    
    analytics = await service.get_analytics_overview("demo_tenant")
    
    print(f"   📈 Knowledge Graphs: {analytics.get('knowledge_graphs', {}).get('total', 0)}")
    print(f"   🔗 Total Entities: {analytics.get('entities', {}).get('total', 0)}")
    print(f"   🔀 Total Relationships: {analytics.get('relationships', {}).get('total', 0)}")
    print(f"   📊 Queries Today: {analytics.get('queries', {}).get('today', 0)}")
    
    print("\n🎉 Quick start demo completed successfully!")
    print("Next steps:")
    print("  - Explore the User Guide for advanced features")
    print("  - Check out API documentation for integration")
    print("  - Try the visualization capabilities")
    
    # Cleanup
    await service.cleanup()

# Run the complete demo
if __name__ == "__main__":
    asyncio.run(complete_quickstart_demo())
```

## 🎯 Next Steps

Congratulations! You've successfully:

1. ✅ Set up APG GraphRAG
2. ✅ Created a knowledge graph
3. ✅ Processed documents with AI extraction
4. ✅ Queried with multi-hop reasoning
5. ✅ Generated visualizations

### What's Next?

1. **[User Guide](./user_guide.md)** - Explore advanced features
2. **[API Reference](./api_reference.md)** - Complete API documentation
3. **[Examples](./examples/)** - More code examples and tutorials
4. **[Configuration](./configuration.md)** - Advanced configuration options
5. **[Performance Tuning](./performance.md)** - Optimize for your use case

### Advanced Features to Explore

- **Real-time Updates** - Update knowledge graphs incrementally
- **Collaborative Curation** - Expert-driven knowledge validation
- **Contextual Intelligence** - Adaptive learning from user interactions
- **Advanced Visualization** - 3D graphs, temporal views, clustering
- **Multi-hop Reasoning** - Complex reasoning chains with explanation
- **Performance Analytics** - Detailed usage and performance metrics

### Integration Options

- **Flask-AppBuilder** - Web interface integration
- **REST API** - Programmatic access with 40+ endpoints
- **Visualization** - Interactive graph exploration
- **Export/Import** - Multiple format support

## 🆘 Troubleshooting

### Common Issues

**PostgreSQL Connection Error**
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Verify Apache AGE extension
psql -d graphrag_db -c "SELECT * FROM pg_extension WHERE extname = 'age';"
```

**Ollama Model Not Found**
```bash
# List installed models
ollama list

# Pull missing models
ollama pull bge-m3
ollama pull qwen3
```

**Import Errors**
```bash
# Verify installation
pip list | grep graphrag

# Reinstall if needed
pip install -e capabilities/common/graphrag --force-reinstall
```

For more help, see our [Troubleshooting Guide](./troubleshooting.md).

---

**Ready to build revolutionary knowledge graphs?** Continue with the [User Guide](./user_guide.md) for advanced features!