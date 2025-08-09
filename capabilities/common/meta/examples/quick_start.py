#!/usr/bin/env python3
"""
APG Metadata Management - Quick Start Example
Demonstrates how to quickly get started with the metadata management capability

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
from typing import Dict, Any

# Import the metadata capability
from capabilities.common.meta import (
    initialize_capability,
    discover_database,
    search_assets,
    get_asset_lineage,
    get_capability_info
)


async def quick_start_example():
    """
    Quick start example showing basic metadata management operations
    """
    print("🚀 APG Metadata Management - Quick Start Example")
    print("=" * 60)
    
    # Step 1: Initialize the capability
    print("\n1. Initializing Metadata Management Capability...")
    
    config = {
        "database": {
            "postgresql_url": "postgresql://localhost/apg_metadata",
            "neo4j_url": "bolt://localhost:7687",
            "redis_url": "redis://localhost:6379"
        },
        "discovery": {
            "enable_auto_discovery": True,
            "discovery_interval_hours": 24
        },
        "ai_classifier": {
            "enable_ai_classification": True,
            "ollama_base_url": "http://localhost:11434"
        }
    }
    
    try:
        service = await initialize_capability(config)
        print("✓ Metadata service initialized successfully!")
        
        # Get capability info
        info = get_capability_info()
        print(f"✓ Capability: {info['display_name']} v{info['version']}")
        print(f"✓ Features: {', '.join(info['capabilities'])}")
        
    except Exception as e:
        print(f"❌ Failed to initialize service: {str(e)}")
        return
    
    # Step 2: Discover a database
    print("\n2. Discovering Database Assets...")
    
    database_config = {
        "name": "sample_postgresql_db",
        "type": "postgresql",
        "host": "localhost",
        "port": "5432",
        "database": "sample_db",
        "username": "postgres",
        "password": "password"
    }
    
    try:
        job_id = await discover_database(database_config, tenant_id="demo_tenant")
        print(f"✓ Discovery job started: {job_id}")
        print("  This will scan the database and create metadata assets")
        
    except Exception as e:
        print(f"❌ Discovery failed: {str(e)}")
    
    # Step 3: Search for assets
    print("\n3. Searching Metadata Assets...")
    
    search_queries = [
        "customer data with email addresses",
        "user tables",
        "payment information",
        "tables containing phone numbers"
    ]
    
    for query in search_queries:
        try:
            print(f"\n   Searching: '{query}'")
            results = await search_assets(
                query_text=query,
                tenant_id="demo_tenant",
                limit=5
            )
            
            if results.get('results'):
                print(f"   ✓ Found {len(results['results'])} assets")
                for asset in results['results'][:2]:  # Show first 2
                    print(f"     - {asset['name']} ({asset['asset_type']})")
            else:
                print("   • No assets found")
                
        except Exception as e:
            print(f"   ❌ Search failed: {str(e)}")
    
    # Step 4: Get lineage for an asset
    print("\n4. Analyzing Asset Lineage...")
    
    # This would use a real asset_id from discovery results
    sample_asset_id = "asset_123"
    
    try:
        lineage = await get_asset_lineage(
            asset_id=sample_asset_id,
            tenant_id="demo_tenant",
            direction="both",
            max_depth=3
        )
        
        if lineage:
            print(f"✓ Found {len(lineage)} lineage paths")
            for i, path in enumerate(lineage[:2], 1):
                print(f"   Path {i}: {path.get('description', 'Data flow path')}")
        else:
            print("  • No lineage found (asset may not exist yet)")
            
    except Exception as e:
        print(f"❌ Lineage analysis failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Quick Start Example Completed!")
    print("\nNext Steps:")
    print("1. Check the web UI at http://localhost:5000/metadata/dashboard")
    print("2. Explore the API documentation at http://localhost:5000/api/v1/docs/")
    print("3. Review more examples in the examples/ directory")


async def discovery_example():
    """
    Detailed discovery example with multiple data sources
    """
    print("\n🔍 Advanced Discovery Example")
    print("=" * 40)
    
    # Multiple data source configurations
    data_sources = [
        {
            "name": "Production PostgreSQL",
            "type": "postgresql",
            "host": "prod-db.company.com",
            "port": "5432",
            "database": "production",
            "username": "readonly_user",
            "password": "secure_password"
        },
        {
            "name": "Analytics MySQL",
            "type": "mysql",
            "host": "analytics.company.com",
            "port": "3306",
            "database": "analytics",
            "username": "analyst",
            "password": "mysql_password"
        },
        {
            "name": "MongoDB Documents",
            "type": "mongodb",
            "host": "mongo.company.com",
            "port": "27017",
            "database": "documents",
            "username": "mongo_user",
            "password": "mongo_password"
        }
    ]
    
    discovery_jobs = []
    
    for source in data_sources:
        try:
            print(f"\nDiscovering: {source['name']} ({source['type']})")
            
            job_id = await discover_database(source, tenant_id="production")
            discovery_jobs.append(job_id)
            
            print(f"✓ Started discovery job: {job_id}")
            
        except Exception as e:
            print(f"❌ Failed to start discovery for {source['name']}: {str(e)}")
    
    print(f"\n✅ Started {len(discovery_jobs)} discovery jobs")
    print("These jobs will run in the background and populate the metadata catalog")
    
    return discovery_jobs


async def search_and_classification_example():
    """
    Example showing AI-powered search and classification
    """
    print("\n🧠 AI-Powered Search & Classification Example")
    print("=" * 50)
    
    # Natural language search queries
    ai_queries = [
        "Find all tables that might contain personally identifiable information",
        "Show me customer data sources with high quality scores",
        "Tables related to financial transactions or payments",
        "Data sources that were updated in the last week",
        "Assets with poor data quality that need attention"
    ]
    
    for query in ai_queries:
        print(f"\n🔍 AI Query: '{query}'")
        
        try:
            results = await search_assets(
                query_text=query,
                tenant_id="production",
                filters={"enable_ai_analysis": True},
                limit=10
            )
            
            if results.get('results'):
                print(f"   ✓ Found {results['total_results']} relevant assets")
                print(f"   ✓ Query time: {results.get('query_time_ms', 0):.1f}ms")
                
                # Show top results with AI classification
                for asset in results['results'][:3]:
                    print(f"     📊 {asset['name']}")
                    print(f"         Type: {asset['asset_type']}")
                    print(f"         Quality: {asset.get('quality_score', 0) * 100:.0f}%")
                    
                    if asset.get('ai_classification'):
                        classifications = asset['ai_classification']
                        print(f"         AI Tags: {', '.join(classifications[:3])}")
                        
            else:
                print("   • No matching assets found")
                
        except Exception as e:
            print(f"   ❌ AI search failed: {str(e)}")


async def lineage_analysis_example():
    """
    Example showing comprehensive lineage analysis
    """
    print("\n🔗 Advanced Lineage Analysis Example")
    print("=" * 45)
    
    # Sample asset IDs (in real usage, these would come from discovery)
    sample_assets = [
        {"id": "table_customers", "name": "customers table"},
        {"id": "table_orders", "name": "orders table"},
        {"id": "view_customer_metrics", "name": "customer_metrics view"}
    ]
    
    for asset in sample_assets:
        print(f"\n📈 Analyzing lineage for: {asset['name']}")
        
        try:
            # Get upstream lineage (data sources)
            upstream = await get_asset_lineage(
                asset_id=asset['id'],
                tenant_id="production",
                direction="upstream",
                max_depth=5
            )
            
            # Get downstream lineage (data consumers)
            downstream = await get_asset_lineage(
                asset_id=asset['id'],
                tenant_id="production", 
                direction="downstream",
                max_depth=5
            )
            
            print(f"   ✓ Upstream dependencies: {len(upstream)} found")
            print(f"   ✓ Downstream consumers: {len(downstream)} found")
            
            # Show sample paths
            if upstream:
                print("   📥 Sample upstream path:")
                path = upstream[0]
                steps = [step['asset_name'] for step in path.get('steps', [])]
                print(f"      {' → '.join(steps)}")
            
            if downstream:
                print("   📤 Sample downstream path:")
                path = downstream[0]
                steps = [step['asset_name'] for step in path.get('steps', [])]
                print(f"      {' → '.join(steps)}")
                
        except Exception as e:
            print(f"   ❌ Lineage analysis failed: {str(e)}")


async def main():
    """Main example runner"""
    print("🌟 APG Metadata Management - Comprehensive Examples")
    print("=" * 70)
    
    try:
        # Run all examples
        await quick_start_example()
        
        print("\n" + "=" * 70)
        await discovery_example()
        
        print("\n" + "=" * 70)
        await search_and_classification_example()
        
        print("\n" + "=" * 70)
        await lineage_analysis_example()
        
        print("\n" + "=" * 70)
        print("🎉 All examples completed successfully!")
        print("\nFor more information:")
        print("- Web UI: http://localhost:5000/metadata/")
        print("- API Docs: http://localhost:5000/api/v1/docs/")
        print("- GitHub: https://github.com/datacraft/apg-metadata")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Example execution failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())