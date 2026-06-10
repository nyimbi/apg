#!/usr/bin/env python3
"""
APG Metadata Management - REST API Client Examples
Demonstrates how to interact with the metadata management API using HTTP requests

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class APIClient:
    """REST API client for APG Metadata Management"""
    base_url: str = "http://localhost:5000"
    api_version: str = "v1"
    tenant_id: str = "default"
    
    def __post_init__(self):
        self.api_url = f"{self.base_url}/api/{self.api_version}/metadata"
        self.headers = {
            "Content-Type": "application/json",
            "X-Tenant-ID": self.tenant_id,
            "User-Agent": "APG-Metadata-Client/1.0"
        }
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
        """Make HTTP request to the API"""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data, params=params)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, json=data, params=params)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=self.headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"   Error details: {error_detail}")
                except Exception:
                    print(f"   Response: {e.response.text}")
            raise
    
    def get_health(self) -> Dict:
        """Get service health status"""
        return self._make_request('GET', '/health')
    
    def get_metrics(self) -> Dict:
        """Get service performance metrics"""
        return self._make_request('GET', '/metrics')
    
    def create_discovery_schedule(self, schedule_config: Dict) -> Dict:
        """Create a new discovery schedule"""
        return self._make_request('POST', '/discovery/schedules', data=schedule_config)
    
    def run_discovery_job(self, schedule_id: str, override_config: Dict = None) -> Dict:
        """Run a discovery job"""
        data = {'override_config': override_config} if override_config else {}
        return self._make_request('POST', f'/discovery/jobs/{schedule_id}/run', data=data)
    
    def get_discovery_job_status(self, job_id: str) -> Dict:
        """Get discovery job status"""
        return self._make_request('GET', f'/discovery/jobs/{job_id}')
    
    def search_assets(self, query_text: str, filters: Dict = None, limit: int = 50, 
                     enable_natural_language: bool = True) -> Dict:
        """Search metadata assets"""
        data = {
            'query_text': query_text,
            'filters': filters or {},
            'limit': limit,
            'enable_natural_language': enable_natural_language
        }
        return self._make_request('POST', '/search', data=data)
    
    def list_assets(self, filters: Dict = None, limit: int = 100, offset: int = 0) -> Dict:
        """List metadata assets"""
        params = {'limit': limit, 'offset': offset}
        if filters:
            params.update(filters)
        return self._make_request('GET', '/assets', params=params)
    
    def get_asset(self, asset_id: str) -> Dict:
        """Get asset by ID"""
        return self._make_request('GET', f'/assets/{asset_id}')
    
    def get_asset_lineage(self, asset_id: str, direction: str = 'both', 
                         max_depth: int = 5) -> Dict:
        """Get asset lineage"""
        params = {'direction': direction, 'max_depth': max_depth}
        return self._make_request('GET', f'/assets/{asset_id}/lineage', params=params)
    
    def analyze_asset_impact(self, asset_id: str, change_type: str = 'schema_change',
                           change_details: Dict = None) -> Dict:
        """Analyze asset impact"""
        data = {
            'change_type': change_type,
            'change_details': change_details or {}
        }
        return self._make_request('POST', f'/assets/{asset_id}/impact', data=data)
    
    def add_lineage_relationship(self, source_asset_id: str, target_asset_id: str,
                               lineage_type: str, transformation_logic: str = None) -> Dict:
        """Add lineage relationship"""
        data = {
            'source_asset_id': source_asset_id,
            'target_asset_id': target_asset_id,
            'lineage_type': lineage_type,
            'transformation_logic': transformation_logic
        }
        return self._make_request('POST', '/lineage', data=data)
    
    def classify_column_data(self, column_name: str, data_type: str, 
                           sample_data: List, context: Dict = None) -> Dict:
        """Classify column data using AI"""
        data = {
            'column_name': column_name,
            'data_type': data_type,
            'sample_data': sample_data,
            'context': context or {}
        }
        return self._make_request('POST', '/classification/classify', data=data)


def demo_health_and_metrics():
    """Demo: Check service health and metrics"""
    print("\n🏥 Health and Metrics Demo")
    print("=" * 35)
    
    client = APIClient()
    
    try:
        # Check health
        health = client.get_health()
        print(f"Service Status: {health['status']}")
        print(f"Uptime: {health['uptime_seconds']:.1f} seconds")
        print(f"Total Assets: {health['metrics']['total_assets']}")
        
        # Get metrics
        metrics = client.get_metrics()
        print(f"Request Count: {metrics.get('request_count', 0)}")
        print(f"Avg Response Time: {metrics.get('avg_response_time_ms', 0):.1f}ms")
        print(f"Error Rate: {metrics.get('error_rate', 0):.2%}")
        
        print("✅ Health check successful")
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")


def demo_discovery_workflow():
    """Demo: Complete discovery workflow"""
    print("\n🔍 Discovery Workflow Demo")
    print("=" * 35)
    
    client = APIClient()
    
    # Create discovery schedule
    schedule_config = {
        'name': 'Demo PostgreSQL Discovery',
        'description': 'Discover sample PostgreSQL database',
        'connector_type': 'postgresql',
        'connection_params': {
            'host': 'localhost',
            'port': '5432',
            'database': 'demo_db',
            'username': 'postgres',
            'password': 'password'
        },
        'schedule_cron': '0 2 * * *',  # Daily at 2 AM
        'is_enabled': True,
        'is_one_time': False
    }
    
    try:
        print("Creating discovery schedule...")
        schedule_result = client.create_discovery_schedule(schedule_config)
        schedule_id = schedule_result['job_id']
        print(f"✅ Schedule created: {schedule_id}")
        
        # Run discovery job
        print("Starting discovery job...")
        job_result = client.run_discovery_job(schedule_id)
        job_id = job_result['job_id']
        print(f"✅ Job started: {job_id}")
        
        # Monitor job status
        print("Monitoring job progress...")
        for i in range(10):  # Check up to 10 times
            status = client.get_discovery_job_status(job_id)
            print(f"   Status: {status.get('status', 'unknown')}")
            
            if status.get('status') in ['completed', 'failed']:
                break
                
            time.sleep(2)  # Wait 2 seconds between checks
        
        print("✅ Discovery workflow completed")
        
    except Exception as e:
        print(f"❌ Discovery workflow failed: {str(e)}")


def demo_intelligent_search():
    """Demo: Intelligent search capabilities"""
    print("\n🔍 Intelligent Search Demo")
    print("=" * 32)
    
    client = APIClient()
    
    # Natural language search queries
    search_queries = [
        "customer information with email addresses",
        "tables containing payment data", 
        "high quality data sources",
        "recently updated assets",
        "files with sensitive information"
    ]
    
    for query in search_queries:
        try:
            print(f"\n🔍 Searching: '{query}'")
            
            results = client.search_assets(
                query_text=query,
                enable_natural_language=True,
                limit=5
            )
            
            if results.get('results'):
                print(f"   ✅ Found {results['total_results']} assets")
                print(f"   ⏱️  Query time: {results.get('query_time_ms', 0):.1f}ms")
                
                # Show top results
                for asset in results['results'][:3]:
                    print(f"     📊 {asset['name']} ({asset['asset_type']})")
                    
                    if asset.get('quality_score'):
                        quality = asset['quality_score'] * 100
                        print(f"        Quality: {quality:.0f}%")
                        
            else:
                print("   • No results found")
                
        except Exception as e:
            print(f"   ❌ Search failed: {str(e)}")


def demo_lineage_analysis():
    """Demo: Lineage analysis and impact assessment"""
    print("\n🔗 Lineage Analysis Demo")
    print("=" * 30)
    
    client = APIClient()
    
    try:
        # First, list some assets to get real asset IDs
        print("Getting sample assets...")
        assets = client.list_assets(limit=5)
        
        if not assets.get('assets'):
            print("⚠️  No assets found. Run discovery first.")
            return
        
        sample_asset = assets['assets'][0]
        asset_id = sample_asset['id']
        asset_name = sample_asset['name']
        
        print(f"📊 Analyzing lineage for: {asset_name}")
        
        # Get lineage
        lineage = client.get_asset_lineage(
            asset_id=asset_id,
            direction='both',
            max_depth=3
        )
        
        print(f"✅ Found {len(lineage.get('lineage_paths', []))} lineage paths")
        
        # Perform impact analysis
        print("🎯 Performing impact analysis...")
        impact = client.analyze_asset_impact(
            asset_id=asset_id,
            change_type='schema_change',
            change_details={'column_added': 'new_field'}
        )
        
        affected_count = impact.get('total_impacted_assets', 0)
        print(f"✅ Impact analysis complete: {affected_count} assets affected")
        
        if impact.get('impacted_assets'):
            print("   📈 Top impacted assets:")
            for affected in impact['impacted_assets'][:3]:
                print(f"     - {affected['asset_name']} ({affected['impact_type']})")
        
    except Exception as e:
        print(f"❌ Lineage analysis failed: {str(e)}")


def demo_ai_classification():
    """Demo: AI-powered data classification"""
    print("\n🧠 AI Classification Demo")
    print("=" * 30)
    
    client = APIClient()
    
    # Sample data for classification
    classification_samples = [
        {
            'column_name': 'email_address',
            'data_type': 'varchar',
            'sample_data': ['john@example.com', 'jane@company.org', 'user@domain.net']
        },
        {
            'column_name': 'phone_number',
            'data_type': 'varchar',
            'sample_data': ['555-123-4567', '+1-555-987-6543', '(555) 111-2222']
        },
        {
            'column_name': 'credit_card',
            'data_type': 'varchar',
            'sample_data': ['4532-1234-5678-9012', '5555-4444-3333-2222', '378234123456789']
        },
        {
            'column_name': 'customer_id',
            'data_type': 'integer',
            'sample_data': [12345, 67890, 11111]
        }
    ]
    
    for sample in classification_samples:
        try:
            print(f"\n🔬 Classifying: {sample['column_name']}")
            
            result = client.classify_column_data(
                column_name=sample['column_name'],
                data_type=sample['data_type'],
                sample_data=sample['sample_data'],
                context={'source_table': 'customer_data'}
            )
            
            classification = result.get('classification', 'unknown')
            confidence = result.get('confidence_score', 0) * 100
            
            print(f"   ✅ Classification: {classification}")
            print(f"   🎯 Confidence: {confidence:.0f}%")
            
            if result.get('tags'):
                print(f"   🏷️  Tags: {', '.join(result['tags'][:3])}")
                
        except Exception as e:
            print(f"   ❌ Classification failed: {str(e)}")


def demo_asset_management():
    """Demo: Asset browsing and management"""
    print("\n📊 Asset Management Demo")
    print("=" * 30)
    
    client = APIClient()
    
    try:
        # List assets with filtering
        print("📋 Listing assets...")
        
        filters = {'asset_type': 'table', 'limit': 10}
        assets = client.list_assets(filters=filters)
        
        total_assets = assets.get('pagination', {}).get('total', 0)
        print(f"✅ Found {total_assets} total assets")
        
        # Show sample assets
        for asset in assets.get('assets', [])[:5]:
            print(f"   📊 {asset['name']}")
            print(f"      Type: {asset['asset_type']}")
            print(f"      System: {asset.get('source_system', 'unknown')}")
            
            if asset.get('quality_score'):
                quality = asset['quality_score'] * 100
                print(f"      Quality: {quality:.0f}%")
            
            # Get detailed asset information
            try:
                asset_detail = client.get_asset(asset['id'])
                description = asset_detail.get('description', 'No description')
                print(f"      Description: {description[:50]}{'...' if len(description) > 50 else ''}")
                
            except Exception:
                pass  # Skip if asset detail fetch fails
                
        print("✅ Asset management demo completed")
        
    except Exception as e:
        print(f"❌ Asset management demo failed: {str(e)}")


def main():
    """Run all API demos"""
    print("🌟 APG Metadata Management - REST API Demos")
    print("=" * 55)
    
    try:
        demo_health_and_metrics()
        demo_discovery_workflow()
        demo_intelligent_search()
        demo_lineage_analysis()
        demo_ai_classification()
        demo_asset_management()
        
        print("\n" + "=" * 55)
        print("🎉 All API demos completed!")
        print("\nNext Steps:")
        print("1. Explore the interactive API docs: http://localhost:5000/api/v1/docs/")
        print("2. Check out the web UI: http://localhost:5000/metadata/dashboard")
        print("3. Integrate these examples into your applications")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demos interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo execution failed: {str(e)}")


if __name__ == "__main__":
    main()