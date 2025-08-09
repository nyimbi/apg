#!/usr/bin/env python3
"""
APG Metadata Management - Integration Patterns
Advanced integration examples showing how to integrate with various systems and frameworks

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# APG Metadata imports
from capabilities.common.meta import (
    initialize_capability,
    get_capability_instance,
    APGMetadataService
)
from capabilities.common.meta.connectors import ConnectorConfig
from capabilities.common.meta.discovery import DiscoverySchedule


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for various integrations"""
    airflow_dag_folder: str = "/opt/airflow/dags"
    jupyter_notebooks_path: str = "/data/notebooks"
    git_repositories: List[str] = None
    slack_webhook: str = None
    email_notifications: bool = True


class AirflowIntegration:
    """
    Integration with Apache Airflow for data pipeline metadata
    """
    
    def __init__(self, metadata_service: APGMetadataService, config: IntegrationConfig):
        self.metadata_service = metadata_service
        self.config = config
    
    async def discover_dags_metadata(self) -> Dict[str, Any]:
        """
        Discover metadata from Airflow DAGs
        """
        logger.info("🔍 Discovering Airflow DAGs metadata...")
        
        # This would typically scan DAG files and extract metadata
        # For demo purposes, we'll simulate some DAG metadata
        
        sample_dags = [
            {
                "dag_id": "customer_etl_pipeline",
                "description": "ETL pipeline for customer data processing",
                "schedule_interval": "@daily",
                "owner": "data_team",
                "tags": ["etl", "customer", "daily"],
                "tasks": [
                    {
                        "task_id": "extract_customer_data",
                        "operator": "PostgreSQLOperator",
                        "sql_query": "SELECT * FROM customers WHERE updated_at > {{ ds }}"
                    },
                    {
                        "task_id": "transform_customer_data", 
                        "operator": "PythonOperator",
                        "function": "transform_customer_data"
                    },
                    {
                        "task_id": "load_to_warehouse",
                        "operator": "BigQueryInsertJobOperator",
                        "destination_table": "analytics.customers"
                    }
                ]
            },
            {
                "dag_id": "sales_reporting_pipeline",
                "description": "Generate daily sales reports",
                "schedule_interval": "0 6 * * *",
                "owner": "analytics_team",
                "tags": ["reporting", "sales", "daily"],
                "tasks": [
                    {
                        "task_id": "aggregate_sales_data",
                        "operator": "SQLOperator",
                        "sql_query": "SELECT date, sum(amount) FROM sales GROUP BY date"
                    },
                    {
                        "task_id": "generate_report",
                        "operator": "PythonOperator", 
                        "function": "generate_sales_report"
                    }
                ]
            }
        ]
        
        # Process each DAG and create lineage relationships
        lineage_count = 0
        
        for dag in sample_dags:
            logger.info(f"   Processing DAG: {dag['dag_id']}")
            
            # Create asset for the DAG itself
            dag_asset = {
                "name": dag['dag_id'],
                "display_name": dag['dag_id'].replace('_', ' ').title(),
                "description": dag['description'],
                "asset_type": "pipeline",
                "source_system": "airflow",
                "tags": dag['tags'],
                "owner": dag['owner'],
                "custom_attributes": {
                    "schedule_interval": dag['schedule_interval'],
                    "dag_type": "etl_pipeline"
                }
            }
            
            # In a real implementation, you would use the metadata service to create assets
            logger.info(f"     Created pipeline asset: {dag_asset['name']}")
            
            # Process tasks and create lineage
            for task in dag['tasks']:
                if 'sql_query' in task:
                    # Extract table names from SQL (simplified)
                    sql = task['sql_query'].lower()
                    if 'from' in sql:
                        # Simple parsing - in reality you'd use a proper SQL parser
                        source_tables = self._extract_table_names(sql)
                        logger.info(f"     Found source tables: {source_tables}")
                        lineage_count += len(source_tables)
                
                if task.get('destination_table'):
                    logger.info(f"     Found destination table: {task['destination_table']}")
                    lineage_count += 1
        
        logger.info(f"✅ Discovered {len(sample_dags)} DAGs with {lineage_count} lineage relationships")
        
        return {
            "dags_discovered": len(sample_dags),
            "lineage_relationships": lineage_count,
            "discovery_time": datetime.utcnow().isoformat()
        }
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL (simplified implementation)"""
        # This is a very basic implementation - use a proper SQL parser in production
        import re
        
        # Find table names after FROM, JOIN keywords
        pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        
        return [match.strip() for match in matches]


class JupyterIntegration:
    """
    Integration with Jupyter notebooks for data science workflow metadata
    """
    
    def __init__(self, metadata_service: APGMetadataService, config: IntegrationConfig):
        self.metadata_service = metadata_service
        self.config = config
    
    async def discover_notebooks_metadata(self) -> Dict[str, Any]:
        """
        Discover metadata from Jupyter notebooks
        """
        logger.info("📓 Discovering Jupyter notebooks metadata...")
        
        # Sample notebook metadata (would typically scan .ipynb files)
        sample_notebooks = [
            {
                "filename": "customer_analysis.ipynb",
                "title": "Customer Behavior Analysis",
                "author": "data_scientist@company.com",
                "created_date": "2024-12-01T10:00:00Z",
                "last_modified": "2024-12-15T14:30:00Z",
                "description": "Analysis of customer purchase patterns and segmentation",
                "tags": ["analysis", "customers", "segmentation"],
                "data_sources": [
                    "postgresql://prod-db/customers",
                    "postgresql://prod-db/orders",
                    "s3://analytics-bucket/customer-events/"
                ],
                "outputs": [
                    "customer_segments.csv",
                    "analysis_results.json"
                ],
                "libraries_used": ["pandas", "scikit-learn", "matplotlib", "seaborn"],
                "cell_count": 45,
                "markdown_cells": 15,
                "code_cells": 30
            },
            {
                "filename": "sales_forecasting.ipynb",
                "title": "Sales Forecasting Model",
                "author": "ml_engineer@company.com", 
                "created_date": "2024-11-15T09:00:00Z",
                "last_modified": "2024-12-10T16:45:00Z",
                "description": "Machine learning model for sales forecasting",
                "tags": ["ml", "forecasting", "sales", "time-series"],
                "data_sources": [
                    "postgresql://prod-db/sales_history",
                    "api://weather-service/historical-data"
                ],
                "outputs": [
                    "models/sales_forecast_model.pkl",
                    "predictions/sales_forecast_2025.csv"
                ],
                "libraries_used": ["pandas", "tensorflow", "prophet", "plotly"],
                "cell_count": 67,
                "markdown_cells": 22,
                "code_cells": 45
            }
        ]
        
        assets_created = 0
        lineage_relationships = 0
        
        for notebook in sample_notebooks:
            logger.info(f"   Processing notebook: {notebook['filename']}")
            
            # Create notebook asset
            notebook_asset = {
                "name": notebook['filename'],
                "display_name": notebook['title'],
                "description": notebook['description'],
                "asset_type": "notebook",
                "source_system": "jupyter",
                "tags": notebook['tags'],
                "owner": notebook['author'],
                "custom_attributes": {
                    "cell_count": notebook['cell_count'],
                    "markdown_cells": notebook['markdown_cells'],
                    "code_cells": notebook['code_cells'],
                    "libraries_used": notebook['libraries_used'],
                    "last_modified": notebook['last_modified']
                }
            }
            
            assets_created += 1
            logger.info(f"     Created notebook asset: {notebook_asset['name']}")
            
            # Create lineage for data sources
            for source in notebook['data_sources']:
                logger.info(f"     Data source dependency: {source}")
                lineage_relationships += 1
            
            # Create lineage for outputs
            for output in notebook['outputs']:
                logger.info(f"     Output generated: {output}")
                lineage_relationships += 1
        
        logger.info(f"✅ Discovered {len(sample_notebooks)} notebooks with {lineage_relationships} lineage relationships")
        
        return {
            "notebooks_discovered": len(sample_notebooks),
            "assets_created": assets_created,
            "lineage_relationships": lineage_relationships,
            "discovery_time": datetime.utcnow().isoformat()
        }


class GitIntegration:
    """
    Integration with Git repositories for code-based data assets
    """
    
    def __init__(self, metadata_service: APGMetadataService, config: IntegrationConfig):
        self.metadata_service = metadata_service
        self.config = config
    
    async def discover_data_code_metadata(self) -> Dict[str, Any]:
        """
        Discover metadata from code repositories (SQL files, data scripts, etc.)
        """
        logger.info("📂 Discovering data code metadata from Git repositories...")
        
        # Sample repository analysis results
        sample_repos = [
            {
                "repo_name": "data-warehouse-sql",
                "repo_url": "https://github.com/company/data-warehouse-sql",
                "branch": "main",
                "last_commit": "2024-12-15T10:30:00Z",
                "sql_files": [
                    {
                        "path": "models/customers/customer_360_view.sql",
                        "description": "Customer 360 degree view combining multiple data sources",
                        "tables_referenced": ["customers", "orders", "customer_interactions"],
                        "creates_table": "customer_360_view",
                        "author": "data_engineer@company.com",
                        "last_modified": "2024-12-10T14:20:00Z"
                    },
                    {
                        "path": "etl/daily_sales_aggregation.sql",
                        "description": "Daily sales aggregation for reporting",
                        "tables_referenced": ["sales_transactions", "products", "customers"],
                        "creates_table": "daily_sales_summary",
                        "author": "analytics_engineer@company.com",
                        "last_modified": "2024-12-08T09:15:00Z"
                    }
                ],
                "python_scripts": [
                    {
                        "path": "scripts/data_quality_checks.py",
                        "description": "Data quality validation scripts",
                        "tables_accessed": ["all_tables"],
                        "author": "data_engineer@company.com",
                        "last_modified": "2024-12-12T11:45:00Z"
                    }
                ]
            }
        ]
        
        assets_created = 0
        lineage_relationships = 0
        
        for repo in sample_repos:
            logger.info(f"   Processing repository: {repo['repo_name']}")
            
            # Process SQL files
            for sql_file in repo['sql_files']:
                logger.info(f"     Processing SQL file: {sql_file['path']}")
                
                # Create asset for SQL file
                sql_asset = {
                    "name": sql_file['path'],
                    "display_name": sql_file['path'].split('/')[-1].replace('.sql', '').replace('_', ' ').title(),
                    "description": sql_file['description'],
                    "asset_type": "sql_script",
                    "source_system": "git",
                    "tags": ["sql", "etl", "transformation"],
                    "owner": sql_file['author'],
                    "custom_attributes": {
                        "repository": repo['repo_name'],
                        "file_path": sql_file['path'],
                        "last_commit": repo['last_commit'],
                        "creates_table": sql_file.get('creates_table')
                    }
                }
                
                assets_created += 1
                
                # Create lineage relationships
                for table in sql_file['tables_referenced']:
                    logger.info(f"       References table: {table}")
                    lineage_relationships += 1
                
                if sql_file.get('creates_table'):
                    logger.info(f"       Creates table: {sql_file['creates_table']}")
                    lineage_relationships += 1
            
            # Process Python scripts
            for script in repo['python_scripts']:
                logger.info(f"     Processing Python script: {script['path']}")
                
                script_asset = {
                    "name": script['path'],
                    "display_name": script['path'].split('/')[-1].replace('.py', '').replace('_', ' ').title(),
                    "description": script['description'],
                    "asset_type": "python_script",
                    "source_system": "git",
                    "tags": ["python", "data_processing", "script"],
                    "owner": script['author'],
                    "custom_attributes": {
                        "repository": repo['repo_name'],
                        "file_path": script['path'],
                        "last_commit": repo['last_commit']
                    }
                }
                
                assets_created += 1
        
        logger.info(f"✅ Discovered {assets_created} code assets with {lineage_relationships} lineage relationships")
        
        return {
            "repositories_scanned": len(sample_repos),
            "assets_created": assets_created,
            "lineage_relationships": lineage_relationships,
            "discovery_time": datetime.utcnow().isoformat()
        }


class EventDrivenIntegration:
    """
    Event-driven integration for real-time metadata updates
    """
    
    def __init__(self, metadata_service: APGMetadataService, config: IntegrationConfig):
        self.metadata_service = metadata_service
        self.config = config
    
    async def setup_event_listeners(self):
        """
        Setup event listeners for real-time metadata updates
        """
        logger.info("🔔 Setting up event-driven metadata integration...")
        
        # Simulate setting up various event listeners
        event_sources = [
            {
                "source": "database_changes",
                "description": "Listen for database schema changes",
                "event_types": ["table_created", "column_added", "index_created"]
            },
            {
                "source": "data_pipeline_events", 
                "description": "Listen for data pipeline execution events",
                "event_types": ["pipeline_started", "pipeline_completed", "pipeline_failed"]
            },
            {
                "source": "data_quality_events",
                "description": "Listen for data quality check results",
                "event_types": ["quality_check_passed", "quality_check_failed", "anomaly_detected"]
            }
        ]
        
        for source in event_sources:
            logger.info(f"   📡 Setting up listener for: {source['source']}")
            logger.info(f"     Event types: {', '.join(source['event_types'])}")
            
            # In a real implementation, you would set up actual event listeners
            # For demo, we'll just log the setup
        
        logger.info("✅ Event listeners configured")
        
        # Simulate processing some events
        await self._simulate_event_processing()
    
    async def _simulate_event_processing(self):
        """Simulate processing real-time events"""
        logger.info("🔄 Simulating real-time event processing...")
        
        sample_events = [
            {
                "event_type": "table_created",
                "source": "production_db",
                "details": {
                    "table_name": "user_preferences",
                    "schema": "public",
                    "columns": ["user_id", "preference_key", "preference_value", "created_at"]
                },
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "event_type": "pipeline_completed",
                "source": "airflow",
                "details": {
                    "dag_id": "customer_etl_pipeline",
                    "run_id": "manual_2024-12-15",
                    "duration_seconds": 1847,
                    "records_processed": 125043
                },
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "event_type": "quality_check_failed",
                "source": "great_expectations",
                "details": {
                    "table_name": "orders",
                    "check_name": "expect_column_values_to_not_be_null",
                    "column": "customer_id",
                    "failure_count": 23
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        
        for event in sample_events:
            logger.info(f"   📥 Processing event: {event['event_type']}")
            
            # Process the event and update metadata
            await self._process_metadata_event(event)
            
            # Simulate some processing time
            await asyncio.sleep(0.1)
        
        logger.info("✅ Event processing simulation completed")
    
    async def _process_metadata_event(self, event: Dict[str, Any]):
        """Process a metadata event and update the catalog"""
        event_type = event['event_type']
        details = event['details']
        
        if event_type == "table_created":
            logger.info(f"     Creating metadata for new table: {details['table_name']}")
            # Would create new asset in metadata catalog
            
        elif event_type == "pipeline_completed":
            logger.info(f"     Updating pipeline execution metrics: {details['dag_id']}")
            # Would update pipeline metadata with execution statistics
            
        elif event_type == "quality_check_failed":
            logger.info(f"     Updating data quality metrics: {details['table_name']}")
            # Would update asset quality score and create data quality issues


class NotificationIntegration:
    """
    Integration for sending notifications about metadata events
    """
    
    def __init__(self, metadata_service: APGMetadataService, config: IntegrationConfig):
        self.metadata_service = metadata_service
        self.config = config
    
    async def send_discovery_notification(self, discovery_results: Dict[str, Any]):
        """Send notification about discovery results"""
        logger.info("📧 Sending discovery completion notification...")
        
        # Create notification message
        message = f"""
🔍 Metadata Discovery Completed

📊 Summary:
• Assets discovered: {discovery_results.get('assets_discovered', 0)}
• Lineage relationships: {discovery_results.get('lineage_relationships', 0)}
• Data quality issues: {discovery_results.get('quality_issues', 0)}
• Discovery time: {discovery_results.get('discovery_duration_minutes', 0)} minutes

🎯 Key Findings:
• High quality assets: {discovery_results.get('high_quality_assets', 0)}
• Assets needing attention: {discovery_results.get('attention_needed', 0)}
• New sensitive data found: {discovery_results.get('sensitive_data_found', 0)}

🔗 View Results: http://localhost:5000/metadata/dashboard
        """.strip()
        
        # Send to configured channels
        await self._send_to_slack(message)
        await self._send_email_notification(message)
        
        logger.info("✅ Notifications sent")
    
    async def _send_to_slack(self, message: str):
        """Send message to Slack (simulated)"""
        if self.config.slack_webhook:
            logger.info("   📱 Sent notification to Slack")
        else:
            logger.info("   📱 Slack webhook not configured, skipping")
    
    async def _send_email_notification(self, message: str):
        """Send email notification (simulated)"""
        if self.config.email_notifications:
            logger.info("   📧 Sent email notification")
        else:
            logger.info("   📧 Email notifications disabled, skipping")


async def comprehensive_integration_example():
    """
    Comprehensive example showing multiple integration patterns
    """
    print("🔗 Comprehensive Integration Patterns Demo")
    print("=" * 50)
    
    # Initialize metadata service
    logger.info("Initializing metadata service...")
    service = await initialize_capability()
    
    # Setup integration configuration
    config = IntegrationConfig(
        airflow_dag_folder="/opt/airflow/dags",
        jupyter_notebooks_path="/data/notebooks",
        git_repositories=["https://github.com/company/data-warehouse-sql"],
        slack_webhook="https://hooks.slack.com/services/...",
        email_notifications=True
    )
    
    # Initialize integrations
    airflow_integration = AirflowIntegration(service, config)
    jupyter_integration = JupyterIntegration(service, config)
    git_integration = GitIntegration(service, config)
    event_integration = EventDrivenIntegration(service, config)
    notification_integration = NotificationIntegration(service, config)
    
    try:
        # Run all integrations
        logger.info("\n🚀 Starting comprehensive integration workflow...")
        
        # 1. Discover Airflow metadata
        airflow_results = await airflow_integration.discover_dags_metadata()
        
        # 2. Discover Jupyter notebooks
        jupyter_results = await jupyter_integration.discover_notebooks_metadata()
        
        # 3. Discover Git repository metadata
        git_results = await git_integration.discover_data_code_metadata()
        
        # 4. Setup event-driven integration
        await event_integration.setup_event_listeners()
        
        # 5. Send completion notifications
        combined_results = {
            "assets_discovered": (
                airflow_results.get('dags_discovered', 0) +
                jupyter_results.get('notebooks_discovered', 0) +
                git_results.get('assets_created', 0)
            ),
            "lineage_relationships": (
                airflow_results.get('lineage_relationships', 0) +
                jupyter_results.get('lineage_relationships', 0) +
                git_results.get('lineage_relationships', 0)
            ),
            "discovery_duration_minutes": 15,  # Simulated
            "high_quality_assets": 45,
            "attention_needed": 3,
            "sensitive_data_found": 8,
            "quality_issues": 2
        }
        
        await notification_integration.send_discovery_notification(combined_results)
        
        logger.info("\n✅ Comprehensive integration workflow completed!")
        
        # Print summary
        print("\n📋 Integration Summary:")
        print(f"   • Total assets discovered: {combined_results['assets_discovered']}")
        print(f"   • Total lineage relationships: {combined_results['lineage_relationships']}")
        print(f"   • Event listeners configured: 3")
        print(f"   • Notifications sent: 2")
        
        print("\n🎯 Integration Benefits:")
        print("   • Automated metadata discovery from multiple sources")
        print("   • Real-time updates through event-driven integration")
        print("   • Proactive notifications for important changes")
        print("   • Comprehensive data lineage across the entire stack")
        
    except Exception as e:
        logger.error(f"❌ Integration workflow failed: {str(e)}")
        raise


async def main():
    """Main integration patterns demo"""
    try:
        await comprehensive_integration_example()
        
        print("\n🌟 Integration Patterns Demo Completed!")
        print("\nKey Takeaways:")
        print("1. APG Metadata Management integrates with your entire data stack")
        print("2. Event-driven architecture enables real-time metadata updates")
        print("3. Automated discovery reduces manual metadata management overhead")
        print("4. Comprehensive lineage tracking across tools and systems")
        print("5. Proactive notifications keep teams informed of important changes")
        
    except Exception as e:
        logger.error(f"❌ Integration demo failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())