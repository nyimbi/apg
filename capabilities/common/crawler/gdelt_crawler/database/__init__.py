"""
GDELT Database Integration Module
=================================

Comprehensive database integration for GDELT data with ML Deep Scorer support,
optimized queries, and production-ready ETL pipeline components.

Key Components:
- **ETL Pipeline**: GDELTDatabaseETL for data loading and processing
- **Database Models**: SQLAlchemy models for GDELT data structures
- **Optimized Queries**: High-performance queries for analysis and monitoring
- **ML Integration**: Support for ML Deep Scorer event extraction

Features:
- PostgreSQL optimization with connection pooling
- Batch processing with configurable sizes
- ML Deep Scorer integration for event extraction
- Conflict analysis and monitoring capabilities
- Real-time analytics and reporting
- Comprehensive error handling and logging

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

from .etl import GDELTDatabaseETL, ETLConfig, ETLMetrics, create_gdelt_etl
from .models import (
    Base, DataSource, InformationUnit, ProcessingLog, GDELTEventSummary,
    create_tables, drop_tables, get_table_info
)
from .queries import GDELTQueryOptimizer, QueryResult, execute_custom_query

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__company__ = "Datacraft"
__website__ = "www.datacraft.co.ke"
__license__ = "MIT"

# Export all public components
__all__ = [
    # ETL Components
    "GDELTDatabaseETL",
    "ETLConfig", 
    "ETLMetrics",
    "create_gdelt_etl",
    
    # Database Models
    "Base",
    "DataSource",
    "InformationUnit",
    "ProcessingLog", 
    "GDELTEventSummary",
    "create_tables",
    "drop_tables",
    "get_table_info",
    
    # Query Components
    "GDELTQueryOptimizer",
    "QueryResult",
    "execute_custom_query",
    
    # Version info
    "__version__",
    "__author__",
    "__company__",
    "__website__", 
    "__license__"
]