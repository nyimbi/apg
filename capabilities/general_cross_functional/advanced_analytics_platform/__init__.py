"""
APG Advanced Analytics Platform

Self-service analytics, data science workbench, and machine learning
platform for enterprise-wide analytics and insights.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import List, Dict, Any, Optional
from enum import Enum

# Advanced Analytics Platform Metadata
__version__ = "1.0.0"
__capability_id__ = "advanced_analytics_platform"
__description__ = "Enterprise analytics and data science platform"

class AnalyticsType(str, Enum):
	"""Types of analytics supported."""
	DESCRIPTIVE = "descriptive"
	DIAGNOSTIC = "diagnostic"
	PREDICTIVE = "predictive"
	PRESCRIPTIVE = "prescriptive"
	REAL_TIME = "real_time"
	STREAMING = "streaming"

class DataSourceType(str, Enum):
	"""Types of data sources."""
	DATABASE = "database"
	API = "api"
	FILE_UPLOAD = "file_upload"
	STREAMING = "streaming"
	EXTERNAL_SERVICE = "external_service"
	CLOUD_STORAGE = "cloud_storage"

class ModelType(str, Enum):
	"""Types of ML models supported."""
	REGRESSION = "regression"
	CLASSIFICATION = "classification"
	CLUSTERING = "clustering"
	TIME_SERIES = "time_series"
	DEEP_LEARNING = "deep_learning"
	NLP = "nlp"
	COMPUTER_VISION = "computer_vision"

# Sub-capability Registry
SUBCAPABILITIES = [
	"data_discovery_catalog",
	"visual_analytics_builder",
	"ml_workbench",
	"predictive_modeling",
	"real_time_streaming_analytics",
	"self_service_reporting",
	"data_science_collaboration",
	"model_deployment_management"
]

# Feature Set
PLATFORM_FEATURES = {
	"data_discovery": {
		"description": "Automated data discovery and cataloging",
		"capabilities": [
			"Data profiling and quality assessment",
			"Automated schema discovery",
			"Data lineage tracking",
			"Metadata management",
			"Data governance integration"
		]
	},
	"visual_analytics": {
		"description": "Drag-and-drop visual analytics builder",
		"capabilities": [
			"Interactive dashboards",
			"Custom visualizations",
			"Real-time data refresh",
			"Embedded analytics",
			"Mobile-responsive design"
		]
	},
	"ml_workbench": {
		"description": "Collaborative machine learning environment",
		"capabilities": [
			"Jupyter notebook integration",
			"AutoML capabilities",
			"Model versioning",
			"Experiment tracking",
			"Collaborative development"
		]
	},
	"predictive_modeling": {
		"description": "Advanced predictive modeling platform",
		"capabilities": [
			"Time series forecasting",
			"Customer behavior prediction",
			"Risk modeling",
			"Optimization algorithms",
			"What-if scenario analysis"
		]
	}
}

# APG Composition Engine Registration
CAPABILITY_METADATA = {
	"capability_id": "general_cross_functional.advanced_analytics_platform",
	"version": __version__,
	"category": "analytics_platform",
	"subcapabilities": SUBCAPABILITIES,
	"dependencies": [
		"auth_rbac",
		"audit_compliance",
		"general_cross_functional.integration_api_management",
		"emerging_technologies.artificial_intelligence"
	],
	"provides_services": [
		"data_discovery_services",
		"visual_analytics_services",
		"machine_learning_services",
		"predictive_analytics_services",
		"real_time_analytics_services",
		"self_service_reporting_services"
	],
	"integrates_with": [
		"All APG Capabilities",
		"External BI Tools",
		"Cloud Analytics Services",
		"Data Warehouses"
	],
	"data_models": ["AADataSource", "AADataset", "AAModel", "AAExperiment", "AADashboard"],
	"features": PLATFORM_FEATURES
}

def get_capability_info() -> Dict[str, Any]:
	"""Get advanced analytics platform capability information."""
	return CAPABILITY_METADATA

def list_subcapabilities() -> List[str]:
	"""List all available subcapabilities."""
	return SUBCAPABILITIES.copy()

def get_analytics_types() -> List[AnalyticsType]:
	"""Get supported analytics types."""
	return list(AnalyticsType)

def get_data_source_types() -> List[DataSourceType]:
	"""Get supported data source types."""
	return list(DataSourceType)

def get_model_types() -> List[ModelType]:
	"""Get supported ML model types."""
	return list(ModelType)

def get_platform_features() -> Dict[str, Any]:
	"""Get platform feature details."""
	return PLATFORM_FEATURES.copy()

__all__ = [
	"AnalyticsType",
	"DataSourceType",
	"ModelType",
	"SUBCAPABILITIES",
	"PLATFORM_FEATURES",
	"CAPABILITY_METADATA",
	"get_capability_info",
	"list_subcapabilities",
	"get_analytics_types",
	"get_data_source_types",
	"get_model_types",
	"get_platform_features"
]