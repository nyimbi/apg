"""
Intelligence and Analytics Capability

Comprehensive intelligence gathering, analysis, presentation and management platform.
Supports multiple intelligence disciplines (OSINT, SIGINT, HUMINT, GEOINT, CYBINT, FININT)
with advanced analytics, correlation, and reporting capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Intelligence and Analytics',
	'code': 'INTEL',
	'version': '1.0.0',
	'description': 'Comprehensive intelligence gathering, analysis, presentation and management platform',
	'industry_focus': 'Government, Defense, Security, Law Enforcement, Corporate Intelligence',
	'classification_levels': ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP_SECRET'],
	'intelligence_disciplines': [
		'OSINT',  # Open Source Intelligence
		'SIGINT', # Signals Intelligence  
		'HUMINT', # Human Intelligence
		'GEOINT', # Geospatial Intelligence
		'CYBINT', # Cyber Intelligence
		'FININT', # Financial Intelligence
		'SOCINT', # Social Media Intelligence
	],
	'subcapabilities': [
		# Collection Capabilities
		'osint',       # Open Source Intelligence
		'sigint',      # Signals Intelligence
		'humint',      # Human Intelligence  
		'geoint',      # Geospatial Intelligence
		'cybint',      # Cyber Intelligence
		'finint',      # Financial Intelligence
		'socint',      # Social Media Intelligence
		'darkweb',     # Dark Web Monitoring
		'radio',       # Radio Intelligence Listener
		'surveillance', # Digital Surveillance
		'monitoring',  # Real-Time Monitoring
		
		# Analysis & Processing  
		'fusion',      # Intelligence Fusion
		'analytics',   # Intelligence Analytics
		'correlation', # Data Correlation
		'prediction',  # Predictive Intelligence
		'threats',     # Threat Intelligence
		
		# Presentation & Management
		'reporting',   # Intelligence Reporting
		'dashboard',   # Intelligence Dashboard
		'alerts',      # Alert Management
		'search',      # Intelligence Search
		'archive',     # Intelligence Archive
	],
	'implemented_subcapabilities': [
		'crawler',     # Web Crawler (legacy)
		'osint',       # Open Source Intelligence
		'sigint',      # Signals Intelligence
		'humint',      # Human Intelligence
		'geoint',      # Geospatial Intelligence
		'cybint',      # Cyber Intelligence
		'radio',       # Radio Intelligence Listener
	],
	'security_requirements': {
		'classification_handling': True,
		'compartmentalized_access': True,
		'need_to_know': True,
		'audit_logging': True,
		'data_sanitization': True,
		'secure_communications': True,
		'multi_level_security': True
	},
	'dependencies': [
		'auth_rbac',
		'audit_compliance', 
		'computer_vision',
		'nlp',
		'rag',
		'notification'
	],
	'optional_dependencies': [
		'graphrag',
		'real_time_collaboration',
		'geographical_location_services',
		'biometric'
	],
	'database_prefix': 'intel_',
	'menu_category': 'Intelligence',
	'menu_icon': 'fa-eye'
}

def get_capability_info() -> Dict[str, Any]:
	"""Get intelligence capability information"""
	return CAPABILITY_META

def get_intelligence_disciplines() -> List[str]:
	"""Get supported intelligence disciplines"""
	return CAPABILITY_META['intelligence_disciplines']

def get_subcapabilities() -> List[str]:
	"""Get list of available sub-capabilities"""
	return CAPABILITY_META['subcapabilities']

def get_implemented_subcapabilities() -> List[str]:
	"""Get list of currently implemented sub-capabilities"""
	return CAPABILITY_META['implemented_subcapabilities']

def get_collection_capabilities() -> List[str]:
	"""Get intelligence collection capabilities"""
	return [
		'osint', 'sigint', 'humint', 'geoint', 'cybint', 'finint', 
		'socint', 'darkweb', 'radio', 'surveillance', 'monitoring'
	]

def get_analysis_capabilities() -> List[str]:
	"""Get intelligence analysis capabilities"""
	return [
		'fusion', 'analytics', 'correlation', 'prediction', 'threats'
	]

def get_presentation_capabilities() -> List[str]:
	"""Get intelligence presentation capabilities"""
	return [
		'reporting', 'dashboard', 'alerts', 'search', 'archive'
	]

def validate_security_clearance(user_clearance: str, data_classification: str) -> bool:
	"""Validate if user clearance allows access to classified data"""
	clearance_hierarchy = {
		'UNCLASSIFIED': 0,
		'CONFIDENTIAL': 1,
		'SECRET': 2,
		'TOP_SECRET': 3
	}
	
	user_level = clearance_hierarchy.get(user_clearance, -1)
	data_level = clearance_hierarchy.get(data_classification, 999)
	
	return user_level >= data_level

def get_intelligence_cycle() -> List[Dict[str, Any]]:
	"""Get the standard intelligence cycle phases"""
	return [
		{
			'phase': 'Planning & Direction',
			'description': 'Identify intelligence requirements and priorities',
			'capabilities': ['requirements', 'planning', 'direction']
		},
		{
			'phase': 'Collection',
			'description': 'Gather raw information from various sources',
			'capabilities': ['osint', 'sigint', 'humint', 'geoint', 'cybint', 'finint', 'socint']
		},
		{
			'phase': 'Processing',
			'description': 'Convert raw data into usable formats',
			'capabilities': ['processing', 'translation', 'decryption', 'formatting']
		},
		{
			'phase': 'Analysis & Production', 
			'description': 'Analyze processed information and produce intelligence',
			'capabilities': ['fusion', 'analytics', 'correlation', 'prediction', 'threats']
		},
		{
			'phase': 'Dissemination',
			'description': 'Deliver intelligence products to decision makers',
			'capabilities': ['reporting', 'dashboard', 'alerts', 'briefings']
		},
		{
			'phase': 'Evaluation & Feedback',
			'description': 'Assess effectiveness and refine processes',
			'capabilities': ['assessment', 'feedback', 'improvement']
		}
	]

def get_threat_categories() -> List[Dict[str, Any]]:
	"""Get threat intelligence categories"""
	return [
		{
			'category': 'Cyber Threats',
			'subcategories': ['Malware', 'APT', 'Ransomware', 'Data Breaches', 'DDoS'],
			'priority': 'High'
		},
		{
			'category': 'Physical Security',
			'subcategories': ['Terrorism', 'Sabotage', 'Theft', 'Espionage', 'Violence'],
			'priority': 'High'
		},
		{
			'category': 'Financial Crime',
			'subcategories': ['Fraud', 'Money Laundering', 'Corruption', 'Embezzlement'],
			'priority': 'Medium'
		},
		{
			'category': 'Geopolitical',
			'subcategories': ['State Actors', 'Sanctions', 'Trade Wars', 'Political Instability'],
			'priority': 'Medium'
		},
		{
			'category': 'Reputational',
			'subcategories': ['Social Media', 'Misinformation', 'Brand Attacks', 'Leaks'],
			'priority': 'Medium'
		}
	]

def validate_composition(subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate a composition of intelligence sub-capabilities"""
	errors = []
	warnings = []
	
	# Check if requested sub-capabilities exist
	available = get_subcapabilities()
	for subcap in subcapabilities:
		if subcap not in available:
			errors.append(f"Unknown sub-capability: {subcap}")
	
	# Check for recommended combinations
	collection_caps = get_collection_capabilities()
	analysis_caps = get_analysis_capabilities()
	
	has_collection = any(cap in subcapabilities for cap in collection_caps)
	has_analysis = any(cap in subcapabilities for cap in analysis_caps)
	
	if has_collection and not has_analysis:
		warnings.append("Collection capabilities without analysis may limit intelligence value")
	
	if has_analysis and not has_collection:
		warnings.append("Analysis capabilities require data collection sources")
	
	# Security requirements
	if any(cap in subcapabilities for cap in ['sigint', 'humint', 'cybint']):
		warnings.append("Classified intelligence capabilities require appropriate security clearances")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize Intelligence capability with Flask-AppBuilder"""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)

# Import implemented sub-capabilities for discovery
from . import crawler  # Legacy web crawler
from . import radio    # Radio Intelligence Listener

__all__ = [
	'get_capability_info',
	'get_intelligence_disciplines',
	'get_subcapabilities', 
	'get_implemented_subcapabilities',
	'get_collection_capabilities',
	'get_analysis_capabilities',
	'get_presentation_capabilities',
	'validate_security_clearance',
	'get_intelligence_cycle',
	'get_threat_categories',
	'validate_composition',
	'init_capability'
]
