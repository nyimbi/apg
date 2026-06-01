"""
Financial Technology (FinTech) Capability

Comprehensive financial technology platform providing modern digital financial services,
blockchain solutions, regulatory compliance, and innovative financial products.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Financial Technology',
	'code': 'FINTECH',
	'version': '2.0.0',
	'description': 'Comprehensive financial technology platform with digital payments, lending, banking, and blockchain services',
	'industry_focus': 'FinTech, Banking, Financial Services, Insurance, Investment Management',
	'regulatory_compliance': ['PCI DSS', 'PSD2', 'Open Banking', 'GDPR', 'SOX', 'Basel III', 'MiFID II', 'CCPA'],
	'service_categories': [
		'Digital Payments',
		'Digital Banking', 
		'Lending & Credit',
		'Investment & Wealth',
		'Insurance Technology',
		'Regulatory Technology',
		'Blockchain & Crypto',
		'Risk & Compliance'
	],
	'subcapabilities': [
		# Core Payment Services
		'payments',      # Digital Payments
		'cards',         # Digital Cards
		'wallets',       # Digital Wallets
		'mobile',        # Mobile Banking
		'remittance',    # Cross-Border Remittance
		'apis',          # Banking APIs
		'embedded',      # Embedded Finance
		
		# Banking & Lending
		'neobanking',    # Neo Banking
		'lending',       # Digital Lending
		'bnpl',          # Buy Now Pay Later
		'tms',           # Treasury Management System
		'agency',        # Agency Banking
		'terminal',      # Terminal Management System
		
		# Investment & Wealth
		'wealth',        # Wealth Management
		'robo',          # Robo Advisory
		'trading',       # Algorithmic Trading
		'portfolio',     # Portfolio Management
		'crowdfunding',  # Crowdfunding Platform
		
		# Insurance & Risk
		'insurance',     # InsurTech
		'risk',          # Risk Management
		'fraud',         # Fraud Detection
		
		# Regulatory & Compliance
		'regtech',       # Regulatory Technology
		'kyc',           # Know Your Customer
		'aml',           # Anti Money Laundering
		'compliance',    # Compliance Automation
		
		# Blockchain & Crypto
		'blockchain',    # Blockchain Services
		'crypto',        # Cryptocurrency
		'defi',          # Decentralized Finance
	],
	'implemented_subcapabilities': [
		'payments',      # Digital Payments
		'cards',         # Digital Cards
		'wallets',       # Digital Wallets
		'mobile',        # Mobile Banking
		'apis',          # Banking APIs
		'embedded',      # Embedded Finance
		'wealth',        # Wealth Management
		'robo',          # Robo Advisory
		'portfolio',     # Portfolio Management
		'trading',       # Algorithmic Trading
		'crowdfunding',  # Crowdfunding Platform
		'neobanking',    # Neo Banking
		'lending',       # Digital Lending
		'bnpl',          # Buy Now Pay Later
		'agency',        # Agency Banking
		'remittance',    # Cross-Border Remittance
		'insurance',     # InsurTech
		'risk',          # Risk Management
		'regtech',       # Regulatory Technology
		'kyc',           # Know Your Customer
		'aml',           # Anti Money Laundering
		'fraud',         # Fraud Detection
		'compliance',    # Compliance Automation
		'blockchain',    # Blockchain Services
		'crypto',        # Cryptocurrency
		'tms',           # Treasury Management System (existing)
		'gateway',       # Payment Gateway (existing)
		'switch',        # Payment Switch (existing)
	],
	'technology_stack': {
		'blockchain': ['Ethereum', 'Polygon', 'Solana', 'Bitcoin', 'Hyperledger'],
		'ai_ml': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'XGBoost'],
		'cloud': ['AWS', 'Azure', 'GCP', 'Kubernetes'],
		'databases': ['PostgreSQL', 'MongoDB', 'Redis', 'InfluxDB'],
		'messaging': ['Bytewax', 'RabbitMQ', 'AWS SQS'],
		'apis': ['REST', 'GraphQL', 'gRPC', 'WebSockets']
	},
	'security_standards': {
		'encryption': 'AES-256, RSA-2048',
		'authentication': 'OAuth 2.0, OpenID Connect',
		'compliance': 'PCI DSS Level 1, ISO 27001',
		'data_protection': 'GDPR, CCPA compliant',
		'audit': 'SOC 2 Type II'
	},
	'dependencies': [
		'auth_rbac',
		'audit_compliance',
		'notification',
		'computer_vision',
		'nlp',
		'fin'  # Core Financial Management
	],
	'optional_dependencies': [
		'blockchain_security',
		'biometric',
		'mfa',
		'geographical_location_services',
		'real_time_collaboration'
	],
	'database_prefix': 'fintech_',
	'menu_category': 'FinTech',
	'menu_icon': 'fa-coins'
}

def get_capability_info() -> Dict[str, Any]:
	"""Get FinTech capability information"""
	return CAPABILITY_META

def get_service_categories() -> List[str]:
	"""Get FinTech service categories"""
	return CAPABILITY_META['service_categories']

def get_subcapabilities() -> List[str]:
	"""Get list of available sub-capabilities"""
	return CAPABILITY_META['subcapabilities']

def get_implemented_subcapabilities() -> List[str]:
	"""Get list of currently implemented sub-capabilities"""
	return CAPABILITY_META['implemented_subcapabilities']

def get_payment_capabilities() -> List[str]:
	"""Get payment and transaction capabilities"""
	return [
		'payments', 'cards', 'wallets', 'mobile', 'remittance', 'apis', 'embedded'
	]

def get_banking_capabilities() -> List[str]:
	"""Get banking and lending capabilities"""
	return [
		'neobanking', 'lending', 'bnpl', 'tms', 'agency', 'terminal'
	]

def get_investment_capabilities() -> List[str]:
	"""Get investment and wealth management capabilities"""
	return [
		'wealth', 'robo', 'trading', 'portfolio', 'crowdfunding'
	]

def get_compliance_capabilities() -> List[str]:
	"""Get regulatory and compliance capabilities"""
	return [
		'regtech', 'kyc', 'aml', 'compliance', 'fraud', 'risk'
	]

def get_blockchain_capabilities() -> List[str]:
	"""Get blockchain and cryptocurrency capabilities"""
	return [
		'blockchain', 'crypto', 'defi'
	]

def get_regulatory_frameworks() -> List[Dict[str, Any]]:
	"""Get supported regulatory frameworks"""
	return [
		{
			'framework': 'PCI DSS',
			'description': 'Payment Card Industry Data Security Standard',
			'scope': 'Payment Processing',
			'requirements': ['Data Protection', 'Network Security', 'Vulnerability Management']
		},
		{
			'framework': 'PSD2',
			'description': 'Payment Services Directive 2',
			'scope': 'European Payment Services',
			'requirements': ['Strong Customer Authentication', 'Open Banking APIs', 'Data Sharing']
		},
		{
			'framework': 'Open Banking',
			'description': 'Open Banking Standards',
			'scope': 'Banking APIs',
			'requirements': ['API Standards', 'Customer Consent', 'Data Security']
		},
		{
			'framework': 'Basel III',
			'description': 'International Banking Regulations',
			'scope': 'Banking Risk Management',
			'requirements': ['Capital Requirements', 'Liquidity Coverage', 'Leverage Ratios']
		},
		{
			'framework': 'MiFID II',
			'description': 'Markets in Financial Instruments Directive',
			'scope': 'Investment Services',
			'requirements': ['Best Execution', 'Transparency', 'Investor Protection']
		}
	]

def get_fintech_products() -> List[Dict[str, Any]]:
	"""Get FinTech product categories"""
	return [
		{
			'category': 'Digital Payments',
			'products': ['Mobile Payments', 'Contactless Cards', 'QR Payments', 'P2P Transfers'],
			'market_size': '$7.5T globally',
			'growth_rate': '12% CAGR'
		},
		{
			'category': 'Digital Banking',
			'products': ['Neo Banks', 'Banking-as-a-Service', 'Embedded Banking', 'Open Banking'],
			'market_size': '$8.2T globally',
			'growth_rate': '15% CAGR'
		},
		{
			'category': 'Digital Lending',
			'products': ['P2P Lending', 'SME Lending', 'Consumer Credit', 'BNPL'],
			'market_size': '$350B globally',
			'growth_rate': '25% CAGR'
		},
		{
			'category': 'Wealth Management',
			'products': ['Robo-Advisors', 'Digital Investment', 'Portfolio Management', 'Trading'],
			'market_size': '$4.2T globally',
			'growth_rate': '8% CAGR'
		},
		{
			'category': 'InsurTech',
			'products': ['Digital Insurance', 'Parametric Insurance', 'Claims Automation', 'Risk Assessment'],
			'market_size': '$5.4T globally',
			'growth_rate': '20% CAGR'
		},
		{
			'category': 'RegTech',
			'products': ['Compliance Automation', 'KYC/AML', 'Risk Management', 'Regulatory Reporting'],
			'market_size': '$120B globally',
			'growth_rate': '18% CAGR'
		},
		{
			'category': 'Blockchain & DeFi',
			'products': ['Cryptocurrency', 'DeFi Protocols', 'NFTs', 'Smart Contracts'],
			'market_size': '$3T globally',
			'growth_rate': '45% CAGR'
		}
	]

def get_technology_trends() -> List[Dict[str, Any]]:
	"""Get emerging FinTech technology trends"""
	return [
		{
			'trend': 'Embedded Finance',
			'description': 'Financial services integrated into non-financial platforms',
			'impact': 'High',
			'adoption_timeline': '2024-2026'
		},
		{
			'trend': 'Central Bank Digital Currencies (CBDCs)',
			'description': 'Government-issued digital currencies',
			'impact': 'Very High',
			'adoption_timeline': '2025-2030'
		},
		{
			'trend': 'AI-Powered Financial Services',
			'description': 'Machine learning for credit scoring, fraud detection, and personalization',
			'impact': 'High',
			'adoption_timeline': '2024-2025'
		},
		{
			'trend': 'Quantum Computing in Finance',
			'description': 'Quantum algorithms for risk modeling and optimization',
			'impact': 'Medium',
			'adoption_timeline': '2028-2035'
		},
		{
			'trend': 'Sustainable Finance Technology',
			'description': 'ESG scoring, green bonds, carbon credits trading',
			'impact': 'Medium',
			'adoption_timeline': '2024-2027'
		}
	]

def validate_regulatory_compliance(subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate regulatory compliance for FinTech capabilities"""
	errors = []
	warnings = []
	
	# Check payment capabilities compliance
	payment_caps = get_payment_capabilities()
	if any(cap in subcapabilities for cap in payment_caps):
		if 'fraud' not in subcapabilities:
			errors.append("Fraud detection is required for payment capabilities")
		if 'compliance' not in subcapabilities:
			warnings.append("Compliance automation recommended for payment services")
	
	# Check lending capabilities compliance
	lending_caps = ['lending', 'bnpl', 'crowdfunding']
	if any(cap in subcapabilities for cap in lending_caps):
		if 'kyc' not in subcapabilities:
			errors.append("KYC is required for lending capabilities")
		if 'risk' not in subcapabilities:
			errors.append("Risk management is required for lending capabilities")
	
	# Check crypto/blockchain compliance
	crypto_caps = get_blockchain_capabilities()
	if any(cap in subcapabilities for cap in crypto_caps):
		if 'aml' not in subcapabilities:
			errors.append("AML compliance is required for cryptocurrency capabilities")
		if 'kyc' not in subcapabilities:
			errors.append("KYC is required for cryptocurrency capabilities")
	
	# Check wealth management compliance
	wealth_caps = get_investment_capabilities()
	if any(cap in subcapabilities for cap in wealth_caps):
		if 'regtech' not in subcapabilities:
			warnings.append("RegTech recommended for investment services")
	
	return {
		'compliant': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def validate_composition(subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate a composition of FinTech sub-capabilities"""
	errors = []
	warnings = []
	
	# Check if requested sub-capabilities exist
	available = get_subcapabilities()
	for subcap in subcapabilities:
		if subcap not in available:
			errors.append(f"Unknown sub-capability: {subcap}")
	
	# Regulatory compliance validation
	compliance_result = validate_regulatory_compliance(subcapabilities)
	errors.extend(compliance_result['errors'])
	warnings.extend(compliance_result['warnings'])
	
	# Check for recommended technology combinations
	if 'payments' in subcapabilities and 'wallets' not in subcapabilities:
		warnings.append("Digital wallets recommended for payment services")
	
	if 'trading' in subcapabilities and 'risk' not in subcapabilities:
		warnings.append("Risk management strongly recommended for trading services")
	
	if 'neobanking' in subcapabilities:
		banking_deps = ['payments', 'cards', 'mobile', 'kyc', 'fraud']
		missing_deps = [dep for dep in banking_deps if dep not in subcapabilities]
		if missing_deps:
			warnings.append(f"Neo banking typically requires: {', '.join(missing_deps)}")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings,
		'compliance_status': compliance_result['compliant']
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize FinTech capability with the host application builder."""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)

def _optional_subcapability(name: str):
	"""Import an optional fintech sub-capability when it exists."""
	try:
		module = __import__(f"{__name__}.{name}", fromlist=[name])
	except ImportError:
		return None
	return module


# Import implemented sub-capabilities for discovery without blocking package imports.
tms = _optional_subcapability("tms")            # Treasury Management System
terminal = _optional_subcapability("terminal")  # Terminal Management System
agency = _optional_subcapability("agency")      # Agency Banking

__all__ = [
	'get_capability_info',
	'get_service_categories',
	'get_subcapabilities',
	'get_implemented_subcapabilities',
	'get_payment_capabilities',
	'get_banking_capabilities', 
	'get_investment_capabilities',
	'get_compliance_capabilities',
	'get_blockchain_capabilities',
	'get_regulatory_frameworks',
	'get_fintech_products',
	'get_technology_trends',
	'validate_regulatory_compliance',
	'validate_composition',
	'init_capability'
]
