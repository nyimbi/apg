"""
Agency Banking FinTech Capability

Comprehensive agency banking platform enabling third-party agents to provide 
basic banking services on behalf of financial institutions. Supports cash-in/cash-out,
bill payments, account opening, and other financial services through agent networks.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, List, Any

# Agency Banking metadata
SUBCAPABILITY_META = {
	'name': 'Agency Banking',
	'code': 'AGENCY',
	'version': '1.0.0',
	'capability': 'fintech',
	'description': 'Comprehensive agency banking platform for financial inclusion through agent networks',
	'service_types': [
		'CASH_IN',           # Deposit money
		'CASH_OUT',          # Withdraw money
		'MONEY_TRANSFER',    # Send/receive money
		'BILL_PAYMENT',      # Utility and service payments
		'AIRTIME_TOPUP',     # Mobile airtime purchase
		'LOAN_DISBURSEMENT', # Microfinance loan disbursement
		'LOAN_COLLECTION',   # Loan repayment collection
		'ACCOUNT_OPENING',   # Basic account registration
		'BALANCE_INQUIRY',   # Account balance check
		'MINI_STATEMENT',    # Transaction history
		'CARD_SERVICES',     # Debit card issuance/management
		'INSURANCE',         # Micro-insurance products
		'SAVINGS_PRODUCTS',  # Savings account management
		'GOVERNMENT_PAYMENTS' # Social benefit disbursements
	],
	'agent_types': [
		'RETAIL_SHOP',       # Small retail stores
		'PHARMACY',          # Pharmacy chains
		'SUPERMARKET',       # Supermarket chains
		'PETROL_STATION',    # Fuel stations
		'MOBILE_MONEY_AGENT', # Mobile money operators
		'POST_OFFICE',       # Postal service points
		'COOPERATIVE',       # Agricultural cooperatives
		'MICROFINANCE',      # MFI branches
		'COMMUNITY_BANK',    # Community banking points
		'MOBILE_AGENT'       # Roving/mobile agents
	],
	'regulatory_compliance': [
		'KYC_SIMPLIFIED',    # Simplified Know Your Customer
		'AML_BASIC',         # Basic Anti-Money Laundering
		'TRANSACTION_LIMITS', # Daily/monthly transaction limits
		'AGENT_LICENSING',   # Agent registration and licensing
		'CASH_MANAGEMENT',   # Cash float management rules
		'AUDIT_TRAIL',       # Complete transaction audit trail
		'DATA_PROTECTION',   # Customer data protection
		'CONSUMER_PROTECTION' # Consumer rights and protection
	],
	'technology_features': [
		'offline_capability',
		'biometric_authentication',
		'multi_language_support',
		'real_time_settlement',
		'agent_commission_management',
		'inventory_management',
		'risk_monitoring',
		'customer_onboarding',
		'dispute_resolution',
		'performance_analytics'
	],
	'supported_channels': [
		'POS_TERMINAL',      # Point of Sale devices
		'MOBILE_APP',        # Smartphone applications
		'USSD',             # Unstructured Supplementary Service Data
		'SMS',              # Short Message Service
		'WEB_PORTAL',       # Web-based interface
		'TABLET',           # Tablet applications
		'FEATURE_PHONE'     # Basic phone support
	],
	'settlement_models': [
		'REAL_TIME',        # Immediate settlement
		'BATCH_HOURLY',     # Hourly batch settlement
		'BATCH_DAILY',      # End of day settlement
		'BILATERAL',        # Direct bank-to-bank
		'CENTRAL_SWITCH'    # Through central switching platform
	],
	'dependencies': [
		'payments',
		'kyc',
		'fraud',
		'mobile',
		'compliance'
	],
	'optional_dependencies': [
		'biometric',
		'geographical_location_services',
		'notification',
		'analytics'
	]
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get agency banking information"""
	return SUBCAPABILITY_META

def get_service_types() -> List[str]:
	"""Get supported agency banking services"""
	return SUBCAPABILITY_META['service_types']

def get_agent_types() -> List[str]:
	"""Get supported agent types"""
	return SUBCAPABILITY_META['agent_types']

def get_supported_channels() -> List[str]:
	"""Get supported service delivery channels"""
	return SUBCAPABILITY_META['supported_channels']

def get_agent_requirements() -> List[Dict[str, Any]]:
	"""Get agent onboarding requirements"""
	return [
		{
			'requirement': 'Business Registration',
			'description': 'Valid business license or registration',
			'mandatory': True,
			'documents': ['Business permit', 'Tax certificate']
		},
		{
			'requirement': 'Physical Location',
			'description': 'Fixed business location accessible to customers',
			'mandatory': True,
			'criteria': ['Visible signage', 'Safe environment', 'Operating hours']
		},
		{
			'requirement': 'Financial Capacity',
			'description': 'Adequate working capital for cash float',
			'mandatory': True,
			'minimum_float': '$500-$5,000 depending on location'
		},
		{
			'requirement': 'Technology Infrastructure',
			'description': 'Reliable connectivity and payment devices',
			'mandatory': True,
			'equipment': ['POS terminal or smartphone', 'Internet connection']
		},
		{
			'requirement': 'Staff Training',
			'description': 'Trained personnel for banking services',
			'mandatory': True,
			'training_areas': ['Customer service', 'KYC procedures', 'System operations']
		},
		{
			'requirement': 'Security Measures',
			'description': 'Basic security protocols for cash handling',
			'mandatory': True,
			'measures': ['Safe storage', 'CCTV (recommended)', 'Insurance coverage']
		}
	]

def get_transaction_limits() -> Dict[str, Dict[str, Any]]:
	"""Get regulatory transaction limits"""
	return {
		'TIER_1_CUSTOMER': {
			'daily_limit': 50000,  # Local currency units
			'monthly_limit': 200000,
			'kyc_requirements': ['Phone verification', 'Basic ID'],
			'services': ['Cash in/out', 'Bill payments', 'Airtime']
		},
		'TIER_2_CUSTOMER': {
			'daily_limit': 100000,
			'monthly_limit': 500000,
			'kyc_requirements': ['Full ID verification', 'Address proof'],
			'services': ['All Tier 1', 'Money transfers', 'Account opening']
		},
		'TIER_3_CUSTOMER': {
			'daily_limit': 200000,
			'monthly_limit': 1000000,
			'kyc_requirements': ['Enhanced due diligence', 'Income verification'],
			'services': ['All services', 'Loan products', 'Investment products']
		}
	}

def get_commission_structure() -> List[Dict[str, Any]]:
	"""Get agent commission structure"""
	return [
		{
			'service': 'CASH_IN',
			'commission_rate': '0.5%',
			'minimum_fee': 5,
			'maximum_fee': 50,
			'revenue_share': {'agent': '60%', 'bank': '40%'}
		},
		{
			'service': 'CASH_OUT',
			'commission_rate': '1.0%',
			'minimum_fee': 10,
			'maximum_fee': 100,
			'revenue_share': {'agent': '70%', 'bank': '30%'}
		},
		{
			'service': 'BILL_PAYMENT',
			'commission_rate': '2.0%',
			'minimum_fee': 5,
			'maximum_fee': 200,
			'revenue_share': {'agent': '50%', 'bank': '30%', 'utility': '20%'}
		},
		{
			'service': 'MONEY_TRANSFER',
			'commission_rate': '1.5%',
			'minimum_fee': 15,
			'maximum_fee': 150,
			'revenue_share': {'agent': '65%', 'bank': '35%'}
		},
		{
			'service': 'ACCOUNT_OPENING',
			'fixed_fee': 50,
			'revenue_share': {'agent': '80%', 'bank': '20%'}
		}
	]

def calculate_agent_performance() -> Dict[str, Any]:
	"""Calculate agent performance metrics"""
	return {
		'transaction_volume': {
			'description': 'Total monthly transaction value',
			'target': 50000,
			'weight': 30
		},
		'transaction_count': {
			'description': 'Number of transactions per month',
			'target': 200,
			'weight': 25
		},
		'customer_acquisition': {
			'description': 'New customers onboarded monthly',
			'target': 10,
			'weight': 20
		},
		'service_quality': {
			'description': 'Customer satisfaction score',
			'target': 4.5,  # out of 5
			'weight': 15
		},
		'compliance_score': {
			'description': 'KYC and regulatory compliance',
			'target': 95,  # percentage
			'weight': 10
		}
	}

def validate_agent_application(agent_data: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate agent application"""
	errors = []
	warnings = []
	
	# Required fields
	required_fields = [
		'business_name', 'business_type', 'location', 
		'contact_person', 'phone_number', 'proposed_float'
	]
	
	for field in required_fields:
		if not agent_data.get(field):
			errors.append(f"Required field missing: {field}")
	
	# Agent type validation
	if agent_data.get('business_type') not in get_agent_types():
		errors.append(f"Invalid business type: {agent_data.get('business_type')}")
	
	# Float capacity validation
	proposed_float = agent_data.get('proposed_float', 0)
	if proposed_float < 500:
		errors.append("Minimum float requirement: $500")
	elif proposed_float > 50000:
		warnings.append("High float amount may require additional security measures")
	
	# Location validation
	location_data = agent_data.get('location', {})
	if not all(k in location_data for k in ['latitude', 'longitude', 'address']):
		errors.append("Complete location information required")
	
	return {
		'approved': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

__version__ = "1.0.0"
__status__ = "Development"