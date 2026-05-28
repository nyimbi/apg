#!/usr/bin/env python3
"""
APG Key Management - Flask-AppBuilder Views
Management dashboard and UI following APG patterns

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, abort
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget
from flask_appbuilder.actions import action
from flask_babel import lazy_gettext
from sqlalchemy import func, text
from datetime import datetime, timedelta
import json
import asyncio
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

from .models import KeyAlgorithm, KeyUsage, KeyState, SecurityLevel, ComplianceFramework, create_key_spec_async
from .service import KeyManagementService
from .ai_lifecycle import AILifecycleManager, LifecycleDecision
from .security_intelligence import SecurityIntelligenceEngine, AnomalyAlert
from .policy_engine import IntelligentPolicyEngine, PolicyEvaluationResult
try:
	from .cloud_federation import CloudKeyFederationManager
except Exception:
	CloudKeyFederationManager = None
try:
	from .hsm_integration import HSMIntegrationManager
except Exception:
	HSMIntegrationManager = None
try:
	from .quantum_safe import QuantumSafeCryptographyManager
except Exception:
	QuantumSafeCryptographyManager = None


_runtime_keym_service: KeyManagementService | None = None


def set_key_management_service(service: KeyManagementService | None) -> None:
	"""Register the runtime key-management service used by views and API helpers."""
	global _runtime_keym_service
	_runtime_keym_service = service


def _enum_value(value: Any) -> Any:
	"""Return enum values while leaving plain values unchanged."""
	return value.value if hasattr(value, "value") else value


def _run_async(coro):
	"""Run a coroutine from sync Flask-AppBuilder handlers."""
	try:
		asyncio.get_running_loop()
	except RuntimeError:
		return asyncio.run(coro)
	raise RuntimeError("Cannot run KEYM async service operation from an active event loop")


def _get_runtime_service(view: Any = None) -> KeyManagementService | None:
	"""Resolve an injected or globally registered key-management service."""
	if view is not None and getattr(view, "_keym_service", None) is not None:
		return view._keym_service
	return _runtime_keym_service


def _get_or_create_runtime_service(view: Any = None) -> KeyManagementService:
	"""Resolve or initialize an in-process KEYM service for view operations."""
	service = _get_runtime_service(view)
	if service is not None:
		return service

	service = KeyManagementService()
	_run_async(service.initialize({"tenant_id": "default"}))
	set_key_management_service(service)
	if view is not None:
		view._keym_service = service
	return service


def _service_keys(service: KeyManagementService | None) -> List[Any]:
	"""Return key objects from service runtime state."""
	if service is None:
		return []
	keys = getattr(service, "keys", {})
	if isinstance(keys, dict):
		return list(keys.values())
	return list(keys or [])


def _service_key(service: KeyManagementService | None, key_id: str) -> Any | None:
	"""Return one key object from service runtime state."""
	if service is None:
		return None
	keys = getattr(service, "keys", {})
	if isinstance(keys, dict):
		return keys.get(key_id)
	for key in keys or []:
		if getattr(getattr(key, "spec", None), "id", None) == key_id:
			return key
	return None


def _service_usage_stats(service: KeyManagementService | None, key_id: str) -> Any | None:
	"""Return usage stats for a key from service runtime state."""
	if service is None:
		return None
	return getattr(service, "usage_stats", {}).get(key_id)


def _key_record(key: Any, stats: Any = None) -> Dict[str, Any]:
	"""Normalize a runtime Key model into the list/detail view shape."""
	spec = key.spec
	metadata = spec.metadata
	policy = spec.policy
	return {
		'id': spec.id,
		'name': metadata.name,
		'algorithm': _enum_value(spec.algorithm),
		'key_size': spec.key_size,
		'state': _enum_value(spec.state),
		'usage': [_enum_value(usage) for usage in spec.usage],
		'created_at': spec.created_at,
		'created_by': spec.created_by,
		'last_used': getattr(stats, "last_used", None) or getattr(key, "last_used", None),
		'usage_count': getattr(stats, "total_operations", getattr(key, "usage_count", 0)),
		'security_level': _enum_value(spec.security_level),
		'auto_rotate': policy.auto_rotate,
		'rotation_interval': policy.rotation_interval_days,
		'next_rotation': getattr(key, "next_rotation", None),
		'hsm_backed': getattr(key, "hsm_key_id", None) is not None,
		'compliance_frameworks': [_enum_value(framework) for framework in policy.compliance_frameworks],
		'metadata': {
			'project': metadata.project_id,
			'environment': metadata.environment,
			'cost_center': metadata.cost_center,
			'tags': metadata.tags,
			'description': metadata.description
		}
	}


def _service_dashboard_snapshot(service: KeyManagementService | None) -> Dict[str, Any]:
	"""Build dashboard data from current service runtime state."""
	keys = _service_keys(service)
	stats_by_id = getattr(service, "usage_stats", {}) if service else {}
	threats = list(getattr(service, "threats", {}).values()) if service else []
	audit_events = list(getattr(service, "audit_events", [])) if service else []

	state_counts = Counter(_enum_value(key.spec.state) for key in keys)
	algorithm_counts = Counter(_enum_value(key.spec.algorithm) for key in keys)
	pending_rotation = sum(
		1
		for key in keys
		if getattr(key, "next_rotation", None) and key.next_rotation <= datetime.utcnow() + timedelta(days=30)
	)
	violation_events = [
		event for event in audit_events
		if "violation" in getattr(event, "event_type", "").lower()
		or getattr(event, "outcome", "") == "violation"
	]
	severity_counts = Counter(getattr(threat, "severity", "low") for threat in threats)
	total_ops = sum(getattr(stats, "total_operations", 0) for stats in stats_by_id.values())

	return {
		'timestamp': datetime.utcnow().isoformat(),
		'summary': {
			'total_keys': len(keys),
			'active_keys': state_counts.get(KeyState.ACTIVE.value, 0),
			'pending_rotation': pending_rotation,
			'compliance_violations': len(violation_events),
			'security_alerts': len(threats),
			'total_operations': total_ops,
			'audit_events': len(audit_events)
		},
		'algorithm_distribution': dict(algorithm_counts),
		'security_metrics': {
			'threat_level': _derive_threat_level(severity_counts),
			'recent_anomalies': len([
				threat for threat in threats
				if getattr(threat, "detected_at", datetime.min) >= datetime.utcnow() - timedelta(hours=24)
			]),
			'policy_violations': len(violation_events),
			'hsm_health': _derive_hsm_health(service)
		},
		'compliance_status': _derive_compliance_status(keys, violation_events)
	}


def _derive_threat_level(severity_counts: Counter) -> str:
	"""Derive a threat level from runtime threat severities."""
	if severity_counts.get("critical", 0):
		return "critical"
	if severity_counts.get("high", 0):
		return "high"
	if severity_counts.get("medium", 0):
		return "medium"
	if sum(severity_counts.values()):
		return "low"
	return "none"


def _derive_hsm_health(service: KeyManagementService | None) -> float:
	"""Calculate HSM health from configured runtime HSM entries."""
	if service is None:
		return 0.0
	configs = getattr(service, "hsm_configs", {}) or {}
	if not configs:
		return 0.0
	healthy = 0
	for config in configs.values():
		status = config.get("health_status") or config.get("status") or ("healthy" if config.get("enabled") else "disabled")
		if status in {"healthy", "connected", "online"} or config.get("enabled"):
			healthy += 1
	return round((healthy / len(configs)) * 100, 2)


def _derive_compliance_status(keys: List[Any], violation_events: List[Any]) -> Dict[str, str]:
	"""Calculate compliance framework status from key policies and violations."""
	frameworks = {framework.value for framework in ComplianceFramework}
	declared = Counter(
		_enum_value(framework)
		for key in keys
		for framework in key.spec.policy.compliance_frameworks
	)
	violated = Counter(
		_enum_value(framework)
		for event in violation_events
		for framework in getattr(event, "compliance_frameworks", [])
	)
	return {
		framework: "violation" if violated.get(framework, 0) else ("compliant" if declared.get(framework, 0) else "not_configured")
		for framework in sorted(frameworks)
	}


# APG Blueprint for Key Management
keym_bp = Blueprint(
	'keym', 
	__name__, 
	template_folder='templates', 
	static_folder='static',
	url_prefix='/keym'
)


class KeyManagementDashboardView(BaseView):
	"""Main dashboard for key management"""
	
	route_base = '/dashboard'
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard view"""
		try:
			# Get dashboard data
			dashboard_data = self._get_dashboard_data()
			
			return self.render_template(
				'keym/dashboard.html',
				title='Key Management Dashboard',
				dashboard_data=dashboard_data
			)
			
		except Exception as e:
			flash(f"Error loading dashboard: {str(e)}", "error")
			return self.render_template('keym/error.html', error=str(e))
	
	def _get_dashboard_data(self) -> Dict[str, Any]:
		"""Gather comprehensive dashboard data"""
		return _service_dashboard_snapshot(_get_runtime_service(self))


class KeyListView(BaseView):
	"""Key management list and operations view"""
	
	route_base = '/keys'
	
	@expose('/')
	@has_access
	def list(self):
		"""List all keys with filtering and search"""
		# Get filter parameters
		algorithm_filter = request.args.get('algorithm', '')
		state_filter = request.args.get('state', '')
		search_term = request.args.get('search', '')
		page = int(request.args.get('page', 1))
		per_page = int(request.args.get('per_page', 25))
		
		keys_data = self._get_keys_data(
			algorithm_filter, state_filter, search_term, page, per_page
		)
		
		return self.render_template(
			'keym/key_list.html',
			title='Key Management',
			keys_data=keys_data,
			algorithms=KeyAlgorithm,
			states=KeyState,
			current_filters={
				'algorithm': algorithm_filter,
				'state': state_filter,
				'search': search_term
			}
		)
	
	@expose('/create', methods=['GET', 'POST'])
	@has_access
	def create(self):
		"""Create new cryptographic key"""
		if request.method == 'POST':
			try:
				# Get form data
				key_data = {
					'name': request.form.get('name'),
					'algorithm': request.form.get('algorithm'),
					'key_size': int(request.form.get('key_size', 256)),
					'usage': request.form.getlist('usage'),
					'description': request.form.get('description'),
					'security_level': request.form.get('security_level'),
					'auto_rotate': request.form.get('auto_rotate') == 'on'
				}
				
				# Validate and create key
				success, message = self._create_key(key_data)
				
				if success:
					flash(message, "success")
					return redirect(url_for('keym.KeyListView.list'))
				else:
					flash(message, "error")
					
			except Exception as e:
				flash(f"Error creating key: {str(e)}", "error")
		
		return self.render_template(
			'keym/key_create.html',
			title='Create New Key',
			algorithms=KeyAlgorithm,
			usages=KeyUsage,
			security_levels=SecurityLevel
		)
	
	@expose('/detail/<key_id>')
	@has_access
	def detail(self, key_id: str):
		"""Show detailed key information"""
		try:
			key_detail = self._get_key_detail(key_id)
			if not key_detail:
				abort(404)
				
			return self.render_template(
				'keym/key_detail.html',
				title=f'Key Details - {key_id}',
				key_detail=key_detail
			)
			
		except Exception as e:
			flash(f"Error loading key details: {str(e)}", "error")
			return redirect(url_for('keym.KeyListView.list'))
	
	@expose('/rotate/<key_id>', methods=['POST'])
	@has_access
	def rotate(self, key_id: str):
		"""Rotate specific key"""
		try:
			success, message = self._rotate_key(key_id)
			flash(message, "success" if success else "error")
			
		except Exception as e:
			flash(f"Error rotating key: {str(e)}", "error")
		
		return redirect(url_for('keym.KeyListView.detail', key_id=key_id))
	
	def _get_keys_data(self, algorithm_filter: str, state_filter: str, 
					   search_term: str, page: int, per_page: int) -> Dict[str, Any]:
		"""Get paginated keys data with filters"""
		service = _get_runtime_service(self)
		key_records = [
			_key_record(key, _service_usage_stats(service, key.spec.id))
			for key in _service_keys(service)
		]
		
		filtered_keys = key_records
		if algorithm_filter:
			filtered_keys = [k for k in filtered_keys if k['algorithm'] == algorithm_filter]
		if state_filter:
			filtered_keys = [k for k in filtered_keys if k['state'] == state_filter]
		if search_term:
			filtered_keys = [k for k in filtered_keys if search_term.lower() in k['name'].lower()]
		
		# Pagination
		total_count = len(filtered_keys)
		start_idx = (page - 1) * per_page
		end_idx = start_idx + per_page
		page_keys = filtered_keys[start_idx:end_idx]
		
		return {
			'keys': page_keys,
			'pagination': {
				'page': page,
				'per_page': per_page,
				'total': total_count,
				'pages': (total_count + per_page - 1) // per_page,
				'has_prev': page > 1,
				'has_next': end_idx < total_count
			}
		}
	
	def _create_key(self, key_data: Dict[str, Any]) -> tuple[bool, str]:
		"""Create new key with validation"""
		# Validate required fields
		if not key_data.get('name'):
			return False, "Key name is required"
		
		if not key_data.get('algorithm'):
			return False, "Algorithm selection is required"
		
		try:
			service = _get_or_create_runtime_service(self)
			algorithm = KeyAlgorithm(key_data['algorithm'])
			usage = [KeyUsage(value) for value in key_data.get('usage', [])] or [KeyUsage.ENCRYPT]
			security_level = SecurityLevel(key_data.get('security_level') or SecurityLevel.INTERNAL.value)
			tenant_id = getattr(service, "config", {}).get("tenant_id", "default")
			spec = _run_async(create_key_spec_async(
				tenant_id=tenant_id,
				algorithm=algorithm,
				usage=usage,
				name=key_data['name'],
				created_by="ui",
				key_size=key_data.get('key_size'),
				description=key_data.get('description'),
				security_level=security_level,
				auto_rotate=key_data.get('auto_rotate', True)
			))
			key = _run_async(service.create_key(spec, user_id="ui"))
			return True, f"Key '{key_data['name']}' created successfully with ID: {key.spec.id}"
		except Exception as exc:
			return False, f"Key creation failed: {exc}"
	
	def _get_key_detail(self, key_id: str) -> Dict[str, Any] | None:
		"""Get detailed key information"""
		service = _get_runtime_service(self)
		key = _service_key(service, key_id)
		if key is None:
			return None
		return _key_record(key, _service_usage_stats(service, key_id))
	
	def _rotate_key(self, key_id: str) -> tuple[bool, str]:
		"""Rotate specific key"""
		service = _get_runtime_service(self)
		if service is None or _service_key(service, key_id) is None:
			return False, f"Key {key_id} not found"
		try:
			_run_async(service.rotate_key(key_id, user_id="ui"))
			return True, f"Key {key_id} rotated successfully"
		except Exception as exc:
			return False, f"Key rotation failed: {exc}"


class SecurityDashboardView(BaseView):
	"""Security monitoring and intelligence dashboard"""
	
	route_base = '/security'
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Security intelligence dashboard"""
		security_data = self._get_security_dashboard_data()
		
		return self.render_template(
			'keym/security_dashboard.html',
			title='Security Intelligence',
			security_data=security_data
		)
	
	@expose('/anomalies')
	@has_access
	def anomalies(self):
		"""Security anomalies and alerts"""
		anomalies_data = self._get_anomalies_data()
		
		return self.render_template(
			'keym/security_anomalies.html', 
			title='Security Anomalies',
			anomalies_data=anomalies_data
		)
	
	@expose('/threats')
	@has_access
	def threats(self):
		"""Threat intelligence and correlation"""
		threats_data = self._get_threats_data()
		
		return self.render_template(
			'keym/security_threats.html',
			title='Threat Intelligence',
			threats_data=threats_data
		)
	
	def _get_security_dashboard_data(self) -> Dict[str, Any]:
		"""Get security dashboard data"""
		return {
			'alert_summary': {
				'critical': 2,
				'high': 5, 
				'medium': 12,
				'low': 8
			},
			'threat_trends': {
				'last_24h': 15,
				'last_7d': 89,
				'last_30d': 234
			},
			'user_risk_distribution': {
				'high_risk': 3,
				'medium_risk': 12,
				'low_risk': 85
			},
			'recent_events': [
				{
					'type': 'anomaly_detection',
					'severity': 'high',
					'description': 'Unusual access pattern detected for key_001',
					'timestamp': datetime.utcnow() - timedelta(minutes=15)
				},
				{
					'type': 'policy_violation',
					'severity': 'medium',
					'description': 'MFA requirement bypassed for key_002',
					'timestamp': datetime.utcnow() - timedelta(hours=2)
				}
			]
		}
	
	def _get_anomalies_data(self) -> Dict[str, Any]:
		"""Get security anomalies data"""
		return {
			'active_anomalies': [
				{
					'id': 'anom_001',
					'type': 'unusual_volume',
					'severity': 'high',
					'affected_keys': ['key_001', 'key_005'],
					'detection_time': datetime.utcnow() - timedelta(minutes=30),
					'description': '500% increase in key operations',
					'confidence': 0.89
				},
				{
					'id': 'anom_002',
					'type': 'suspicious_ip',
					'severity': 'medium', 
					'affected_keys': ['key_003'],
					'detection_time': datetime.utcnow() - timedelta(hours=1),
					'description': 'Access from new geographic location',
					'confidence': 0.72
				}
			],
			'anomaly_stats': {
				'total_detected': 156,
				'resolved': 144,
				'false_positives': 12,
				'accuracy_rate': 92.3
			}
		}
	
	def _get_threats_data(self) -> Dict[str, Any]:
		"""Get threat intelligence data"""
		return {
			'active_threats': [
				{
					'threat_id': 'thr_001',
					'type': 'account_compromise',
					'severity': 'critical',
					'affected_keys': ['key_001', 'key_002'],
					'confidence': 0.95,
					'detected_at': datetime.utcnow() - timedelta(hours=3),
					'indicators': ['multiple_failed_auth', 'unusual_access_time', 'new_device']
				}
			],
			'threat_intelligence': {
				'global_threat_level': 'moderate',
				'known_attack_vectors': ['quantum_computing', 'social_engineering', 'supply_chain'],
				'industry_alerts': 2
			}
		}


class ComplianceDashboardView(BaseView):
	"""Compliance monitoring and reporting dashboard"""
	
	route_base = '/compliance'
	
	@expose('/')
	@has_access 
	def dashboard(self):
		"""Compliance overview dashboard"""
		compliance_data = self._get_compliance_dashboard_data()
		
		return self.render_template(
			'keym/compliance_dashboard.html',
			title='Compliance Dashboard',
			compliance_data=compliance_data
		)
	
	@expose('/reports')
	@has_access
	def reports(self):
		"""Compliance reports and audit trails"""
		reports_data = self._get_compliance_reports_data()
		
		return self.render_template(
			'keym/compliance_reports.html',
			title='Compliance Reports',
			reports_data=reports_data
		)
	
	@expose('/policies')
	@has_access
	def policies(self):
		"""Policy management and validation"""
		policies_data = self._get_policies_data()
		
		return self.render_template(
			'keym/compliance_policies.html',
			title='Policy Management',
			policies_data=policies_data
		)
	
	def _get_compliance_dashboard_data(self) -> Dict[str, Any]:
		"""Get compliance dashboard data"""
		return {
			'framework_status': {
				'GDPR': {'status': 'compliant', 'score': 98},
				'HIPAA': {'status': 'partial', 'score': 87},
				'PCI_DSS': {'status': 'compliant', 'score': 95},
				'SOX': {'status': 'compliant', 'score': 92},
				'FIPS_140_2': {'status': 'compliant', 'score': 99}
			},
			'recent_violations': [
				{
					'framework': 'HIPAA',
					'rule': 'minimum_necessary_access',
					'severity': 'medium',
					'affected_keys': ['key_003'],
					'detected_at': datetime.utcnow() - timedelta(hours=6)
				}
			],
			'audit_summary': {
				'total_events': 12543,
				'policy_evaluations': 8934,
				'violations_detected': 23,
				'auto_remediated': 18
			}
		}
	
	def _get_compliance_reports_data(self) -> Dict[str, Any]:
		"""Get compliance reports data"""
		return {
			'available_reports': [
				{
					'name': 'GDPR Compliance Report',
					'framework': 'GDPR',
					'generated_at': datetime.utcnow() - timedelta(days=1),
					'status': 'complete',
					'file_url': '/reports/gdpr_compliance_2025.pdf'
				},
				{
					'name': 'PCI DSS Audit Trail',
					'framework': 'PCI_DSS', 
					'generated_at': datetime.utcnow() - timedelta(days=7),
					'status': 'complete',
					'file_url': '/reports/pci_audit_2025.pdf'
				}
			],
			'scheduled_reports': [
				{
					'name': 'Monthly Compliance Summary',
					'frequency': 'monthly',
					'next_generation': datetime.utcnow() + timedelta(days=5),
					'recipients': ['compliance@company.com', 'security@company.com']
				}
			]
		}
	
	def _get_policies_data(self) -> Dict[str, Any]:
		"""Get policy management data"""
		return {
			'active_policies': [
				{
					'policy_id': 'pol_001',
					'name': 'Production Key Access Policy',
					'framework': 'GDPR',
					'status': 'active',
					'last_updated': datetime.utcnow() - timedelta(days=15),
					'violations_24h': 2
				},
				{
					'policy_id': 'pol_002',
					'name': 'Healthcare Data Encryption Policy', 
					'framework': 'HIPAA',
					'status': 'active',
					'last_updated': datetime.utcnow() - timedelta(days=30),
					'violations_24h': 0
				}
			],
			'policy_effectiveness': {
				'average_compliance_rate': 94.2,
				'policy_violations_trend': 'decreasing',
				'auto_remediation_rate': 78.3
			}
		}


class CloudFederationView(BaseView):
	"""Multi-cloud key federation management"""
	
	route_base = '/federation'
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Cloud federation dashboard"""
		federation_data = self._get_federation_dashboard_data()
		
		return self.render_template(
			'keym/federation_dashboard.html',
			title='Cloud Federation',
			federation_data=federation_data
		)
	
	@expose('/providers')
	@has_access
	def providers(self):
		"""Cloud provider management"""
		providers_data = self._get_providers_data()
		
		return self.render_template(
			'keym/federation_providers.html',
			title='Cloud Providers',
			providers_data=providers_data
		)
	
	def _get_federation_dashboard_data(self) -> Dict[str, Any]:
		"""Get federation dashboard data"""
		return {
			'provider_status': {
				'aws': {'status': 'online', 'keys': 89, 'sync_status': 'in_sync'},
				'azure': {'status': 'online', 'keys': 67, 'sync_status': 'in_sync'}, 
				'gcp': {'status': 'online', 'keys': 45, 'sync_status': 'syncing'}
			},
			'federation_health': {
				'overall_status': 'healthy',
				'total_federated_keys': 156,
				'sync_success_rate': 98.7,
				'last_sync': datetime.utcnow() - timedelta(minutes=5)
			},
			'cost_optimization': {
				'monthly_cost': 1250.00,
				'potential_savings': 180.00,
				'cost_trend': 'decreasing'
			}
		}
	
	def _get_providers_data(self) -> Dict[str, Any]:
		"""Get cloud providers data"""
		return {
			'configured_providers': [
				{
					'provider': 'aws',
					'region': 'us-east-1',
					'service': 'KMS',
					'status': 'connected',
					'keys_count': 89,
					'last_health_check': datetime.utcnow() - timedelta(minutes=2)
				},
				{
					'provider': 'azure',
					'region': 'eastus',
					'service': 'Key Vault',
					'status': 'connected',
					'keys_count': 67,
					'last_health_check': datetime.utcnow() - timedelta(minutes=3)
				}
			]
		}


class QuantumSafeView(BaseView):
	"""Quantum-safe cryptography management"""
	
	route_base = '/quantum'
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Quantum-safe dashboard"""
		quantum_data = self._get_quantum_dashboard_data()
		
		return self.render_template(
			'keym/quantum_dashboard.html',
			title='Quantum-Safe Cryptography',
			quantum_data=quantum_data
		)
	
	@expose('/migration')
	@has_access
	def migration(self):
		"""Quantum-safe migration management"""
		migration_data = self._get_migration_data()
		
		return self.render_template(
			'keym/quantum_migration.html',
			title='Quantum-Safe Migration',
			migration_data=migration_data
		)
	
	def _get_quantum_dashboard_data(self) -> Dict[str, Any]:
		"""Get quantum-safe dashboard data"""
		return {
			'threat_assessment': {
				'current_threat_level': 'minimal',
				'vulnerable_keys': 125,
				'quantum_safe_keys': 45,
				'migration_priority_critical': 15
			},
			'algorithm_readiness': {
				'RSA-2048': {'status': 'vulnerable', 'replacement': 'Kyber-768'},
				'ECDSA-P256': {'status': 'vulnerable', 'replacement': 'Dilithium-2'},
				'AES-256': {'status': 'partially_resistant', 'replacement': None}
			},
			'migration_progress': {
				'total_migration_plans': 3,
				'completed_migrations': 1,
				'in_progress': 1,
				'average_success_rate': 96.5
			}
		}
	
	def _get_migration_data(self) -> Dict[str, Any]:
		"""Get quantum-safe migration data"""
		return {
			'active_plans': [
				{
					'plan_id': 'qmig_001',
					'name': 'Production Keys Migration',
					'strategy': 'hybrid',
					'progress': 65,
					'target_date': datetime.utcnow() + timedelta(days=30),
					'keys_migrated': 45,
					'total_keys': 70
				}
			],
			'recommended_actions': [
				{
					'priority': 'critical',
					'action': 'Migrate ECDSA keys immediately',
					'affected_keys': 15,
					'timeline': '30 days'
				},
				{
					'priority': 'high',
					'action': 'Plan RSA key migration',
					'affected_keys': 89,
					'timeline': '90 days'
				}
			]
		}


# API Routes for AJAX calls
@keym_bp.route('/api/dashboard/stats')
def api_dashboard_stats():
	"""API endpoint for dashboard statistics"""
	try:
		snapshot = _service_dashboard_snapshot(_get_runtime_service())
		summary = snapshot['summary']
		compliance_values = list(snapshot['compliance_status'].values())
		compliant = len([status for status in compliance_values if status == "compliant"])
		configured = len([status for status in compliance_values if status != "not_configured"])
		stats = {
			'total_keys': summary['total_keys'],
			'active_keys': summary['active_keys'],
			'security_alerts': summary['security_alerts'],
			'compliance_score': round((compliant / configured) * 100, 2) if configured else 0.0
		}
		return jsonify({'success': True, 'data': stats})
		
	except Exception as e:
		return jsonify({'success': False, 'error': str(e)}), 500


@keym_bp.route('/api/security/alerts')
def api_security_alerts():
	"""API endpoint for real-time security alerts"""
	try:
		service = _get_runtime_service()
		alerts = []
		for threat in list(getattr(service, "threats", {}).values()) if service else []:
			alerts.append({
				'id': threat.threat_id,
				'type': threat.threat_type,
				'severity': threat.severity,
				'message': f"{threat.threat_type} detected for {len(threat.affected_keys)} key(s)",
				'timestamp': threat.detected_at.isoformat()
			})
		return jsonify({'success': True, 'data': alerts})
		
	except Exception as e:
		return jsonify({'success': False, 'error': str(e)}), 500


@keym_bp.route('/api/keys/<key_id>/health')
def api_key_health(key_id: str):
	"""API endpoint for key health status"""
	try:
		service = _get_runtime_service()
		key = _service_key(service, key_id)
		if key is None:
			return jsonify({'success': False, 'error': 'Key not found'}), 404

		stats = _service_usage_stats(service, key_id)
		rotation_due = bool(getattr(key, "next_rotation", None) and key.next_rotation <= datetime.utcnow())
		health = {
			'key_id': key_id,
			'status': 'rotation_due' if rotation_due else _enum_value(key.spec.state),
			'last_used': (getattr(stats, "last_used", None) or getattr(key, "last_used", None)).isoformat()
				if (getattr(stats, "last_used", None) or getattr(key, "last_used", None)) else None,
			'usage_count': getattr(stats, "total_operations", getattr(key, "usage_count", 0)),
			'rotation_due': rotation_due
		}
		return jsonify({'success': True, 'data': health})
		
	except Exception as e:
		return jsonify({'success': False, 'error': str(e)}), 500


# Register views with Flask-AppBuilder
def init_views(appbuilder):
	"""Initialize all views with Flask-AppBuilder"""
	
	# Main views
	appbuilder.add_view(
		KeyManagementDashboardView,
		"Dashboard",
		icon="fa-dashboard",
		category="Key Management"
	)
	
	appbuilder.add_view(
		KeyListView,
		"Keys",
		icon="fa-key", 
		category="Key Management"
	)
	
	appbuilder.add_view(
		SecurityDashboardView,
		"Security Intelligence",
		icon="fa-shield",
		category="Security"
	)
	
	appbuilder.add_view(
		ComplianceDashboardView,
		"Compliance",
		icon="fa-check-circle",
		category="Compliance"
	)
	
	appbuilder.add_view(
		CloudFederationView,
		"Cloud Federation",
		icon="fa-cloud",
		category="Infrastructure"
	)
	
	appbuilder.add_view(
		QuantumSafeView,
		"Quantum-Safe",
		icon="fa-atom",
		category="Advanced"
	)


# Export views and blueprint
__all__ = [
	'keym_bp', 'init_views', 'KeyManagementDashboardView', 'KeyListView',
	'SecurityDashboardView', 'ComplianceDashboardView', 'CloudFederationView',
	'QuantumSafeView'
]
