"""
Enterprise Integration API Endpoints

Provides REST API endpoints for enterprise integrations:
- SSO authentication endpoints
- Compliance management APIs
- Risk assessment APIs
- Data governance APIs
- Audit and security APIs

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, session, redirect, url_for
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.security.decorators import permission_name
from pydantic import ValidationError
import structlog

from .enterprise_integration import (
	EnterpriseIntegrationManager, LDAPConfig, SAMLConfig, OIDCConfig,
	EnterpriseDBConfig, AuthenticationMethod, ComplianceFramework,
	AuditEvent, SecurityPolicy, enterprise_integration
)
from .sso_connectors import (
	SSOConfiguration, SSOProvider, SSOConnectorFactory, sso_session_manager
)
from .compliance_governance import (
	CompliancePolicy, RiskAssessment, DataInventoryItem, ComplianceControl,
	ComplianceIncident, RiskLevel, DataClassification, PolicyStatus,
	compliance_governance
)

logger = structlog.get_logger(__name__)

# Create Flask Blueprint
enterprise_bp = Blueprint(
	'enterprise',
	__name__,
	url_prefix='/api/enterprise'
)


# =============================================================================
# SSO Authentication Endpoints
# =============================================================================

@enterprise_bp.route('/sso/providers', methods=['GET'])
async def get_sso_providers():
	"""Get available SSO providers"""
	try:
		providers = [
			{
				"id": provider.value,
				"name": provider.value.replace("_", " ").title(),
				"description": f"{provider.value.replace('_', ' ').title()} Single Sign-On"
			}
			for provider in SSOProvider
		]
		
		return jsonify({
			"success": True,
			"data": providers
		})
		
	except Exception as e:
		logger.error(f"Get SSO providers error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/sso/initiate/<provider>', methods=['POST'])
async def initiate_sso(provider: str):
	"""Initiate SSO authentication"""
	try:
		data = request.get_json()
		redirect_uri = data.get('redirect_uri')
		
		if not redirect_uri:
			return jsonify({
				"success": False,
				"error": "redirect_uri is required"
			}), 400
		
		# Create SSO configuration (would normally be retrieved from settings)
		config = SSOConfiguration(
			provider=SSOProvider(provider),
			client_id=data.get('client_id', ''),
			client_secret=data.get('client_secret', ''),
			tenant_id=data.get('tenant_id'),
			domain=data.get('domain'),
			scopes=data.get('scopes', [])
		)
		
		connector = SSOConnectorFactory.create_connector(config)
		
		# Generate state for CSRF protection
		state = f"sso_{provider}_{datetime.utcnow().timestamp()}"
		session['sso_state'] = state
		
		# Get authorization URL
		auth_url = await connector.get_authorization_url(state, redirect_uri)
		
		return jsonify({
			"success": True,
			"data": {
				"authorization_url": auth_url,
				"state": state
			}
		})
		
	except Exception as e:
		logger.error(f"SSO initiation error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/sso/callback/<provider>', methods=['POST'])
async def sso_callback(provider: str):
	"""Handle SSO callback"""
	try:
		data = request.get_json()
		code = data.get('code')
		state = data.get('state')
		redirect_uri = data.get('redirect_uri')
		
		# Verify state for CSRF protection
		if state != session.get('sso_state'):
			return jsonify({
				"success": False,
				"error": "Invalid state parameter"
			}), 400
		
		# Create SSO configuration
		config = SSOConfiguration(
			provider=SSOProvider(provider),
			client_id=data.get('client_id', ''),
			client_secret=data.get('client_secret', ''),
			tenant_id=data.get('tenant_id'),
			domain=data.get('domain')
		)
		
		connector = SSOConnectorFactory.create_connector(config)
		
		# Exchange code for tokens
		success, sso_session_obj = await connector.exchange_code(code, redirect_uri)
		
		if success and sso_session_obj:
			# Create session in database
			session_id = await sso_session_manager.create_session(sso_session_obj)
			
			# Store session ID in Flask session
			session['sso_session_id'] = session_id
			session['user_id'] = sso_session_obj.user_id
			
			return jsonify({
				"success": True,
				"data": {
					"session_id": session_id,
					"user": {
						"id": sso_session_obj.user_id,
						"email": sso_session_obj.email,
						"display_name": sso_session_obj.display_name,
						"roles": sso_session_obj.roles,
						"groups": sso_session_obj.groups
					}
				}
			})
		else:
			return jsonify({
				"success": False,
				"error": "SSO authentication failed"
			}), 401
			
	except Exception as e:
		logger.error(f"SSO callback error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/sso/logout', methods=['POST'])
async def sso_logout():
	"""Logout from SSO session"""
	try:
		session_id = session.get('sso_session_id')
		
		if session_id:
			await sso_session_manager.invalidate_session(session_id)
		
		# Clear Flask session
		session.clear()
		
		return jsonify({
			"success": True,
			"message": "Logged out successfully"
		})
		
	except Exception as e:
		logger.error(f"SSO logout error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Enterprise Database Endpoints
# =============================================================================

@enterprise_bp.route('/database/connections', methods=['POST'])
async def create_database_connection():
	"""Create enterprise database connection"""
	try:
		data = request.get_json()
		
		config = EnterpriseDBConfig(
			db_type=data['db_type'],
			host=data['host'],
			port=data['port'],
			database=data['database'],
			username=data['username'],
			password=data['password'],
			connection_string=data.get('connection_string'),
			ssl_config=data.get('ssl_config'),
			pool_config=data.get('pool_config'),
			advanced_options=data.get('advanced_options')
		)
		
		connection_id = await enterprise_integration.create_database_connection(config)
		
		# Log database connection creation
		audit_event = AuditEvent(
			event_type="database_connection",
			user_id=session.get('user_id'),
			action="connection_created",
			result="success",
			resource_type="database_connection",
			resource_id=connection_id,
			details={
				"db_type": config.db_type.value,
				"host": config.host,
				"database": config.database
			},
			risk_level="medium"
		)
		await enterprise_integration.log_audit_event(audit_event)
		
		return jsonify({
			"success": True,
			"data": {
				"connection_id": connection_id
			}
		})
		
	except Exception as e:
		logger.error(f"Database connection creation error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/database/connections/<connection_id>/query', methods=['POST'])
async def execute_database_query(connection_id: str):
	"""Execute query on enterprise database"""
	try:
		data = request.get_json()
		query = data.get('query')
		parameters = data.get('parameters', {})
		
		if not query:
			return jsonify({
				"success": False,
				"error": "Query is required"
			}), 400
		
		results = await enterprise_integration.execute_database_query(
			connection_id, query, parameters
		)
		
		# Log database query execution
		audit_event = AuditEvent(
			event_type="database_query",
			user_id=session.get('user_id'),
			action="query_executed",
			result="success",
			resource_type="database_connection",
			resource_id=connection_id,
			details={
				"query_type": query.split()[0].upper() if query else "UNKNOWN",
				"parameter_count": len(parameters)
			},
			risk_level="low"
		)
		await enterprise_integration.log_audit_event(audit_event)
		
		return jsonify({
			"success": True,
			"data": {
				"results": results,
				"row_count": len(results)
			}
		})
		
	except Exception as e:
		logger.error(f"Database query execution error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Compliance Management Endpoints
# =============================================================================

@enterprise_bp.route('/compliance/policies', methods=['GET'])
async def get_compliance_policies():
	"""Get compliance policies"""
	try:
		framework = request.args.get('framework')
		status = request.args.get('status')
		limit = int(request.args.get('limit', 50))
		offset = int(request.args.get('offset', 0))
		
		# Get current tenant from context
		tenant_id = g.get('tenant_id')
		if not tenant_id:
			return jsonify({"success": False, "error": "Tenant not found"}), 400
		
		# Use APG audit compliance service to get policies
		try:
			from capabilities.common.audit_compliance.service import AuditComplianceService
			compliance_service = AuditComplianceService()
			
			# Build query filters
			filters = {"tenant_id": tenant_id}
			if framework:
				filters['compliance_framework'] = framework
			if status:
				filters['status'] = status
			
			# Query compliance policies from database
			policies = await compliance_service.get_compliance_policies(
				filters=filters,
				limit=limit,
				offset=offset
			)
			
			# Get total count for pagination
			total_count = await compliance_service.count_compliance_policies(filters)
			
			# Format policies for response
			formatted_policies = []
			for policy in policies:
				formatted_policies.append({
					"id": policy.get("id"),
					"name": policy.get("name"),
					"framework": policy.get("compliance_framework"),
					"status": policy.get("status"),
					"description": policy.get("description"),
					"severity": policy.get("severity", "medium"),
					"requirements": policy.get("requirements", []),
					"last_updated": policy.get("updated_at"),
					"compliance_score": policy.get("compliance_score", 0.0)
				})
			
			return jsonify({
				"success": True,
				"data": {
					"policies": formatted_policies,
					"total": total_count,
					"limit": limit,
					"offset": offset,
					"filters": filters
				}
			})
			
		except ImportError:
			# Fallback when compliance service not available
			logger.warning("Audit compliance service not available, using database fallback")
			
			# Direct database query as fallback
			from sqlalchemy import text
			from .database import get_async_session
			
			async with get_async_session() as session:
				# Query compliance policies table
				query = text("""
					SELECT id, name, compliance_framework, status, description, 
						   severity, requirements, updated_at, compliance_score
					FROM wo_compliance_policies 
					WHERE tenant_id = :tenant_id
				""")
				params = {"tenant_id": tenant_id}
				
				if framework:
					query = text(str(query) + " AND compliance_framework = :framework")
					params["framework"] = framework
				if status:
					query = text(str(query) + " AND status = :status")
					params["status"] = status
				
				query = text(str(query) + " ORDER BY updated_at DESC LIMIT :limit OFFSET :offset")
				params.update({"limit": limit, "offset": offset})
				
				result = await session.execute(query, params)
				policies = result.fetchall()
				
				# Count query
				count_query = text("""
					SELECT COUNT(*) FROM wo_compliance_policies 
					WHERE tenant_id = :tenant_id
				""")
				count_params = {"tenant_id": tenant_id}
				if framework:
					count_query = text(str(count_query) + " AND compliance_framework = :framework")
					count_params["framework"] = framework
				if status:
					count_query = text(str(count_query) + " AND status = :status")
					count_params["status"] = status
				
				count_result = await session.execute(count_query, count_params)
				total_count = count_result.scalar()
				
				# Format policies
				formatted_policies = []
				for policy in policies:
					formatted_policies.append({
						"id": policy.id,
						"name": policy.name,
						"framework": policy.compliance_framework,
						"status": policy.status,
						"description": policy.description,
						"severity": policy.severity,
						"requirements": policy.requirements or [],
						"last_updated": policy.updated_at.isoformat() if policy.updated_at else None,
						"compliance_score": policy.compliance_score or 0.0
					})
				
				return jsonify({
					"success": True,
					"data": {
						"policies": formatted_policies,
						"total": total_count,
						"limit": limit,
						"offset": offset,
						"filters": {"framework": framework, "status": status}
					}
				})
		
	except Exception as e:
		logger.error(f"Get compliance policies error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/compliance/policies', methods=['POST'])
async def create_compliance_policy():
	"""Create compliance policy"""
	try:
		data = request.get_json()
		
		policy = CompliancePolicy(
			name=data['name'],
			description=data['description'],
			framework=ComplianceFramework(data['framework']),
			version=data.get('version', '1.0'),
			status=PolicyStatus(data.get('status', 'draft')),
			effective_date=datetime.fromisoformat(data['effective_date']),
			review_date=datetime.fromisoformat(data['review_date']),
			expiry_date=datetime.fromisoformat(data['expiry_date']) if data.get('expiry_date') else None,
			owner=data['owner'],
			approver=data.get('approver'),
			scope=data.get('scope', []),
			requirements=data.get('requirements', []),
			controls=data.get('controls', []),
			exceptions=data.get('exceptions', []),
			tenant_id=session.get('tenant_id')
		)
		
		policy_id = await compliance_governance.create_policy(policy)
		
		return jsonify({
			"success": True,
			"data": {
				"policy_id": policy_id
			}
		})
		
	except ValidationError as e:
		return jsonify({
			"success": False,
			"error": "Validation error",
			"details": e.errors()
		}), 400
	except Exception as e:
		logger.error(f"Create compliance policy error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/compliance/risk-assessments', methods=['POST'])
async def create_risk_assessment():
	"""Create risk assessment"""
	try:
		data = request.get_json()
		
		assessment = RiskAssessment(
			name=data['name'],
			description=data['description'],
			asset_type=data['asset_type'],
			asset_id=data['asset_id'],
			risk_category=data['risk_category'],
			threat_sources=data.get('threat_sources', []),
			vulnerabilities=data.get('vulnerabilities', []),
			likelihood=RiskLevel(data['likelihood']),
			impact=RiskLevel(data['impact']),
			inherent_risk=RiskLevel(data['inherent_risk']),
			residual_risk=RiskLevel(data['residual_risk']),
			risk_tolerance=RiskLevel(data['risk_tolerance']),
			mitigation_controls=data.get('mitigation_controls', []),
			treatment_plan=data.get('treatment_plan'),
			owner=data['owner'],
			assessor=data['assessor'],
			next_review_date=datetime.fromisoformat(data['next_review_date']),
			tenant_id=session.get('tenant_id')
		)
		
		assessment_id = await compliance_governance.create_risk_assessment(assessment)
		
		return jsonify({
			"success": True,
			"data": {
				"assessment_id": assessment_id
			}
		})
		
	except ValidationError as e:
		return jsonify({
			"success": False,
			"error": "Validation error",
			"details": e.errors()
		}), 400
	except Exception as e:
		logger.error(f"Create risk assessment error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/compliance/data-inventory', methods=['POST'])
async def create_data_inventory_item():
	"""Create data inventory item"""
	try:
		data = request.get_json()
		
		item = DataInventoryItem(
			name=data['name'],
			description=data['description'],
			data_type=data['data_type'],
			classification=DataClassification(data['classification']),
			location=data['location'],
			owner=data['owner'],
			steward=data['steward'],
			retention_period=data.get('retention_period'),
			purpose=data['purpose'],
			legal_basis=data.get('legal_basis'),
			processing_activities=data.get('processing_activities', []),
			sharing_agreements=data.get('sharing_agreements', []),
			protection_measures=data.get('protection_measures', []),
			tenant_id=session.get('tenant_id')
		)
		
		item_id = await compliance_governance.create_data_inventory_item(item)
		
		return jsonify({
			"success": True,
			"data": {
				"item_id": item_id
			}
		})
		
	except ValidationError as e:
		return jsonify({
			"success": False,
			"error": "Validation error",
			"details": e.errors()
		}), 400
	except Exception as e:
		logger.error(f"Create data inventory item error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/compliance/data-classification', methods=['POST'])
async def classify_data():
	"""Automatically classify data"""
	try:
		data = request.get_json()
		data_sample = data.get('data_sample', '')
		location = data.get('location', '')
		
		classification, indicators = await compliance_governance.data_governance.classify_data_automatically(
			data_sample, location
		)
		
		return jsonify({
			"success": True,
			"data": {
				"classification": classification.value,
				"indicators": indicators,
				"confidence": "high" if len(indicators) > 0 else "medium"
			}
		})
		
	except Exception as e:
		logger.error(f"Data classification error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Monitoring and Reporting Endpoints
# =============================================================================

@enterprise_bp.route('/compliance/dashboard', methods=['GET'])
async def get_compliance_dashboard():
	"""Get compliance dashboard data"""
	try:
		dashboard_data = await compliance_governance.get_compliance_dashboard()
		
		return jsonify({
			"success": True,
			"data": dashboard_data
		})
		
	except Exception as e:
		logger.error(f"Compliance dashboard error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/compliance/monitoring/run-checks', methods=['POST'])
async def run_compliance_checks():
	"""Run compliance monitoring checks"""
	try:
		alerts = await compliance_governance.run_compliance_monitoring()
		
		return jsonify({
			"success": True,
			"data": {
				"alerts": alerts,
				"alert_count": len(alerts),
				"timestamp": datetime.utcnow().isoformat()
			}
		})
		
	except Exception as e:
		logger.error(f"Compliance monitoring error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/compliance/reports/<framework>', methods=['GET'])
async def generate_compliance_report(framework: str):
	"""Generate compliance report for framework"""
	try:
		start_date_str = request.args.get('start_date')
		end_date_str = request.args.get('end_date')
		
		if not start_date_str or not end_date_str:
			return jsonify({
				"success": False,
				"error": "start_date and end_date are required"
			}), 400
		
		start_date = datetime.fromisoformat(start_date_str)
		end_date = datetime.fromisoformat(end_date_str)
		
		report = await compliance_governance.generate_framework_report(
			ComplianceFramework(framework), start_date, end_date
		)
		
		return jsonify({
			"success": True,
			"data": report
		})
		
	except ValueError as e:
		return jsonify({
			"success": False,
			"error": f"Invalid framework or date format: {e}"
		}), 400
	except Exception as e:
		logger.error(f"Compliance report generation error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Audit and Security Endpoints
# =============================================================================

@enterprise_bp.route('/audit/events', methods=['POST'])
async def log_audit_event():
	"""Log audit event"""
	try:
		data = request.get_json()
		
		audit_event = AuditEvent(
			event_type=data['event_type'],
			user_id=data.get('user_id') or session.get('user_id'),
			session_id=data.get('session_id') or session.get('sso_session_id'),
			source_ip=data.get('source_ip') or request.remote_addr,
			user_agent=data.get('user_agent') or request.headers.get('User-Agent'),
			resource_type=data.get('resource_type'),
			resource_id=data.get('resource_id'),
			action=data['action'],
			result=data['result'],
			details=data.get('details', {}),
			risk_level=data.get('risk_level', 'low'),
			compliance_tags=data.get('compliance_tags', []),
			tenant_id=data.get('tenant_id') or session.get('tenant_id')
		)
		
		event_id = await enterprise_integration.log_audit_event(audit_event)
		
		return jsonify({
			"success": True,
			"data": {
				"event_id": event_id
			}
		})
		
	except ValidationError as e:
		return jsonify({
			"success": False,
			"error": "Validation error",
			"details": e.errors()
		}), 400
	except Exception as e:
		logger.error(f"Audit event logging error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/security/policies', methods=['POST'])
async def create_security_policy():
	"""Create security policy"""
	try:
		data = request.get_json()
		
		policy = SecurityPolicy(
			name=data['name'],
			description=data['description'],
			policy_type=data['policy_type'],
			rules=data['rules'],
			enabled=data.get('enabled', True),
			enforcement_level=data.get('enforcement_level', 'strict'),
			applicable_roles=data.get('applicable_roles', []),
			applicable_resources=data.get('applicable_resources', [])
		)
		
		policy_id = await enterprise_integration.create_security_policy(policy)
		
		return jsonify({
			"success": True,
			"data": {
				"policy_id": policy_id
			}
		})
		
	except ValidationError as e:
		return jsonify({
			"success": False,
			"error": "Validation error",
			"details": e.errors()
		}), 400
	except Exception as e:
		logger.error(f"Security policy creation error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


@enterprise_bp.route('/security/evaluate-access', methods=['POST'])
async def evaluate_access():
	"""Evaluate access based on security policies"""
	try:
		data = request.get_json()
		
		user_id = data.get('user_id') or session.get('user_id')
		resource_type = data['resource_type']
		action = data['action']
		context = data.get('context', {})
		
		# Add session context
		context.update({
			"source_ip": request.remote_addr,
			"user_agent": request.headers.get('User-Agent'),
			"session": {
				"session_id": session.get('sso_session_id'),
				"user_id": user_id
			}
		})
		
		allowed, violations = await enterprise_integration.evaluate_access(
			user_id, resource_type, action, context
		)
		
		# Log access evaluation
		audit_event = AuditEvent(
			event_type="access_evaluation",
			user_id=user_id,
			action=f"evaluate_{action}",
			result="allowed" if allowed else "denied",
			resource_type=resource_type,
			details={
				"action": action,
				"violations": violations,
				"context_keys": list(context.keys())
			},
			risk_level="medium" if not allowed else "low"
		)
		await enterprise_integration.log_audit_event(audit_event)
		
		return jsonify({
			"success": True,
			"data": {
				"allowed": allowed,
				"violations": violations,
				"timestamp": datetime.utcnow().isoformat()
			}
		})
		
	except Exception as e:
		logger.error(f"Access evaluation error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Health and Status Endpoints
# =============================================================================

@enterprise_bp.route('/health', methods=['GET'])
async def health_check():
	"""Enterprise integration health check"""
	try:
		health_status = {
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"services": {
				"enterprise_integration": "healthy",
				"sso_manager": "healthy",
				"compliance_governance": "healthy",
				"audit_system": "healthy"
			}
		}
		
		return jsonify(health_status)
		
	except Exception as e:
		return jsonify({
			"status": "unhealthy",
			"error": str(e),
			"timestamp": datetime.utcnow().isoformat()
		}), 500


@enterprise_bp.route('/status', methods=['GET'])
async def get_status():
	"""Get enterprise integration status"""
	try:
		status = {
			"enterprise_integration": {
				"active_connections": 0,  # Would be populated from actual data
				"audit_events_today": 0,
				"security_policies": 0
			},
			"sso": {
				"active_sessions": len(sso_session_manager.active_sessions),
				"supported_providers": len(SSOProvider)
			},
			"compliance": {
				"frameworks_supported": len(ComplianceFramework),
				"active_policies": 0,
				"pending_reviews": 0
			}
		}
		
		return jsonify({
			"success": True,
			"data": status
		})
		
	except Exception as e:
		logger.error(f"Status check error: {e}")
		return jsonify({
			"success": False,
			"error": str(e)
		}), 500


# =============================================================================
# Flask-AppBuilder Views (for UI integration)
# =============================================================================

class EnterpriseIntegrationView(BaseView):
	"""Enterprise integration management view"""
	
	route_base = "/enterprise"
	
	@expose("/dashboard")
	@has_access
	def dashboard(self):
		"""Enterprise integration dashboard"""
		return self.render_template(
			"enterprise/dashboard.html",
			title="Enterprise Integration Dashboard"
		)
	
	@expose("/sso")
	@has_access
	def sso_management(self):
		"""SSO management interface"""
		return self.render_template(
			"enterprise/sso.html",
			title="SSO Management"
		)
	
	@expose("/compliance")
	@has_access
	def compliance_management(self):
		"""Compliance management interface"""
		return self.render_template(
			"enterprise/compliance.html",
			title="Compliance Management"
		)
	
	@expose("/audit")
	@has_access
	def audit_management(self):
		"""Audit management interface"""
		return self.render_template(
			"enterprise/audit.html",
			title="Audit Management"
		)


def register_enterprise_views(appbuilder):
	"""Register enterprise views with Flask-AppBuilder"""
	appbuilder.add_view(
		EnterpriseIntegrationView,
		"Enterprise Dashboard",
		icon="fa-building",
		category="Enterprise",
		category_icon="fa-shield"
	)


# Register blueprint routes
def register_enterprise_routes(app):
	"""Register enterprise routes with Flask app"""
	app.register_blueprint(enterprise_bp)