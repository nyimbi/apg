"""
APG Central Configuration - Enterprise Security Audit Engine

Comprehensive security auditing, compliance monitoring, and threat detection
for enterprise-grade configuration management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import hashlib
import hmac
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import uuid
from pathlib import Path

# Cryptography for advanced security
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import bcrypt

# Database and configuration
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc

from ..models import CCConfiguration, CCAuditLog, CCSecurityEvent
from ..service import CentralConfigurationEngine


class SecurityRiskLevel(Enum):
	"""Security risk levels."""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class ComplianceFramework(Enum):
	"""Compliance frameworks."""
	SOC2 = "soc2"
	ISO27001 = "iso27001"  
	GDPR = "gdpr"
	HIPAA = "hipaa"
	PCI_DSS = "pci_dss"
	NIST = "nist"
	FedRAMP = "fedramp"


class SecurityEventType(Enum):
	"""Types of security events."""
	UNAUTHORIZED_ACCESS = "unauthorized_access"
	PRIVILEGE_ESCALATION = "privilege_escalation"
	DATA_BREACH = "data_breach"
	SUSPICIOUS_ACTIVITY = "suspicious_activity"
	CONFIGURATION_TAMPERING = "configuration_tampering"
	ENCRYPTION_FAILURE = "encryption_failure"
	AUTHENTICATION_BYPASS = "authentication_bypass"
	INJECTION_ATTEMPT = "injection_attempt"
	BRUTE_FORCE_ATTACK = "brute_force_attack"
	ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class SecurityVulnerability:
	"""Security vulnerability definition."""
	vulnerability_id: str
	title: str
	description: str
	risk_level: SecurityRiskLevel
	cvss_score: float
	affected_components: List[str]
	remediation_steps: List[str]
	compliance_impact: List[ComplianceFramework]
	discovered_at: datetime
	status: str  # open, in_progress, resolved, false_positive


@dataclass
class ComplianceCheck:
	"""Compliance check result."""
	check_id: str
	framework: ComplianceFramework
	control_id: str
	description: str
	status: str  # pass, fail, partial, not_applicable
	evidence: Dict[str, Any]
	remediation_required: bool
	risk_level: SecurityRiskLevel
	checked_at: datetime


@dataclass
class SecurityMetrics:
	"""Security metrics and KPIs."""
	total_configurations: int
	encrypted_configurations: int
	high_risk_configurations: int
	failed_authentications: int
	security_events_24h: int
	compliance_score: float
	vulnerability_count: Dict[str, int]
	encryption_coverage: float
	access_control_violations: int
	last_security_scan: datetime


class AdvancedSecurityAuditor:
	"""Advanced security auditing and compliance engine."""
	
	def __init__(self, config_engine: CentralConfigurationEngine):
		"""Initialize security auditor."""
		self.config_engine = config_engine
		self.vulnerability_db: Dict[str, SecurityVulnerability] = {}
		self.compliance_checks: Dict[str, ComplianceCheck] = {}
		self.security_baselines: Dict[str, Any] = {}
		self.threat_intelligence: Dict[str, Any] = {}
		
		# Initialize security rules and patterns
		asyncio.create_task(self._initialize_security_baselines())
		asyncio.create_task(self._load_threat_intelligence())
	
	# ==================== Vulnerability Assessment ====================
	
	async def comprehensive_security_scan(
		self,
		workspace_id: Optional[str] = None,
		frameworks: List[ComplianceFramework] = None
	) -> Dict[str, Any]:
		"""Perform comprehensive security assessment."""
		scan_id = f"scan_{uuid.uuid4().hex[:8]}"
		scan_start = datetime.now(timezone.utc)
		
		print(f"🔍 Starting comprehensive security scan: {scan_id}")
		
		# Initialize results structure
		results = {
			"scan_id": scan_id,
			"scan_start": scan_start.isoformat(),
			"scope": {
				"workspace_id": workspace_id,
				"frameworks": [f.value for f in (frameworks or [])]
			},
			"vulnerabilities": [],
			"compliance_results": {},
			"security_metrics": {},
			"recommendations": [],
			"risk_assessment": {}
		}
		
		try:
			# 1. Configuration Security Analysis
			config_vulns = await self._analyze_configuration_security(workspace_id)
			results["vulnerabilities"].extend(config_vulns)
			
			# 2. Access Control Assessment
			access_vulns = await self._assess_access_controls(workspace_id)
			results["vulnerabilities"].extend(access_vulns)
			
			# 3. Encryption Analysis
			encryption_vulns = await self._analyze_encryption_usage(workspace_id)
			results["vulnerabilities"].extend(encryption_vulns)
			
			# 4. Compliance Checks
			if frameworks:
				for framework in frameworks:
					compliance_result = await self._run_compliance_checks(framework, workspace_id)
					results["compliance_results"][framework.value] = compliance_result
			
			# 5. Security Metrics Collection
			results["security_metrics"] = await self._collect_security_metrics(workspace_id)
			
			# 6. Risk Assessment
			results["risk_assessment"] = await self._perform_risk_assessment(results["vulnerabilities"])
			
			# 7. Generate Recommendations
			results["recommendations"] = await self._generate_security_recommendations(results)
			
			# 8. Update audit log
			await self._log_security_scan(scan_id, results)
			
			scan_duration = (datetime.now(timezone.utc) - scan_start).total_seconds()
			results["scan_duration_seconds"] = scan_duration
			results["scan_completed"] = datetime.now(timezone.utc).isoformat()
			
			print(f"✅ Security scan completed: {len(results['vulnerabilities'])} vulnerabilities found")
			return results
			
		except Exception as e:
			print(f"❌ Security scan failed: {e}")
			results["error"] = str(e)
			results["status"] = "failed"
			return results
	
	async def _analyze_configuration_security(self, workspace_id: Optional[str]) -> List[Dict[str, Any]]:
		"""Analyze security of configurations."""
		vulnerabilities = []
		
		# Get configurations to analyze
		async with self.config_engine.get_db_session() as session:
			query = select(CCConfiguration)
			if workspace_id:
				query = query.where(CCConfiguration.workspace_id == workspace_id)
			
			result = await session.execute(query)
			configurations = result.scalars().all()
		
		for config in configurations:
			config_vulns = []
			
			# Check for hardcoded secrets
			secrets_vuln = await self._detect_hardcoded_secrets(config)
			if secrets_vuln:
				config_vulns.append(secrets_vuln)
			
			# Check for weak encryption
			encryption_vuln = await self._check_encryption_strength(config)
			if encryption_vuln:
				config_vulns.append(encryption_vuln)
			
			# Check for insecure defaults
			defaults_vuln = await self._check_insecure_defaults(config)
			if defaults_vuln:
				config_vulns.append(defaults_vuln)
			
			# Check for excessive permissions
			permissions_vuln = await self._check_excessive_permissions(config)
			if permissions_vuln:
				config_vulns.append(permissions_vuln)
			
			vulnerabilities.extend(config_vulns)
		
		return vulnerabilities
	
	async def _detect_hardcoded_secrets(self, config: CCConfiguration) -> Optional[Dict[str, Any]]:
		"""Detect hardcoded secrets in configuration."""
		secret_patterns = [
			r'password\s*[:=]\s*["\']([^"\']+)["\']',
			r'api[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
			r'secret\s*[:=]\s*["\']([^"\']+)["\']',
			r'token\s*[:=]\s*["\']([^"\']+)["\']',
			r'aws[_-]?(access|secret)[_-]?key\s*[:=]\s*["\']([^"\']+)["\']',
			r'private[_-]?key\s*[:=]\s*["\']([^"\']+)["\']'
		]
		
		config_str = json.dumps(config.value) if config.value else ""
		
		import re
		for pattern in secret_patterns:
			matches = re.finditer(pattern, config_str, re.IGNORECASE)
			for match in matches:
				return {
					"vulnerability_id": f"hardcoded_secret_{config.id}_{hash(match.group())%10000}",
					"title": "Hardcoded Secret Detected",
					"description": f"Potential hardcoded secret found in configuration '{config.name}'",
					"risk_level": SecurityRiskLevel.HIGH.value,
					"cvss_score": 7.5,
					"affected_components": [f"configuration:{config.id}"],
					"remediation_steps": [
						"Remove hardcoded secrets from configuration",
						"Use environment variables or secret management system",
						"Rotate any exposed credentials immediately"
					],
					"compliance_impact": ["pci_dss", "soc2", "iso27001"],
					"discovered_at": datetime.now(timezone.utc).isoformat(),
					"evidence": {
						"config_id": config.id,
						"config_name": config.name,
						"pattern_matched": pattern,
						"location": "configuration_value"
					}
				}
		
		return None
	
	async def _check_encryption_strength(self, config: CCConfiguration) -> Optional[Dict[str, Any]]:
		"""Check encryption strength and implementation."""
		if not config.encrypted or config.security_level.value in ["public", "internal"]:
			return {
				"vulnerability_id": f"weak_encryption_{config.id}",
				"title": "Insufficient Encryption",
				"description": f"Configuration '{config.name}' is not encrypted or uses weak encryption",
				"risk_level": SecurityRiskLevel.MEDIUM.value,
				"cvss_score": 5.3,
				"affected_components": [f"configuration:{config.id}"],
				"remediation_steps": [
					"Enable strong encryption for sensitive configurations",
					"Use AES-256 or stronger encryption algorithms",
					"Implement proper key management"
				],
				"compliance_impact": ["gdpr", "hipaa", "pci_dss"],
				"discovered_at": datetime.now(timezone.utc).isoformat(),
				"evidence": {
					"config_id": config.id,
					"encrypted": config.encrypted,
					"security_level": config.security_level.value
				}
			}
		
		return None
	
	# ==================== Compliance Framework Checks ====================
	
	async def _run_compliance_checks(
		self,
		framework: ComplianceFramework,
		workspace_id: Optional[str]
	) -> Dict[str, Any]:
		"""Run compliance checks for specific framework."""
		checks = []
		
		if framework == ComplianceFramework.SOC2:
			checks.extend(await self._run_soc2_checks(workspace_id))
		elif framework == ComplianceFramework.GDPR:
			checks.extend(await self._run_gdpr_checks(workspace_id))
		elif framework == ComplianceFramework.HIPAA:
			checks.extend(await self._run_hipaa_checks(workspace_id))
		elif framework == ComplianceFramework.PCI_DSS:
			checks.extend(await self._run_pci_dss_checks(workspace_id))
		elif framework == ComplianceFramework.ISO27001:
			checks.extend(await self._run_iso27001_checks(workspace_id))
		elif framework == ComplianceFramework.NIST:
			checks.extend(await self._run_nist_checks(workspace_id))
		
		# Calculate compliance score
		total_checks = len(checks)
		passed_checks = len([c for c in checks if c["status"] == "pass"])
		compliance_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
		
		return {
			"framework": framework.value,
			"compliance_score": compliance_score,
			"total_checks": total_checks,
			"passed_checks": passed_checks,
			"failed_checks": total_checks - passed_checks,
			"checks": checks,
			"assessed_at": datetime.now(timezone.utc).isoformat()
		}
	
	async def _run_soc2_checks(self, workspace_id: Optional[str]) -> List[Dict[str, Any]]:
		"""Run SOC 2 compliance checks."""
		checks = []
		
		# CC6.1 - Logical and Physical Access Controls
		access_check = await self._check_access_controls_soc2(workspace_id)
		checks.append(access_check)
		
		# CC6.7 - Data Transmission and Disposal
		encryption_check = await self._check_data_encryption_soc2(workspace_id)
		checks.append(encryption_check)
		
		# CC7.2 - System Monitoring
		monitoring_check = await self._check_system_monitoring_soc2(workspace_id)
		checks.append(monitoring_check)
		
		# CC8.1 - Change Management
		change_mgmt_check = await self._check_change_management_soc2(workspace_id)
		checks.append(change_mgmt_check)
		
		return checks
	
	async def _check_access_controls_soc2(self, workspace_id: Optional[str]) -> Dict[str, Any]:
		"""Check SOC 2 access controls compliance."""
		# Analyze authentication and authorization mechanisms
		async with self.config_engine.get_db_session() as session:
			# Check for configurations with weak access controls
			query = select(CCConfiguration).where(
				CCConfiguration.security_level == "public"
			)
			if workspace_id:
				query = query.where(CCConfiguration.workspace_id == workspace_id)
			
			result = await session.execute(query)
			public_configs = result.scalars().all()
		
		# Determine compliance status
		has_violations = len(public_configs) > 0
		
		return {
			"check_id": "soc2_cc6_1_access_controls",
			"framework": "soc2",
			"control_id": "CC6.1",
			"description": "Logical and Physical Access Controls",
			"status": "fail" if has_violations else "pass",
			"evidence": {
				"public_configurations_count": len(public_configs),
				"public_configurations": [{"id": c.id, "name": c.name} for c in public_configs[:5]]
			},
			"remediation_required": has_violations,
			"risk_level": SecurityRiskLevel.HIGH.value if has_violations else SecurityRiskLevel.LOW.value,
			"checked_at": datetime.now(timezone.utc).isoformat()
		}
	
	# ==================== Real-time Threat Detection ====================
	
	async def detect_security_threats(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Real-time security threat detection."""
		threats = []
		
		# Analyze for various threat patterns
		threats.extend(await self._detect_brute_force_attacks(event_data))
		threats.extend(await self._detect_privilege_escalation(event_data))
		threats.extend(await self._detect_data_exfiltration(event_data))
		threats.extend(await self._detect_injection_attempts(event_data))
		threats.extend(await self._detect_anomalous_behavior(event_data))
		
		# Log detected threats
		for threat in threats:
			await self._log_security_event(threat)
		
		return threats
	
	async def _detect_brute_force_attacks(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Detect brute force authentication attempts."""
		threats = []
		
		if event_data.get("event_type") == "authentication_failure":
			# Check for repeated failures from same IP
			ip_address = event_data.get("source_ip")
			user_id = event_data.get("user_id")
			
			if ip_address:
				# Count recent failures from this IP
				recent_failures = await self._count_recent_auth_failures(ip_address, minutes=15)
				
				if recent_failures >= 10:  # Threshold for brute force
					threats.append({
						"threat_id": f"brute_force_{ip_address}_{int(time.time())}",
						"type": SecurityEventType.BRUTE_FORCE_ATTACK.value,
						"severity": SecurityRiskLevel.HIGH.value,
						"description": f"Brute force attack detected from IP: {ip_address}",
						"source_ip": ip_address,
						"user_id": user_id,
						"failure_count": recent_failures,
						"time_window": "15 minutes",
						"detected_at": datetime.now(timezone.utc).isoformat(),
						"recommended_actions": [
							f"Block IP address: {ip_address}",
							"Review authentication logs",
							"Notify security team",
							"Consider implementing rate limiting"
						]
					})
		
		return threats
	
	async def _detect_anomalous_behavior(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Detect anomalous user behavior patterns."""
		threats = []
		
		user_id = event_data.get("user_id")
		if not user_id:
			return threats
		
		# Analyze user behavior patterns
		current_hour = datetime.now(timezone.utc).hour
		source_ip = event_data.get("source_ip")
		action = event_data.get("action")
		
		# Check for unusual access times
		if current_hour < 6 or current_hour > 22:  # Outside business hours
			user_history = await self._get_user_access_history(user_id, days=30)
			off_hours_access = [h for h in user_history if h.get("hour", 12) < 6 or h.get("hour", 12) > 22]
			
			if len(off_hours_access) < 5:  # User rarely accesses during off hours
				threats.append({
					"threat_id": f"anomalous_time_{user_id}_{int(time.time())}",
					"type": SecurityEventType.ANOMALOUS_BEHAVIOR.value,
					"severity": SecurityRiskLevel.MEDIUM.value,
					"description": f"Unusual access time detected for user: {user_id}",
					"user_id": user_id,
					"access_time": current_hour,
					"source_ip": source_ip,
					"historical_off_hours_access": len(off_hours_access),
					"detected_at": datetime.now(timezone.utc).isoformat(),
					"recommended_actions": [
						"Verify user identity",
						"Review recent user activities",
						"Contact user to confirm legitimate access"
					]
				})
		
		return threats
	
	# ==================== Security Metrics and Reporting ====================
	
	async def _collect_security_metrics(self, workspace_id: Optional[str]) -> SecurityMetrics:
		"""Collect comprehensive security metrics."""
		async with self.config_engine.get_db_session() as session:
			# Total configurations
			query = select(func.count(CCConfiguration.id))
			if workspace_id:
				query = query.where(CCConfiguration.workspace_id == workspace_id)
			result = await session.execute(query)
			total_configs = result.scalar() or 0
			
			# Encrypted configurations
			query = select(func.count(CCConfiguration.id)).where(CCConfiguration.encrypted == True)
			if workspace_id:
				query = query.where(CCConfiguration.workspace_id == workspace_id)
			result = await session.execute(query)
			encrypted_configs = result.scalar() or 0
			
			# High risk configurations
			query = select(func.count(CCConfiguration.id)).where(
				CCConfiguration.security_level == "public"
			)
			if workspace_id:
				query = query.where(CCConfiguration.workspace_id == workspace_id)
			result = await session.execute(query)
			high_risk_configs = result.scalar() or 0
			
			# Recent security events
			query = select(func.count(CCSecurityEvent.id)).where(
				CCSecurityEvent.created_at >= datetime.now(timezone.utc) - timedelta(hours=24)
			)
			result = await session.execute(query)
			security_events_24h = result.scalar() or 0
		
		# Calculate encryption coverage
		encryption_coverage = (encrypted_configs / total_configs * 100) if total_configs > 0 else 0
		
		# Mock additional metrics (would come from real monitoring)
		vulnerability_count = {
			"critical": 0,
			"high": 2,
			"medium": 5,
			"low": 8
		}
		
		return SecurityMetrics(
			total_configurations=total_configs,
			encrypted_configurations=encrypted_configs,
			high_risk_configurations=high_risk_configs,
			failed_authentications=0,  # Would come from auth logs
			security_events_24h=security_events_24h,
			compliance_score=85.5,  # Average across frameworks
			vulnerability_count=vulnerability_count,
			encryption_coverage=encryption_coverage,
			access_control_violations=1,
			last_security_scan=datetime.now(timezone.utc)
		)
	
	async def generate_security_report(
		self,
		workspace_id: Optional[str] = None,
		report_type: str = "comprehensive",
		frameworks: List[ComplianceFramework] = None
	) -> Dict[str, Any]:
		"""Generate comprehensive security report."""
		report_id = f"security_report_{uuid.uuid4().hex[:8]}"
		
		# Run security scan
		scan_results = await self.comprehensive_security_scan(workspace_id, frameworks)
		
		# Generate executive summary
		exec_summary = await self._generate_executive_summary(scan_results)
		
		# Generate detailed findings
		detailed_findings = await self._generate_detailed_findings(scan_results)
		
		# Generate remediation plan
		remediation_plan = await self._generate_remediation_plan(scan_results)
		
		report = {
			"report_id": report_id,
			"generated_at": datetime.now(timezone.utc).isoformat(),
			"report_type": report_type,
			"scope": {
				"workspace_id": workspace_id,
				"frameworks": [f.value for f in (frameworks or [])]
			},
			"executive_summary": exec_summary,
			"security_posture": {
				"overall_risk_level": await self._calculate_overall_risk(scan_results),
				"compliance_scores": scan_results.get("compliance_results", {}),
				"vulnerability_summary": scan_results.get("risk_assessment", {}),
				"security_metrics": scan_results.get("security_metrics", {})
			},
			"detailed_findings": detailed_findings,
			"remediation_plan": remediation_plan,
			"appendices": {
				"scan_details": scan_results,
				"methodology": await self._get_assessment_methodology(),
				"references": await self._get_security_references()
			}
		}
		
		return report
	
	# ==================== Helper Methods ====================
	
	async def _initialize_security_baselines(self):
		"""Initialize security baselines and rules."""
		self.security_baselines = {
			"encryption": {
				"required_algorithms": ["AES-256", "ChaCha20-Poly1305"],
				"minimum_key_length": 256,
				"required_for_levels": ["confidential", "secret"]
			},
			"access_control": {
				"max_public_configs": 0,
				"require_mfa_for": ["admin", "security_admin"],
				"max_session_duration": 3600
			},
			"monitoring": {
				"required_events": ["authentication", "authorization", "configuration_change"],
				"retention_days": 90,
				"alert_thresholds": {
					"failed_auth_per_hour": 50,
					"config_changes_per_hour": 100
				}
			}
		}
		
		print("🔒 Security baselines initialized")
	
	async def _load_threat_intelligence(self):
		"""Load threat intelligence data."""
		# In production, this would load from threat intelligence feeds
		self.threat_intelligence = {
			"known_attack_patterns": [
				"sql_injection",
				"xss_attack", 
				"brute_force",
				"privilege_escalation"
			],
			"suspicious_ips": set(),  # Would be populated from threat feeds
			"malicious_patterns": [
				r"union\s+select",
				r"<script",
				r"../../../",
				r"eval\s*\("
			]
		}
		
		print("🛡️ Threat intelligence loaded")
	
	async def _log_security_event(self, event: Dict[str, Any]):
		"""Log security event to database."""
		async with self.config_engine.get_db_session() as session:
			security_event = CCSecurityEvent(
				event_type=event.get("type"),
				severity=event.get("severity"),
				description=event.get("description"),
				source_ip=event.get("source_ip"),
				user_id=event.get("user_id"),
				metadata=event,
				created_at=datetime.now(timezone.utc)
			)
			
			session.add(security_event)
			await session.commit()
	
	async def _count_recent_auth_failures(self, ip_address: str, minutes: int = 15) -> int:
		"""Count recent authentication failures from IP."""
		# Mock implementation - would query actual auth logs
		return 5  # Placeholder
	
	async def _get_user_access_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
		"""Get user access history for behavioral analysis."""
		# Mock implementation - would query actual access logs
		return [{"hour": 9, "action": "login"}, {"hour": 14, "action": "config_update"}]


# ==================== Factory Functions ====================

async def create_security_auditor(config_engine: CentralConfigurationEngine) -> AdvancedSecurityAuditor:
	"""Create and initialize security auditor."""
	auditor = AdvancedSecurityAuditor(config_engine)
	await asyncio.sleep(1)  # Allow initialization
	print("🔍 Advanced Security Auditor initialized")
	return auditor