"""
Regulatory Compliance Service

Business logic for pharmaceutical regulatory compliance including submission management,
audit coordination, deviation handling, and compliance monitoring.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy import and_, or_, func, desc
from sqlalchemy.orm import Session

from ....auth_rbac.models import db
from .models import (
	PHRCRegulatoryFramework, PHRCSubmission, PHRCSubmissionDocument,
	PHRCAudit, PHRCAuditFinding, PHRCDeviation, PHRCCorrectiveAction,
	PHRCComplianceControl, PHRCRegulatoryContact, PHRCInspection,
	PHRCRegulatoryReport
)


class RegulatoryComplianceService:
	"""Service for managing regulatory compliance operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
	
	def _log_activity(self, activity: str, details: Dict[str, Any]) -> None:
		"""Log regulatory compliance activity"""
		print(f"[REGULATORY] {activity}: {details}")
	
	# Regulatory Framework Management
	
	def get_active_frameworks(self) -> List[PHRCRegulatoryFramework]:
		"""Get all active regulatory frameworks"""
		return PHRCRegulatoryFramework.query.filter_by(
			tenant_id=self.tenant_id,
			is_active=True
		).order_by(PHRCRegulatoryFramework.framework_name).all()
	
	def create_framework(self, framework_data: Dict[str, Any]) -> PHRCRegulatoryFramework:
		"""Create a new regulatory framework"""
		assert 'framework_code' in framework_data, "Framework code is required"
		assert 'framework_name' in framework_data, "Framework name is required"
		
		framework = PHRCRegulatoryFramework(
			tenant_id=self.tenant_id,
			**framework_data
		)
		
		db.session.add(framework)
		db.session.commit()
		
		self._log_activity("Framework Created", {
			'framework_id': framework.framework_id,
			'code': framework.framework_code,
			'name': framework.framework_name
		})
		
		return framework
	
	# Submission Management
	
	def create_submission(self, submission_data: Dict[str, Any]) -> PHRCSubmission:
		"""Create a regulatory submission"""
		assert 'submission_type' in submission_data, "Submission type is required"
		assert 'submission_title' in submission_data, "Submission title is required"
		assert 'framework_id' in submission_data, "Regulatory framework is required"
		
		# Generate submission number
		if 'submission_number' not in submission_data:
			submission_data['submission_number'] = self._generate_submission_number(
				submission_data['submission_type']
			)
		
		submission = PHRCSubmission(
			tenant_id=self.tenant_id,
			**submission_data
		)
		
		db.session.add(submission)
		db.session.commit()
		
		self._log_activity("Submission Created", {
			'submission_id': submission.submission_id,
			'number': submission.submission_number,
			'type': submission.submission_type
		})
		
		return submission
	
	def submit_to_authority(self, submission_id: str, submission_date: date = None) -> bool:
		"""Submit application to regulatory authority"""
		submission = PHRCSubmission.query.filter_by(
			submission_id=submission_id,
			tenant_id=self.tenant_id
		).first()
		
		if not submission:
			return False
		
		# Validate submission is ready
		validation_result = self._validate_submission_readiness(submission)
		if not validation_result['is_ready']:
			raise ValueError(f"Submission not ready: {validation_result['issues']}")
		
		# Update submission status
		submission.status = 'Submitted'
		submission.submission_date = submission_date or date.today()
		submission.target_response_date = self._calculate_target_response_date(
			submission.submission_type,
			submission.submission_date
		)
		
		db.session.commit()
		
		self._log_activity("Submission Filed", {
			'submission_id': submission_id,
			'submission_date': submission.submission_date,
			'target_response': submission.target_response_date
		})
		
		return True
	
	def get_submission_status(self, submission_id: str) -> Dict[str, Any]:
		"""Get detailed submission status"""
		submission = PHRCSubmission.query.filter_by(
			submission_id=submission_id,
			tenant_id=self.tenant_id
		).first()
		
		if not submission:
			return {}
		
		# Calculate days remaining
		days_remaining = None
		if submission.target_response_date:
			days_remaining = (submission.target_response_date - date.today()).days
		
		return {
			'submission_id': submission.submission_id,
			'number': submission.submission_number,
			'status': submission.status,
			'submission_date': submission.submission_date,
			'target_response_date': submission.target_response_date,
			'days_remaining': days_remaining,
			'document_count': len(submission.documents),
			'is_overdue': days_remaining is not None and days_remaining < 0
		}
	
	def _generate_submission_number(self, submission_type: str) -> str:
		"""Generate unique submission number"""
		year = datetime.now().year
		
		# Count existing submissions of this type this year
		count = PHRCSubmission.query.filter(
			PHRCSubmission.tenant_id == self.tenant_id,
			PHRCSubmission.submission_type == submission_type,
			func.extract('year', PHRCSubmission.created_at) == year
		).count()
		
		return f"{submission_type}-{year}-{count + 1:04d}"
	
	def _validate_submission_readiness(self, submission: PHRCSubmission) -> Dict[str, Any]:
		"""Validate submission is ready for filing"""
		issues = []
		
		# Check required documents
		if not submission.documents:
			issues.append("No documents attached")
		else:
			# Check all documents are final
			draft_docs = [doc for doc in submission.documents if not doc.is_final]
			if draft_docs:
				issues.append(f"{len(draft_docs)} documents still in draft status")
		
		# Check required fields
		if not submission.product_name:
			issues.append("Product name is required")
		
		if not submission.indication:
			issues.append("Indication is required")
		
		return {
			'is_ready': len(issues) == 0,
			'issues': issues
		}
	
	def _calculate_target_response_date(self, submission_type: str, submission_date: date) -> date:
		"""Calculate target response date based on submission type"""
		response_days = {
			'IND': 30,
			'NDA': 180,
			'BLA': 180,
			'ANDA': 300,
			'MAA': 210,
			'DMF': 60
		}
		
		days = response_days.get(submission_type, 180)  # Default 180 days
		return submission_date + timedelta(days=days)
	
	# Audit Management
	
	def create_audit(self, audit_data: Dict[str, Any]) -> PHRCAudit:
		"""Create a regulatory audit"""
		assert 'audit_title' in audit_data, "Audit title is required"
		assert 'audit_type' in audit_data, "Audit type is required"
		assert 'framework_id' in audit_data, "Regulatory framework is required"
		
		# Generate audit number
		if 'audit_number' not in audit_data:
			audit_data['audit_number'] = self._generate_audit_number(audit_data['audit_type'])
		
		audit = PHRCAudit(
			tenant_id=self.tenant_id,
			**audit_data
		)
		
		db.session.add(audit)
		db.session.commit()
		
		self._log_activity("Audit Created", {
			'audit_id': audit.audit_id,
			'number': audit.audit_number,
			'type': audit.audit_type
		})
		
		return audit
	
	def add_audit_finding(self, finding_data: Dict[str, Any]) -> PHRCAuditFinding:
		"""Add finding to audit"""
		assert 'audit_id' in finding_data, "Audit ID is required"
		assert 'finding_title' in finding_data, "Finding title is required"
		assert 'severity' in finding_data, "Severity is required"
		
		# Generate finding number
		if 'finding_number' not in finding_data:
			finding_data['finding_number'] = self._generate_finding_number(
				finding_data['audit_id']
			)
		
		finding = PHRCAuditFinding(
			tenant_id=self.tenant_id,
			**finding_data
		)
		
		db.session.add(finding)
		db.session.commit()
		
		self._log_activity("Audit Finding Added", {
			'finding_id': finding.finding_id,
			'audit_id': finding.audit_id,
			'severity': finding.severity
		})
		
		return finding
	
	def get_audit_summary(self, audit_id: str) -> Dict[str, Any]:
		"""Get audit summary with findings"""
		audit = PHRCAudit.query.filter_by(
			audit_id=audit_id,
			tenant_id=self.tenant_id
		).first()
		
		if not audit:
			return {}
		
		# Count findings by severity
		finding_counts = {
			'Critical': 0,
			'Major': 0,
			'Minor': 0,
			'Observation': 0
		}
		
		for finding in audit.findings:
			if finding.severity in finding_counts:
				finding_counts[finding.severity] += 1
		
		# Count open findings
		open_findings = len([f for f in audit.findings if f.status == 'Open'])
		
		return {
			'audit_id': audit.audit_id,
			'audit_number': audit.audit_number,
			'status': audit.status,
			'overall_rating': audit.overall_rating,
			'total_findings': len(audit.findings),
			'open_findings': open_findings,
			'findings_by_severity': finding_counts,
			'completion_percentage': self._calculate_audit_completion(audit)
		}
	
	def _generate_audit_number(self, audit_type: str) -> str:
		"""Generate unique audit number"""
		year = datetime.now().year
		
		count = PHRCAudit.query.filter(
			PHRCAudit.tenant_id == self.tenant_id,
			func.extract('year', PHRCAudit.created_at) == year
		).count()
		
		return f"AUD-{audit_type[:3].upper()}-{year}-{count + 1:04d}"
	
	def _generate_finding_number(self, audit_id: str) -> str:
		"""Generate finding number within audit"""
		count = PHRCAuditFinding.query.filter_by(
			audit_id=audit_id,
			tenant_id=self.tenant_id
		).count()
		
		return f"F-{count + 1:03d}"
	
	def _calculate_audit_completion(self, audit: PHRCAudit) -> float:
		"""Calculate audit completion percentage"""
		if not audit.findings:
			return 100.0 if audit.status == 'Completed' else 0.0
		
		closed_findings = len([f for f in audit.findings if f.status == 'Closed'])
		return (closed_findings / len(audit.findings)) * 100.0
	
	# Deviation Management
	
	def create_deviation(self, deviation_data: Dict[str, Any]) -> PHRCDeviation:
		"""Create a quality deviation"""
		assert 'deviation_title' in deviation_data, "Deviation title is required"
		assert 'description' in deviation_data, "Description is required"
		assert 'severity' in deviation_data, "Severity is required"
		assert 'discovered_by' in deviation_data, "Discoverer is required"
		
		# Generate deviation number
		if 'deviation_number' not in deviation_data:
			deviation_data['deviation_number'] = self._generate_deviation_number()
		
		# Set discovered date if not provided
		if 'discovered_date' not in deviation_data:
			deviation_data['discovered_date'] = date.today()
		
		deviation = PHRCDeviation(
			tenant_id=self.tenant_id,
			**deviation_data
		)
		
		db.session.add(deviation)
		db.session.commit()
		
		self._log_activity("Deviation Created", {
			'deviation_id': deviation.deviation_id,
			'number': deviation.deviation_number,
			'severity': deviation.severity
		})
		
		# Auto-assign investigation if critical
		if deviation.severity == 'Critical':
			self._auto_assign_investigation(deviation)
		
		return deviation
	
	def assign_investigation(self, deviation_id: str, investigator_id: str, 
						   deadline: date = None) -> bool:
		"""Assign deviation investigation"""
		deviation = PHRCDeviation.query.filter_by(
			deviation_id=deviation_id,
			tenant_id=self.tenant_id
		).first()
		
		if not deviation:
			return False
		
		deviation.assigned_investigator = investigator_id
		deviation.investigation_deadline = deadline or self._calculate_investigation_deadline(
			deviation.severity
		)
		deviation.status = 'Under Investigation'
		
		db.session.commit()
		
		self._log_activity("Investigation Assigned", {
			'deviation_id': deviation_id,
			'investigator': investigator_id,
			'deadline': deviation.investigation_deadline
		})
		
		return True
	
	def complete_investigation(self, deviation_id: str, root_cause: str,
							 impact_assessment: str = None) -> bool:
		"""Complete deviation investigation"""
		deviation = PHRCDeviation.query.filter_by(
			deviation_id=deviation_id,
			tenant_id=self.tenant_id
		).first()
		
		if not deviation:
			return False
		
		deviation.root_cause = root_cause
		if impact_assessment:
			deviation.impact_assessment = impact_assessment
		
		# Determine if CAPA is required
		if self._capa_required(deviation):
			deviation.status = 'CAPA Required'
		else:
			deviation.status = 'Closed'
			deviation.closure_date = date.today()
		
		db.session.commit()
		
		self._log_activity("Investigation Completed", {
			'deviation_id': deviation_id,
			'status': deviation.status,
			'capa_required': deviation.status == 'CAPA Required'
		})
		
		return True
	
	def _generate_deviation_number(self) -> str:
		"""Generate unique deviation number"""
		year = datetime.now().year
		
		count = PHRCDeviation.query.filter(
			PHRCDeviation.tenant_id == self.tenant_id,
			func.extract('year', PHRCDeviation.created_at) == year
		).count()
		
		return f"DEV-{year}-{count + 1:06d}"
	
	def _auto_assign_investigation(self, deviation: PHRCDeviation) -> None:
		"""Auto-assign critical deviation investigation"""
		# In a real implementation, this would use business rules to assign
		# For now, we just set the deadline
		deviation.investigation_deadline = self._calculate_investigation_deadline(
			deviation.severity
		)
		db.session.commit()
	
	def _calculate_investigation_deadline(self, severity: str) -> date:
		"""Calculate investigation deadline based on severity"""
		days_map = {
			'Critical': 1,   # 24 hours
			'Major': 3,      # 72 hours  
			'Minor': 7       # 7 days
		}
		
		days = days_map.get(severity, 7)
		return date.today() + timedelta(days=days)
	
	def _capa_required(self, deviation: PHRCDeviation) -> bool:
		"""Determine if CAPA is required for deviation"""
		# CAPA required for Critical and Major deviations
		return deviation.severity in ['Critical', 'Major']
	
	# CAPA Management
	
	def create_corrective_action(self, action_data: Dict[str, Any]) -> PHRCCorrectiveAction:
		"""Create corrective/preventive action"""
		assert 'action_title' in action_data, "Action title is required"
		assert 'description' in action_data, "Description is required"
		assert 'assigned_to' in action_data, "Assignee is required"
		assert 'planned_completion_date' in action_data, "Completion date is required"
		
		# Generate action number
		if 'action_number' not in action_data:
			action_data['action_number'] = self._generate_action_number()
		
		action = PHRCCorrectiveAction(
			tenant_id=self.tenant_id,
			**action_data
		)
		
		db.session.add(action)
		db.session.commit()
		
		self._log_activity("CAPA Created", {
			'action_id': action.action_id,
			'number': action.action_number,
			'type': action.action_type
		})
		
		return action
	
	def complete_action(self, action_id: str, completion_notes: str = None) -> bool:
		"""Complete corrective action"""
		action = PHRCCorrectiveAction.query.filter_by(
			action_id=action_id,
			tenant_id=self.tenant_id
		).first()
		
		if not action:
			return False
		
		action.status = 'Completed'
		action.actual_completion_date = date.today()
		if completion_notes:
			action.implementation_notes = completion_notes
		
		db.session.commit()
		
		self._log_activity("CAPA Completed", {
			'action_id': action_id,
			'completion_date': action.actual_completion_date
		})
		
		return True
	
	def verify_effectiveness(self, action_id: str, is_effective: bool,
						   verification_notes: str = None) -> bool:
		"""Verify CAPA effectiveness"""
		action = PHRCCorrectiveAction.query.filter_by(
			action_id=action_id,
			tenant_id=self.tenant_id
		).first()
		
		if not action:
			return False
		
		action.effectiveness_verified = is_effective
		action.effectiveness_check_date = date.today()
		if verification_notes:
			action.effectiveness_notes = verification_notes
		
		if is_effective:
			action.status = 'Closed'
		else:
			action.status = 'In Progress'  # Requires additional action
		
		db.session.commit()
		
		self._log_activity("CAPA Effectiveness Verified", {
			'action_id': action_id,
			'effective': is_effective,
			'status': action.status
		})
		
		return True
	
	def _generate_action_number(self) -> str:
		"""Generate unique action number"""
		year = datetime.now().year
		
		count = PHRCCorrectiveAction.query.filter(
			PHRCCorrectiveAction.tenant_id == self.tenant_id,
			func.extract('year', PHRCCorrectiveAction.created_at) == year
		).count()
		
		return f"CAPA-{year}-{count + 1:06d}"
	
	# Compliance Monitoring
	
	def get_compliance_dashboard(self) -> Dict[str, Any]:
		"""Get compliance dashboard data"""
		today = date.today()
		
		# Submission metrics
		submissions_stats = self._get_submission_stats()
		
		# Audit metrics
		audit_stats = self._get_audit_stats()
		
		# Deviation metrics
		deviation_stats = self._get_deviation_stats()
		
		# CAPA metrics
		capa_stats = self._get_capa_stats()
		
		return {
			'submissions': submissions_stats,
			'audits': audit_stats,
			'deviations': deviation_stats,
			'capas': capa_stats,
			'compliance_score': self._calculate_compliance_score(),
			'alerts': self._get_compliance_alerts()
		}
	
	def _get_submission_stats(self) -> Dict[str, Any]:
		"""Get submission statistics"""
		total = PHRCSubmission.query.filter_by(tenant_id=self.tenant_id).count()
		
		pending = PHRCSubmission.query.filter_by(
			tenant_id=self.tenant_id,
			status='Under Review'
		).count()
		
		approved = PHRCSubmission.query.filter_by(
			tenant_id=self.tenant_id,
			status='Approved'
		).count()
		
		return {
			'total': total,
			'pending': pending,
			'approved': approved,
			'success_rate': (approved / total * 100) if total > 0 else 0
		}
	
	def _get_audit_stats(self) -> Dict[str, Any]:
		"""Get audit statistics"""
		total = PHRCAudit.query.filter_by(tenant_id=self.tenant_id).count()
		
		active = PHRCAudit.query.filter_by(
			tenant_id=self.tenant_id,
			status='In Progress'
		).count()
		
		open_findings = PHRCAuditFinding.query.filter_by(
			tenant_id=self.tenant_id,
			status='Open'
		).count()
		
		return {
			'total': total,
			'active': active,
			'open_findings': open_findings
		}
	
	def _get_deviation_stats(self) -> Dict[str, Any]:
		"""Get deviation statistics"""
		total = PHRCDeviation.query.filter_by(tenant_id=self.tenant_id).count()
		
		open_deviations = PHRCDeviation.query.filter_by(
			tenant_id=self.tenant_id,
			status='Open'
		).count()
		
		overdue = PHRCDeviation.query.filter(
			PHRCDeviation.tenant_id == self.tenant_id,
			PHRCDeviation.investigation_deadline < date.today(),
			PHRCDeviation.status.in_(['Open', 'Under Investigation'])
		).count()
		
		return {
			'total': total,
			'open': open_deviations,
			'overdue': overdue
		}
	
	def _get_capa_stats(self) -> Dict[str, Any]:
		"""Get CAPA statistics"""
		total = PHRCCorrectiveAction.query.filter_by(tenant_id=self.tenant_id).count()
		
		active = PHRCCorrectiveAction.query.filter_by(
			tenant_id=self.tenant_id,
			status='In Progress'
		).count()
		
		overdue = PHRCCorrectiveAction.query.filter(
			PHRCCorrectiveAction.tenant_id == self.tenant_id,
			PHRCCorrectiveAction.planned_completion_date < date.today(),
			PHRCCorrectiveAction.status.in_(['Planned', 'In Progress'])
		).count()
		
		return {
			'total': total,
			'active': active,
			'overdue': overdue
		}
	
	def _calculate_compliance_score(self) -> float:
		"""Calculate overall compliance score"""
		# Simple scoring algorithm - in practice this would be more sophisticated
		
		scores = []
		
		# Submission success rate (40% weight)
		submission_stats = self._get_submission_stats()
		if submission_stats['total'] > 0:
			scores.append(submission_stats['success_rate'] * 0.4)
		
		# Audit findings resolution (30% weight)
		open_findings = PHRCAuditFinding.query.filter_by(
			tenant_id=self.tenant_id,
			status='Open'
		).count()
		total_findings = PHRCAuditFinding.query.filter_by(tenant_id=self.tenant_id).count()
		
		if total_findings > 0:
			resolution_rate = ((total_findings - open_findings) / total_findings) * 100
			scores.append(resolution_rate * 0.3)
		
		# Deviation response time (30% weight)
		overdue_deviations = PHRCDeviation.query.filter(
			PHRCDeviation.tenant_id == self.tenant_id,
			PHRCDeviation.investigation_deadline < date.today(),
			PHRCDeviation.status.in_(['Open', 'Under Investigation'])
		).count()
		total_deviations = PHRCDeviation.query.filter_by(tenant_id=self.tenant_id).count()
		
		if total_deviations > 0:
			on_time_rate = ((total_deviations - overdue_deviations) / total_deviations) * 100
			scores.append(on_time_rate * 0.3)
		
		return sum(scores) if scores else 100.0
	
	def _get_compliance_alerts(self) -> List[Dict[str, Any]]:
		"""Get compliance alerts and warnings"""
		alerts = []
		
		# Overdue submissions
		overdue_submissions = PHRCSubmission.query.filter(
			PHRCSubmission.tenant_id == self.tenant_id,
			PHRCSubmission.target_response_date < date.today(),
			PHRCSubmission.status == 'Under Review'
		).all()
		
		for submission in overdue_submissions:
			days_overdue = (date.today() - submission.target_response_date).days
			alerts.append({
				'type': 'overdue_submission',
				'severity': 'High',
				'message': f"Submission {submission.submission_number} is {days_overdue} days overdue",
				'entity_id': submission.submission_id
			})
		
		# Critical deviations without investigation
		unassigned_critical = PHRCDeviation.query.filter_by(
			tenant_id=self.tenant_id,
			severity='Critical',
			status='Open'
		).filter(PHRCDeviation.assigned_investigator.is_(None)).all()
		
		for deviation in unassigned_critical:
			alerts.append({
				'type': 'unassigned_critical_deviation',
				'severity': 'Critical',
				'message': f"Critical deviation {deviation.deviation_number} requires investigation assignment",
				'entity_id': deviation.deviation_id
			})
		
		return alerts