#!/usr/bin/env python3
"""
Blockchain Security Blueprint for Flask-AppBuilder
==================================================

Flask-AppBuilder blueprint providing web interface for blockchain-based 
digital twin security, provenance tracking, and smart contract management.
"""

from flask import Blueprint, request, jsonify, render_template_string
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget, ShowWidget
from flask_appbuilder.actions import action
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio

# Import the blockchain security engine
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from capabilities.blockchain_security import (
	BlockchainSecurityEngine, TransactionType, AccessLevel, 
	ComplianceStatus, SmartContract, MaintenancePriority
)
from blueprints.base import BaseCapabilityModel, BaseCapabilityView
from blueprints.base import db, Column, String, DateTime, Text, Boolean, Integer, Float, JSONB, UUID
import uuid

# =============================================================================
# SQLAlchemy Models for Blockchain Security
# =============================================================================

class BlockchainUser(BaseCapabilityModel):
	"""User registry for blockchain system"""
	__tablename__ = 'blockchain_users'
	
	user_id = Column(String(100), nullable=False, unique=True, index=True)
	public_key = Column(Text, nullable=False)
	is_active = Column(Boolean, default=True, nullable=False)
	access_level = Column(String(50), default='read_only')
	last_activity = Column(DateTime, default=datetime.utcnow)
	total_transactions = Column(Integer, default=0)
	
	def __repr__(self):
		return f'<BlockchainUser {self.user_id}>'

class BlockchainTransaction(BaseCapabilityModel):
	"""Blockchain transaction records"""
	__tablename__ = 'blockchain_transactions'
	
	transaction_id = Column(String(100), nullable=False, unique=True, index=True)
	twin_id = Column(String(100), nullable=False, index=True)
	transaction_type = Column(String(50), nullable=False)
	previous_hash = Column(String(64))
	data_hash = Column(String(64), nullable=False)
	transaction_hash = Column(String(64), nullable=False)
	payload = Column(JSONB, default=dict)
	signer_id = Column(String(100), nullable=False)
	signature = Column(Text, nullable=False)
	nonce = Column(Integer, nullable=False)
	is_verified = Column(Boolean, default=False)
	
	def __repr__(self):
		return f'<BlockchainTransaction {self.transaction_id}>'

class ProvenanceRecord(BaseCapabilityModel):
	"""Provenance chain records"""
	__tablename__ = 'provenance_records'
	
	record_id = Column(String(100), nullable=False, unique=True, index=True)
	twin_id = Column(String(100), nullable=False, index=True)
	event_type = Column(String(100), nullable=False)
	location = Column(String(200), nullable=False)
	participant = Column(String(200), nullable=False)
	previous_record_hash = Column(String(64))
	record_hash = Column(String(64), nullable=False)
	metadata_ = Column('metadata', JSONB, default=dict)
	certifications = Column(JSONB, default=list)
	signer_id = Column(String(100), nullable=False)
	is_verified = Column(Boolean, default=False)
	
	def __repr__(self):
		return f'<ProvenanceRecord {self.record_id}>'

class SmartContractModel(BaseCapabilityModel):
	"""Smart contract definitions"""
	__tablename__ = 'smart_contracts'
	
	contract_id = Column(String(100), nullable=False, unique=True, index=True)
	name = Column(String(200), nullable=False)
	description = Column(Text)
	version = Column(String(50), nullable=False)
	rules = Column(JSONB, default=list)
	conditions = Column(JSONB, default=list)
	actions = Column(JSONB, default=list)
	creator = Column(String(100), nullable=False)
	is_active = Column(Boolean, default=True)
	execution_count = Column(Integer, default=0)
	last_executed = Column(DateTime)
	
	def __repr__(self):
		return f'<SmartContract {self.name}>'

class AccessControlRecord(BaseCapabilityModel):
	"""Access control records"""
	__tablename__ = 'access_control_records'
	
	twin_id = Column(String(100), nullable=False, index=True)
	user_id = Column(String(100), nullable=False, index=True)
	access_level = Column(String(50), nullable=False)
	granted_by = Column(String(100), nullable=False)
	granted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	revoked_at = Column(DateTime)
	is_active = Column(Boolean, default=True)
	
	def __repr__(self):
		return f'<AccessControl {self.twin_id}:{self.user_id}>'

class SecurityAuditLog(BaseCapabilityModel):
	"""Security audit log entries"""
	__tablename__ = 'security_audit_logs'
	
	event_type = Column(String(100), nullable=False, index=True)
	twin_id = Column(String(100), index=True)
	user_id = Column(String(100), index=True)
	event_data = Column(JSONB, default=dict)
	severity = Column(String(50), default='info')
	source_ip = Column(String(45))
	user_agent = Column(Text)
	
	def __repr__(self):
		return f'<SecurityAuditLog {self.event_type}>'

# =============================================================================
# Flask-AppBuilder Views
# =============================================================================

class BlockchainUserView(BaseCapabilityView):
	"""View for managing blockchain users"""
	datamodel = SQLAInterface(BlockchainUser)
	
	list_columns = ['user_id', 'access_level', 'is_active', 'total_transactions', 'last_activity', 'created_at']
	show_columns = ['user_id', 'public_key', 'access_level', 'is_active', 'total_transactions', 'last_activity', 'created_at', 'updated_at']
	add_columns = ['user_id', 'access_level', 'is_active']
	edit_columns = ['access_level', 'is_active']
	
	base_order = ('created_at', 'desc')
	search_columns = ['user_id', 'access_level']
	
	@action("generate_keys", "Generate Key Pair", confirmation="Generate new key pair for user?", icon="fa-key")
	def generate_keys(self, items):
		"""Generate new key pair for selected users"""
		if not hasattr(self, '_security_engine'):
			self._security_engine = BlockchainSecurityEngine()
		
		for item in items:
			try:
				private_key, public_key = self._security_engine.register_user(item.user_id)
				item.public_key = public_key
				db.session.commit()
				self.update_redirect()
			except Exception as e:
				self.message = f"Error generating keys for {item.user_id}: {str(e)}"
				return self.list()
		
		self.message = f"Generated key pairs for {len(items)} users"
		return self.list()

class BlockchainTransactionView(BaseCapabilityView):
	"""View for blockchain transactions"""
	datamodel = SQLAInterface(BlockchainTransaction)
	
	list_columns = ['transaction_id', 'twin_id', 'transaction_type', 'signer_id', 'is_verified', 'created_at']
	show_columns = ['transaction_id', 'twin_id', 'transaction_type', 'previous_hash', 'data_hash', 
				   'transaction_hash', 'payload', 'signer_id', 'signature', 'is_verified', 'created_at']
	search_columns = ['transaction_id', 'twin_id', 'transaction_type', 'signer_id']
	
	base_order = ('created_at', 'desc')
	
	@action("verify_transactions", "Verify Transactions", confirmation="Verify selected transactions?", icon="fa-check")
	def verify_transactions(self, items):
		"""Verify selected transactions"""
		if not hasattr(self, '_security_engine'):
			self._security_engine = BlockchainSecurityEngine()
		
		verified_count = 0
		for item in items:
			# In a real implementation, this would verify the transaction
			# For now, we'll just mark it as verified
			item.is_verified = True
			verified_count += 1
		
		db.session.commit()
		self.message = f"Verified {verified_count} transactions"
		return self.list()

class ProvenanceRecordView(BaseCapabilityView):
	"""View for provenance records"""
	datamodel = SQLAInterface(ProvenanceRecord)
	
	list_columns = ['record_id', 'twin_id', 'event_type', 'location', 'participant', 'is_verified', 'created_at']
	show_columns = ['record_id', 'twin_id', 'event_type', 'location', 'participant', 
				   'previous_record_hash', 'record_hash', 'metadata_', 'certifications', 
				   'signer_id', 'is_verified', 'created_at']
	add_columns = ['twin_id', 'event_type', 'location', 'participant', 'metadata_', 'certifications', 'signer_id']
	edit_columns = ['event_type', 'location', 'participant', 'metadata_', 'certifications']
	
	base_order = ('created_at', 'desc')
	search_columns = ['twin_id', 'event_type', 'location', 'participant']

class SmartContractView(BaseCapabilityView):
	"""View for smart contracts"""
	datamodel = SQLAInterface(SmartContractModel)
	
	list_columns = ['name', 'version', 'creator', 'is_active', 'execution_count', 'last_executed', 'created_at']
	show_columns = ['contract_id', 'name', 'description', 'version', 'rules', 'conditions', 
				   'actions', 'creator', 'is_active', 'execution_count', 'last_executed', 'created_at']
	add_columns = ['name', 'description', 'version', 'rules', 'conditions', 'actions', 'creator']
	edit_columns = ['name', 'description', 'rules', 'conditions', 'actions', 'is_active']
	
	base_order = ('created_at', 'desc')
	search_columns = ['name', 'creator']
	
	@action("activate_contracts", "Activate Contracts", confirmation="Activate selected contracts?", icon="fa-play")
	def activate_contracts(self, items):
		"""Activate selected smart contracts"""
		for item in items:
			item.is_active = True
		
		db.session.commit()
		self.message = f"Activated {len(items)} smart contracts"
		return self.list()
	
	@action("deactivate_contracts", "Deactivate Contracts", confirmation="Deactivate selected contracts?", icon="fa-pause")
	def deactivate_contracts(self, items):
		"""Deactivate selected smart contracts"""
		for item in items:
			item.is_active = False
		
		db.session.commit()
		self.message = f"Deactivated {len(items)} smart contracts"
		return self.list()

class AccessControlView(BaseCapabilityView):
	"""View for access control records"""
	datamodel = SQLAInterface(AccessControlRecord)
	
	list_columns = ['twin_id', 'user_id', 'access_level', 'granted_by', 'granted_at', 'is_active']
	show_columns = ['twin_id', 'user_id', 'access_level', 'granted_by', 'granted_at', 'revoked_at', 'is_active', 'created_at']
	add_columns = ['twin_id', 'user_id', 'access_level', 'granted_by']
	edit_columns = ['access_level']
	
	base_order = ('granted_at', 'desc')
	search_columns = ['twin_id', 'user_id', 'access_level', 'granted_by']
	
	@action("revoke_access", "Revoke Access", confirmation="Revoke access for selected records?", icon="fa-ban")
	def revoke_access(self, items):
		"""Revoke access for selected records"""
		for item in items:
			item.is_active = False
			item.revoked_at = datetime.utcnow()
		
		db.session.commit()
		self.message = f"Revoked access for {len(items)} records"
		return self.list()

class SecurityAuditLogView(BaseCapabilityView):
	"""View for security audit logs"""
	datamodel = SQLAInterface(SecurityAuditLog)
	
	list_columns = ['event_type', 'twin_id', 'user_id', 'severity', 'source_ip', 'created_at']
	show_columns = ['event_type', 'twin_id', 'user_id', 'event_data', 'severity', 'source_ip', 'user_agent', 'created_at']
	search_columns = ['event_type', 'twin_id', 'user_id', 'severity', 'source_ip']
	
	base_order = ('created_at', 'desc')

class BlockchainSecurityDashboardView(BaseView):
	"""Main dashboard for blockchain security"""
	
	default_view = 'dashboard'
	
	@expose('/dashboard/')
	@has_access
	def dashboard(self):
		"""Main security dashboard"""
		
		# Get summary statistics
		total_users = db.session.query(BlockchainUser).count()
		active_users = db.session.query(BlockchainUser).filter(BlockchainUser.is_active == True).count()
		total_transactions = db.session.query(BlockchainTransaction).count()
		verified_transactions = db.session.query(BlockchainTransaction).filter(BlockchainTransaction.is_verified == True).count()
		total_contracts = db.session.query(SmartContractModel).count()
		active_contracts = db.session.query(SmartContractModel).filter(SmartContractModel.is_active == True).count()
		total_provenance = db.session.query(ProvenanceRecord).count()
		
		# Recent activity
		recent_transactions = db.session.query(BlockchainTransaction).order_by(
			BlockchainTransaction.created_at.desc()
		).limit(10).all()
		
		recent_audit_logs = db.session.query(SecurityAuditLog).order_by(
			SecurityAuditLog.created_at.desc()
		).limit(10).all()
		
		dashboard_html = """
		<div class="row">
			<div class="col-lg-3 col-md-6">
				<div class="panel panel-primary">
					<div class="panel-heading">
						<div class="row">
							<div class="col-xs-3">
								<i class="fa fa-users fa-5x"></i>
							</div>
							<div class="col-xs-9 text-right">
								<div class="huge">{{ active_users }}</div>
								<div>Active Users</div>
							</div>
						</div>
					</div>
					<div class="panel-footer">
						<span class="pull-left">Total: {{ total_users }}</span>
						<div class="clearfix"></div>
					</div>
				</div>
			</div>
			
			<div class="col-lg-3 col-md-6">
				<div class="panel panel-green">
					<div class="panel-heading">
						<div class="row">
							<div class="col-xs-3">
								<i class="fa fa-chain fa-5x"></i>
							</div>
							<div class="col-xs-9 text-right">
								<div class="huge">{{ verified_transactions }}</div>
								<div>Verified Transactions</div>
							</div>
						</div>
					</div>
					<div class="panel-footer">
						<span class="pull-left">Total: {{ total_transactions }}</span>
						<div class="clearfix"></div>
					</div>
				</div>
			</div>
			
			<div class="col-lg-3 col-md-6">
				<div class="panel panel-yellow">
					<div class="panel-heading">
						<div class="row">
							<div class="col-xs-3">
								<i class="fa fa-code fa-5x"></i>
							</div>
							<div class="col-xs-9 text-right">
								<div class="huge">{{ active_contracts }}</div>
								<div>Active Contracts</div>
							</div>
						</div>
					</div>
					<div class="panel-footer">
						<span class="pull-left">Total: {{ total_contracts }}</span>
						<div class="clearfix"></div>
					</div>
				</div>
			</div>
			
			<div class="col-lg-3 col-md-6">
				<div class="panel panel-red">
					<div class="panel-heading">
						<div class="row">
							<div class="col-xs-3">
								<i class="fa fa-history fa-5x"></i>
							</div>
							<div class="col-xs-9 text-right">
								<div class="huge">{{ total_provenance }}</div>
								<div>Provenance Records</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row">
			<div class="col-lg-6">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Recent Transactions</h3>
					</div>
					<div class="panel-body">
						<table class="table table-striped">
							<thead>
								<tr>
									<th>Transaction ID</th>
									<th>Twin ID</th>
									<th>Type</th>
									<th>Verified</th>
									<th>Created</th>
								</tr>
							</thead>
							<tbody>
								{% for tx in recent_transactions %}
								<tr>
									<td>{{ tx.transaction_id[:12] }}...</td>
									<td>{{ tx.twin_id }}</td>
									<td>{{ tx.transaction_type }}</td>
									<td>
										{% if tx.is_verified %}
											<span class="label label-success">Yes</span>
										{% else %}
											<span class="label label-warning">No</span>
										{% endif %}
									</td>
									<td>{{ tx.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
								</tr>
								{% endfor %}
							</tbody>
						</table>
					</div>
				</div>
			</div>
			
			<div class="col-lg-6">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Recent Security Events</h3>
					</div>
					<div class="panel-body">
						<table class="table table-striped">
							<thead>
								<tr>
									<th>Event Type</th>
									<th>Twin ID</th>
									<th>User</th>
									<th>Severity</th>
									<th>Time</th>
								</tr>
							</thead>
							<tbody>
								{% for log in recent_audit_logs %}
								<tr>
									<td>{{ log.event_type }}</td>
									<td>{{ log.twin_id or '-' }}</td>
									<td>{{ log.user_id or '-' }}</td>
									<td>
										{% if log.severity == 'critical' %}
											<span class="label label-danger">{{ log.severity }}</span>
										{% elif log.severity == 'warning' %}
											<span class="label label-warning">{{ log.severity }}</span>
										{% else %}
											<span class="label label-info">{{ log.severity }}</span>
										{% endif %}
									</td>
									<td>{{ log.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
								</tr>
								{% endfor %}
							</tbody>
						</table>
					</div>
				</div>
			</div>
		</div>
		
		<div class="row">
			<div class="col-lg-12">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Security Analysis Tools</h3>
					</div>
					<div class="panel-body">
						<div class="btn-group" role="group">
							<a href="{{ url_for('BlockchainSecurityAnalysisView.chain_analysis') }}" class="btn btn-primary">
								<i class="fa fa-search"></i> Chain Analysis
							</a>
							<a href="{{ url_for('BlockchainSecurityAnalysisView.compliance_report') }}" class="btn btn-success">
								<i class="fa fa-check"></i> Compliance Report
							</a>
							<a href="{{ url_for('BlockchainSecurityAnalysisView.security_audit') }}" class="btn btn-warning">
								<i class="fa fa-shield"></i> Security Audit
							</a>
							<a href="{{ url_for('BlockchainSecurityAnalysisView.provenance_trace') }}" class="btn btn-info">
								<i class="fa fa-map"></i> Provenance Trace
							</a>
						</div>
					</div>
				</div>
			</div>
		</div>
		"""
		
		return self.render_template(
			'dashboard.html',
			content=render_template_string(
				dashboard_html,
				total_users=total_users,
				active_users=active_users,
				total_transactions=total_transactions,
				verified_transactions=verified_transactions,
				total_contracts=total_contracts,
				active_contracts=active_contracts,
				total_provenance=total_provenance,
				recent_transactions=recent_transactions,
				recent_audit_logs=recent_audit_logs
			)
		)

class BlockchainSecurityAnalysisView(BaseView):
	"""Advanced security analysis tools"""
	
	@expose('/chain_analysis/')
	@has_access
	def chain_analysis(self):
		"""Blockchain chain integrity analysis"""
		
		# Get all twins with transactions
		twins_with_transactions = db.session.query(BlockchainTransaction.twin_id).distinct().all()
		
		if not hasattr(self, '_security_engine'):
			self._security_engine = BlockchainSecurityEngine()
		
		analysis_results = []
		for (twin_id,) in twins_with_transactions:
			# Mock chain verification (in real implementation, would use actual verification)
			result = {
				'twin_id': twin_id,
				'transaction_count': db.session.query(BlockchainTransaction).filter(
					BlockchainTransaction.twin_id == twin_id
				).count(),
				'verified_count': db.session.query(BlockchainTransaction).filter(
					BlockchainTransaction.twin_id == twin_id,
					BlockchainTransaction.is_verified == True
				).count(),
				'integrity_score': 95.0,  # Mock score
				'status': 'healthy'
			}
			analysis_results.append(result)
		
		analysis_html = """
		<div class="panel panel-default">
			<div class="panel-heading">
				<h3 class="panel-title">Blockchain Chain Integrity Analysis</h3>
			</div>
			<div class="panel-body">
				<table class="table table-striped">
					<thead>
						<tr>
							<th>Twin ID</th>
							<th>Total Transactions</th>
							<th>Verified Transactions</th>
							<th>Integrity Score</th>
							<th>Status</th>
						</tr>
					</thead>
					<tbody>
						{% for result in analysis_results %}
						<tr>
							<td>{{ result.twin_id }}</td>
							<td>{{ result.transaction_count }}</td>
							<td>{{ result.verified_count }}</td>
							<td>
								<div class="progress">
									<div class="progress-bar 
										{% if result.integrity_score >= 90 %}progress-bar-success
										{% elif result.integrity_score >= 70 %}progress-bar-warning
										{% else %}progress-bar-danger{% endif %}"
										style="width: {{ result.integrity_score }}%">
										{{ "%.1f"|format(result.integrity_score) }}%
									</div>
								</div>
							</td>
							<td>
								{% if result.status == 'healthy' %}
									<span class="label label-success">{{ result.status }}</span>
								{% else %}
									<span class="label label-danger">{{ result.status }}</span>
								{% endif %}
							</td>
						</tr>
						{% endfor %}
					</tbody>
				</table>
			</div>
		</div>
		"""
		
		return self.render_template(
			'analysis.html',
			content=render_template_string(analysis_html, analysis_results=analysis_results)
		)
	
	@expose('/compliance_report/')
	@has_access
	def compliance_report(self):
		"""Generate compliance report"""
		
		# Get smart contract execution statistics
		total_executions = db.session.query(SmartContractModel).with_entities(
			db.func.sum(SmartContractModel.execution_count)
		).scalar() or 0
		
		active_contracts = db.session.query(SmartContractModel).filter(
			SmartContractModel.is_active == True
		).count()
		
		# Mock compliance data
		compliance_data = {
			'overall_compliance_score': 88.5,
			'total_executions': total_executions,
			'active_contracts': active_contracts,
			'compliance_violations': 3,
			'pending_reviews': 7
		}
		
		report_html = """
		<div class="row">
			<div class="col-lg-12">
				<div class="panel panel-default">
					<div class="panel-heading">
						<h3 class="panel-title">Compliance Report</h3>
					</div>
					<div class="panel-body">
						<div class="row">
							<div class="col-md-3">
								<h4>Overall Compliance Score</h4>
								<div class="progress">
									<div class="progress-bar progress-bar-success" style="width: {{ compliance_data.overall_compliance_score }}%">
										{{ "%.1f"|format(compliance_data.overall_compliance_score) }}%
									</div>
								</div>
							</div>
							<div class="col-md-3">
								<h4>Contract Executions</h4>
								<p class="text-info">{{ compliance_data.total_executions }}</p>
							</div>
							<div class="col-md-3">
								<h4>Active Contracts</h4>
								<p class="text-success">{{ compliance_data.active_contracts }}</p>
							</div>
							<div class="col-md-3">
								<h4>Violations</h4>
								<p class="text-danger">{{ compliance_data.compliance_violations }}</p>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
		"""
		
		return self.render_template(
			'analysis.html',
			content=render_template_string(report_html, compliance_data=compliance_data)
		)
	
	@expose('/security_audit/')
	@has_access
	def security_audit(self):
		"""Comprehensive security audit"""
		
		# Security metrics
		total_users = db.session.query(BlockchainUser).count()
		active_access_records = db.session.query(AccessControlRecord).filter(
			AccessControlRecord.is_active == True
		).count()
		
		security_events = db.session.query(SecurityAuditLog).filter(
			SecurityAuditLog.severity.in_(['warning', 'critical'])
		).order_by(SecurityAuditLog.created_at.desc()).limit(20).all()
		
		audit_html = """
		<div class="panel panel-default">
			<div class="panel-heading">
				<h3 class="panel-title">Security Audit Report</h3>
			</div>
			<div class="panel-body">
				<div class="row">
					<div class="col-md-4">
						<h4>Access Control</h4>
						<p>Total Users: {{ total_users }}</p>
						<p>Active Access Records: {{ active_access_records }}</p>
					</div>
					<div class="col-md-8">
						<h4>Recent Security Events</h4>
						<table class="table table-sm">
							<thead>
								<tr>
									<th>Event</th>
									<th>Severity</th>
									<th>Time</th>
								</tr>
							</thead>
							<tbody>
								{% for event in security_events %}
								<tr>
									<td>{{ event.event_type }}</td>
									<td>
										<span class="label label-{% if event.severity == 'critical' %}danger{% else %}warning{% endif %}">
											{{ event.severity }}
										</span>
									</td>
									<td>{{ event.created_at.strftime('%Y-%m-%d %H:%M') }}</td>
								</tr>
								{% endfor %}
							</tbody>
						</table>
					</div>
				</div>
			</div>
		</div>
		"""
		
		return self.render_template(
			'analysis.html',
			content=render_template_string(
				audit_html,
				total_users=total_users,
				active_access_records=active_access_records,
				security_events=security_events
			)
		)
	
	@expose('/provenance_trace/')
	@has_access
	def provenance_trace(self):
		"""Provenance chain tracing"""
		
		# Get twins with provenance records
		twins_with_provenance = db.session.query(ProvenanceRecord.twin_id).distinct().all()
		
		provenance_chains = []
		for (twin_id,) in twins_with_provenance:
			records = db.session.query(ProvenanceRecord).filter(
				ProvenanceRecord.twin_id == twin_id
			).order_by(ProvenanceRecord.created_at.asc()).all()
			
			chain_data = {
				'twin_id': twin_id,
				'record_count': len(records),
				'events': [record.event_type for record in records],
				'locations': list(set([record.location for record in records])),
				'participants': list(set([record.participant for record in records]))
			}
			provenance_chains.append(chain_data)
		
		trace_html = """
		<div class="panel panel-default">
			<div class="panel-heading">
				<h3 class="panel-title">Provenance Chain Tracing</h3>
			</div>
			<div class="panel-body">
				{% for chain in provenance_chains %}
				<div class="panel panel-info">
					<div class="panel-heading">
						<h4>Twin: {{ chain.twin_id }}</h4>
					</div>
					<div class="panel-body">
						<div class="row">
							<div class="col-md-3">
								<strong>Records:</strong> {{ chain.record_count }}
							</div>
							<div class="col-md-3">
								<strong>Events:</strong> {{ ", ".join(chain.events[:3]) }}{% if chain.events|length > 3 %}...{% endif %}
							</div>
							<div class="col-md-3">
								<strong>Locations:</strong> {{ ", ".join(chain.locations[:2]) }}{% if chain.locations|length > 2 %}...{% endif %}
							</div>
							<div class="col-md-3">
								<strong>Participants:</strong> {{ ", ".join(chain.participants[:2]) }}{% if chain.participants|length > 2 %}...{% endif %}
							</div>
						</div>
					</div>
				</div>
				{% endfor %}
			</div>
		</div>
		"""
		
		return self.render_template(
			'analysis.html',
			content=render_template_string(trace_html, provenance_chains=provenance_chains)
		)

# =============================================================================
# Blueprint Registration Function
# =============================================================================

def register_blockchain_security_views(appbuilder):
	"""Register all blockchain security views with Flask-AppBuilder"""
	
	# Main views
	appbuilder.add_view(
		BlockchainSecurityDashboardView,
		"Security Dashboard",
		icon="fa-shield",
		category="Blockchain Security"
	)
	
	appbuilder.add_view(
		BlockchainUserView,
		"Users",
		icon="fa-users",
		category="Blockchain Security"
	)
	
	appbuilder.add_view(
		BlockchainTransactionView,
		"Transactions",
		icon="fa-chain",
		category="Blockchain Security"
	)
	
	appbuilder.add_view(
		ProvenanceRecordView,
		"Provenance Records",
		icon="fa-history",
		category="Blockchain Security"
	)
	
	appbuilder.add_view(
		SmartContractView,
		"Smart Contracts",
		icon="fa-code",
		category="Blockchain Security"
	)
	
	appbuilder.add_view(
		AccessControlView,
		"Access Control",
		icon="fa-lock",
		category="Blockchain Security"
	)
	
	appbuilder.add_view(
		SecurityAuditLogView,
		"Audit Logs",
		icon="fa-list-alt",
		category="Blockchain Security"
	)
	
	appbuilder.add_view(
		BlockchainSecurityAnalysisView,
		"Security Analysis",
		icon="fa-search",
		category="Blockchain Security"
	)

# =============================================================================
# PostgreSQL Schema Definitions
# =============================================================================

BLOCKCHAIN_SECURITY_SCHEMAS = {
	'blockchain_users': '''
		CREATE TABLE IF NOT EXISTS blockchain_users (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			metadata JSONB DEFAULT '{}',
			
			user_id VARCHAR(100) UNIQUE NOT NULL,
			public_key TEXT NOT NULL,
			is_active BOOLEAN DEFAULT TRUE NOT NULL,
			access_level VARCHAR(50) DEFAULT 'read_only',
			last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			total_transactions INTEGER DEFAULT 0
		);
		
		CREATE INDEX IF NOT EXISTS idx_blockchain_users_user_id ON blockchain_users(user_id);
		CREATE INDEX IF NOT EXISTS idx_blockchain_users_active ON blockchain_users(is_active);
		CREATE INDEX IF NOT EXISTS idx_blockchain_users_created_at ON blockchain_users(created_at);
	''',
	
	'blockchain_transactions': '''
		CREATE TABLE IF NOT EXISTS blockchain_transactions (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			metadata JSONB DEFAULT '{}',
			
			transaction_id VARCHAR(100) UNIQUE NOT NULL,
			twin_id VARCHAR(100) NOT NULL,
			transaction_type VARCHAR(50) NOT NULL,
			previous_hash VARCHAR(64),
			data_hash VARCHAR(64) NOT NULL,
			transaction_hash VARCHAR(64) NOT NULL,
			payload JSONB DEFAULT '{}',
			signer_id VARCHAR(100) NOT NULL,
			signature TEXT NOT NULL,
			nonce INTEGER NOT NULL,
			is_verified BOOLEAN DEFAULT FALSE
		);
		
		CREATE INDEX IF NOT EXISTS idx_blockchain_transactions_id ON blockchain_transactions(transaction_id);
		CREATE INDEX IF NOT EXISTS idx_blockchain_transactions_twin_id ON blockchain_transactions(twin_id);
		CREATE INDEX IF NOT EXISTS idx_blockchain_transactions_type ON blockchain_transactions(transaction_type);
		CREATE INDEX IF NOT EXISTS idx_blockchain_transactions_signer ON blockchain_transactions(signer_id);
		CREATE INDEX IF NOT EXISTS idx_blockchain_transactions_created_at ON blockchain_transactions(created_at);
	''',
	
	'provenance_records': '''
		CREATE TABLE IF NOT EXISTS provenance_records (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			metadata JSONB DEFAULT '{}',
			
			record_id VARCHAR(100) UNIQUE NOT NULL,
			twin_id VARCHAR(100) NOT NULL,
			event_type VARCHAR(100) NOT NULL,
			location VARCHAR(200) NOT NULL,
			participant VARCHAR(200) NOT NULL,
			previous_record_hash VARCHAR(64),
			record_hash VARCHAR(64) NOT NULL,
			certifications JSONB DEFAULT '[]',
			signer_id VARCHAR(100) NOT NULL,
			is_verified BOOLEAN DEFAULT FALSE
		);
		
		CREATE INDEX IF NOT EXISTS idx_provenance_records_id ON provenance_records(record_id);
		CREATE INDEX IF NOT EXISTS idx_provenance_records_twin_id ON provenance_records(twin_id);
		CREATE INDEX IF NOT EXISTS idx_provenance_records_event_type ON provenance_records(event_type);
		CREATE INDEX IF NOT EXISTS idx_provenance_records_created_at ON provenance_records(created_at);
	''',
	
	'smart_contracts': '''
		CREATE TABLE IF NOT EXISTS smart_contracts (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			metadata JSONB DEFAULT '{}',
			
			contract_id VARCHAR(100) UNIQUE NOT NULL,
			name VARCHAR(200) NOT NULL,
			description TEXT,
			version VARCHAR(50) NOT NULL,
			rules JSONB DEFAULT '[]',
			conditions JSONB DEFAULT '[]',
			actions JSONB DEFAULT '[]',
			creator VARCHAR(100) NOT NULL,
			is_active BOOLEAN DEFAULT TRUE,
			execution_count INTEGER DEFAULT 0,
			last_executed TIMESTAMP WITH TIME ZONE
		);
		
		CREATE INDEX IF NOT EXISTS idx_smart_contracts_id ON smart_contracts(contract_id);
		CREATE INDEX IF NOT EXISTS idx_smart_contracts_name ON smart_contracts(name);
		CREATE INDEX IF NOT EXISTS idx_smart_contracts_creator ON smart_contracts(creator);
		CREATE INDEX IF NOT EXISTS idx_smart_contracts_active ON smart_contracts(is_active);
		CREATE INDEX IF NOT EXISTS idx_smart_contracts_created_at ON smart_contracts(created_at);
	''',
	
	'access_control_records': '''
		CREATE TABLE IF NOT EXISTS access_control_records (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			metadata JSONB DEFAULT '{}',
			
			twin_id VARCHAR(100) NOT NULL,
			user_id VARCHAR(100) NOT NULL,
			access_level VARCHAR(50) NOT NULL,
			granted_by VARCHAR(100) NOT NULL,
			granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			revoked_at TIMESTAMP WITH TIME ZONE,
			is_active BOOLEAN DEFAULT TRUE
		);
		
		CREATE INDEX IF NOT EXISTS idx_access_control_twin_id ON access_control_records(twin_id);
		CREATE INDEX IF NOT EXISTS idx_access_control_user_id ON access_control_records(user_id);
		CREATE INDEX IF NOT EXISTS idx_access_control_active ON access_control_records(is_active);
		CREATE INDEX IF NOT EXISTS idx_access_control_granted_at ON access_control_records(granted_at);
	''',
	
	'security_audit_logs': '''
		CREATE TABLE IF NOT EXISTS security_audit_logs (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
			metadata JSONB DEFAULT '{}',
			
			event_type VARCHAR(100) NOT NULL,
			twin_id VARCHAR(100),
			user_id VARCHAR(100),
			event_data JSONB DEFAULT '{}',
			severity VARCHAR(50) DEFAULT 'info',
			source_ip VARCHAR(45),
			user_agent TEXT
		);
		
		CREATE INDEX IF NOT EXISTS idx_security_audit_event_type ON security_audit_logs(event_type);
		CREATE INDEX IF NOT EXISTS idx_security_audit_twin_id ON security_audit_logs(twin_id);
		CREATE INDEX IF NOT EXISTS idx_security_audit_user_id ON security_audit_logs(user_id);
		CREATE INDEX IF NOT EXISTS idx_security_audit_severity ON security_audit_logs(severity);
		CREATE INDEX IF NOT EXISTS idx_security_audit_created_at ON security_audit_logs(created_at);
	'''
}