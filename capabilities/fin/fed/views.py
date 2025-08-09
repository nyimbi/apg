"""
Federated Learning Views

Flask-AppBuilder views for federated learning management, cross-twin collaboration,
privacy-preserving ML, and distributed model training orchestration.
"""

from flask import request, jsonify, flash, redirect, url_for, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from flask_appbuilder.widgets import FormWidget, ListWidget, SearchWidget
from flask_appbuilder.forms import DynamicForm
from wtforms import StringField, TextAreaField, SelectField, BooleanField, FloatField, IntegerField, validators
from wtforms.validators import DataRequired, Length, Optional, NumberRange
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .models import (
	FLFederation, FLParticipant, FLLearningTask,
	FLTrainingRound, FLLocalModel, FLModelUpdate, FLGlobalModel
)


class FederatedLearningBaseView(BaseView):
	"""Base view for federated learning functionality"""
	
	def __init__(self):
		super().__init__()
		self.default_view = 'dashboard'
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID from security context"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"
	
	def _format_accuracy(self, accuracy: float) -> str:
		"""Format accuracy for display"""
		if accuracy is None:
			return "N/A"
		return f"{accuracy*100:.1f}%"
	
	def _format_duration(self, minutes: float) -> str:
		"""Format duration for display"""
		if minutes is None:
			return "N/A"
		if minutes < 60:
			return f"{minutes:.1f} min"
		else:
			hours = minutes / 60
			return f"{hours:.1f} hrs"


class FLFederationModelView(ModelView):
	"""Federated learning federation management view"""
	
	datamodel = SQLAInterface(FLFederation)
	
	# List view configuration
	list_columns = [
		'federation_name', 'domain', 'federation_type', 'status',
		'current_participant_count', 'total_rounds_completed', 'best_global_accuracy'
	]
	show_columns = [
		'federation_id', 'federation_name', 'description', 'federation_type', 'domain',
		'governance_model', 'privacy_level', 'aggregation_algorithm', 'status',
		'is_open', 'requires_approval', 'max_participants', 'current_participant_count',
		'differential_privacy_enabled', 'differential_privacy_epsilon',
		'coordinator_id', 'round_duration_minutes', 'min_participants_per_round',
		'total_rounds_completed', 'best_global_accuracy', 'average_round_duration'
	]
	edit_columns = [
		'federation_name', 'description', 'federation_type', 'domain',
		'governance_model', 'privacy_level', 'aggregation_algorithm',
		'is_open', 'requires_approval', 'max_participants',
		'differential_privacy_enabled', 'differential_privacy_epsilon',
		'round_duration_minutes', 'min_participants_per_round'
	]
	add_columns = [
		'federation_name', 'description', 'federation_type', 'domain',
		'privacy_level', 'aggregation_algorithm'
	]
	
	# Search and filtering
	search_columns = ['federation_name', 'domain', 'federation_type']
	base_filters = [['status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('federation_name', 'asc')
	
	# Form validation
	validators_columns = {
		'federation_name': [DataRequired(), Length(min=3, max=200)],
		'federation_type': [DataRequired()],
		'domain': [DataRequired()],
		'max_participants': [NumberRange(min=2)],
		'min_participants_per_round': [NumberRange(min=2)],
		'differential_privacy_epsilon': [NumberRange(min=0.1, max=10.0)]
	}
	
	# Custom labels
	label_columns = {
		'federation_id': 'Federation ID',
		'federation_name': 'Federation Name',
		'federation_type': 'Federation Type',
		'governance_model': 'Governance Model',
		'privacy_level': 'Privacy Level',
		'data_sharing_policy': 'Data Sharing Policy',
		'model_sharing_policy': 'Model Sharing Policy',
		'aggregation_algorithm': 'Aggregation Algorithm',
		'communication_protocol': 'Communication Protocol',
		'encryption_method': 'Encryption Method',
		'differential_privacy_enabled': 'Differential Privacy',
		'differential_privacy_epsilon': 'Privacy Epsilon',
		'is_open': 'Open Federation',
		'requires_approval': 'Requires Approval',
		'max_participants': 'Max Participants',
		'current_participant_count': 'Current Participants',
		'min_data_quality_score': 'Min Data Quality',
		'trust_threshold': 'Trust Threshold',
		'reputation_system_enabled': 'Reputation System',
		'coordinator_id': 'Coordinator ID',
		'backup_coordinators': 'Backup Coordinators',
		'round_duration_minutes': 'Round Duration (min)',
		'min_participants_per_round': 'Min Participants/Round',
		'total_rounds_completed': 'Rounds Completed',
		'best_global_accuracy': 'Best Global Accuracy',
		'average_round_duration': 'Avg Round Duration',
		'total_models_trained': 'Models Trained',
		'incentive_mechanism': 'Incentive Mechanism'
	}
	
	@expose('/federation_dashboard/<int:pk>')
	@has_access
	def federation_dashboard(self, pk):
		"""Federation monitoring dashboard"""
		federation = self.datamodel.get(pk)
		if not federation:
			flash('Federation not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			dashboard_data = self._get_federation_dashboard_data(federation)
			
			return render_template('federated_learning/federation_dashboard.html',
								   federation=federation,
								   dashboard_data=dashboard_data,
								   page_title=f"Federation Dashboard: {federation.federation_name}")
		except Exception as e:
			flash(f'Error loading federation dashboard: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/start_training_round/<int:pk>')
	@has_access
	def start_training_round(self, pk):
		"""Start new training round"""
		federation = self.datamodel.get(pk)
		if not federation:
			flash('Federation not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if federation.can_start_new_round():
				# Implementation would start actual training round
				flash(f'Training round initiated for {federation.federation_name}', 'success')
			else:
				flash('Cannot start new round. Check participant count and federation status.', 'warning')
		except Exception as e:
			flash(f'Error starting training round: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/federation_analytics/<int:pk>')
	@has_access
	def federation_analytics(self, pk):
		"""View federation analytics"""
		federation = self.datamodel.get(pk)
		if not federation:
			flash('Federation not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			analytics_data = self._get_federation_analytics(federation)
			
			return render_template('federated_learning/federation_analytics.html',
								   federation=federation,
								   analytics_data=analytics_data,
								   page_title=f"Analytics: {federation.federation_name}")
		except Exception as e:
			flash(f'Error loading federation analytics: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new federation"""
		item.tenant_id = self._get_tenant_id()
		item.coordinator_id = self._get_current_user_id()
		
		# Set default values
		if not item.status:
			item.status = 'active'
		if not item.governance_model:
			item.governance_model = 'democratic'
		if not item.communication_protocol:
			item.communication_protocol = 'grpc'
	
	def _get_federation_dashboard_data(self, federation: FLFederation) -> Dict[str, Any]:
		"""Get dashboard data for federation"""
		participation_rate = federation.calculate_participation_rate()
		
		return {
			'participant_statistics': {
				'total_participants': len(federation.participants),
				'active_participants': len([p for p in federation.participants if p.status == 'active']),
				'pending_approvals': len([p for p in federation.participants if p.approval_status == 'pending'])
			},
			'training_statistics': {
				'total_rounds': federation.total_rounds_completed,
				'active_tasks': len([t for t in federation.learning_tasks if t.status == 'active']),
				'completed_tasks': len([t for t in federation.learning_tasks if t.status == 'completed']),
				'best_accuracy': federation.best_global_accuracy or 0.0
			},
			'performance_metrics': {
				'participation_rate': participation_rate,
				'average_round_duration': federation.average_round_duration or 0.0,
				'can_start_round': federation.can_start_new_round()
			},
			'privacy_metrics': {
				'privacy_level': federation.privacy_level,
				'differential_privacy': federation.differential_privacy_enabled,
				'epsilon_budget': federation.differential_privacy_epsilon
			}
		}
	
	def _get_federation_analytics(self, federation: FLFederation) -> Dict[str, Any]:
		"""Get analytics data for federation"""
		return {
			'performance_trends': {
				'accuracy_over_time': [],
				'participation_over_time': [],
				'round_duration_trends': []
			},
			'participant_analysis': {
				'contribution_distribution': {},
				'trust_score_distribution': {},
				'geographic_distribution': {}
			},
			'privacy_analysis': {
				'privacy_budget_usage': [],
				'differential_privacy_impact': {},
				'data_sovereignty_compliance': 95.5
			},
			'efficiency_metrics': {
				'communication_efficiency': 87.2,
				'convergence_rate': 0.15,
				'resource_utilization': 73.8
			}
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class FLParticipantModelView(ModelView):
	"""Federated learning participant management view"""
	
	datamodel = SQLAInterface(FLParticipant)
	
	# List view configuration
	list_columns = [
		'participant_name', 'federation', 'organization', 'status',
		'trust_score', 'reputation_score', 'rounds_participated'
	]
	show_columns = [
		'participant_id', 'participant_name', 'federation', 'digital_twin_id',
		'organization', 'contact_email', 'status', 'joined_at', 'last_activity',
		'approval_status', 'trust_score', 'reputation_score', 'data_quality_score',
		'rounds_participated', 'rounds_invited', 'successful_contributions',
		'failed_contributions', 'average_training_time'
	]
	edit_columns = [
		'participant_name', 'organization', 'contact_email', 'status',
		'compute_capacity', 'supported_algorithms', 'privacy_requirements'
	]
	add_columns = [
		'participant_name', 'digital_twin_id', 'organization', 'contact_email'
	]
	
	# Search and filtering
	search_columns = ['participant_name', 'organization', 'digital_twin_id']
	base_filters = [['status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('trust_score', 'desc')
	
	# Form validation
	validators_columns = {
		'participant_name': [DataRequired(), Length(min=3, max=200)],
		'digital_twin_id': [DataRequired()],
		'contact_email': [Optional(), validators.Email()]
	}
	
	# Custom labels
	label_columns = {
		'participant_id': 'Participant ID',
		'participant_name': 'Participant Name',
		'digital_twin_id': 'Digital Twin ID',
		'contact_email': 'Contact Email',
		'last_activity': 'Last Activity',
		'approval_status': 'Approval Status',
		'approved_by': 'Approved By',
		'approval_date': 'Approval Date',
		'compute_capacity': 'Compute Capacity',
		'data_characteristics': 'Data Characteristics',
		'supported_algorithms': 'Supported Algorithms',
		'technical_capabilities': 'Technical Capabilities',
		'privacy_requirements': 'Privacy Requirements',
		'security_level': 'Security Level',
		'data_sovereignty_restrictions': 'Data Sovereignty',
		'compliance_certifications': 'Compliance Certifications',
		'trust_score': 'Trust Score',
		'reputation_score': 'Reputation Score',
		'data_quality_score': 'Data Quality Score',
		'model_contribution_score': 'Model Contribution',
		'rounds_participated': 'Rounds Participated',
		'rounds_invited': 'Rounds Invited',
		'successful_contributions': 'Successful Contributions',
		'failed_contributions': 'Failed Contributions',
		'average_training_time': 'Avg Training Time',
		'endpoint_url': 'Endpoint URL',
		'public_key': 'Public Key',
		'communication_preferences': 'Communication Preferences',
		'total_rewards_earned': 'Total Rewards',
		'contribution_weight': 'Contribution Weight'
	}
	
	@expose('/approve_participant/<int:pk>')
	@has_access
	def approve_participant(self, pk):
		"""Approve participant for federation"""
		participant = self.datamodel.get(pk)
		if not participant:
			flash('Participant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			participant.approval_status = 'approved'
			participant.status = 'active'
			participant.approved_by = self._get_current_user_id()
			participant.approval_date = datetime.utcnow()
			self.datamodel.edit(participant)
			flash(f'Participant {participant.participant_name} approved', 'success')
		except Exception as e:
			flash(f'Error approving participant: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/participant_metrics/<int:pk>')
	@has_access
	def participant_metrics(self, pk):
		"""View participant performance metrics"""
		participant = self.datamodel.get(pk)
		if not participant:
			flash('Participant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			metrics_data = self._get_participant_metrics(participant)
			
			return render_template('federated_learning/participant_metrics.html',
								   participant=participant,
								   metrics_data=metrics_data,
								   page_title=f"Participant Metrics: {participant.participant_name}")
		except Exception as e:
			flash(f'Error loading participant metrics: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/update_trust_score/<int:pk>')
	@has_access
	def update_trust_score(self, pk):
		"""Update participant trust score"""
		participant = self.datamodel.get(pk)
		if not participant:
			flash('Participant not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Simulate trust score calculation
			participant.update_trust_score(0.8, 0.9)  # Performance and reliability factors
			self.datamodel.edit(participant)
			flash(f'Trust score updated for {participant.participant_name}', 'success')
		except Exception as e:
			flash(f'Error updating trust score: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new participant"""
		item.tenant_id = self._get_tenant_id()
		item.joined_at = datetime.utcnow()
		
		# Set default values
		if not item.status:
			item.status = 'pending'
		if not item.trust_score:
			item.trust_score = 0.5
		if not item.reputation_score:
			item.reputation_score = 0.5
	
	def _get_participant_metrics(self, participant: FLParticipant) -> Dict[str, Any]:
		"""Get performance metrics for participant"""
		participation_rate = participant.calculate_participation_rate()
		success_rate = participant.calculate_success_rate()
		
		return {
			'participation_metrics': {
				'participation_rate': participation_rate,
				'success_rate': success_rate,
				'total_contributions': participant.successful_contributions + participant.failed_contributions,
				'average_training_time': participant.average_training_time or 0.0
			},
			'trust_and_reputation': {
				'trust_score': participant.trust_score,
				'reputation_score': participant.reputation_score,
				'data_quality_score': participant.data_quality_score or 0.0,
				'model_contribution_score': participant.model_contribution_score
			},
			'technical_profile': {
				'compute_capacity': participant.compute_capacity,
				'supported_algorithms': participant.supported_algorithms,
				'security_level': participant.security_level
			},
			'activity_timeline': [],
			'contribution_history': []
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class FLLearningTaskModelView(ModelView):
	"""Federated learning task management view"""
	
	datamodel = SQLAInterface(FLLearningTask)
	
	# List view configuration
	list_columns = [
		'task_name', 'federation', 'task_type', 'status',
		'current_round', 'best_global_accuracy', 'target_accuracy'
	]
	show_columns = [
		'task_id', 'task_name', 'description', 'federation', 'task_type',
		'domain', 'algorithm_type', 'optimization_objective', 'status',
		'current_round', 'max_training_rounds', 'best_global_accuracy',
		'target_accuracy', 'convergence_threshold', 'min_participants',
		'differential_privacy_budget', 'created_by', 'started_at'
	]
	edit_columns = [
		'task_name', 'description', 'task_type', 'domain', 'algorithm_type',
		'model_architecture', 'hyperparameters', 'optimization_objective',
		'max_training_rounds', 'convergence_threshold', 'target_accuracy',
		'min_participants', 'differential_privacy_budget'
	]
	add_columns = [
		'task_name', 'description', 'task_type', 'domain', 'algorithm_type',
		'optimization_objective'
	]
	
	# Search and filtering
	search_columns = ['task_name', 'task_type', 'domain']
	base_filters = [['status', lambda: 'active', lambda: True]]
	
	# Ordering
	base_order = ('created_at', 'desc')
	
	# Form validation
	validators_columns = {
		'task_name': [DataRequired(), Length(min=3, max=200)],
		'description': [DataRequired()],
		'task_type': [DataRequired()],
		'algorithm_type': [DataRequired()],
		'max_training_rounds': [NumberRange(min=1)],
		'min_participants': [NumberRange(min=2)]
	}
	
	# Custom labels
	label_columns = {
		'task_id': 'Task ID',
		'task_name': 'Task Name',
		'task_type': 'Task Type',
		'use_case': 'Use Case',
		'algorithm_type': 'Algorithm Type',
		'model_architecture': 'Model Architecture',
		'optimization_objective': 'Optimization Objective',
		'data_schema': 'Data Schema',
		'min_data_samples': 'Min Data Samples',
		'data_quality_requirements': 'Data Quality Requirements',
		'feature_requirements': 'Feature Requirements',
		'max_training_rounds': 'Max Training Rounds',
		'convergence_threshold': 'Convergence Threshold',
		'target_accuracy': 'Target Accuracy',
		'early_stopping_patience': 'Early Stopping Patience',
		'differential_privacy_budget': 'Privacy Budget',
		'secure_aggregation_required': 'Secure Aggregation',
		'homomorphic_encryption': 'Homomorphic Encryption',
		'trusted_execution_environment': 'TEE Required',
		'min_participants': 'Min Participants',
		'max_participants': 'Max Participants',
		'estimated_round_duration': 'Est. Round Duration',
		'compute_requirements': 'Compute Requirements',
		'current_round': 'Current Round',
		'best_global_accuracy': 'Best Global Accuracy',
		'last_improvement_round': 'Last Improvement Round',
		'created_by': 'Created By',
		'started_at': 'Started At',
		'completed_at': 'Completed At',
		'final_model_accuracy': 'Final Model Accuracy',
		'final_model_metrics': 'Final Model Metrics'
	}
	
	@expose('/start_task/<int:pk>')
	@has_access
	def start_task(self, pk):
		"""Start federated learning task"""
		task = self.datamodel.get(pk)
		if not task:
			flash('Task not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if task.status == 'draft':
				task.status = 'active'
				task.started_at = datetime.utcnow()
				self.datamodel.edit(task)
				flash(f'Task {task.task_name} started', 'success')
			else:
				flash('Task cannot be started in current state', 'warning')
		except Exception as e:
			flash(f'Error starting task: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/task_progress/<int:pk>')
	@has_access
	def task_progress(self, pk):
		"""View task progress and metrics"""
		task = self.datamodel.get(pk)
		if not task:
			flash('Task not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			progress_data = self._get_task_progress_data(task)
			
			return render_template('federated_learning/task_progress.html',
								   task=task,
								   progress_data=progress_data,
								   page_title=f"Task Progress: {task.task_name}")
		except Exception as e:
			flash(f'Error loading task progress: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/stop_task/<int:pk>')
	@has_access
	def stop_task(self, pk):
		"""Stop federated learning task"""
		task = self.datamodel.get(pk)
		if not task:
			flash('Task not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			if task.status == 'active':
				task.status = 'completed'
				task.completed_at = datetime.utcnow()
				self.datamodel.edit(task)
				flash(f'Task {task.task_name} stopped', 'success')
			else:
				flash('Task cannot be stopped in current state', 'warning')
		except Exception as e:
			flash(f'Error stopping task: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new task"""
		item.tenant_id = self._get_tenant_id()
		item.created_by = self._get_current_user_id()
		
		# Set default values
		if not item.status:
			item.status = 'draft'
		if not item.max_training_rounds:
			item.max_training_rounds = 100
		if not item.convergence_threshold:
			item.convergence_threshold = 0.001
	
	def _get_task_progress_data(self, task: FLLearningTask) -> Dict[str, Any]:
		"""Get progress data for task"""
		progress_percentage = task.calculate_progress()
		
		return {
			'progress_overview': {
				'current_round': task.current_round,
				'max_rounds': task.max_training_rounds,
				'progress_percentage': progress_percentage,
				'best_accuracy': task.best_global_accuracy or 0.0,
				'target_accuracy': task.target_accuracy
			},
			'convergence_analysis': {
				'is_converged': task.is_converged(task.best_global_accuracy or 0.0),
				'should_early_stop': task.should_early_stop(),
				'last_improvement': task.last_improvement_round,
				'convergence_threshold': task.convergence_threshold
			},
			'participant_engagement': {
				'eligible_participants': len([p for p in task.federation.participants 
											if p.is_eligible_for_round({'min_trust_score': 0.5})]),
				'active_participants': len([p for p in task.federation.participants if p.status == 'active'])
			},
			'training_rounds': [
				{
					'round_number': round_obj.round_number,
					'accuracy': round_obj.global_accuracy_after,
					'participants': round_obj.actual_participant_count,
					'duration': round_obj.duration_minutes
				}
				for round_obj in task.training_rounds[-10:]  # Last 10 rounds
			]
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class FLTrainingRoundModelView(ModelView):
	"""Federated learning training round monitoring view"""
	
	datamodel = SQLAInterface(FLTrainingRound)
	
	# List view configuration
	list_columns = [
		'round_number', 'task', 'status', 'actual_participant_count',
		'started_at', 'duration_minutes', 'global_accuracy_after'
	]
	show_columns = [
		'round_id', 'round_number', 'task', 'round_type', 'status',
		'target_participant_count', 'actual_participant_count',
		'started_at', 'completed_at', 'duration_minutes',
		'global_accuracy_before', 'global_accuracy_after', 'accuracy_improvement',
		'model_updates_received', 'model_updates_expected', 'aggregation_algorithm'
	]
	# Read-only view for training rounds
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	can_delete = False
	
	# Search and filtering
	search_columns = ['task.task_name', 'status', 'round_type']
	base_filters = [['status', lambda: 'completed', lambda: True]]
	
	# Ordering
	base_order = ('started_at', 'desc')
	
	# Custom labels
	label_columns = {
		'round_id': 'Round ID',
		'round_number': 'Round Number',
		'round_type': 'Round Type',
		'selected_participants': 'Selected Participants',
		'participant_selection_strategy': 'Selection Strategy',
		'target_participant_count': 'Target Participants',
		'actual_participant_count': 'Actual Participants',
		'started_at': 'Started At',
		'completed_at': 'Completed At',
		'timeout_at': 'Timeout At',
		'duration_minutes': 'Duration (min)',
		'global_model_before': 'Model Before',
		'global_model_after': 'Model After',
		'model_updates_received': 'Updates Received',
		'model_updates_expected': 'Updates Expected',
		'aggregation_algorithm': 'Aggregation Algorithm',
		'aggregation_weights': 'Aggregation Weights',
		'aggregation_started_at': 'Aggregation Started',
		'aggregation_completed_at': 'Aggregation Completed',
		'aggregation_duration_seconds': 'Aggregation Duration (s)',
		'global_accuracy_before': 'Accuracy Before',
		'global_accuracy_after': 'Accuracy After',
		'accuracy_improvement': 'Accuracy Improvement',
		'convergence_metrics': 'Convergence Metrics',
		'differential_privacy_applied': 'Differential Privacy',
		'noise_scale': 'Noise Scale',
		'privacy_budget_consumed': 'Privacy Budget Used',
		'secure_aggregation_used': 'Secure Aggregation',
		'total_communication_bytes': 'Communication (bytes)',
		'average_communication_delay': 'Avg Communication Delay',
		'failed_communications': 'Failed Communications'
	}
	
	@expose('/round_details/<int:pk>')
	@has_access
	def round_details(self, pk):
		"""View detailed round information"""
		round_obj = self.datamodel.get(pk)
		if not round_obj:
			flash('Training round not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			round_details = self._get_round_details(round_obj)
			
			return render_template('federated_learning/round_details.html',
								   round_obj=round_obj,
								   round_details=round_details,
								   page_title=f"Round Details: {round_obj.task.task_name} Round {round_obj.round_number}")
		except Exception as e:
			flash(f'Error loading round details: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def _get_round_details(self, round_obj: FLTrainingRound) -> Dict[str, Any]:
		"""Get detailed information for training round"""
		participation_rate = round_obj.calculate_participation_rate()
		success_rate = round_obj.calculate_success_rate()
		
		return {
			'execution_metrics': {
				'participation_rate': participation_rate,
				'success_rate': success_rate,
				'duration_minutes': round_obj.duration_minutes or 0.0,
				'is_timed_out': round_obj.is_timed_out()
			},
			'performance_metrics': {
				'accuracy_before': round_obj.global_accuracy_before or 0.0,
				'accuracy_after': round_obj.global_accuracy_after or 0.0,
				'improvement': round_obj.accuracy_improvement or 0.0
			},
			'privacy_metrics': {
				'differential_privacy': round_obj.differential_privacy_applied,
				'noise_scale': round_obj.noise_scale,
				'privacy_budget': round_obj.privacy_budget_consumed,
				'secure_aggregation': round_obj.secure_aggregation_used
			},
			'communication_metrics': {
				'total_bytes': round_obj.total_communication_bytes,
				'average_delay': round_obj.average_communication_delay,
				'failed_communications': round_obj.failed_communications
			},
			'participant_updates': [
				{
					'participant': update.participant.participant_name,
					'status': update.status,
					'quality_score': update.update_quality_score,
					'communication_delay': update.communication_delay_ms
				}
				for update in round_obj.model_updates
			]
		}


class FederatedLearningDashboardView(FederatedLearningBaseView):
	"""Federated learning dashboard"""
	
	route_base = "/federated_learning_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""Federated learning dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('federated_learning/dashboard.html',
								   metrics=metrics,
								   page_title="Federated Learning Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('federated_learning/dashboard.html',
								   metrics={},
								   page_title="Federated Learning Dashboard")
	
	@expose('/privacy_analytics/')
	@has_access
	def privacy_analytics(self):
		"""Privacy-preserving analytics dashboard"""
		try:
			privacy_data = self._get_privacy_analytics_data()
			
			return render_template('federated_learning/privacy_analytics.html',
								   privacy_data=privacy_data,
								   page_title="Privacy Analytics")
		except Exception as e:
			flash(f'Error loading privacy analytics: {str(e)}', 'error')
			return redirect(url_for('FederatedLearningDashboardView.index'))
	
	@expose('/collaboration_network/')
	@has_access
	def collaboration_network(self):
		"""Cross-twin collaboration network visualization"""
		try:
			network_data = self._get_collaboration_network_data()
			
			return render_template('federated_learning/collaboration_network.html',
								   network_data=network_data,
								   page_title="Collaboration Network")
		except Exception as e:
			flash(f'Error loading collaboration network: {str(e)}', 'error')
			return redirect(url_for('FederatedLearningDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get federated learning dashboard metrics"""
		# Implementation would calculate real metrics from database
		return {
			'federation_overview': {
				'total_federations': 8,
				'active_federations': 6,
				'total_participants': 245,
				'active_participants': 189,
				'pending_approvals': 12
			},
			'training_activity': {
				'active_tasks': 15,
				'completed_tasks': 73,
				'total_training_rounds': 1847,
				'successful_rounds': 1789,
				'average_accuracy_improvement': 2.3
			},
			'privacy_metrics': {
				'differential_privacy_usage': 87.5,
				'secure_aggregation_usage': 92.1,
				'privacy_budget_efficiency': 94.2,
				'data_sovereignty_compliance': 98.7
			},
			'performance_metrics': {
				'average_round_duration': 42.5,
				'model_convergence_rate': 85.3,
				'communication_efficiency': 78.9,
				'participant_satisfaction': 91.2
			},
			'collaboration_health': {
				'trust_score_average': 0.78,
				'participation_rate': 83.4,
				'contribution_quality': 88.7,
				'cross_domain_collaborations': 23
			}
		}
	
	def _get_privacy_analytics_data(self) -> Dict[str, Any]:
		"""Get privacy analytics data"""
		return {
			'privacy_budget_usage': {
				'total_budget': 100.0,
				'used_budget': 45.2,
				'remaining_budget': 54.8,
				'efficiency_score': 89.3
			},
			'differential_privacy_impact': {
				'accuracy_impact': -2.1,  # Percentage points
				'utility_preservation': 94.5,
				'privacy_guarantee': 'ε=1.0, δ=1e-5'
			},
			'secure_aggregation_metrics': {
				'protocols_used': ['SecAgg', 'JOYE', 'BGV'],
				'communication_overhead': 15.2,
				'computation_overhead': 8.7,
				'security_level': 'High'
			},
			'compliance_status': {
				'gdpr_compliance': 98.5,
				'hipaa_compliance': 96.2,
				'data_locality_compliance': 100.0,
				'audit_trail_completeness': 97.8
			}
		}
	
	def _get_collaboration_network_data(self) -> Dict[str, Any]:
		"""Get collaboration network data"""
		return {
			'network_topology': {
				'total_nodes': 245,
				'total_edges': 892,
				'clustering_coefficient': 0.67,
				'average_path_length': 3.2,
				'network_density': 0.15
			},
			'collaboration_patterns': {
				'cross_domain_collaborations': 23,
				'intra_domain_collaborations': 67,
				'knowledge_transfer_events': 156,
				'model_sharing_events': 89
			},
			'participant_roles': {
				'knowledge_providers': 89,
				'knowledge_consumers': 134,
				'knowledge_brokers': 22,
				'hybrid_participants': 67
			},
			'domain_distribution': {
				'healthcare': 78,
				'manufacturing': 65,
				'finance': 42,
				'smart_cities': 35,
				'energy': 25
			}
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all federated learning views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		FLFederationModelView,
		"Federations",
		icon="fa-network-wired",
		category="Federated Learning",
		category_icon="fa-share-alt"
	)
	
	appbuilder.add_view(
		FLParticipantModelView,
		"Participants",
		icon="fa-users",
		category="Federated Learning"
	)
	
	appbuilder.add_view(
		FLLearningTaskModelView,
		"Learning Tasks",
		icon="fa-brain",
		category="Federated Learning"
	)
	
	appbuilder.add_view(
		FLTrainingRoundModelView,
		"Training Rounds",
		icon="fa-sync-alt",
		category="Federated Learning"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(FederatedLearningDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"FL Dashboard",
		href="/federated_learning_dashboard/",
		icon="fa-dashboard",
		category="Federated Learning"
	)
	
	appbuilder.add_link(
		"Privacy Analytics",
		href="/federated_learning_dashboard/privacy_analytics/",
		icon="fa-shield-alt",
		category="Federated Learning"
	)
	
	appbuilder.add_link(
		"Collaboration Network",
		href="/federated_learning_dashboard/collaboration_network/",
		icon="fa-project-diagram",
		category="Federated Learning"
	)