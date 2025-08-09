"""
APG Facial Recognition - Collaborative Verification Engine

Revolutionary multi-person verification system with supervisor approval workflows,
crowd-sourced validation, and consensus-based identity confirmation.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid_extensions import uuid7str
from enum import Enum

try:
	from sklearn.cluster import KMeans
	from sklearn.metrics import silhouette_score
	import numpy as np
except ImportError as e:
	print(f"Optional ML dependencies not available: {e}")

class CollaborativeVerificationType(Enum):
	SUPERVISOR_APPROVAL = "supervisor_approval"
	PEER_VERIFICATION = "peer_verification"
	CROWD_CONSENSUS = "crowd_consensus"
	EXPERT_REVIEW = "expert_review"
	AUTOMATED_CONSENSUS = "automated_consensus"

class VerificationParticipantRole(Enum):
	PRIMARY_VERIFIER = "primary_verifier"
	SUPERVISOR = "supervisor"
	PEER_REVIEWER = "peer_reviewer"
	EXPERT = "expert"
	AUTOMATED_SYSTEM = "automated_system"

class CollaborationWorkflowStatus(Enum):
	INITIATED = "initiated"
	IN_PROGRESS = "in_progress"
	PENDING_APPROVAL = "pending_approval"
	APPROVED = "approved"
	REJECTED = "rejected"
	ESCALATED = "escalated"
	COMPLETED = "completed"

class CollaborativeVerificationEngine:
	"""Multi-person collaborative verification system"""
	
	def __init__(self, tenant_id: str):
		"""Initialize collaborative verification engine"""
		assert tenant_id, "Tenant ID cannot be empty"
		
		self.tenant_id = tenant_id
		self.collaboration_enabled = True
		self.supervisor_approval_threshold = 0.7
		self.consensus_threshold = 0.8
		self.max_reviewers = 5
		
		# Active collaboration sessions
		self.active_collaborations = {}
		self.collaboration_history = {}
		self.participant_ratings = {}
		self.workflow_templates = {}
		
		self._initialize_workflows()
		self._log_engine_initialized()
	
	def _initialize_workflows(self) -> None:
		"""Initialize collaboration workflow templates"""
		try:
			# Default workflow templates
			self.workflow_templates = {
				'high_security': {
					'name': 'High Security Verification',
					'required_approvals': 2,
					'required_roles': [VerificationParticipantRole.SUPERVISOR, VerificationParticipantRole.EXPERT],
					'timeout_minutes': 30,
					'escalation_threshold': 0.5,
					'consensus_required': True
				},
				'standard_approval': {
					'name': 'Standard Supervisor Approval',
					'required_approvals': 1,
					'required_roles': [VerificationParticipantRole.SUPERVISOR],
					'timeout_minutes': 15,
					'escalation_threshold': 0.7,
					'consensus_required': False
				},
				'peer_review': {
					'name': 'Peer Review Verification',
					'required_approvals': 3,
					'required_roles': [VerificationParticipantRole.PEER_REVIEWER],
					'timeout_minutes': 20,
					'escalation_threshold': 0.6,
					'consensus_required': True
				},
				'expert_validation': {
					'name': 'Expert Validation',
					'required_approvals': 1,
					'required_roles': [VerificationParticipantRole.EXPERT],
					'timeout_minutes': 45,
					'escalation_threshold': 0.3,
					'consensus_required': False
				},
				'crowd_consensus': {
					'name': 'Crowd-Sourced Consensus',
					'required_approvals': 5,
					'required_roles': [VerificationParticipantRole.PEER_REVIEWER],
					'timeout_minutes': 60,
					'escalation_threshold': 0.8,
					'consensus_required': True
				}
			}
			
		except Exception as e:
			print(f"Failed to initialize workflows: {e}")
	
	def _log_engine_initialized(self) -> None:
		"""Log engine initialization"""
		print(f"Collaborative Verification Engine initialized for tenant {self.tenant_id}")
	
	def _log_collaboration_operation(self, operation: str, collaboration_id: str | None = None, result: str | None = None) -> None:
		"""Log collaboration operations"""
		collab_info = f" (Collaboration: {collaboration_id})" if collaboration_id else ""
		result_info = f" [{result}]" if result else ""
		print(f"Collaborative Verification {operation}{collab_info}{result_info}")
	
	async def initiate_collaborative_verification(self, verification_request: Dict[str, Any]) -> Dict[str, Any]:
		"""Initiate collaborative verification process"""
		try:
			assert verification_request, "Verification request cannot be empty"
			assert verification_request.get('user_id'), "User ID is required"
			assert verification_request.get('verification_type'), "Verification type is required"
			
			collaboration_id = uuid7str()
			start_time = datetime.now(timezone.utc)
			
			# Determine workflow based on verification context
			workflow_type = self._determine_workflow_type(verification_request)
			workflow_template = self.workflow_templates.get(workflow_type, self.workflow_templates['standard_approval'])
			
			# Create collaboration session
			collaboration_session = {
				'collaboration_id': collaboration_id,
				'tenant_id': self.tenant_id,
				'user_id': verification_request['user_id'],
				'verification_type': verification_request['verification_type'],
				'workflow_type': workflow_type,
				'workflow_template': workflow_template,
				'status': CollaborationWorkflowStatus.INITIATED,
				'initiated_at': start_time.isoformat(),
				'timeout_at': (start_time + timedelta(minutes=workflow_template['timeout_minutes'])).isoformat(),
				'primary_verification_result': verification_request.get('primary_result', {}),
				'context': verification_request.get('context', {}),
				'participants': [],
				'approvals': [],
				'rejections': [],
				'comments': [],
				'consensus_score': 0.0,
				'final_decision': None,
				'metadata': verification_request.get('metadata', {})
			}
			
			# Store active collaboration
			self.active_collaborations[collaboration_id] = collaboration_session
			
			# Identify and invite participants
			participants = await self._identify_participants(collaboration_session)
			collaboration_session['participants'] = participants
			
			# Send invitations to participants
			invitations_sent = await self._send_participant_invitations(collaboration_session)
			
			# Update status
			collaboration_session['status'] = CollaborationWorkflowStatus.IN_PROGRESS
			collaboration_session['invitations_sent'] = invitations_sent
			
			self._log_collaboration_operation(
				"INITIATE",
				collaboration_id,
				f"Workflow: {workflow_type}, Participants: {len(participants)}"
			)
			
			return {
				'success': True,
				'collaboration_id': collaboration_id,
				'workflow_type': workflow_type,
				'participants_invited': len(participants),
				'timeout_at': collaboration_session['timeout_at'],
				'required_approvals': workflow_template['required_approvals'],
				'status': collaboration_session['status'].value
			}
			
		except Exception as e:
			print(f"Failed to initiate collaborative verification: {e}")
			return {'success': False, 'error': str(e)}
	
	def _determine_workflow_type(self, verification_request: Dict[str, Any]) -> str:
		"""Determine appropriate workflow type based on verification context"""
		try:
			context = verification_request.get('context', {})
			primary_result = verification_request.get('primary_result', {})
			
			# Risk-based workflow selection
			confidence_score = primary_result.get('confidence_score', 1.0)
			risk_score = context.get('risk_score', 0.0)
			business_sensitivity = context.get('business_sensitivity', 'low')
			user_role = context.get('user_role', 'standard')
			
			# High-security scenarios
			if (business_sensitivity == 'high' or 
				user_role in ['admin', 'executive'] or 
				risk_score > 0.7):
				return 'high_security'
			
			# Expert validation for complex cases
			elif confidence_score < 0.6 or context.get('anomaly_detected', False):
				return 'expert_validation'
			
			# Peer review for medium-risk scenarios
			elif confidence_score < 0.8 or risk_score > 0.4:
				return 'peer_review'
			
			# Crowd consensus for identity disputes
			elif context.get('identity_disputed', False):
				return 'crowd_consensus'
			
			# Standard approval for normal cases
			else:
				return 'standard_approval'
				
		except Exception as e:
			print(f"Failed to determine workflow type: {e}")
			return 'standard_approval'
	
	async def _identify_participants(self, collaboration_session: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Identify appropriate participants for collaboration"""
		try:
			workflow_template = collaboration_session['workflow_template']
			required_roles = workflow_template['required_roles']
			required_approvals = workflow_template['required_approvals']
			
			participants = []
			
			# For each required role, find qualified participants
			for role in required_roles:
				role_participants = await self._find_participants_by_role(role, collaboration_session)
				participants.extend(role_participants)
			
			# Ensure we have enough participants
			while len(participants) < required_approvals:
				backup_participant = await self._find_backup_participant(collaboration_session, participants)
				if backup_participant:
					participants.append(backup_participant)
				else:
					break
			
			# Limit to max reviewers
			if len(participants) > self.max_reviewers:
				participants = self._prioritize_participants(participants, collaboration_session)[:self.max_reviewers]
			
			return participants
			
		except Exception as e:
			print(f"Failed to identify participants: {e}")
			return []
	
	async def _find_participants_by_role(self, role: VerificationParticipantRole, collaboration_session: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Find participants with specific role"""
		try:
			participants = []
			
			# This would integrate with APG user management system
			# For now, we'll simulate participant discovery
			
			if role == VerificationParticipantRole.SUPERVISOR:
				participants = [
					{
						'participant_id': 'supervisor_001',
						'role': role.value,
						'name': 'Security Supervisor',
						'qualification_score': 0.9,
						'availability_score': 0.8,
						'expertise_areas': ['identity_verification', 'security_protocols'],
						'average_response_time_minutes': 5
					}
				]
			elif role == VerificationParticipantRole.EXPERT:
				participants = [
					{
						'participant_id': 'expert_001',
						'role': role.value,
						'name': 'Biometric Expert',
						'qualification_score': 0.95,
						'availability_score': 0.6,
						'expertise_areas': ['facial_recognition', 'liveness_detection', 'spoofing_detection'],
						'average_response_time_minutes': 15
					}
				]
			elif role == VerificationParticipantRole.PEER_REVIEWER:
				participants = [
					{
						'participant_id': f'peer_{i:03d}',
						'role': role.value,
						'name': f'Peer Reviewer {i+1}',
						'qualification_score': 0.7 + (i * 0.05),
						'availability_score': 0.8 - (i * 0.1),
						'expertise_areas': ['general_verification'],
						'average_response_time_minutes': 10 + (i * 2)
					}
					for i in range(5)
				]
			
			return participants
			
		except Exception as e:
			print(f"Failed to find participants by role: {e}")
			return []
	
	async def _find_backup_participant(self, collaboration_session: Dict[str, Any], existing_participants: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
		"""Find backup participant if needed"""
		try:
			existing_ids = {p['participant_id'] for p in existing_participants}
			
			# Find available backup participants
			backup_participant = {
				'participant_id': f'backup_{len(existing_participants):03d}',
				'role': VerificationParticipantRole.PEER_REVIEWER.value,
				'name': f'Backup Reviewer {len(existing_participants)}',
				'qualification_score': 0.6,
				'availability_score': 0.9,
				'expertise_areas': ['general_verification'],
				'average_response_time_minutes': 20
			}
			
			if backup_participant['participant_id'] not in existing_ids:
				return backup_participant
				
			return None
			
		except Exception as e:
			print(f"Failed to find backup participant: {e}")
			return None
	
	def _prioritize_participants(self, participants: List[Dict[str, Any]], collaboration_session: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Prioritize participants based on qualification and availability"""
		try:
			# Calculate priority score for each participant
			for participant in participants:
				qualification = participant.get('qualification_score', 0.5)
				availability = participant.get('availability_score', 0.5)
				response_time = participant.get('average_response_time_minutes', 30)
				
				# Lower response time is better
				time_score = max(0.1, 1.0 - (response_time / 60))
				
				participant['priority_score'] = (
					qualification * 0.4 + 
					availability * 0.3 + 
					time_score * 0.3
				)
			
			# Sort by priority score (descending)
			participants.sort(key=lambda p: p['priority_score'], reverse=True)
			
			return participants
			
		except Exception as e:
			print(f"Failed to prioritize participants: {e}")
			return participants
	
	async def _send_participant_invitations(self, collaboration_session: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Send invitations to selected participants"""
		try:
			invitations = []
			
			for participant in collaboration_session['participants']:
				invitation = {
					'invitation_id': uuid7str(),
					'collaboration_id': collaboration_session['collaboration_id'],
					'participant_id': participant['participant_id'],
					'participant_name': participant['name'],
					'role': participant['role'],
					'invited_at': datetime.now(timezone.utc).isoformat(),
					'expires_at': collaboration_session['timeout_at'],
					'status': 'sent',
					'verification_context': {
						'user_id': collaboration_session['user_id'],
						'verification_type': collaboration_session['verification_type'],
						'primary_result': collaboration_session['primary_verification_result'],
						'risk_factors': collaboration_session['context'].get('risk_factors', [])
					}
				}
				
				# In real implementation, this would send actual notifications
				# via email, SMS, push notifications, etc.
				await self._send_invitation_notification(invitation)
				
				invitations.append(invitation)
			
			return invitations
			
		except Exception as e:
			print(f"Failed to send participant invitations: {e}")
			return []
	
	async def _send_invitation_notification(self, invitation: Dict[str, Any]) -> bool:
		"""Send notification to participant (simulated)"""
		try:
			# This would integrate with notification service
			# For now, we'll just log the invitation
			self._log_collaboration_operation(
				"INVITE_PARTICIPANT",
				invitation['collaboration_id'],
				f"Participant: {invitation['participant_id']}"
			)
			
			return True
			
		except Exception as e:
			print(f"Failed to send invitation notification: {e}")
			return False
	
	async def submit_participant_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Submit participant response to collaboration"""
		try:
			assert response_data.get('collaboration_id'), "Collaboration ID is required"
			assert response_data.get('participant_id'), "Participant ID is required"
			assert response_data.get('decision') in ['approve', 'reject'], "Decision must be 'approve' or 'reject'"
			
			collaboration_id = response_data['collaboration_id']
			participant_id = response_data['participant_id']
			decision = response_data['decision']
			
			# Get collaboration session
			if collaboration_id not in self.active_collaborations:
				return {'success': False, 'error': 'Collaboration session not found'}
			
			collaboration_session = self.active_collaborations[collaboration_id]
			
			# Verify participant is authorized
			authorized_participant = None
			for participant in collaboration_session['participants']:
				if participant['participant_id'] == participant_id:
					authorized_participant = participant
					break
			
			if not authorized_participant:
				return {'success': False, 'error': 'Participant not authorized for this collaboration'}
			
			# Check if already responded
			existing_responses = collaboration_session['approvals'] + collaboration_session['rejections']
			for response in existing_responses:
				if response['participant_id'] == participant_id:
					return {'success': False, 'error': 'Participant has already responded'}
			
			# Create response record
			response_record = {
				'response_id': uuid7str(),
				'participant_id': participant_id,
				'participant_name': authorized_participant['name'],
				'participant_role': authorized_participant['role'],
				'decision': decision,
				'confidence_score': response_data.get('confidence_score', 0.8),
				'reasoning': response_data.get('reasoning', ''),
				'additional_evidence': response_data.get('additional_evidence', {}),
				'response_time_seconds': response_data.get('response_time_seconds', 0),
				'submitted_at': datetime.now(timezone.utc).isoformat()
			}
			
			# Add to appropriate list
			if decision == 'approve':
				collaboration_session['approvals'].append(response_record)
			else:
				collaboration_session['rejections'].append(response_record)
			
			# Add any comments
			if response_data.get('comments'):
				comment_record = {
					'comment_id': uuid7str(),
					'participant_id': participant_id,
					'participant_name': authorized_participant['name'],
					'comment': response_data['comments'],
					'timestamp': datetime.now(timezone.utc).isoformat()
				}
				collaboration_session['comments'].append(comment_record)
			
			# Check if collaboration is complete
			workflow_result = await self._evaluate_collaboration_status(collaboration_session)
			
			self._log_collaboration_operation(
				"PARTICIPANT_RESPONSE",
				collaboration_id,
				f"Participant: {participant_id}, Decision: {decision}"
			)
			
			return {
				'success': True,
				'response_recorded': True,
				'collaboration_status': workflow_result['status'],
				'final_decision': workflow_result.get('final_decision'),
				'completion_status': workflow_result
			}
			
		except Exception as e:
			print(f"Failed to submit participant response: {e}")
			return {'success': False, 'error': str(e)}
	
	async def _evaluate_collaboration_status(self, collaboration_session: Dict[str, Any]) -> Dict[str, Any]:
		"""Evaluate current collaboration status and determine if complete"""
		try:
			workflow_template = collaboration_session['workflow_template']
			required_approvals = workflow_template['required_approvals']
			consensus_required = workflow_template['consensus_required']
			
			total_approvals = len(collaboration_session['approvals'])
			total_rejections = len(collaboration_session['rejections'])
			total_responses = total_approvals + total_rejections
			
			# Calculate consensus score
			consensus_score = self._calculate_consensus_score(collaboration_session)
			collaboration_session['consensus_score'] = consensus_score
			
			# Check for completion conditions
			status_result = {
				'status': collaboration_session['status'].value,
				'total_responses': total_responses,
				'approvals': total_approvals,
				'rejections': total_rejections,
				'consensus_score': consensus_score,
				'required_approvals': required_approvals,
				'consensus_required': consensus_required
			}
			
			# Check if we have enough approvals
			if total_approvals >= required_approvals:
				if consensus_required:
					if consensus_score >= self.consensus_threshold:
						# Consensus reached with enough approvals
						collaboration_session['status'] = CollaborationWorkflowStatus.APPROVED
						collaboration_session['final_decision'] = 'approved'
						status_result['status'] = 'approved'
						status_result['final_decision'] = 'approved'
						status_result['completion_reason'] = 'consensus_approval'
					else:
						# Need better consensus
						collaboration_session['status'] = CollaborationWorkflowStatus.PENDING_APPROVAL
						status_result['status'] = 'pending_approval'
						status_result['completion_reason'] = 'awaiting_consensus'
				else:
					# Enough approvals without consensus requirement
					collaboration_session['status'] = CollaborationWorkflowStatus.APPROVED
					collaboration_session['final_decision'] = 'approved'
					status_result['status'] = 'approved'
					status_result['final_decision'] = 'approved'
					status_result['completion_reason'] = 'sufficient_approvals'
			
			# Check for rejection threshold
			elif total_rejections > (len(collaboration_session['participants']) - required_approvals):
				# Too many rejections to reach required approvals
				collaboration_session['status'] = CollaborationWorkflowStatus.REJECTED
				collaboration_session['final_decision'] = 'rejected'
				status_result['status'] = 'rejected'
				status_result['final_decision'] = 'rejected'
				status_result['completion_reason'] = 'insufficient_approvals'
			
			# Check for timeout
			elif datetime.now(timezone.utc) > datetime.fromisoformat(collaboration_session['timeout_at'].replace('Z', '+00:00')):
				if total_approvals > 0 and consensus_score >= workflow_template['escalation_threshold']:
					# Escalate for review
					collaboration_session['status'] = CollaborationWorkflowStatus.ESCALATED
					status_result['status'] = 'escalated'
					status_result['completion_reason'] = 'timeout_escalation'
				else:
					# Timeout rejection
					collaboration_session['status'] = CollaborationWorkflowStatus.REJECTED
					collaboration_session['final_decision'] = 'rejected'
					status_result['status'] = 'rejected'
					status_result['final_decision'] = 'rejected'
					status_result['completion_reason'] = 'timeout'
			
			# If completed, move to history
			if collaboration_session['status'] in [
				CollaborationWorkflowStatus.APPROVED,
				CollaborationWorkflowStatus.REJECTED,
				CollaborationWorkflowStatus.COMPLETED
			]:
				await self._complete_collaboration(collaboration_session)
			
			return status_result
			
		except Exception as e:
			print(f"Failed to evaluate collaboration status: {e}")
			return {'status': 'error', 'error': str(e)}
	
	def _calculate_consensus_score(self, collaboration_session: Dict[str, Any]) -> float:
		"""Calculate consensus score from participant responses"""
		try:
			approvals = collaboration_session['approvals']
			rejections = collaboration_session['rejections']
			
			if not approvals and not rejections:
				return 0.0
			
			total_responses = len(approvals) + len(rejections)
			
			# Basic consensus: ratio of approvals
			approval_ratio = len(approvals) / total_responses
			
			# Weight by participant confidence scores
			weighted_approvals = sum(approval.get('confidence_score', 0.8) for approval in approvals)
			weighted_rejections = sum(rejection.get('confidence_score', 0.8) for rejection in rejections)
			total_weighted = weighted_approvals + weighted_rejections
			
			if total_weighted > 0:
				weighted_consensus = weighted_approvals / total_weighted
			else:
				weighted_consensus = approval_ratio
			
			# Weight by participant roles
			role_weighted_score = self._calculate_role_weighted_consensus(collaboration_session)
			
			# Combine scores
			final_consensus = (
				approval_ratio * 0.4 +
				weighted_consensus * 0.4 +
				role_weighted_score * 0.2
			)
			
			return min(1.0, max(0.0, final_consensus))
			
		except Exception as e:
			print(f"Failed to calculate consensus score: {e}")
			return 0.0
	
	def _calculate_role_weighted_consensus(self, collaboration_session: Dict[str, Any]) -> float:
		"""Calculate consensus score weighted by participant roles"""
		try:
			# Role weights
			role_weights = {
				VerificationParticipantRole.EXPERT.value: 1.0,
				VerificationParticipantRole.SUPERVISOR.value: 0.8,
				VerificationParticipantRole.PEER_REVIEWER.value: 0.6,
				VerificationParticipantRole.AUTOMATED_SYSTEM.value: 0.4
			}
			
			total_weighted_approval = 0.0
			total_weight = 0.0
			
			# Weight approvals by role
			for approval in collaboration_session['approvals']:
				role = approval['participant_role']
				weight = role_weights.get(role, 0.5)
				confidence = approval.get('confidence_score', 0.8)
				
				total_weighted_approval += weight * confidence
				total_weight += weight
			
			# Weight rejections by role (negative contribution)
			for rejection in collaboration_session['rejections']:
				role = rejection['participant_role']
				weight = role_weights.get(role, 0.5)
				confidence = rejection.get('confidence_score', 0.8)
				
				total_weight += weight
				# Rejections reduce the weighted approval score
			
			if total_weight > 0:
				role_weighted_consensus = total_weighted_approval / total_weight
			else:
				role_weighted_consensus = 0.0
			
			return min(1.0, max(0.0, role_weighted_consensus))
			
		except Exception as e:
			print(f"Failed to calculate role-weighted consensus: {e}")
			return 0.0
	
	async def _complete_collaboration(self, collaboration_session: Dict[str, Any]) -> None:
		"""Complete collaboration and move to history"""
		try:
			collaboration_id = collaboration_session['collaboration_id']
			
			# Add completion metadata
			collaboration_session['completed_at'] = datetime.now(timezone.utc).isoformat()
			collaboration_session['total_duration_minutes'] = (
				datetime.now(timezone.utc) - 
				datetime.fromisoformat(collaboration_session['initiated_at'].replace('Z', '+00:00'))
			).total_seconds() / 60
			
			# Calculate participant ratings
			await self._update_participant_ratings(collaboration_session)
			
			# Move to history
			self.collaboration_history[collaboration_id] = collaboration_session
			
			# Remove from active collaborations
			if collaboration_id in self.active_collaborations:
				del self.active_collaborations[collaboration_id]
			
			self._log_collaboration_operation(
				"COMPLETE",
				collaboration_id,
				f"Status: {collaboration_session['status'].value}, Decision: {collaboration_session.get('final_decision', 'none')}"
			)
			
		except Exception as e:
			print(f"Failed to complete collaboration: {e}")
	
	async def _update_participant_ratings(self, collaboration_session: Dict[str, Any]) -> None:
		"""Update participant performance ratings"""
		try:
			consensus_score = collaboration_session.get('consensus_score', 0.5)
			final_decision = collaboration_session.get('final_decision')
			
			# Rate participants based on alignment with consensus
			for approval in collaboration_session['approvals']:
				participant_id = approval['participant_id']
				
				# Positive rating if approval aligns with final decision
				rating_adjustment = 0.1 if final_decision == 'approved' else -0.05
				await self._update_participant_rating(participant_id, rating_adjustment)
			
			for rejection in collaboration_session['rejections']:
				participant_id = rejection['participant_id']
				
				# Positive rating if rejection aligns with final decision
				rating_adjustment = 0.1 if final_decision == 'rejected' else -0.05
				await self._update_participant_rating(participant_id, rating_adjustment)
			
		except Exception as e:
			print(f"Failed to update participant ratings: {e}")
	
	async def _update_participant_rating(self, participant_id: str, rating_adjustment: float) -> None:
		"""Update individual participant rating"""
		try:
			if participant_id not in self.participant_ratings:
				self.participant_ratings[participant_id] = {
					'overall_rating': 0.7,
					'total_collaborations': 0,
					'successful_collaborations': 0,
					'average_response_time': 0.0,
					'consensus_alignment': 0.0
				}
			
			participant_rating = self.participant_ratings[participant_id]
			participant_rating['total_collaborations'] += 1
			
			# Update overall rating with exponential moving average
			current_rating = participant_rating['overall_rating']
			new_rating = current_rating + (rating_adjustment * 0.1)
			participant_rating['overall_rating'] = max(0.0, min(1.0, new_rating))
			
		except Exception as e:
			print(f"Failed to update participant rating: {e}")
	
	async def get_collaboration_status(self, collaboration_id: str) -> Dict[str, Any]:
		"""Get current status of collaboration"""
		try:
			# Check active collaborations first
			if collaboration_id in self.active_collaborations:
				collaboration = self.active_collaborations[collaboration_id]
			elif collaboration_id in self.collaboration_history:
				collaboration = self.collaboration_history[collaboration_id]
			else:
				return {'error': 'Collaboration not found'}
			
			# Calculate current status
			status_info = {
				'collaboration_id': collaboration_id,
				'status': collaboration['status'].value if hasattr(collaboration['status'], 'value') else collaboration['status'],
				'workflow_type': collaboration['workflow_type'],
				'initiated_at': collaboration['initiated_at'],
				'timeout_at': collaboration['timeout_at'],
				'participants': len(collaboration['participants']),
				'responses_received': len(collaboration['approvals']) + len(collaboration['rejections']),
				'approvals': len(collaboration['approvals']),
				'rejections': len(collaboration['rejections']),
				'consensus_score': collaboration.get('consensus_score', 0.0),
				'final_decision': collaboration.get('final_decision'),
				'comments': len(collaboration['comments'])
			}
			
			# Add completion info if available
			if 'completed_at' in collaboration:
				status_info['completed_at'] = collaboration['completed_at']
				status_info['total_duration_minutes'] = collaboration.get('total_duration_minutes', 0)
			
			return status_info
			
		except Exception as e:
			print(f"Failed to get collaboration status: {e}")
			return {'error': str(e)}
	
	async def cancel_collaboration(self, collaboration_id: str, reason: str) -> Dict[str, Any]:
		"""Cancel active collaboration"""
		try:
			if collaboration_id not in self.active_collaborations:
				return {'success': False, 'error': 'Active collaboration not found'}
			
			collaboration_session = self.active_collaborations[collaboration_id]
			
			# Update status
			collaboration_session['status'] = CollaborationWorkflowStatus.REJECTED
			collaboration_session['final_decision'] = 'cancelled'
			collaboration_session['cancellation_reason'] = reason
			collaboration_session['cancelled_at'] = datetime.now(timezone.utc).isoformat()
			
			# Complete the collaboration
			await self._complete_collaboration(collaboration_session)
			
			self._log_collaboration_operation("CANCEL", collaboration_id, reason)
			
			return {'success': True, 'status': 'cancelled', 'reason': reason}
			
		except Exception as e:
			print(f"Failed to cancel collaboration: {e}")
			return {'success': False, 'error': str(e)}
	
	async def get_participant_workload(self, participant_id: str) -> Dict[str, Any]:
		"""Get current workload for participant"""
		try:
			workload = {
				'participant_id': participant_id,
				'active_collaborations': 0,
				'pending_responses': 0,
				'overdue_responses': 0,
				'average_response_time_minutes': 0,
				'recent_activity': []
			}
			
			current_time = datetime.now(timezone.utc)
			
			# Check active collaborations
			for collaboration in self.active_collaborations.values():
				for participant in collaboration['participants']:
					if participant['participant_id'] == participant_id:
						workload['active_collaborations'] += 1
						
						# Check if response is pending
						has_responded = any(
							response['participant_id'] == participant_id
							for response in collaboration['approvals'] + collaboration['rejections']
						)
						
						if not has_responded:
							workload['pending_responses'] += 1
							
							# Check if overdue
							timeout = datetime.fromisoformat(collaboration['timeout_at'].replace('Z', '+00:00'))
							if current_time > timeout:
								workload['overdue_responses'] += 1
			
			# Get participant ratings
			if participant_id in self.participant_ratings:
				rating_info = self.participant_ratings[participant_id]
				workload['overall_rating'] = rating_info['overall_rating']
				workload['total_collaborations'] = rating_info['total_collaborations']
				workload['average_response_time_minutes'] = rating_info.get('average_response_time', 0)
			
			return workload
			
		except Exception as e:
			print(f"Failed to get participant workload: {e}")
			return {'error': str(e)}
	
	def get_collaboration_analytics(self, days: int = 30) -> Dict[str, Any]:
		"""Get collaboration analytics for specified period"""
		try:
			cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
			
			analytics = {
				'period_days': days,
				'total_collaborations': 0,
				'successful_collaborations': 0,
				'failed_collaborations': 0,
				'average_duration_minutes': 0.0,
				'average_consensus_score': 0.0,
				'workflow_breakdown': {},
				'participant_performance': {},
				'response_time_metrics': {}
			}
			
			relevant_collaborations = []
			
			# Collect relevant collaborations
			for collaboration in self.collaboration_history.values():
				initiated_date = datetime.fromisoformat(collaboration['initiated_at'].replace('Z', '+00:00'))
				if initiated_date >= cutoff_date:
					relevant_collaborations.append(collaboration)
			
			if not relevant_collaborations:
				return analytics
			
			analytics['total_collaborations'] = len(relevant_collaborations)
			
			# Calculate metrics
			durations = []
			consensus_scores = []
			workflows = {}
			
			for collaboration in relevant_collaborations:
				# Success/failure
				if collaboration.get('final_decision') == 'approved':
					analytics['successful_collaborations'] += 1
				else:
					analytics['failed_collaborations'] += 1
				
				# Duration
				if 'total_duration_minutes' in collaboration:
					durations.append(collaboration['total_duration_minutes'])
				
				# Consensus
				consensus_scores.append(collaboration.get('consensus_score', 0.0))
				
				# Workflow breakdown
				workflow_type = collaboration['workflow_type']
				workflows[workflow_type] = workflows.get(workflow_type, 0) + 1
			
			# Calculate averages
			if durations:
				analytics['average_duration_minutes'] = sum(durations) / len(durations)
			
			if consensus_scores:
				analytics['average_consensus_score'] = sum(consensus_scores) / len(consensus_scores)
			
			analytics['workflow_breakdown'] = workflows
			
			# Participant performance summary
			analytics['participant_performance'] = {
				participant_id: {
					'overall_rating': rating['overall_rating'],
					'total_collaborations': rating['total_collaborations'],
					'success_rate': rating.get('successful_collaborations', 0) / max(1, rating['total_collaborations'])
				}
				for participant_id, rating in self.participant_ratings.items()
			}
			
			return analytics
			
		except Exception as e:
			print(f"Failed to get collaboration analytics: {e}")
			return {'error': str(e)}
	
	def get_engine_statistics(self) -> Dict[str, Any]:
		"""Get collaborative verification engine statistics"""
		return {
			'tenant_id': self.tenant_id,
			'collaboration_enabled': self.collaboration_enabled,
			'active_collaborations': len(self.active_collaborations),
			'completed_collaborations': len(self.collaboration_history),
			'registered_participants': len(self.participant_ratings),
			'workflow_templates': list(self.workflow_templates.keys()),
			'consensus_threshold': self.consensus_threshold,
			'supervisor_approval_threshold': self.supervisor_approval_threshold,
			'max_reviewers': self.max_reviewers
		}

# Export for use in other modules
__all__ = ['CollaborativeVerificationEngine', 'CollaborativeVerificationType', 'VerificationParticipantRole', 'CollaborationWorkflowStatus']