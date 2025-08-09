"""
Product Lifecycle Management (PLM) Capability - Service Tests

Async service tests with APG capability integration testing.
Tests all PLM services with real objects and external API mocking.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid_extensions import uuid7str
from unittest.mock import AsyncMock, patch
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from ..service import PLMProductService, PLMEngineeringChangeService, PLMCollaborationService
from ..ai_service import PLMAIService
from ..models import PLProduct, PLEngineeringChange, PLCollaborationSession
from .conftest import assert_async_result


class TestPLMProductService:
	"""Test PLMProductService with APG integration"""
	
	async def test_create_product_success(
		self,
		plm_product_service: PLMProductService,
		sample_product_data: Dict[str, Any],
		test_tenant_id: str,
		test_user_id: str
	):
		"""Test successful product creation"""
		# Test product creation
		product = await plm_product_service.create_product(
			tenant_id=test_tenant_id,
			product_data=sample_product_data,
			user_id=test_user_id
		)
		
		# Verify product creation
		assert product is not None
		assert product.product_name == sample_product_data['product_name']
		assert product.product_number == sample_product_data['product_number']
		assert product.tenant_id == test_tenant_id
		assert product.created_by == test_user_id
		assert product.lifecycle_phase == sample_product_data['lifecycle_phase']
	
	async def test_create_product_with_auto_digital_twin(
		self,
		plm_product_service: PLMProductService,
		sample_product_data: Dict[str, Any],
		test_tenant_id: str,
		test_user_id: str,
		mock_digital_twin_api: str
	):
		"""Test product creation with automatic digital twin creation"""
		with patch('httpx.AsyncClient.post') as mock_post:
			# Mock digital twin API response
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'success': True,
				'twin_id': f'twin_{uuid7str()[:8]}'
			}
			mock_response.status_code = 201
			mock_post.return_value = mock_response
			
			# Create product with auto digital twin
			product = await plm_product_service.create_product(
				tenant_id=test_tenant_id,
				product_data=sample_product_data,
				user_id=test_user_id,
				auto_create_digital_twin=True
			)
			
			# Verify product and digital twin creation
			assert product is not None
			assert product.digital_twin_id is not None
			assert product.digital_twin_id.startswith('twin_')
	
	async def test_get_product_success(
		self,
		plm_product_service: PLMProductService,
		test_product: PLProduct,
		test_user_id: str
	):
		"""Test successful product retrieval"""
		# Get product
		retrieved_product = await plm_product_service.get_product(
			product_id=test_product.product_id,
			user_id=test_user_id,
			tenant_id=test_product.tenant_id
		)
		
		# Verify retrieval
		assert retrieved_product is not None
		assert retrieved_product.product_id == test_product.product_id
		assert retrieved_product.product_name == test_product.product_name
	
	async def test_get_product_not_found(
		self,
		plm_product_service: PLMProductService,
		test_user_id: str,
		test_tenant_id: str
	):
		"""Test product retrieval with non-existent ID"""
		# Try to get non-existent product
		retrieved_product = await plm_product_service.get_product(
			product_id='non_existent_id',
			user_id=test_user_id,
			tenant_id=test_tenant_id
		)
		
		# Verify not found
		assert retrieved_product is None
	
	async def test_update_product_success(
		self,
		plm_product_service: PLMProductService,
		test_product: PLProduct,
		test_user_id: str
	):
		"""Test successful product update"""
		# Update data
		update_data = {
			'product_name': 'Updated Product Name',
			'target_cost': 1500.00,
			'lifecycle_phase': 'production'
		}
		
		# Update product
		updated_product = await plm_product_service.update_product(
			product_id=test_product.product_id,
			update_data=update_data,
			user_id=test_user_id,
			tenant_id=test_product.tenant_id
		)
		
		# Verify update
		assert updated_product is not None
		assert updated_product.product_name == 'Updated Product Name'
		assert updated_product.target_cost == 1500.00
		assert updated_product.lifecycle_phase == 'production'
		assert updated_product.updated_by == test_user_id
	
	async def test_search_products_with_filters(
		self,
		plm_product_service: PLMProductService,
		multiple_test_products: List[PLProduct],
		test_user_id: str
	):
		"""Test product search with various filters"""
		tenant_id = multiple_test_products[0].tenant_id
		
		# Search with text filter
		result = await plm_product_service.search_products(
			tenant_id=tenant_id,
			user_id=test_user_id,
			search_text='Alpha',
			page=1,
			page_size=10
		)
		
		# Verify search results
		assert result['total_count'] >= 1
		assert len(result['products']) >= 1
		assert 'Alpha' in result['products'][0]['product_name']
		
		# Search with product type filter
		result = await plm_product_service.search_products(
			tenant_id=tenant_id,
			user_id=test_user_id,
			product_type='manufactured',
			page=1,
			page_size=10
		)
		
		# Verify filtered results
		assert result['total_count'] >= 1
		for product in result['products']:
			assert product['product_type'] == 'manufactured'
	
	async def test_search_products_pagination(
		self,
		plm_product_service: PLMProductService,
		multiple_test_products: List[PLProduct],
		test_user_id: str
	):
		"""Test product search pagination"""
		tenant_id = multiple_test_products[0].tenant_id
		
		# Search with small page size
		result = await plm_product_service.search_products(
			tenant_id=tenant_id,
			user_id=test_user_id,
			page=1,
			page_size=2
		)
		
		# Verify pagination
		assert len(result['products']) <= 2
		assert result['total_pages'] >= 1
		assert result['page'] == 1
		assert result['page_size'] == 2
	
	async def test_delete_product_success(
		self,
		plm_product_service: PLMProductService,
		test_product: PLProduct,
		test_user_id: str
	):
		"""Test successful product soft deletion"""
		# Delete product
		success = await plm_product_service.delete_product(
			product_id=test_product.product_id,
			user_id=test_user_id,
			tenant_id=test_product.tenant_id
		)
		
		# Verify deletion
		assert success is True
		
		# Verify product is soft deleted (status changed but not physically deleted)
		deleted_product = await plm_product_service.get_product(
			product_id=test_product.product_id,
			user_id=test_user_id,
			tenant_id=test_product.tenant_id,
			include_deleted=True
		)
		
		assert deleted_product is not None
		assert deleted_product.status == 'deleted'
	
	async def test_create_digital_twin_integration(
		self,
		plm_product_service: PLMProductService,
		test_product: PLProduct,
		test_user_id: str,
		mock_digital_twin_api: str
	):
		"""Test digital twin creation integration"""
		with patch('httpx.AsyncClient.post') as mock_post:
			# Mock successful digital twin creation
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'success': True,
				'twin_id': f'twin_{uuid7str()[:8]}'
			}
			mock_response.status_code = 201
			mock_post.return_value = mock_response
			
			# Create digital twin
			twin_id = await plm_product_service.create_digital_twin(
				product_id=test_product.product_id,
				user_id=test_user_id,
				tenant_id=test_product.tenant_id
			)
			
			# Verify digital twin creation
			assert twin_id is not None
			assert twin_id.startswith('twin_')
			
			# Verify API call was made
			mock_post.assert_called_once()
	
	async def test_get_dashboard_metrics(
		self,
		plm_product_service: PLMProductService,
		multiple_test_products: List[PLProduct],
		test_user_id: str
	):
		"""Test dashboard metrics calculation"""
		tenant_id = multiple_test_products[0].tenant_id
		
		# Get dashboard metrics
		metrics = await plm_product_service.get_dashboard_metrics(
			tenant_id=tenant_id,
			user_id=test_user_id
		)
		
		# Verify metrics structure
		assert 'total_products' in metrics
		assert 'active_products' in metrics
		assert 'products_in_development' in metrics
		assert 'system_performance_score' in metrics
		
		# Verify metrics values
		assert metrics['total_products'] >= len(multiple_test_products)
		assert metrics['active_products'] >= 0
		assert 0 <= metrics['system_performance_score'] <= 100
	
	async def test_get_product_structure(
		self,
		plm_product_service: PLMProductService,
		test_product: PLProduct,
		test_user_id: str
	):
		"""Test product structure (BOM) retrieval"""
		# Get product structure
		structure = await plm_product_service.get_product_structure(
			product_id=test_product.product_id,
			user_id=test_user_id,
			tenant_id=test_product.tenant_id
		)
		
		# Verify structure response
		assert structure is not None
		assert 'product_id' in structure
		assert 'components' in structure
		assert structure['product_id'] == test_product.product_id
	
	async def test_manufacturing_sync_integration(
		self,
		plm_product_service: PLMProductService,
		test_product: PLProduct,
		test_user_id: str,
		mock_manufacturing_api: str
	):
		"""Test manufacturing system sync integration"""
		with patch('httpx.AsyncClient.post') as mock_post:
			# Mock successful manufacturing sync
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'success': True,
				'bom_id': 'mock_bom_123'
			}
			mock_response.status_code = 201
			mock_post.return_value = mock_response
			
			# Sync with manufacturing
			success = await plm_product_service.sync_with_manufacturing(
				product_id=test_product.product_id,
				user_id=test_user_id,
				tenant_id=test_product.tenant_id
			)
			
			# Verify sync success
			assert success is True
			
			# Verify API call was made
			mock_post.assert_called_once()


class TestPLMEngineeringChangeService:
	"""Test PLMEngineeringChangeService with APG integration"""
	
	async def test_create_change_success(
		self,
		plm_change_service: PLMEngineeringChangeService,
		sample_change_data: Dict[str, Any],
		test_tenant_id: str,
		test_user_id: str
	):
		"""Test successful engineering change creation"""
		# Create change
		change = await plm_change_service.create_change(
			tenant_id=test_tenant_id,
			change_data=sample_change_data,
			user_id=test_user_id
		)
		
		# Verify change creation
		assert change is not None
		assert change.change_title == sample_change_data['change_title']
		assert change.change_type == sample_change_data['change_type']
		assert change.status == 'draft'
		assert change.created_by == test_user_id
		assert change.change_number is not None
	
	async def test_submit_change_for_approval(
		self,
		plm_change_service: PLMEngineeringChangeService,
		test_change: PLEngineeringChange,
		test_user_id: str
	):
		"""Test submitting change for approval"""
		# Submit for approval
		success = await plm_change_service.submit_change_for_approval(
			change_id=test_change.change_id,
			user_id=test_user_id,
			tenant_id=test_change.tenant_id
		)
		
		# Verify submission
		assert success is True
		
		# Get updated change
		updated_change = await plm_change_service.get_change(
			change_id=test_change.change_id,
			user_id=test_user_id,
			tenant_id=test_change.tenant_id
		)
		
		assert updated_change.status == 'submitted'
	
	async def test_approve_change_success(
		self,
		plm_change_service: PLMEngineeringChangeService,
		test_change: PLEngineeringChange,
		test_user_id: str
	):
		"""Test successful change approval"""
		# Set change to submitted status first
		test_change.status = 'submitted'
		test_change.approvers = [test_user_id, 'other_approver']
		
		# Approve change
		success = await plm_change_service.approve_change(
			change_id=test_change.change_id,
			user_id=test_user_id,
			tenant_id=test_change.tenant_id,
			comments='Approved for implementation'
		)
		
		# Verify approval
		assert success is True
		
		# Get updated change
		approved_change = await plm_change_service.get_change(
			change_id=test_change.change_id,
			user_id=test_user_id,
			tenant_id=test_change.tenant_id
		)
		
		assert test_user_id in approved_change.approved_by
		assert approved_change.approval_comments[test_user_id] == 'Approved for implementation'
	
	async def test_reject_change(
		self,
		plm_change_service: PLMEngineeringChangeService,
		test_change: PLEngineeringChange,
		test_user_id: str
	):
		"""Test change rejection"""
		# Set change to under review status
		test_change.status = 'under_review'
		
		# Reject change
		success = await plm_change_service.reject_change(
			change_id=test_change.change_id,
			user_id=test_user_id,
			tenant_id=test_change.tenant_id,
			reason='Insufficient technical details'
		)
		
		# Verify rejection
		assert success is True
		
		# Get updated change
		rejected_change = await plm_change_service.get_change(
			change_id=test_change.change_id,
			user_id=test_user_id,
			tenant_id=test_change.tenant_id
		)
		
		assert rejected_change.status == 'rejected'
	
	async def test_implement_change(
		self,
		plm_change_service: PLMEngineeringChangeService,
		test_change: PLEngineeringChange,
		test_user_id: str
	):
		"""Test change implementation"""
		# Set change to approved status
		test_change.status = 'approved'
		
		# Implement change
		success = await plm_change_service.implement_change(
			change_id=test_change.change_id,
			user_id=test_user_id,
			tenant_id=test_change.tenant_id,
			implementation_notes='Successfully implemented in production'
		)
		
		# Verify implementation
		assert success is True
		
		# Get updated change
		implemented_change = await plm_change_service.get_change(
			change_id=test_change.change_id,
			user_id=test_user_id,
			tenant_id=test_change.tenant_id
		)
		
		assert implemented_change.status == 'implemented'
		assert implemented_change.actual_implementation_date is not None
		assert 'production' in implemented_change.implementation_notes
	
	async def test_get_change_impact_analysis(
		self,
		plm_change_service: PLMEngineeringChangeService,
		test_change: PLEngineeringChange,
		test_user_id: str
	):
		"""Test change impact analysis"""
		# Get impact analysis
		impact = await plm_change_service.get_change_impact_analysis(
			change_id=test_change.change_id,
			user_id=test_user_id,
			tenant_id=test_change.tenant_id
		)
		
		# Verify impact analysis structure
		assert impact is not None
		assert 'affected_products' in impact
		assert 'cost_impact' in impact
		assert 'schedule_impact' in impact
		assert 'risk_assessment' in impact
		
		# Verify impact data
		assert len(impact['affected_products']) > 0
		assert impact['cost_impact'] is not None
	
	async def test_search_changes_with_filters(
		self,
		plm_change_service: PLMEngineeringChangeService,
		test_change: PLEngineeringChange,
		test_user_id: str
	):
		"""Test change search with filters"""
		# Search changes
		result = await plm_change_service.search_changes(
			tenant_id=test_change.tenant_id,
			user_id=test_user_id,
			status='draft',
			change_type=test_change.change_type,
			page=1,
			page_size=10
		)
		
		# Verify search results
		assert result['total_count'] >= 1
		assert len(result['changes']) >= 1
		
		# Verify filter application
		for change in result['changes']:
			assert change['status'] == 'draft'
			assert change['change_type'] == test_change.change_type
	
	async def test_audit_compliance_integration(
		self,
		plm_change_service: PLMEngineeringChangeService,
		test_change: PLEngineeringChange,
		test_user_id: str
	):
		"""Test audit compliance integration for changes"""
		with patch('httpx.AsyncClient.post') as mock_post:
			# Mock audit compliance API response
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'success': True,
				'audit_trail_id': f'audit_{uuid7str()[:8]}'
			}
			mock_response.status_code = 201
			mock_post.return_value = mock_response
			
			# Create audit trail
			audit_trail_id = await plm_change_service.create_audit_trail(
				change_id=test_change.change_id,
				user_id=test_user_id,
				tenant_id=test_change.tenant_id,
				action='change_approved'
			)
			
			# Verify audit trail creation
			assert audit_trail_id is not None
			assert audit_trail_id.startswith('audit_')
			
			# Verify API call was made
			mock_post.assert_called_once()


class TestPLMCollaborationService:
	"""Test PLMCollaborationService with APG integration"""
	
	async def test_create_collaboration_session_success(
		self,
		plm_collaboration_service: PLMCollaborationService,
		sample_collaboration_data: Dict[str, Any],
		test_tenant_id: str,
		test_user_id: str
	):
		"""Test successful collaboration session creation"""
		# Create session
		session = await plm_collaboration_service.create_collaboration_session(
			tenant_id=test_tenant_id,
			session_data=sample_collaboration_data,
			user_id=test_user_id
		)
		
		# Verify session creation
		assert session is not None
		assert session.session_name == sample_collaboration_data['session_name']
		assert session.host_user_id == test_user_id
		assert session.status == 'scheduled'
		assert session.created_by == test_user_id
	
	async def test_start_collaboration_session(
		self,
		plm_collaboration_service: PLMCollaborationService,
		test_collaboration_session: PLCollaborationSession,
		test_user_id: str
	):
		"""Test starting a collaboration session"""
		with patch('httpx.AsyncClient.post') as mock_post:
			# Mock real-time collaboration API response
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'success': True,
				'room_id': f'room_{uuid7str()[:8]}'
			}
			mock_response.status_code = 201
			mock_post.return_value = mock_response
			
			# Start session
			room_id = await plm_collaboration_service.start_collaboration_session(
				session_id=test_collaboration_session.session_id,
				user_id=test_user_id,
				tenant_id=test_collaboration_session.tenant_id
			)
			
			# Verify session start
			assert room_id is not None
			assert room_id.startswith('room_')
			
			# Get updated session
			updated_session = await plm_collaboration_service.get_collaboration_session(
				session_id=test_collaboration_session.session_id,
				user_id=test_user_id,
				tenant_id=test_collaboration_session.tenant_id
			)
			
			assert updated_session.status == 'in_progress'
			assert updated_session.actual_start is not None
			assert updated_session.collaboration_room_id == room_id
	
	async def test_join_collaboration_session(
		self,
		plm_collaboration_service: PLMCollaborationService,
		test_collaboration_session: PLCollaborationSession,
		test_user_id: str
	):
		"""Test joining a collaboration session"""
		# Set session to in progress
		test_collaboration_session.status = 'in_progress'
		test_collaboration_session.collaboration_room_id = f'room_{uuid7str()[:8]}'
		
		# Join session
		success = await plm_collaboration_service.join_collaboration_session(
			session_id=test_collaboration_session.session_id,
			user_id=test_user_id,
			tenant_id=test_collaboration_session.tenant_id
		)
		
		# Verify join success
		assert success is True
		
		# Get updated session
		updated_session = await plm_collaboration_service.get_collaboration_session(
			session_id=test_collaboration_session.session_id,
			user_id=test_user_id,
			tenant_id=test_collaboration_session.tenant_id
		)
		
		assert test_user_id in updated_session.participants
	
	async def test_leave_collaboration_session(
		self,
		plm_collaboration_service: PLMCollaborationService,
		test_collaboration_session: PLCollaborationSession,
		test_user_id: str
	):
		"""Test leaving a collaboration session"""
		# Set user as participant
		test_collaboration_session.participants = [test_user_id, 'other_user']
		test_collaboration_session.status = 'in_progress'
		
		# Leave session
		success = await plm_collaboration_service.leave_collaboration_session(
			session_id=test_collaboration_session.session_id,
			user_id=test_user_id,
			tenant_id=test_collaboration_session.tenant_id
		)
		
		# Verify leave success
		assert success is True
		
		# Get updated session
		updated_session = await plm_collaboration_service.get_collaboration_session(
			session_id=test_collaboration_session.session_id,
			user_id=test_user_id,
			tenant_id=test_collaboration_session.tenant_id
		)
		
		assert test_user_id not in updated_session.participants
	
	async def test_end_collaboration_session(
		self,
		plm_collaboration_service: PLMCollaborationService,
		test_collaboration_session: PLCollaborationSession,
		test_user_id: str
	):
		"""Test ending a collaboration session"""
		# Set session to in progress
		test_collaboration_session.status = 'in_progress'
		test_collaboration_session.actual_start = datetime.utcnow()
		
		# End session
		success = await plm_collaboration_service.end_collaboration_session(
			session_id=test_collaboration_session.session_id,
			user_id=test_user_id,
			tenant_id=test_collaboration_session.tenant_id,
			session_notes='Productive design review session completed'
		)
		
		# Verify session end
		assert success is True
		
		# Get updated session
		updated_session = await plm_collaboration_service.get_collaboration_session(
			session_id=test_collaboration_session.session_id,
			user_id=test_user_id,
			tenant_id=test_collaboration_session.tenant_id
		)
		
		assert updated_session.status == 'completed'
		assert updated_session.actual_end is not None
		assert 'Productive' in updated_session.session_notes
	
	async def test_get_collaboration_analytics(
		self,
		plm_collaboration_service: PLMCollaborationService,
		test_collaboration_session: PLCollaborationSession,
		test_user_id: str
	):
		"""Test collaboration analytics"""
		# Get analytics
		analytics = await plm_collaboration_service.get_collaboration_analytics(
			tenant_id=test_collaboration_session.tenant_id,
			user_id=test_user_id
		)
		
		# Verify analytics structure
		assert analytics is not None
		assert 'active_sessions' in analytics
		assert 'total_sessions' in analytics
		assert 'average_duration' in analytics
		assert 'participation_rate' in analytics
		
		# Verify analytics values
		assert analytics['total_sessions'] >= 1
		assert analytics['active_sessions'] >= 0
	
	async def test_send_session_invitations(
		self,
		plm_collaboration_service: PLMCollaborationService,
		test_collaboration_session: PLCollaborationSession,
		test_user_id: str
	):
		"""Test sending session invitations"""
		with patch('httpx.AsyncClient.post') as mock_post:
			# Mock notification API response
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'success': True,
				'notifications_sent': 3
			}
			mock_response.status_code = 200
			mock_post.return_value = mock_response
			
			# Send invitations
			invitees = ['user1@example.com', 'user2@example.com', 'user3@example.com']
			success = await plm_collaboration_service.send_session_invitations(
				session_id=test_collaboration_session.session_id,
				invitees=invitees,
				user_id=test_user_id,
				tenant_id=test_collaboration_session.tenant_id
			)
			
			# Verify invitations sent
			assert success is True
			
			# Verify API call was made
			mock_post.assert_called_once()
	
	async def test_real_time_collaboration_integration(
		self,
		plm_collaboration_service: PLMCollaborationService,
		test_collaboration_session: PLCollaborationSession,
		test_user_id: str
	):
		"""Test real-time collaboration infrastructure integration"""
		with patch('httpx.AsyncClient.post') as mock_post:
			# Mock real-time collaboration setup
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'success': True,
				'websocket_url': 'wss://collab.apg.local/ws/room_123',
				'auth_token': 'collab_token_xyz'
			}
			mock_response.status_code = 200
			mock_post.return_value = mock_response
			
			# Setup real-time collaboration
			collab_config = await plm_collaboration_service.setup_real_time_collaboration(
				session_id=test_collaboration_session.session_id,
				user_id=test_user_id,
				tenant_id=test_collaboration_session.tenant_id
			)
			
			# Verify collaboration setup
			assert collab_config is not None
			assert 'websocket_url' in collab_config
			assert 'auth_token' in collab_config
			assert collab_config['websocket_url'].startswith('wss://')


class TestPLMAIService:
	"""Test PLMAIService with APG AI integration"""
	
	async def test_optimize_product_design(
		self,
		plm_ai_service: PLMAIService,
		test_product: PLProduct,
		test_user_id: str,
		mock_ai_orchestration_api: str
	):
		"""Test AI-powered product design optimization"""
		with patch('httpx.AsyncClient.post') as mock_post:
			# Mock AI optimization response
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'optimization_id': f'opt_{uuid7str()[:8]}',
				'status': 'completed',
				'optimized_parameters': {
					'weight_reduction': 15.5,
					'cost_reduction': 8.2,
					'strength_improvement': 12.1
				},
				'recommendations': [
					'Reduce material thickness by 2mm',
					'Optimize internal geometry',
					'Consider alternative materials'
				]
			}
			mock_response.status_code = 200
			mock_post.return_value = mock_response
			
			# Perform optimization
			optimization_objectives = {
				'minimize_weight': True,
				'minimize_cost': True,
				'maximize_strength': True
			}
			constraints = {
				'max_weight_kg': 5.0,
				'max_cost': 1000.00,
				'min_strength_mpa': 200
			}
			
			result = await plm_ai_service.optimize_product_design(
				product_id=test_product.product_id,
				optimization_objectives=optimization_objectives,
				constraints=constraints,
				user_id=test_user_id
			)
			
			# Verify optimization result
			assert result is not None
			assert 'optimization_id' in result
			assert 'optimized_parameters' in result
			assert 'recommendations' in result
			assert len(result['recommendations']) > 0
			assert result['optimized_parameters']['weight_reduction'] > 0
	
	async def test_get_innovation_insights(
		self,
		plm_ai_service: PLMAIService,
		test_tenant_id: str,
		mock_ai_orchestration_api: str
	):
		"""Test AI innovation insights"""
		with patch('httpx.AsyncClient.get') as mock_get:
			# Mock innovation insights response
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'insights': [
					'Market trending towards sustainable materials',
					'Additive manufacturing adoption increasing',
					'IoT integration becoming standard'
				],
				'confidence_scores': [0.89, 0.76, 0.82],
				'trend_analysis': {
					'sustainability': {'growth_rate': 0.25, 'market_size': '45B'},
					'additive_manufacturing': {'growth_rate': 0.18, 'adoption_rate': 0.34}
				}
			}
			mock_response.status_code = 200
			mock_get.return_value = mock_response
			
			# Get innovation insights
			insights = await plm_ai_service.get_innovation_insights(test_tenant_id)
			
			# Verify insights
			assert insights is not None
			assert 'insights' in insights
			assert 'confidence_scores' in insights
			assert 'trend_analysis' in insights
			assert len(insights['insights']) > 0
			assert all(score > 0.5 for score in insights['confidence_scores'])
	
	async def test_predict_product_failure(
		self,
		plm_ai_service: PLMAIService,
		test_product: PLProduct,
		test_user_id: str
	):
		"""Test AI-powered failure prediction"""
		with patch('httpx.AsyncClient.post') as mock_post:
			# Mock failure prediction response
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'prediction_id': f'pred_{uuid7str()[:8]}',
				'failure_probability': 0.15,
				'predicted_failure_modes': [
					{'mode': 'material_fatigue', 'probability': 0.08, 'time_to_failure_days': 1200},
					{'mode': 'thermal_stress', 'probability': 0.05, 'time_to_failure_days': 800},
					{'mode': 'wear', 'probability': 0.02, 'time_to_failure_days': 2000}
				],
				'preventive_actions': [
					'Regular inspection of high-stress components',
					'Implement thermal monitoring',
					'Consider surface hardening treatment'
				]
			}
			mock_response.status_code = 200
			mock_post.return_value = mock_response
			
			# Predict failure
			operating_conditions = {
				'temperature_range': {'min': -20, 'max': 60},
				'load_cycles_per_day': 1000,
				'environment': 'industrial'
			}
			
			prediction = await plm_ai_service.predict_product_failure(
				product_id=test_product.product_id,
				operating_conditions=operating_conditions,
				user_id=test_user_id
			)
			
			# Verify prediction
			assert prediction is not None
			assert 'failure_probability' in prediction
			assert 'predicted_failure_modes' in prediction
			assert 'preventive_actions' in prediction
			assert 0 <= prediction['failure_probability'] <= 1
			assert len(prediction['predicted_failure_modes']) > 0
	
	async def test_get_cost_optimization_insights(
		self,
		plm_ai_service: PLMAIService,
		test_tenant_id: str
	):
		"""Test AI cost optimization insights"""
		with patch('httpx.AsyncClient.get') as mock_get:
			# Mock cost optimization response
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'total_savings_potential': 125000.00,
				'optimization_opportunities': [
					{
						'category': 'material_optimization',
						'savings_potential': 50000.00,
						'confidence': 0.85,
						'recommendations': ['Switch to aluminum alloy', 'Reduce wall thickness']
					},
					{
						'category': 'manufacturing_process',
						'savings_potential': 35000.00,
						'confidence': 0.78,
						'recommendations': ['Implement lean manufacturing', 'Automate assembly']
					}
				],
				'roi_analysis': {
					'implementation_cost': 25000.00,
					'payback_period_months': 6,
					'roi_percentage': 400
				}
			}
			mock_response.status_code = 200
			mock_get.return_value = mock_response
			
			# Get cost optimization insights
			insights = await plm_ai_service.get_cost_optimization_insights(test_tenant_id)
			
			# Verify insights
			assert insights is not None
			assert 'total_savings_potential' in insights
			assert 'optimization_opportunities' in insights
			assert 'roi_analysis' in insights
			assert insights['total_savings_potential'] > 0
			assert len(insights['optimization_opportunities']) > 0
	
	async def test_get_supplier_intelligence(
		self,
		plm_ai_service: PLMAIService,
		test_tenant_id: str
	):
		"""Test AI supplier intelligence insights"""
		with patch('httpx.AsyncClient.get') as mock_get:
			# Mock supplier intelligence response
			mock_response = AsyncMock()
			mock_response.json.return_value = {
				'supplier_risk_assessment': {
					'high_risk_suppliers': 2,
					'medium_risk_suppliers': 5,
					'low_risk_suppliers': 18
				},
				'market_insights': {
					'price_trends': {'aluminum': 'increasing', 'steel': 'stable', 'plastic': 'decreasing'},
					'supply_chain_disruptions': ['port_congestion', 'weather_delays'],
					'emerging_suppliers': ['GreenTech Materials', 'Precision Components Inc']
				},
				'recommendations': [
					'Diversify aluminum suppliers to reduce price risk',
					'Consider local suppliers for critical components',
					'Evaluate emerging suppliers for cost advantages'
				]
			}
			mock_response.status_code = 200
			mock_get.return_value = mock_response
			
			# Get supplier intelligence
			intelligence = await plm_ai_service.get_supplier_intelligence_insights(test_tenant_id)
			
			# Verify intelligence
			assert intelligence is not None
			assert 'supplier_risk_assessment' in intelligence
			assert 'market_insights' in intelligence
			assert 'recommendations' in intelligence
			assert intelligence['supplier_risk_assessment']['high_risk_suppliers'] >= 0


# Integration Tests

class TestServiceIntegration:
	"""Test service integration with APG capabilities"""
	
	async def test_end_to_end_product_lifecycle(
		self,
		plm_product_service: PLMProductService,
		plm_change_service: PLMEngineeringChangeService,
		sample_product_data: Dict[str, Any],
		test_tenant_id: str,
		test_user_id: str
	):
		"""Test complete product lifecycle integration"""
		# Create product
		product = await plm_product_service.create_product(
			tenant_id=test_tenant_id,
			product_data=sample_product_data,
			user_id=test_user_id
		)
		assert product is not None
		
		# Create engineering change for the product
		change_data = {
			'change_title': 'Lifecycle Test Change',
			'change_description': 'Test change for end-to-end lifecycle',
			'change_type': 'design',
			'affected_products': [product.product_id],
			'reason_for_change': 'Testing integration',
			'business_impact': 'Improved test coverage'
		}
		
		change = await plm_change_service.create_change(
			tenant_id=test_tenant_id,
			change_data=change_data,
			user_id=test_user_id
		)
		assert change is not None
		
		# Submit and approve change
		submit_success = await plm_change_service.submit_change_for_approval(
			change_id=change.change_id,
			user_id=test_user_id,
			tenant_id=test_tenant_id
		)
		assert submit_success is True
		
		# Update product based on approved change
		update_data = {
			'lifecycle_phase': 'production',
			'current_cost': product.target_cost * 0.95  # 5% cost reduction
		}
		
		updated_product = await plm_product_service.update_product(
			product_id=product.product_id,
			update_data=update_data,
			user_id=test_user_id,
			tenant_id=test_tenant_id
		)
		
		# Verify end-to-end integration
		assert updated_product.lifecycle_phase == 'production'
		assert updated_product.current_cost < product.target_cost
	
	async def test_multi_tenant_isolation(
		self,
		plm_product_service: PLMProductService,
		sample_product_data: Dict[str, Any],
		test_user_id: str
	):
		"""Test multi-tenant isolation across services"""
		tenant1_id = f"tenant_1_{uuid7str()[:8]}"
		tenant2_id = f"tenant_2_{uuid7str()[:8]}"
		
		# Create products in different tenants
		product1_data = sample_product_data.copy()
		product1_data['product_name'] = 'Tenant 1 Product'
		
		product2_data = sample_product_data.copy()
		product2_data['product_name'] = 'Tenant 2 Product'
		
		product1 = await plm_product_service.create_product(
			tenant_id=tenant1_id,
			product_data=product1_data,
			user_id=test_user_id
		)
		
		product2 = await plm_product_service.create_product(
			tenant_id=tenant2_id,
			product_data=product2_data,
			user_id=test_user_id
		)
		
		# Verify tenant isolation - each tenant should only see their own products
		tenant1_products = await plm_product_service.search_products(
			tenant_id=tenant1_id,
			user_id=test_user_id,
			page=1,
			page_size=100
		)
		
		tenant2_products = await plm_product_service.search_products(
			tenant_id=tenant2_id,
			user_id=test_user_id,
			page=1,
			page_size=100
		)
		
		# Verify isolation
		tenant1_product_ids = [p['product_id'] for p in tenant1_products['products']]
		tenant2_product_ids = [p['product_id'] for p in tenant2_products['products']]
		
		assert product1.product_id in tenant1_product_ids
		assert product1.product_id not in tenant2_product_ids
		assert product2.product_id in tenant2_product_ids
		assert product2.product_id not in tenant1_product_ids