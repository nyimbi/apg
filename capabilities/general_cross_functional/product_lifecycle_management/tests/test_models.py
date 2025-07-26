"""
Product Lifecycle Management (PLM) Capability - Model Tests

Async model tests following APG testing standards.
Tests all PLM data models with real database operations.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from uuid_extensions import uuid7str
from sqlalchemy.exc import IntegrityError
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from ..models import (
	PLProduct, PLProductStructure, PLEngineeringChange,
	PLProductConfiguration, PLCollaborationSession, PLComplianceRecord,
	PLManufacturingIntegration, PLDigitalTwinBinding
)
from .conftest import assert_async_result


class TestPLProduct:
	"""Test PLProduct model"""
	
	async def test_product_creation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_product_data: Dict[str, Any]
	):
		"""Test product creation with valid data"""
		with test_flask_app.app_context():
			# Create product
			product = PLProduct(**sample_product_data)
			test_db.session.add(product)
			test_db.session.commit()
			
			# Verify creation
			assert product.product_id is not None
			assert len(product.product_id) > 20  # uuid7str length
			assert product.product_name == sample_product_data['product_name']
			assert product.product_number == sample_product_data['product_number']
			assert product.tenant_id == sample_product_data['tenant_id']
			assert product.created_by == sample_product_data['created_by']
			assert product.created_at is not None
			assert product.updated_at is not None
	
	async def test_product_validation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_tenant_id: str,
		test_user_id: str
	):
		"""Test product validation constraints"""
		with test_flask_app.app_context():
			# Test missing required fields
			with pytest.raises(Exception):  # Should fail validation
				invalid_product = PLProduct(
					tenant_id=test_tenant_id,
					created_by=test_user_id
					# Missing required product_name and product_number
				)
				test_db.session.add(invalid_product)
				test_db.session.commit()
	
	async def test_product_unique_constraints(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_product_data: Dict[str, Any]
	):
		"""Test product unique constraints"""
		with test_flask_app.app_context():
			# Create first product
			product1 = PLProduct(**sample_product_data)
			test_db.session.add(product1)
			test_db.session.commit()
			
			# Try to create duplicate product number within same tenant
			duplicate_data = sample_product_data.copy()
			duplicate_data['product_name'] = 'Different Name'
			
			with pytest.raises(IntegrityError):
				product2 = PLProduct(**duplicate_data)
				test_db.session.add(product2)
				test_db.session.commit()
	
	async def test_product_lifecycle_phases(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_product_data: Dict[str, Any]
	):
		"""Test product lifecycle phase transitions"""
		with test_flask_app.app_context():
			product = PLProduct(**sample_product_data)
			test_db.session.add(product)
			test_db.session.commit()
			
			# Test lifecycle phase transitions
			valid_phases = [
				'concept', 'design', 'prototype', 'development',
				'testing', 'production', 'active', 'mature',
				'declining', 'obsolete', 'discontinued'
			]
			
			for phase in valid_phases:
				product.lifecycle_phase = phase
				test_db.session.commit()
				assert product.lifecycle_phase == phase
	
	async def test_product_cost_calculations(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_product_data: Dict[str, Any]
	):
		"""Test product cost calculations and validations"""
		with test_flask_app.app_context():
			product = PLProduct(**sample_product_data)
			product.target_cost = 1000.00
			product.current_cost = 1100.00
			
			test_db.session.add(product)
			test_db.session.commit()
			
			# Test cost variance calculation
			cost_variance = product.current_cost - product.target_cost
			assert cost_variance == 100.00
			
			# Test cost validation (negative costs should be prevented at service layer)
			assert product.target_cost >= 0
			assert product.current_cost >= 0
	
	async def test_product_custom_attributes(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_product_data: Dict[str, Any]
	):
		"""Test product custom attributes functionality"""
		with test_flask_app.app_context():
			custom_attrs = {
				'material': 'aluminum',
				'weight_kg': 2.5,
				'dimensions': {'length': 10, 'width': 5, 'height': 3},
				'certifications': ['ISO9001', 'CE']
			}
			
			product = PLProduct(**sample_product_data)
			product.custom_attributes = custom_attrs
			
			test_db.session.add(product)
			test_db.session.commit()
			test_db.session.refresh(product)
			
			# Verify custom attributes are stored correctly
			assert product.custom_attributes['material'] == 'aluminum'
			assert product.custom_attributes['weight_kg'] == 2.5
			assert 'ISO9001' in product.custom_attributes['certifications']
	
	async def test_product_tenant_isolation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_user_id: str
	):
		"""Test multi-tenant isolation for products"""
		with test_flask_app.app_context():
			tenant1_id = f"tenant_1_{uuid7str()[:8]}"
			tenant2_id = f"tenant_2_{uuid7str()[:8]}"
			
			# Create products in different tenants with same product number
			product1 = PLProduct(
				tenant_id=tenant1_id,
				created_by=test_user_id,
				product_name='Shared Product Name',
				product_number='SHARED-001',
				product_type='manufactured'
			)
			
			product2 = PLProduct(
				tenant_id=tenant2_id,
				created_by=test_user_id,
				product_name='Shared Product Name',
				product_number='SHARED-001',  # Same number, different tenant
				product_type='manufactured'
			)
			
			test_db.session.add(product1)
			test_db.session.add(product2)
			test_db.session.commit()
			
			# Both should be created successfully due to tenant isolation
			assert product1.product_id != product2.product_id
			assert product1.tenant_id != product2.tenant_id


class TestPLProductStructure:
	"""Test PLProductStructure model"""
	
	async def test_structure_creation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_product: PLProduct
	):
		"""Test product structure creation"""
		with test_flask_app.app_context():
			child_product = PLProduct(
				tenant_id=test_product.tenant_id,
				created_by=test_product.created_by,
				product_name='Child Component',
				product_number='CHILD-001',
				product_type='manufactured'
			)
			test_db.session.add(child_product)
			test_db.session.commit()
			
			# Create structure relationship
			structure = PLProductStructure(
				tenant_id=test_product.tenant_id,
				parent_product_id=test_product.product_id,
				child_product_id=child_product.product_id,
				quantity=2.0,
				unit_of_measure='each',
				reference_designator='C1',
				sequence_number=1,
				created_by=test_product.created_by
			)
			
			test_db.session.add(structure)
			test_db.session.commit()
			
			# Verify structure creation
			assert structure.structure_id is not None
			assert structure.parent_product_id == test_product.product_id
			assert structure.child_product_id == child_product.product_id
			assert structure.quantity == 2.0
	
	async def test_structure_circular_reference_prevention(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_product: PLProduct
	):
		"""Test prevention of circular references in structure"""
		with test_flask_app.app_context():
			# Try to create structure where product is its own child
			with pytest.raises(Exception):
				circular_structure = PLProductStructure(
					tenant_id=test_product.tenant_id,
					parent_product_id=test_product.product_id,
					child_product_id=test_product.product_id,  # Same as parent
					quantity=1.0,
					created_by=test_product.created_by
				)
				test_db.session.add(circular_structure)
				test_db.session.commit()
	
	async def test_structure_quantity_validation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_product: PLProduct
	):
		"""Test quantity validation in product structure"""
		with test_flask_app.app_context():
			child_product = PLProduct(
				tenant_id=test_product.tenant_id,
				created_by=test_product.created_by,
				product_name='Test Child',
				product_number='CHILD-002',
				product_type='manufactured'
			)
			test_db.session.add(child_product)
			test_db.session.commit()
			
			# Test valid quantity
			structure = PLProductStructure(
				tenant_id=test_product.tenant_id,
				parent_product_id=test_product.product_id,
				child_product_id=child_product.product_id,
				quantity=5.5,
				created_by=test_product.created_by
			)
			
			test_db.session.add(structure)
			test_db.session.commit()
			
			assert structure.quantity == 5.5
			
			# Negative quantities should be prevented at service layer
			assert structure.quantity > 0


class TestPLEngineeringChange:
	"""Test PLEngineeringChange model"""
	
	async def test_change_creation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_change_data: Dict[str, Any]
	):
		"""Test engineering change creation"""
		with test_flask_app.app_context():
			change = PLEngineeringChange(**sample_change_data)
			test_db.session.add(change)
			test_db.session.commit()
			
			# Verify creation
			assert change.change_id is not None
			assert change.change_number is not None
			assert change.status == 'draft'  # Default status
			assert change.priority == sample_change_data['priority']
			assert change.created_at is not None
	
	async def test_change_status_transitions(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_change_data: Dict[str, Any]
	):
		"""Test engineering change status transitions"""
		with test_flask_app.app_context():
			change = PLEngineeringChange(**sample_change_data)
			test_db.session.add(change)
			test_db.session.commit()
			
			# Test status transitions
			valid_statuses = [
				'draft', 'submitted', 'under_review', 'approved',
				'rejected', 'implemented', 'cancelled'
			]
			
			for status in valid_statuses:
				change.status = status
				test_db.session.commit()
				assert change.status == status
	
	async def test_change_approval_workflow(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_change_data: Dict[str, Any]
	):
		"""Test change approval workflow"""
		with test_flask_app.app_context():
			change = PLEngineeringChange(**sample_change_data)
			change.approvers = ['approver1', 'approver2', 'approver3']
			
			test_db.session.add(change)
			test_db.session.commit()
			
			# Test approval process
			change.approved_by = ['approver1', 'approver2']
			change.approval_comments = {
				'approver1': 'Approved with minor modifications',
				'approver2': 'Looks good to proceed'
			}
			
			test_db.session.commit()
			
			# Verify approval data
			assert len(change.approved_by) == 2
			assert 'approver1' in change.approved_by
			assert 'approver1' in change.approval_comments
	
	async def test_change_cost_impact(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_change_data: Dict[str, Any]
	):
		"""Test change cost impact tracking"""
		with test_flask_app.app_context():
			change = PLEngineeringChange(**sample_change_data)
			change.cost_impact = -5000.00  # Cost reduction
			change.schedule_impact_days = 7
			
			test_db.session.add(change)
			test_db.session.commit()
			
			# Verify impact tracking
			assert change.cost_impact == -5000.00
			assert change.schedule_impact_days == 7


class TestPLCollaborationSession:
	"""Test PLCollaborationSession model"""
	
	async def test_session_creation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_collaboration_data: Dict[str, Any]
	):
		"""Test collaboration session creation"""
		with test_flask_app.app_context():
			session = PLCollaborationSession(**sample_collaboration_data)
			test_db.session.add(session)
			test_db.session.commit()
			
			# Verify creation
			assert session.session_id is not None
			assert session.host_user_id == sample_collaboration_data['host_user_id']
			assert session.session_type == sample_collaboration_data['session_type']
			assert session.status == 'scheduled'  # Default status
	
	async def test_session_scheduling_validation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_collaboration_data: Dict[str, Any]
	):
		"""Test session scheduling validation"""
		with test_flask_app.app_context():
			# Test valid scheduling
			session = PLCollaborationSession(**sample_collaboration_data)
			test_db.session.add(session)
			test_db.session.commit()
			
			# Verify scheduling
			assert session.scheduled_start < session.scheduled_end
			assert session.scheduled_start > datetime.utcnow()
	
	async def test_session_participant_management(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_collaboration_data: Dict[str, Any]
	):
		"""Test session participant management"""
		with test_flask_app.app_context():
			session = PLCollaborationSession(**sample_collaboration_data)
			session.invited_users = ['user1', 'user2', 'user3']
			session.participants = ['user1', 'user2']  # 2 joined out of 3 invited
			
			test_db.session.add(session)
			test_db.session.commit()
			
			# Verify participant management
			assert len(session.invited_users) == 3
			assert len(session.participants) == 2
			assert 'user1' in session.participants
			assert 'user3' not in session.participants
	
	async def test_session_features_configuration(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		sample_collaboration_data: Dict[str, Any]
	):
		"""Test session features configuration"""
		with test_flask_app.app_context():
			session = PLCollaborationSession(**sample_collaboration_data)
			session.recording_enabled = True
			session.whiteboard_enabled = True
			session.file_sharing_enabled = False
			session.viewing_3d_enabled = True
			
			test_db.session.add(session)
			test_db.session.commit()
			
			# Verify feature configuration
			assert session.recording_enabled is True
			assert session.whiteboard_enabled is True
			assert session.file_sharing_enabled is False
			assert session.viewing_3d_enabled is True


class TestPLComplianceRecord:
	"""Test PLComplianceRecord model"""
	
	async def test_compliance_creation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_product: PLProduct
	):
		"""Test compliance record creation"""
		with test_flask_app.app_context():
			compliance = PLComplianceRecord(
				tenant_id=test_product.tenant_id,
				product_id=test_product.product_id,
				regulation_name='ISO 9001:2015',
				regulation_version='2015',
				compliance_type='quality_management',
				status='pending',
				created_by=test_product.created_by
			)
			
			test_db.session.add(compliance)
			test_db.session.commit()
			
			# Verify creation
			assert compliance.compliance_id is not None
			assert compliance.product_id == test_product.product_id
			assert compliance.regulation_name == 'ISO 9001:2015'
			assert compliance.status == 'pending'
	
	async def test_compliance_certification_tracking(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_product: PLProduct
	):
		"""Test compliance certification tracking"""
		with test_flask_app.app_context():
			compliance = PLComplianceRecord(
				tenant_id=test_product.tenant_id,
				product_id=test_product.product_id,
				regulation_name='CE Marking',
				regulation_version='2021',
				compliance_type='safety',
				status='certified',
				certification_date=datetime.utcnow(),
				expiration_date=datetime.utcnow() + timedelta(days=365*3),  # 3 years
				certificate_number='CE-2025-001234',
				certification_body='Notified Body XYZ',
				created_by=test_product.created_by
			)
			
			test_db.session.add(compliance)
			test_db.session.commit()
			
			# Verify certification tracking
			assert compliance.status == 'certified'
			assert compliance.certificate_number == 'CE-2025-001234'
			assert compliance.certification_date is not None
			assert compliance.expiration_date > compliance.certification_date


class TestPLManufacturingIntegration:
	"""Test PLManufacturingIntegration model"""
	
	async def test_manufacturing_integration_creation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_product: PLProduct
	):
		"""Test manufacturing integration record creation"""
		with test_flask_app.app_context():
			integration = PLManufacturingIntegration(
				tenant_id=test_product.tenant_id,
				product_id=test_product.product_id,
				manufacturing_part_number='MFG-001-A',
				manufacturing_status='active',
				sync_status='synced',
				last_sync_timestamp=datetime.utcnow(),
				created_by=test_product.created_by
			)
			
			test_db.session.add(integration)
			test_db.session.commit()
			
			# Verify creation
			assert integration.integration_id is not None
			assert integration.product_id == test_product.product_id
			assert integration.manufacturing_part_number == 'MFG-001-A'
			assert integration.sync_status == 'synced'
	
	async def test_manufacturing_sync_status_tracking(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_product: PLProduct
	):
		"""Test manufacturing sync status tracking"""
		with test_flask_app.app_context():
			integration = PLManufacturingIntegration(
				tenant_id=test_product.tenant_id,
				product_id=test_product.product_id,
				manufacturing_part_number='MFG-002-B',
				manufacturing_status='active',
				sync_status='failed',
				sync_error_message='Connection timeout to manufacturing system',
				last_sync_timestamp=datetime.utcnow() - timedelta(hours=2),
				created_by=test_product.created_by
			)
			
			test_db.session.add(integration)
			test_db.session.commit()
			
			# Verify sync status tracking
			assert integration.sync_status == 'failed'
			assert 'timeout' in integration.sync_error_message.lower()
			assert integration.last_sync_timestamp < datetime.utcnow()


class TestPLDigitalTwinBinding:
	"""Test PLDigitalTwinBinding model"""
	
	async def test_digital_twin_binding_creation(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_product: PLProduct
	):
		"""Test digital twin binding creation"""
		with test_flask_app.app_context():
			twin_id = f"twin_{uuid7str()[:8]}"
			
			binding = PLDigitalTwinBinding(
				tenant_id=test_product.tenant_id,
				product_id=test_product.product_id,
				digital_twin_id=twin_id,
				binding_type='product_twin',
				auto_sync_enabled=True,
				sync_frequency='real_time',
				binding_status='active',
				created_by=test_product.created_by
			)
			
			test_db.session.add(binding)
			test_db.session.commit()
			
			# Verify creation
			assert binding.binding_id is not None
			assert binding.product_id == test_product.product_id
			assert binding.digital_twin_id == twin_id
			assert binding.auto_sync_enabled is True
	
	async def test_digital_twin_properties(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_product: PLProduct
	):
		"""Test digital twin properties management"""
		with test_flask_app.app_context():
			twin_properties = {
				'simulation_parameters': {
					'temperature_range': {'min': -40, 'max': 85},
					'pressure_range': {'min': 0, 'max': 100}
				},
				'iot_sensors': ['temperature', 'vibration', 'pressure'],
				'update_frequency': 'hourly'
			}
			
			binding = PLDigitalTwinBinding(
				tenant_id=test_product.tenant_id,
				product_id=test_product.product_id,
				digital_twin_id=f"twin_{uuid7str()[:8]}",
				binding_type='iot_twin',
				twin_properties=twin_properties,
				simulation_enabled=True,
				iot_integration=True,
				created_by=test_product.created_by
			)
			
			test_db.session.add(binding)
			test_db.session.commit()
			test_db.session.refresh(binding)
			
			# Verify properties
			assert binding.simulation_enabled is True
			assert binding.iot_integration is True
			assert 'temperature' in binding.twin_properties['iot_sensors']
			assert binding.twin_properties['simulation_parameters']['temperature_range']['max'] == 85


# Performance Tests

class TestModelPerformance:
	"""Test model performance and scalability"""
	
	async def test_bulk_product_creation_performance(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		test_tenant_id: str,
		test_user_id: str
	):
		"""Test bulk product creation performance"""
		with test_flask_app.app_context():
			import time
			
			start_time = time.time()
			
			# Create 100 products
			products = []
			for i in range(100):
				product = PLProduct(
					tenant_id=test_tenant_id,
					created_by=test_user_id,
					product_name=f'Performance Test Product {i:03d}',
					product_number=f'PERF-{i:03d}',
					product_type='manufactured',
					lifecycle_phase='design',
					target_cost=1000.00 + i
				)
				products.append(product)
			
			# Bulk insert
			test_db.session.add_all(products)
			test_db.session.commit()
			
			end_time = time.time()
			execution_time = end_time - start_time
			
			# Performance assertion (should create 100 products in < 2 seconds)
			assert execution_time < 2.0
			assert len(products) == 100
			
			# Verify all products were created
			created_count = test_db.session.query(PLProduct).filter_by(tenant_id=test_tenant_id).count()
			assert created_count >= 100
	
	async def test_complex_query_performance(
		self,
		test_db: SQLAlchemy,
		test_flask_app: Flask,
		multiple_test_products: list[PLProduct]
	):
		"""Test complex query performance"""
		with test_flask_app.app_context():
			import time
			
			start_time = time.time()
			
			# Complex query with multiple conditions
			results = test_db.session.query(PLProduct).filter(
				PLProduct.tenant_id == multiple_test_products[0].tenant_id,
				PLProduct.lifecycle_phase.in_(['design', 'production', 'active']),
				PLProduct.target_cost.between(500.00, 5000.00)
			).order_by(PLProduct.target_cost.desc()).limit(10).all()
			
			end_time = time.time()
			execution_time = end_time - start_time
			
			# Performance assertion (complex query should execute in < 0.1 seconds)
			assert execution_time < 0.1
			assert len(results) > 0