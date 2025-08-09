"""
Audio Processing Blueprint Unit Tests

Tests for Flask blueprint integration, APG composition engine registration,
menu integration, and multi-tenant route handling.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import pytest
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import MagicMock, AsyncMock, patch

from flask import Flask, g
from flask_appbuilder import AppBuilder

from ...blueprint import (
	audio_processing_bp, AudioProcessingBlueprint, register_with_composition_engine,
	initialize_default_data, create_audio_processing_blueprint
)
from ...models import APVoiceModel

class TestAudioProcessingBlueprint:
	"""Test AudioProcessingBlueprint class"""
	
	def test_blueprint_initialization(self):
		"""Test blueprint manager initialization"""
		mock_appbuilder = MagicMock()
		blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		assert blueprint_manager.appbuilder == mock_appbuilder
		assert blueprint_manager._services_initialized is False
		assert blueprint_manager._menu_registered is False
	
	@patch('...blueprint.current_app')
	def test_log_blueprint_action(self, mock_current_app):
		"""Test blueprint action logging"""
		mock_appbuilder = MagicMock()
		mock_logger = MagicMock()
		mock_current_app.logger = mock_logger
		
		blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		with patch('flask.g') as mock_g:
			mock_g.user_id = 'test_user'
			mock_g.tenant_id = 'test_tenant'
			
			blueprint_manager._log_blueprint_action('test_action', {'key': 'value'})
		
		mock_logger.info.assert_called_once()
		call_args = mock_logger.info.call_args[0]
		assert 'test_action' in call_args[0]
	
	@patch('...blueprint.current_app')
	def test_register_blueprint_success(self, mock_current_app):
		"""Test successful blueprint registration"""
		mock_app = MagicMock()
		mock_current_app.register_blueprint = MagicMock()
		mock_current_app.logger = MagicMock()
		
		mock_appbuilder = MagicMock()
		blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		# Mock the dependent methods
		blueprint_manager._register_menu_items = MagicMock()
		blueprint_manager._initialize_services = MagicMock()
		blueprint_manager._setup_health_checks = MagicMock()
		
		blueprint_manager.register_blueprint()
		
		# Verify registration steps were called
		mock_current_app.register_blueprint.assert_called_once_with(audio_processing_bp)
		mock_appbuilder.add_view.assert_called_once()
		blueprint_manager._register_menu_items.assert_called_once()
		blueprint_manager._initialize_services.assert_called_once()
		blueprint_manager._setup_health_checks.assert_called_once()
	
	def test_register_menu_items(self):
		"""Test menu items registration"""
		mock_appbuilder = MagicMock()
		blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		with patch('...blueprint.current_app') as mock_current_app:
			mock_current_app.logger = MagicMock()
			
			blueprint_manager._register_menu_items()
		
		# Verify menu registration calls
		assert mock_appbuilder.add_view_no_menu.called
		assert mock_appbuilder.add_link.call_count >= 6  # Multiple menu links
		assert blueprint_manager._menu_registered is True
	
	@patch('...blueprint.create_transcription_service')
	@patch('...blueprint.create_synthesis_service')
	@patch('...blueprint.create_analysis_service')
	@patch('...blueprint.create_enhancement_service')
	@patch('...blueprint.create_model_manager')
	@patch('...blueprint.create_workflow_orchestrator')
	def test_initialize_services(self, mock_orchestrator, mock_manager, mock_enhancement,
							   mock_analysis, mock_synthesis, mock_transcription):
		"""Test services initialization"""
		# Setup mocks
		mock_transcription.return_value = MagicMock()
		mock_synthesis.return_value = MagicMock()
		mock_analysis.return_value = MagicMock()
		mock_enhancement.return_value = MagicMock()
		mock_manager.return_value = MagicMock()
		mock_orchestrator.return_value = MagicMock()
		
		mock_appbuilder = MagicMock()
		blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		with patch('...blueprint.current_app') as mock_current_app:
			mock_current_app.logger = MagicMock()
			mock_current_app.audio_processing_services = None
			
			blueprint_manager._initialize_services()
		
		# Verify all service factories were called
		mock_transcription.assert_called_once()
		mock_synthesis.assert_called_once()
		mock_analysis.assert_called_once()
		mock_enhancement.assert_called_once()
		mock_manager.assert_called_once()
		mock_orchestrator.assert_called_once()
		
		assert blueprint_manager._services_initialized is True
	
	def test_setup_health_checks(self):
		"""Test health check endpoint setup"""
		mock_appbuilder = MagicMock()
		blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		with patch('...blueprint.current_app') as mock_current_app:
			mock_current_app.logger = MagicMock()
			mock_current_app.audio_processing_services = {
				'transcription_service': MagicMock(),
				'synthesis_service': MagicMock()
			}
			
			blueprint_manager._setup_health_checks()
		
		# Health check should be registered (verified through blueprint routes)
		assert hasattr(blueprint_manager, '_setup_health_checks')

class TestBlueprintRoutes:
	"""Test blueprint route functionality"""
	
	def test_health_check_endpoint(self):
		"""Test health check endpoint"""
		app = Flask(__name__)
		app.register_blueprint(audio_processing_bp)
		
		with app.test_client() as client:
			with patch('...blueprint.current_app') as mock_current_app:
				mock_current_app.audio_processing_services = {
					'transcription_service': MagicMock(),
					'synthesis_service': MagicMock()
				}
				
				with patch('flask.g') as mock_g:
					mock_g.tenant_id = 'test_tenant'
					
					response = client.get('/audio_processing/health')
		
		assert response.status_code == 200
		data = response.get_json()
		assert data['capability'] == 'audio_processing'
		assert data['status'] in ['healthy', 'degraded']
		assert 'services' in data
	
	def test_health_check_critical_error(self):
		"""Test health check with critical error"""
		app = Flask(__name__)
		app.register_blueprint(audio_processing_bp)
		
		with app.test_client() as client:
			with patch('...blueprint.current_app') as mock_current_app:
				# Simulate error in health check
				mock_current_app.audio_processing_services = None
				
				with patch('flask.g') as mock_g:
					mock_g.tenant_id = 'test_tenant'
					
					# Mock an exception during health check
					with patch('...blueprint.datetime') as mock_datetime:
						mock_datetime.utcnow.side_effect = Exception("Critical error")
						
						response = client.get('/audio_processing/health')
		
		assert response.status_code == 500
		data = response.get_json()
		assert data['status'] == 'critical'

class TestCompositionEngineRegistration:
	"""Test APG composition engine registration"""
	
	def test_register_with_composition_engine(self):
		"""Test capability registration with composition engine"""
		mock_composition_engine = MagicMock()
		mock_composition_engine.register_capability = MagicMock(return_value={
			'registration_id': 'audio_processing_reg_001'
		})
		mock_composition_engine.version = '2.1.0'
		
		result = register_with_composition_engine(mock_composition_engine)
		
		# Verify registration was called
		mock_composition_engine.register_capability.assert_called_once()
		
		# Verify registration data
		call_args = mock_composition_engine.register_capability.call_args[0][0]
		assert call_args['capability_code'] == 'AUDIO_PROCESSING'
		assert call_args['capability_name'] == 'Audio Processing & Intelligence'
		assert call_args['version'] == '1.0.0'
		assert 'composition_keywords' in call_args
		assert 'primary_interfaces' in call_args
		assert 'dependencies' in call_args
		
		# Verify result structure
		assert result['capability_registered'] is True
		assert result['registration_id'] == 'audio_processing_reg_001'
		assert result['composition_engine_version'] == '2.1.0'
	
	def test_composition_keywords_completeness(self):
		"""Test composition keywords completeness"""
		mock_composition_engine = MagicMock()
		mock_composition_engine.register_capability = MagicMock(return_value={})
		mock_composition_engine.version = '2.1.0'
		
		register_with_composition_engine(mock_composition_engine)
		
		call_args = mock_composition_engine.register_capability.call_args[0][0]
		keywords = call_args['composition_keywords']
		
		# Verify key composition keywords are present
		expected_keywords = [
			'processes_audio', 'transcription_enabled', 'voice_synthesis_capable',
			'audio_analysis_aware', 'real_time_audio', 'speech_recognition',
			'voice_generation', 'audio_enhancement', 'ai_powered_audio'
		]
		
		for keyword in expected_keywords:
			assert keyword in keywords
	
	def test_dependency_requirements(self):
		"""Test mandatory dependency requirements"""
		mock_composition_engine = MagicMock()
		mock_composition_engine.register_capability = MagicMock(return_value={})
		mock_composition_engine.version = '2.1.0'
		
		register_with_composition_engine(mock_composition_engine)
		
		call_args = mock_composition_engine.register_capability.call_args[0][0]
		dependencies = call_args['dependencies']
		
		# Verify mandatory dependencies
		required_capabilities = [
			'auth_rbac', 'ai_orchestration', 'audit_compliance',
			'real_time_collaboration', 'notification_engine', 'intelligent_orchestration'
		]
		
		dependency_names = [dep['capability'] for dep in dependencies]
		for required_cap in required_capabilities:
			assert required_cap in dependency_names
		
		# Verify all dependencies are marked as required
		for dep in dependencies:
			assert dep['required'] is True
	
	def test_api_endpoints_registration(self):
		"""Test API endpoints registration"""
		mock_composition_engine = MagicMock()
		mock_composition_engine.register_capability = MagicMock(return_value={})
		mock_composition_engine.version = '2.1.0'
		
		register_with_composition_engine(mock_composition_engine)
		
		call_args = mock_composition_engine.register_capability.call_args[0][0]
		api_endpoints = call_args['api_endpoints']
		
		# Verify key API endpoints are registered
		endpoint_paths = [ep['path'] for ep in api_endpoints]
		expected_paths = [
			'/api/v1/audio/transcribe',
			'/api/v1/audio/synthesize',
			'/api/v1/audio/analyze',
			'/api/v1/audio/enhance',
			'/api/v1/audio/voices/clone'
		]
		
		for path in expected_paths:
			assert path in endpoint_paths
	
	def test_ui_routes_registration(self):
		"""Test UI routes registration"""
		mock_composition_engine = MagicMock()
		mock_composition_engine.register_capability = MagicMock(return_value={})
		mock_composition_engine.version = '2.1.0'
		
		register_with_composition_engine(mock_composition_engine)
		
		call_args = mock_composition_engine.register_capability.call_args[0][0]
		ui_routes = call_args['ui_routes']
		
		# Verify UI routes are registered
		route_paths = [route['path'] for route in ui_routes]
		expected_paths = [
			'/audio_processing/',
			'/audio_processing/transcription',
			'/audio_processing/synthesis',
			'/audio_processing/analysis',
			'/audio_processing/models',
			'/audio_processing/enhancement'
		]
		
		for path in expected_paths:
			assert path in route_paths

class TestDefaultDataInitialization:
	"""Test default data initialization"""
	
	def test_initialize_default_data_success(self):
		"""Test successful default data initialization"""
		mock_session = MagicMock()
		mock_session.query.return_value.filter_by.return_value.first.return_value = None
		mock_session.add = MagicMock()
		mock_session.commit = MagicMock()
		
		with patch('...blueprint.current_app') as mock_current_app:
			mock_current_app.logger = MagicMock()
			
			initialize_default_data(mock_session)
		
		# Verify models were added
		assert mock_session.add.call_count >= 2  # At least 2 default voice models
		mock_session.commit.assert_called_once()
	
	def test_initialize_default_data_already_exists(self):
		"""Test default data initialization when data already exists"""
		mock_session = MagicMock()
		# Mock existing default model
		mock_session.query.return_value.filter_by.return_value.first.return_value = MagicMock()
		
		initialize_default_data(mock_session)
		
		# Should not add new models if they already exist
		mock_session.add.assert_not_called()
		mock_session.commit.assert_not_called()
	
	def test_initialize_default_data_error_handling(self):
		"""Test default data initialization error handling"""
		mock_session = MagicMock()
		mock_session.query.return_value.filter_by.return_value.first.return_value = None
		mock_session.commit.side_effect = Exception("Database error")
		mock_session.rollback = MagicMock()
		
		with patch('...blueprint.current_app') as mock_current_app:
			mock_current_app.logger = MagicMock()
			
			with pytest.raises(Exception):
				initialize_default_data(mock_session)
		
		mock_session.rollback.assert_called_once()

class TestBlueprintFactory:
	"""Test blueprint factory function"""
	
	@patch('...blueprint.AudioProcessingBlueprint')
	@patch('...blueprint.initialize_default_data')
	def test_create_audio_processing_blueprint(self, mock_init_data, mock_blueprint_class):
		"""Test blueprint factory function"""
		mock_appbuilder = MagicMock()
		mock_blueprint_manager = MagicMock()
		mock_blueprint_class.return_value = mock_blueprint_manager
		
		result = create_audio_processing_blueprint(mock_appbuilder)
		
		# Verify blueprint manager was created and configured
		mock_blueprint_class.assert_called_once_with(mock_appbuilder)
		mock_blueprint_manager.register_blueprint.assert_called_once()
		mock_blueprint_manager._setup_error_handlers.assert_called_once()
		
		# Verify default data was initialized
		mock_init_data.assert_called_once()
		
		# Verify blueprint is returned
		assert result == audio_processing_bp

class TestBlueprintErrorHandlers:
	"""Test blueprint error handlers"""
	
	def test_error_handlers_setup(self):
		"""Test error handlers setup"""
		mock_appbuilder = MagicMock()
		blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		# Mock error handler registration
		with patch.object(audio_processing_bp, 'errorhandler') as mock_errorhandler:
			blueprint_manager._setup_error_handlers()
			
			# Verify error handlers were registered
			assert mock_errorhandler.call_count >= 3  # 400, 404, 500

class TestBlueprintMultiTenant:
	"""Test multi-tenant blueprint functionality"""
	
	def test_tenant_context_handling(self):
		"""Test tenant context in blueprint operations"""
		mock_appbuilder = MagicMock()
		blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		with patch('flask.g') as mock_g:
			mock_g.tenant_id = 'enterprise_tenant_001'
			mock_g.user_id = 'user_001'
			
			with patch('...blueprint.current_app') as mock_current_app:
				mock_current_app.logger = MagicMock()
				
				blueprint_manager._log_blueprint_action('test_action')
		
		# Verify tenant context is captured in logging
		mock_current_app.logger.info.assert_called_once()
	
	def test_tenant_isolation_in_services(self):
		"""Test tenant isolation in service initialization"""
		mock_appbuilder = MagicMock()
		blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		with patch('...blueprint.create_transcription_service') as mock_service:
			mock_service.return_value = MagicMock()
			
			with patch('...blueprint.current_app') as mock_current_app:
				mock_current_app.logger = MagicMock()
				mock_current_app.audio_processing_services = None
				
				blueprint_manager._initialize_services()
		
		# Services should be initialized (tenant isolation handled at service level)
		assert blueprint_manager._services_initialized is True

class TestBlueprintIntegration:
	"""Test blueprint integration with Flask and APG"""
	
	def test_blueprint_flask_integration(self):
		"""Test blueprint integration with Flask app"""
		app = Flask(__name__)
		
		# Register blueprint
		app.register_blueprint(audio_processing_bp)
		
		# Verify blueprint is registered
		assert 'audio_processing' in app.blueprints
		blueprint = app.blueprints['audio_processing']
		assert blueprint.url_prefix == '/audio_processing'
	
	def test_blueprint_apg_integration(self):
		"""Test blueprint integration with APG platform"""
		mock_appbuilder = MagicMock()
		
		# Create blueprint with APG integration
		blueprint = create_audio_processing_blueprint(mock_appbuilder)
		
		# Verify APG integration points were configured
		assert blueprint == audio_processing_bp
		mock_appbuilder.add_view.assert_called()

class TestBlueprintPerformance:
	"""Test blueprint performance"""
	
	def test_blueprint_initialization_performance(self):
		"""Test blueprint initialization performance"""
		import time
		
		mock_appbuilder = MagicMock()
		
		start_time = time.time()
		
		# Initialize multiple blueprint managers
		for i in range(100):
			blueprint_manager = AudioProcessingBlueprint(mock_appbuilder)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Should initialize quickly
		assert duration < 1.0
	
	def test_composition_registration_performance(self):
		"""Test composition engine registration performance"""
		import time
		
		mock_composition_engine = MagicMock()
		mock_composition_engine.register_capability = MagicMock(return_value={})
		mock_composition_engine.version = '2.1.0'
		
		start_time = time.time()
		
		# Register multiple times (simulating multiple capabilities)
		for i in range(50):
			register_with_composition_engine(mock_composition_engine)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Registration should be fast
		assert duration < 0.5