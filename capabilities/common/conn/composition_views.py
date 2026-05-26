"""
APG Connection Management Capability Composition Views
Flask-AppBuilder views for capability composition management

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import BaseView, ModelView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import FormWidget
from wtforms import Form, StringField, SelectField, TextAreaField, validators
from wtforms.widgets import TextArea
import json
from typing import Dict, Any, List

from .composition_api import (
	ConnectionCapabilityComposer,
	CapabilityInterface,
	CompositionContract,
	CapabilityType,
	IntegrationMethod,
	CapabilityEvent
)
from .service_bridge import with_service_bridge


class JSONTextAreaWidget(TextArea):
	"""Custom widget for JSON input fields"""

	def __call__(self, field, **kwargs):
		kwargs.setdefault('rows', 10)
		kwargs.setdefault('cols', 50)
		kwargs.setdefault('class_', 'form-control json-editor')
		return super().__call__(field, **kwargs)


class JSONTextAreaField(TextAreaField):
	"""Custom field for JSON input with validation"""
	widget = JSONTextAreaWidget()

	def process_formdata(self, valuelist):
		if valuelist:
			try:
				self.data = json.loads(valuelist[0]) if valuelist[0] else {}
			except json.JSONDecodeError:
				self.data = valuelist[0]
				raise validators.ValidationError('Invalid JSON format')
		else:
			self.data = {}


class CapabilityRegistrationForm(Form):
	"""Form for registering new capabilities"""
	name = StringField('Name', [validators.DataRequired(), validators.Length(min=3, max=100)])
	version = StringField('Version', [validators.DataRequired(), validators.Length(min=1, max=20)])
	capability_type = SelectField('Type', choices=[(t.value, t.value) for t in CapabilityType])
	supported_methods = SelectField('Integration Methods',
		choices=[(m.value, m.value) for m in IntegrationMethod],
		render_kw={'multiple': True}
	)
	endpoints = JSONTextAreaField('Endpoints (JSON)')
	event_types = JSONTextAreaField('Event Types (JSON Array)')
	data_formats = JSONTextAreaField('Data Formats (JSON Array)')
	requirements = JSONTextAreaField('Requirements (JSON)')
	metadata = JSONTextAreaField('Metadata (JSON)')


class CompositionContractForm(Form):
	"""Form for creating composition contracts"""
	source_capability = StringField('Source Capability', [validators.DataRequired()])
	target_capability = StringField('Target Capability', [validators.DataRequired()])
	integration_method = SelectField('Integration Method',
		choices=[(m.value, m.value) for m in IntegrationMethod])
	data_flow_direction = SelectField('Data Flow Direction', choices=[
		('bidirectional', 'Bidirectional'),
		('source_to_target', 'Source to Target'),
		('target_to_source', 'Target to Source')
	])
	event_mappings = JSONTextAreaField('Event Mappings (JSON)')
	data_transformations = JSONTextAreaField('Data Transformations (JSON Array)')
	validation_rules = JSONTextAreaField('Validation Rules (JSON Array)')
	error_handling = JSONTextAreaField('Error Handling (JSON)')
	performance_requirements = JSONTextAreaField('Performance Requirements (JSON)')


class CapabilityCompositionView(BaseView):
	"""Main view for capability composition management"""

	route_base = "/composition"
	default_view = "dashboard"

	@expose("/")
	@expose("/dashboard")
	@has_access
	@with_service_bridge
	def dashboard(self, service_bridge=None):
		"""Capability composition dashboard"""
		try:
			# Get composition statistics
			registered_capabilities = []
			active_compositions = []

			if hasattr(service_bridge, 'connection_manager'):
				composer = ConnectionCapabilityComposer(
					service_bridge.connection_manager,
					service_bridge.tenant_id
				)
				registered_capabilities = service_bridge.run_async(
					composer.get_registered_capabilities()
				)
				active_compositions = service_bridge.run_async(
					composer.get_active_compositions()
				)

			stats = {
				'registered_capabilities_count': len(registered_capabilities),
				'active_compositions_count': len(active_compositions),
				'integration_methods': [method.value for method in IntegrationMethod],
				'capability_types': [ctype.value for ctype in CapabilityType]
			}

			return self.render_template(
				"composition_dashboard.html",
				stats=stats,
				registered_capabilities=registered_capabilities,
				active_compositions=active_compositions
			)
		except Exception as e:
			flash(f"Error loading composition dashboard: {str(e)}", "error")
			return self.render_template("composition_dashboard.html", stats={})

	@expose("/register", methods=["GET", "POST"])
	@has_access
	@with_service_bridge
	def register_capability(self, service_bridge=None):
		"""Register a new capability for composition"""
		form = CapabilityRegistrationForm(request.form)

		if request.method == "POST" and form.validate():
			try:
				# Create capability interface
				interface_data = {
					'name': form.name.data,
					'version': form.version.data,
					'capability_type': CapabilityType(form.capability_type.data),
					'supported_methods': [IntegrationMethod(form.supported_methods.data)],
					'endpoints': form.endpoints.data if isinstance(form.endpoints.data, dict) else {},
					'event_types': form.event_types.data if isinstance(form.event_types.data, list) else [],
					'data_formats': form.data_formats.data if isinstance(form.data_formats.data, list) else [],
					'requirements': form.requirements.data if isinstance(form.requirements.data, dict) else {},
					'metadata': form.metadata.data if isinstance(form.metadata.data, dict) else {}
				}

				interface = CapabilityInterface(**interface_data)

				# Register with composer
				if hasattr(service_bridge, 'connection_manager'):
					composer = ConnectionCapabilityComposer(
						service_bridge.connection_manager,
						service_bridge.tenant_id
					)
					success = service_bridge.run_async(composer.register_capability(interface))

					if success:
						flash(f"Capability '{form.name.data}' registered successfully", "success")
						return redirect(url_for("CapabilityCompositionView.dashboard"))
					else:
						flash("Failed to register capability", "error")
				else:
					flash("Service bridge not available", "error")

			except Exception as e:
				flash(f"Error registering capability: {str(e)}", "error")

		return self.render_template(
			"register_capability.html",
			form=form,
			capability_types=[(t.value, t.value) for t in CapabilityType],
			integration_methods=[(m.value, m.value) for m in IntegrationMethod]
		)

	@expose("/compose", methods=["GET", "POST"])
	@has_access
	@with_service_bridge
	def create_composition(self, service_bridge=None):
		"""Create a new capability composition"""
		form = CompositionContractForm(request.form)

		# Get available capabilities for dropdowns
		available_capabilities = []
		if hasattr(service_bridge, 'connection_manager'):
			try:
				composer = ConnectionCapabilityComposer(
					service_bridge.connection_manager,
					service_bridge.tenant_id
				)
				registered = service_bridge.run_async(composer.get_registered_capabilities())
				available_capabilities = [(cap.capability_id, cap.name) for cap in registered]
			except Exception as e:
				flash(f"Error loading capabilities: {str(e)}", "warning")

		if request.method == "POST" and form.validate():
			try:
				# Create composition contract
				contract_data = {
					'source_capability': form.source_capability.data,
					'target_capability': form.target_capability.data,
					'integration_method': IntegrationMethod(form.integration_method.data),
					'data_flow_direction': form.data_flow_direction.data,
					'event_mappings': form.event_mappings.data if isinstance(form.event_mappings.data, dict) else {},
					'data_transformations': form.data_transformations.data if isinstance(form.data_transformations.data, list) else [],
					'validation_rules': form.validation_rules.data if isinstance(form.validation_rules.data, list) else [],
					'error_handling': form.error_handling.data if isinstance(form.error_handling.data, dict) else {},
					'performance_requirements': form.performance_requirements.data if isinstance(form.performance_requirements.data, dict) else {}
				}

				contract = CompositionContract(**contract_data)

				# Create composition
				if hasattr(service_bridge, 'connection_manager'):
					composer = ConnectionCapabilityComposer(
						service_bridge.connection_manager,
						service_bridge.tenant_id
					)
					composition_id = service_bridge.run_async(composer.create_composition(contract))

					flash(f"Composition created successfully: {composition_id}", "success")
					return redirect(url_for("CapabilityCompositionView.dashboard"))
				else:
					flash("Service bridge not available", "error")

			except Exception as e:
				flash(f"Error creating composition: {str(e)}", "error")

		return self.render_template(
			"create_composition.html",
			form=form,
			available_capabilities=available_capabilities,
			integration_methods=[(m.value, m.value) for m in IntegrationMethod]
		)

	@expose("/api/capabilities")
	@has_access
	@with_service_bridge
	def api_list_capabilities(self, service_bridge=None):
		"""API endpoint to list registered capabilities"""
		try:
			if hasattr(service_bridge, 'connection_manager'):
				composer = ConnectionCapabilityComposer(
					service_bridge.connection_manager,
					service_bridge.tenant_id
				)
				capabilities = service_bridge.run_async(composer.get_registered_capabilities())

				return jsonify({
					'status': 'success',
					'data': [cap.model_dump() for cap in capabilities]
				})
			else:
				return jsonify({
					'status': 'error',
					'message': 'Service bridge not available'
				}), 500
		except Exception as e:
			return jsonify({
				'status': 'error',
				'message': str(e)
			}), 500

	@expose("/api/compositions")
	@has_access
	@with_service_bridge
	def api_list_compositions(self, service_bridge=None):
		"""API endpoint to list active compositions"""
		try:
			if hasattr(service_bridge, 'connection_manager'):
				composer = ConnectionCapabilityComposer(
					service_bridge.connection_manager,
					service_bridge.tenant_id
				)
				compositions = service_bridge.run_async(composer.get_active_compositions())

				return jsonify({
					'status': 'success',
					'data': [comp.model_dump() for comp in compositions]
				})
			else:
				return jsonify({
					'status': 'error',
					'message': 'Service bridge not available'
				}), 500
		except Exception as e:
			return jsonify({
				'status': 'error',
				'message': str(e)
			}), 500

	@expose("/api/execute", methods=["POST"])
	@has_access
	@with_service_bridge
	def api_execute_composition(self, service_bridge=None):
		"""API endpoint to execute a capability composition"""
		try:
			data = request.get_json()
			if not data:
				return jsonify({
					'status': 'error',
					'message': 'No JSON data provided'
				}), 400

			composition_id = data.get('composition_id')
			event_data = data.get('event')

			if not composition_id or not event_data:
				return jsonify({
					'status': 'error',
					'message': 'composition_id and event are required'
				}), 400

			# Create event object
			event = CapabilityEvent(**event_data)

			if hasattr(service_bridge, 'connection_manager'):
				composer = ConnectionCapabilityComposer(
					service_bridge.connection_manager,
					service_bridge.tenant_id
				)
				result = service_bridge.run_async(
					composer.execute_composition(composition_id, event)
				)

				return jsonify({
					'status': 'success',
					'data': result
				})
			else:
				return jsonify({
					'status': 'error',
					'message': 'Service bridge not available'
				}), 500

		except Exception as e:
			return jsonify({
				'status': 'error',
				'message': str(e)
			}), 500

	@expose("/api/validate", methods=["POST"])
	@has_access
	@with_service_bridge
	def api_validate_composition(self, service_bridge=None):
		"""API endpoint to validate a composition contract"""
		try:
			data = request.get_json()
			if not data:
				return jsonify({
					'status': 'error',
					'message': 'No JSON data provided'
				}), 400

			# Create contract object
			contract = CompositionContract(**data)

			if hasattr(service_bridge, 'connection_manager'):
				composer = ConnectionCapabilityComposer(
					service_bridge.connection_manager,
					service_bridge.tenant_id
				)
				errors = service_bridge.run_async(composer.validate_composition(contract))

				return jsonify({
					'status': 'success',
					'valid': len(errors) == 0,
					'errors': errors
				})
			else:
				return jsonify({
					'status': 'error',
					'message': 'Service bridge not available'
				}), 500

		except Exception as e:
			return jsonify({
				'status': 'error',
				'message': str(e)
			}), 500

	@expose("/api/interface/<capability_id>")
	@has_access
	@with_service_bridge
	def api_get_capability_interface(self, capability_id, service_bridge=None):
		"""API endpoint to get a specific capability interface"""
		try:
			if hasattr(service_bridge, 'connection_manager'):
				composer = ConnectionCapabilityComposer(
					service_bridge.connection_manager,
					service_bridge.tenant_id
				)
				capabilities = service_bridge.run_async(composer.get_registered_capabilities())

				for cap in capabilities:
					if cap.capability_id == capability_id:
						return jsonify({
							'status': 'success',
							'data': cap.model_dump()
						})

				return jsonify({
					'status': 'error',
					'message': 'Capability not found'
				}), 404
			else:
				return jsonify({
					'status': 'error',
					'message': 'Service bridge not available'
				}), 500
		except Exception as e:
			return jsonify({
				'status': 'error',
				'message': str(e)
			}), 500

	@expose("/monitor")
	@has_access
	@with_service_bridge
	def composition_monitor(self, service_bridge=None):
		"""Real-time composition monitoring dashboard"""
		try:
			# Get composition statistics and health metrics
			if hasattr(service_bridge, 'connection_manager'):
				composer = ConnectionCapabilityComposer(
					service_bridge.connection_manager,
					service_bridge.tenant_id
				)

				compositions = service_bridge.run_async(composer.get_active_compositions())
				capabilities = service_bridge.run_async(composer.get_registered_capabilities())

				# Generate monitoring data
				monitor_data = {
					'total_compositions': len(compositions),
					'total_capabilities': len(capabilities),
					'composition_types': {},
					'integration_methods': {},
					'capability_types': {}
				}

				# Analyze composition patterns
				for comp in compositions:
					method = comp.integration_method.value
					monitor_data['integration_methods'][method] = monitor_data['integration_methods'].get(method, 0) + 1

				for cap in capabilities:
					ctype = cap.capability_type.value
					monitor_data['capability_types'][ctype] = monitor_data['capability_types'].get(ctype, 0) + 1

				return self.render_template(
					"composition_monitor.html",
					monitor_data=monitor_data,
					compositions=compositions,
					capabilities=capabilities
				)
			else:
				flash("Service bridge not available", "error")
				return self.render_template("composition_monitor.html", monitor_data={})

		except Exception as e:
			flash(f"Error loading composition monitor: {str(e)}", "error")
			return self.render_template("composition_monitor.html", monitor_data={})


class CapabilityTestView(BaseView):
	"""View for testing capability compositions"""

	route_base = "/composition/test"

	@expose("/")
	@has_access
	@with_service_bridge
	def test_dashboard(self, service_bridge=None):
		"""Capability testing dashboard"""
		return self.render_template("composition_test.html")

	@expose("/api/test-event", methods=["POST"])
	@has_access
	@with_service_bridge
	def api_test_event(self, service_bridge=None):
		"""Test API for capability event handling"""
		try:
			data = request.get_json()
			if not data:
				return jsonify({'status': 'error', 'message': 'No data provided'}), 400

			# Simulate event processing
			event_type = data.get('event_type')
			connection_id = data.get('connection_id', 'test-connection')
			event_data = data.get('event_data', {})

			if hasattr(service_bridge, 'connection_manager'):
				composer = ConnectionCapabilityComposer(
					service_bridge.connection_manager,
					service_bridge.tenant_id
				)

				# Handle the test event
				service_bridge.run_async(
					composer.handle_connection_event(event_type, connection_id, event_data)
				)

				return jsonify({
					'status': 'success',
					'message': f'Event {event_type} processed successfully'
				})
			else:
				return jsonify({
					'status': 'error',
					'message': 'Service bridge not available'
				}), 500

		except Exception as e:
			return jsonify({
				'status': 'error',
				'message': str(e)
			}), 500