#!/usr/bin/env python3
"""
IoT Management Flask-AppBuilder Blueprint
=========================================

Rich web interface for IoT device management, sensor monitoring,
and real-time data visualization.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.views import ModelView, SimpleFormView
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, BS3TextAreaFieldWidget
from wtforms import StringField, TextAreaField, SelectField, IntegerField, FloatField, BooleanField, HiddenField
from wtforms.validators import DataRequired, Optional as OptionalValidator, NumberRange
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
import plotly.graph_objs as go
import plotly.utils

# Import our IoT management capability
from capabilities.iot_management import (
	IoTManagementCapability,
	DeviceType,
	SensorType,
	ConnectionType,
	DeviceStatus
)

Base = declarative_base()

class DeviceRegistrationForm(DynamicForm):
	"""Form for registering new IoT devices"""
	
	device_id = StringField(
		'Device ID',
		validators=[DataRequired()],
		description='Unique identifier for the device'
	)
	
	name = StringField(
		'Device Name',
		validators=[DataRequired()],
		description='Human-readable name for the device'
	)
	
	device_type = SelectField(
		'Device Type',
		choices=[(dt.value, dt.value.replace('_', ' ').title()) for dt in DeviceType],
		validators=[DataRequired()],
		description='Type of IoT device'
	)
	
	manufacturer = StringField(
		'Manufacturer',
		validators=[OptionalValidator()],
		description='Device manufacturer'
	)
	
	model = StringField(
		'Model',
		validators=[OptionalValidator()],
		description='Device model number'
	)
	
	ip_address = StringField(
		'IP Address',
		validators=[OptionalValidator()],
		description='Device IP address'
	)
	
	connection_type = SelectField(
		'Connection Type',
		choices=[(ct.value, ct.value.replace('_', ' ').title()) for ct in ConnectionType],
		default='wifi',
		description='How the device connects to the network'
	)
	
	location_name = StringField(
		'Location Name',
		validators=[OptionalValidator()],
		description='Physical location of the device'
	)
	
	location_lat = FloatField(
		'Latitude',
		validators=[OptionalValidator(), NumberRange(-90, 90)],
		description='GPS latitude coordinate'
	)
	
	location_lng = FloatField(
		'Longitude',
		validators=[OptionalValidator(), NumberRange(-180, 180)],
		description='GPS longitude coordinate'
	)
	
	sensors = SelectField(
		'Primary Sensor Type',
		choices=[(st.value, st.value.replace('_', ' ').title()) for st in SensorType],
		validators=[OptionalValidator()],
		description='Primary sensor type for this device'
	)

class SensorDataForm(DynamicForm):
	"""Form for recording sensor data"""
	
	device_id = SelectField(
		'Device',
		validators=[DataRequired()],
		description='Select the device'
	)
	
	sensor_id = StringField(
		'Sensor ID',
		validators=[DataRequired()],
		description='Unique sensor identifier'
	)
	
	sensor_type = SelectField(
		'Sensor Type',
		choices=[(st.value, st.value.replace('_', ' ').title()) for st in SensorType],
		validators=[DataRequired()],
		description='Type of sensor reading'
	)
	
	value = FloatField(
		'Sensor Value',
		validators=[DataRequired()],
		description='Sensor reading value'
	)
	
	unit = StringField(
		'Unit',
		validators=[DataRequired()],
		description='Unit of measurement (e.g., °C, %, ppm)'
	)
	
	quality = FloatField(
		'Quality Score',
		default=1.0,
		validators=[OptionalValidator(), NumberRange(0, 1)],
		description='Quality score from 0 to 1'
	)

class DeviceCommandForm(DynamicForm):
	"""Form for sending commands to devices"""
	
	device_id = SelectField(
		'Device',
		validators=[DataRequired()],
		description='Select the target device'
	)
	
	command = StringField(
		'Command',
		validators=[DataRequired()],
		description='Command to send to device'
	)
	
	parameters = TextAreaField(
		'Parameters (JSON)',
		validators=[OptionalValidator()],
		description='Command parameters in JSON format'
	)
	
	priority = IntegerField(
		'Priority',
		default=1,
		validators=[OptionalValidator(), NumberRange(1, 10)],
		description='Command priority (1-10, higher = more urgent)'
	)

class AlertRuleForm(DynamicForm):
	"""Form for creating alert rules"""
	
	name = StringField(
		'Rule Name',
		validators=[DataRequired()],
		description='Name for this alert rule'
	)
	
	description = TextAreaField(
		'Description',
		validators=[OptionalValidator()],
		description='Description of what this rule monitors'
	)
	
	device_id = SelectField(
		'Device',
		validators=[OptionalValidator()],
		description='Specific device to monitor (leave blank for all devices)'
	)
	
	sensor_type = SelectField(
		'Sensor Type',
		choices=[('', 'Any Sensor')] + [(st.value, st.value.replace('_', ' ').title()) for st in SensorType],
		validators=[OptionalValidator()],
		description='Type of sensor to monitor'
	)
	
	condition = StringField(
		'Condition',
		validators=[DataRequired()],
		description='Alert condition (e.g., "value > 30", "status == \'offline\'")'
	)
	
	threshold_value = FloatField(
		'Threshold Value',
		validators=[OptionalValidator()],
		description='Threshold value for numeric comparisons'
	)
	
	action = SelectField(
		'Action',
		choices=[
			('log', 'Log Alert'),
			('email', 'Send Email'),
			('webhook', 'Call Webhook'),
			('command', 'Send Device Command')
		],
		default='log',
		description='Action to take when alert is triggered'
	)
	
	enabled = BooleanField(
		'Enabled',
		default=True,
		description='Whether this rule is active'
	)

class IoTManagementView(BaseView):
	"""Main IoT management interface"""
	
	route_base = '/iot_management'
	default_view = 'dashboard'
	
	def __init__(self):
		super().__init__()
		self.iot_capability = IoTManagementCapability()
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""IoT management dashboard"""
		
		# Get device statistics
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		devices_result = loop.run_until_complete(
			self.iot_capability.list_devices()
		)
		
		loop.close()
		
		devices = devices_result.get('devices', [])
		
		# Calculate statistics
		stats = self._calculate_dashboard_stats(devices)
		
		# Get recent sensor data for charts
		chart_data = self._get_dashboard_charts(devices)
		
		return self.render_template(
			'iot_management/dashboard.html',
			stats=stats,
			devices=devices[:10],  # Show last 10 devices
			chart_data=chart_data
		)
	
	@expose('/devices')
	@has_access
	def devices_list(self):
		"""List all IoT devices"""
		
		# Get device filter parameters
		device_type = request.args.get('type')
		status = request.args.get('status')
		
		# Fetch devices
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		devices_result = loop.run_until_complete(
			self.iot_capability.list_devices(device_type, status)
		)
		
		loop.close()
		
		devices = devices_result.get('devices', [])
		
		return self.render_template(
			'iot_management/devices_list.html',
			devices=devices,
			device_types=DeviceType,
			device_statuses=DeviceStatus,
			current_filters={'type': device_type, 'status': status}
		)
	
	@expose('/devices/<device_id>')
	@has_access
	def device_detail(self, device_id):
		"""Device detail view"""
		
		# Get device information
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		device_result = loop.run_until_complete(
			self.iot_capability.get_device_info(device_id)
		)
		
		# Get recent sensor readings
		readings_result = loop.run_until_complete(
			self.iot_capability.get_sensor_readings(device_id, hours_back=24)
		)
		
		loop.close()
		
		if not device_result.get('success'):
			flash('Device not found', 'danger')
			return redirect(url_for('IoTManagementView.devices_list'))
		
		device = device_result['device']
		readings = readings_result.get('readings', [])
		
		# Create sensor data charts
		sensor_charts = self._create_sensor_charts(readings)
		
		return self.render_template(
			'iot_management/device_detail.html',
			device=device,
			readings=readings[:50],  # Show last 50 readings
			sensor_charts=sensor_charts
		)
	
	@expose('/register_device', methods=['GET', 'POST'])
	@has_access
	def register_device(self):
		"""Register new IoT device"""
		
		form = DeviceRegistrationForm()
		
		if form.validate_on_submit():
			try:
				# Prepare device data
				device_data = {
					'device_id': form.device_id.data,
					'name': form.name.data,
					'device_type': form.device_type.data,
					'manufacturer': form.manufacturer.data or '',
					'model': form.model.data or '',
					'ip_address': form.ip_address.data or '',
					'connection_type': form.connection_type.data,
					'location': {},
					'sensors': []
				}
				
				# Add location if provided
				if form.location_name.data:
					device_data['location']['name'] = form.location_name.data
				if form.location_lat.data is not None:
					device_data['location']['latitude'] = form.location_lat.data
				if form.location_lng.data is not None:
					device_data['location']['longitude'] = form.location_lng.data
				
				# Add sensor if specified
				if form.sensors.data:
					device_data['sensors'] = [form.sensors.data]
				
				# Register device
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				result = loop.run_until_complete(
					self.iot_capability.register_device(device_data)
				)
				
				loop.close()
				
				if result.get('success'):
					flash(f"Device '{form.name.data}' registered successfully!", 'success')
					return redirect(url_for('IoTManagementView.device_detail', device_id=form.device_id.data))
				else:
					flash(f"Failed to register device: {result.get('error', 'Unknown error')}", 'danger')
			
			except Exception as e:
				flash(f"Error registering device: {str(e)}", 'danger')
		
		return self.render_template(
			'iot_management/register_device.html',
			form=form
		)
	
	@expose('/sensor_data', methods=['GET', 'POST'])
	@has_access
	def sensor_data(self):
		"""Record sensor data"""
		
		# Get device list for form
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		devices_result = loop.run_until_complete(
			self.iot_capability.list_devices()
		)
		
		devices = devices_result.get('devices', [])
		device_choices = [(d['device_id'], f"{d['name']} ({d['device_id']})") for d in devices]
		
		form = SensorDataForm()
		form.device_id.choices = device_choices
		
		if form.validate_on_submit():
			try:
				# Prepare sensor data
				sensor_data = {
					'sensor_id': f"{form.device_id.data}_{form.sensor_id.data}",
					'sensor_type': form.sensor_type.data,
					'value': form.value.data,
					'unit': form.unit.data,
					'quality': form.quality.data,
					'metadata': {}
				}
				
				# Record sensor data
				result = loop.run_until_complete(
					self.iot_capability.record_sensor_data(sensor_data)
				)
				
				if result.get('success'):
					flash('Sensor data recorded successfully!', 'success')
					form = SensorDataForm()  # Reset form
					form.device_id.choices = device_choices
				else:
					flash(f"Failed to record sensor data: {result.get('error', 'Unknown error')}", 'danger')
			
			except Exception as e:
				flash(f"Error recording sensor data: {str(e)}", 'danger')
		
		loop.close()
		
		return self.render_template(
			'iot_management/sensor_data.html',
			form=form
		)
	
	@expose('/device_commands', methods=['GET', 'POST'])
	@has_access
	def device_commands(self):
		"""Send commands to devices"""
		
		# Get device list for form
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		devices_result = loop.run_until_complete(
			self.iot_capability.list_devices()
		)
		
		devices = devices_result.get('devices', [])
		device_choices = [(d['device_id'], f"{d['name']} ({d['device_id']})") for d in devices]
		
		form = DeviceCommandForm()
		form.device_id.choices = device_choices
		
		if form.validate_on_submit():
			try:
				# Parse parameters JSON
				parameters = {}
				if form.parameters.data:
					parameters = json.loads(form.parameters.data)
				
				# Send command
				result = loop.run_until_complete(
					self.iot_capability.send_device_command(
						form.device_id.data,
						form.command.data,
						parameters
					)
				)
				
				if result.get('success'):
					flash(f"Command sent successfully! Command ID: {result.get('command_id')}", 'success')
					form = DeviceCommandForm()  # Reset form
					form.device_id.choices = device_choices
				else:
					flash(f"Failed to send command: {result.get('error', 'Unknown error')}", 'danger')
			
			except json.JSONDecodeError:
				flash('Invalid JSON in parameters field', 'danger')
			except Exception as e:
				flash(f"Error sending command: {str(e)}", 'danger')
		
		loop.close()
		
		return self.render_template(
			'iot_management/device_commands.html',
			form=form
		)
	
	@expose('/alerts', methods=['GET', 'POST'])
	@has_access
	def alert_rules(self):
		"""Manage alert rules"""
		
		# Get device list for form
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		devices_result = loop.run_until_complete(
			self.iot_capability.list_devices()
		)
		
		devices = devices_result.get('devices', [])
		device_choices = [('', 'All Devices')] + [(d['device_id'], f"{d['name']} ({d['device_id']})") for d in devices]
		
		form = AlertRuleForm()
		form.device_id.choices = device_choices
		
		if form.validate_on_submit():
			try:
				# Prepare alert rule data
				rule_data = {
					'name': form.name.data,
					'description': form.description.data or '',
					'device_id': form.device_id.data or '',
					'sensor_type': form.sensor_type.data or None,
					'condition': form.condition.data,
					'threshold_value': form.threshold_value.data,
					'action': form.action.data,
					'action_config': {},
					'enabled': form.enabled.data
				}
				
				# Create alert rule
				result = loop.run_until_complete(
					self.iot_capability.create_alert_rule(rule_data)
				)
				
				if result.get('success'):
					flash(f"Alert rule '{form.name.data}' created successfully!", 'success')
					form = AlertRuleForm()  # Reset form
					form.device_id.choices = device_choices
				else:
					flash(f"Failed to create alert rule: {result.get('error', 'Unknown error')}", 'danger')
			
			except Exception as e:
				flash(f"Error creating alert rule: {str(e)}", 'danger')
		
		loop.close()
		
		return self.render_template(
			'iot_management/alert_rules.html',
			form=form
		)
	
	@expose('/analytics')
	@has_access
	def analytics(self):
		"""IoT analytics dashboard"""
		
		# Get comprehensive analytics data
		analytics_data = self._get_analytics_data()
		
		return self.render_template(
			'iot_management/analytics.html',
			analytics=analytics_data
		)
	
	@expose('/api/sensor_data/<device_id>')
	@has_access
	def api_sensor_data(self, device_id):
		"""API endpoint for sensor data"""
		
		hours_back = request.args.get('hours', 24, type=int)
		sensor_type = request.args.get('sensor_type')
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			result = loop.run_until_complete(
				self.iot_capability.get_sensor_readings(
					device_id, sensor_type, hours_back
				)
			)
			
			loop.close()
			
			return jsonify(result)
		
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	@expose('/api/device_status/<device_id>')
	@has_access
	def api_device_status(self, device_id):
		"""API endpoint for device status"""
		
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			result = loop.run_until_complete(
				self.iot_capability.get_device_info(device_id)
			)
			
			loop.close()
			
			return jsonify(result)
		
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	def _calculate_dashboard_stats(self, devices: List[Dict]) -> Dict[str, Any]:
		"""Calculate dashboard statistics"""
		
		total_devices = len(devices)
		online_devices = len([d for d in devices if d.get('status') == 'online'])
		offline_devices = total_devices - online_devices
		
		# Count by device type
		device_type_counts = {}
		for device in devices:
			device_type = device.get('device_type', 'unknown')
			device_type_counts[device_type] = device_type_counts.get(device_type, 0) + 1
		
		# Mock sensor data count (in real implementation, query database)
		total_sensor_readings = 15847
		
		return {
			'total_devices': total_devices,
			'online_devices': online_devices,
			'offline_devices': offline_devices,
			'device_types': device_type_counts,
			'total_sensor_readings': total_sensor_readings,
			'uptime_percentage': 98.7,
			'active_alerts': 3
		}
	
	def _get_dashboard_charts(self, devices: List[Dict]) -> Dict[str, str]:
		"""Create dashboard charts"""
		
		# Device status pie chart
		online_count = len([d for d in devices if d.get('status') == 'online'])
		offline_count = len(devices) - online_count
		
		status_chart = go.Figure(data=[
			go.Pie(
				labels=['Online', 'Offline'],
				values=[online_count, offline_count],
				hole=0.3,
				marker_colors=['#28a745', '#dc3545']
			)
		])
		status_chart.update_layout(title='Device Status', height=300)
		
		# Device type bar chart
		device_types = {}
		for device in devices:
			device_type = device.get('device_type', 'unknown').replace('_', ' ').title()
			device_types[device_type] = device_types.get(device_type, 0) + 1
		
		type_chart = go.Figure(data=[
			go.Bar(
				x=list(device_types.keys()),
				y=list(device_types.values()),
				marker_color='#007bff'
			)
		])
		type_chart.update_layout(title='Devices by Type', height=300)
		
		return {
			'status_chart': json.dumps(status_chart, cls=plotly.utils.PlotlyJSONEncoder),
			'type_chart': json.dumps(type_chart, cls=plotly.utils.PlotlyJSONEncoder)
		}
	
	def _create_sensor_charts(self, readings: List[Dict]) -> Dict[str, str]:
		"""Create sensor data charts"""
		
		if not readings:
			return {}
		
		# Group readings by sensor type
		sensor_data = {}
		for reading in readings:
			sensor_type = reading.get('sensor_type', 'unknown')
			if sensor_type not in sensor_data:
				sensor_data[sensor_type] = {'timestamps': [], 'values': []}
			
			sensor_data[sensor_type]['timestamps'].append(reading.get('timestamp'))
			sensor_data[sensor_type]['values'].append(reading.get('value'))
		
		charts = {}
		
		for sensor_type, data in sensor_data.items():
			chart = go.Figure()
			chart.add_trace(go.Scatter(
				x=data['timestamps'],
				y=data['values'],
				mode='lines+markers',
				name=sensor_type.replace('_', ' ').title(),
				line=dict(width=2)
			))
			
			chart.update_layout(
				title=f'{sensor_type.replace("_", " ").title()} Readings',
				xaxis_title='Time',
				yaxis_title='Value',
				height=300
			)
			
			charts[sensor_type] = json.dumps(chart, cls=plotly.utils.PlotlyJSONEncoder)
		
		return charts
	
	def _get_analytics_data(self) -> Dict[str, Any]:
		"""Get comprehensive analytics data"""
		
		# Mock analytics data (in real implementation, query database)
		return {
			'device_utilization': {
				'peak_hour': '14:00',
				'avg_daily_readings': 2847,
				'data_quality_score': 96.3,
				'network_uptime': 99.2
			},
			'sensor_performance': {
				'temperature': {'accuracy': 98.5, 'reliability': 97.2},
				'humidity': {'accuracy': 96.8, 'reliability': 95.1},
				'pressure': {'accuracy': 99.1, 'reliability': 98.7},
				'motion': {'accuracy': 94.3, 'reliability': 92.8}
			},
			'alert_statistics': {
				'total_alerts': 127,
				'critical_alerts': 8,
				'resolved_alerts': 119,
				'avg_resolution_time': '4.2 minutes'
			},
			'monthly_trends': [
				{'month': 'Jan', 'devices': 45, 'readings': 23456, 'alerts': 12},
				{'month': 'Feb', 'devices': 52, 'readings': 28934, 'alerts': 15},
				{'month': 'Mar', 'devices': 67, 'readings': 35672, 'alerts': 18},
				{'month': 'Apr', 'devices': 73, 'readings': 42189, 'alerts': 14},
				{'month': 'May', 'devices': 81, 'readings': 48526, 'alerts': 11}
			]
		}

# Template definitions for IoT Management
IOT_MANAGEMENT_TEMPLATES = {
	'iot_management/dashboard.html': """
{% extends "appbuilder/base.html" %}

{% block content %}
<div class="container-fluid">
	<div class="row">
		<div class="col-12">
			<h1><i class="fa fa-microchip"></i> IoT Management Dashboard</h1>
		</div>
	</div>
	
	<div class="row">
		<div class="col-md-3">
			<div class="card bg-primary text-white">
				<div class="card-body text-center">
					<h3>{{ stats.total_devices }}</h3>
					<p>Total Devices</p>
				</div>
			</div>
		</div>
		<div class="col-md-3">
			<div class="card bg-success text-white">
				<div class="card-body text-center">
					<h3>{{ stats.online_devices }}</h3>
					<p>Online Devices</p>
				</div>
			</div>
		</div>
		<div class="col-md-3">
			<div class="card bg-info text-white">
				<div class="card-body text-center">
					<h3>{{ stats.total_sensor_readings }}</h3>
					<p>Sensor Readings</p>
				</div>
			</div>
		</div>
		<div class="col-md-3">
			<div class="card bg-warning text-white">
				<div class="card-body text-center">
					<h3>{{ "%.1f"|format(stats.uptime_percentage) }}%</h3>
					<p>Uptime</p>
				</div>
			</div>
		</div>
	</div>
	
	<div class="row mt-4">
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5>Quick Actions</h5>
				</div>
				<div class="card-body">
					<div class="d-grid gap-2">
						<a href="{{ url_for('IoTManagementView.register_device') }}" class="btn btn-primary">
							<i class="fa fa-plus"></i> Register Device
						</a>
						<a href="{{ url_for('IoTManagementView.sensor_data') }}" class="btn btn-success">
							<i class="fa fa-thermometer-half"></i> Record Sensor Data
						</a>
						<a href="{{ url_for('IoTManagementView.device_commands') }}" class="btn btn-warning">
							<i class="fa fa-terminal"></i> Send Commands
						</a>
						<a href="{{ url_for('IoTManagementView.alert_rules') }}" class="btn btn-danger">
							<i class="fa fa-bell"></i> Manage Alerts
						</a>
						<a href="{{ url_for('IoTManagementView.analytics') }}" class="btn btn-info">
							<i class="fa fa-chart-line"></i> Analytics
						</a>
					</div>
				</div>
			</div>
		</div>
		
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5>Device Status</h5>
				</div>
				<div class="card-body">
					<div id="device-status-chart"></div>
				</div>
			</div>
		</div>
	</div>
	
	<div class="row mt-4">
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5>Recent Devices</h5>
				</div>
				<div class="card-body">
					<div class="table-responsive">
						<table class="table table-sm">
							<thead>
								<tr>
									<th>Name</th>
									<th>Type</th>
									<th>Status</th>
									<th>Last Seen</th>
								</tr>
							</thead>
							<tbody>
								{% for device in devices %}
								<tr>
									<td>
										<a href="{{ url_for('IoTManagementView.device_detail', device_id=device.device_id) }}">
											{{ device.name }}
										</a>
									</td>
									<td>{{ device.device_type.replace('_', ' ').title() }}</td>
									<td>
										<span class="badge bg-{{ 'success' if device.status == 'online' else 'danger' }}">
											{{ device.status.title() }}
										</span>
									</td>
									<td>{{ device.last_seen or 'Never' }}</td>
								</tr>
								{% endfor %}
							</tbody>
						</table>
					</div>
					<a href="{{ url_for('IoTManagementView.devices_list') }}" class="btn btn-sm btn-outline-primary">
						View All Devices
					</a>
				</div>
			</div>
		</div>
		
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5>Devices by Type</h5>
				</div>
				<div class="card-body">
					<div id="device-type-chart"></div>
				</div>
			</div>
		</div>
	</div>
</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
	// Render device status chart
	var statusChartData = {{ chart_data.status_chart|safe }};
	Plotly.newPlot('device-status-chart', statusChartData.data, statusChartData.layout, {responsive: true});
	
	// Render device type chart
	var typeChartData = {{ chart_data.type_chart|safe }};
	Plotly.newPlot('device-type-chart', typeChartData.data, typeChartData.layout, {responsive: true});
</script>
{% endblock %}
"""
}

# Blueprint registration
iot_management_bp = Blueprint(
	'iot_management',
	__name__,
	template_folder='templates',
	static_folder='static'
)

# Export the view class for AppBuilder registration
__all__ = ['IoTManagementView', 'iot_management_bp']