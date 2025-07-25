#!/usr/bin/env python3
"""
APG Vision & IoT Example Application
====================================

Complete Flask-AppBuilder application showcasing computer vision and IoT capabilities
with rich dashboards, real-time monitoring, and integrated workflows.
"""

import os
import logging
from flask import Flask
from flask_appbuilder import AppBuilder, SQLA
from flask_appbuilder.models.sqla import Model
from flask_appbuilder.views import ModelView, BaseView, expose, has_access
from flask_appbuilder.menu import Menu
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base

# Import our blueprints
from blueprints.computer_vision_blueprint import ComputerVisionView
from blueprints.iot_management_blueprint import IoTManagementView

# Configure logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# Flask app configuration
class Config(object):
	"""Application configuration"""
	
	# Flask settings
	SECRET_KEY = '\2\1thisismyscretkey\1\2\e\y\y\h'
	
	# Database settings
	SQLALCHEMY_DATABASE_URI = 'sqlite:///vision_iot_app.db'
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	
	# Flask-AppBuilder settings
	APP_NAME = "APG Vision & IoT Platform"
	APP_THEME = "cerulean.css"  # bootstrap theme
	APP_ICON = "static/img/logo.jpg"
	
	# Upload settings
	UPLOAD_FOLDER = 'static/uploads/'
	MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
	
	# Security settings
	AUTH_TYPE = 1  # Database authentication
	AUTH_ROLE_ADMIN = 'Admin'
	AUTH_ROLE_PUBLIC = 'Public'
	PUBLIC_REGISTER = True
	
	# Enable CSV export
	FAB_ROLES = {
		"ReadOnly": [
			["can_show", "ComputerVisionView"],
			["can_show", "IoTManagementView"],
			["can_list", "ComputerVisionView"],
			["can_list", "IoTManagementView"]
		]
	}

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db = SQLA(app)

class ProcessingLog(Model):
	"""Model for tracking processing operations"""
	
	id = Column(Integer, primary_key=True)
	operation_type = Column(String(50), nullable=False)  # 'computer_vision', 'iot_data'
	operation_name = Column(String(100), nullable=False)
	input_file = Column(String(255))
	output_file = Column(String(255))
	parameters = Column(Text)
	results = Column(Text)
	processing_time_ms = Column(Float)
	status = Column(String(20), default='pending')  # pending, completed, failed
	error_message = Column(Text)
	created_at = Column(DateTime, default=db.func.current_timestamp())
	
	def __repr__(self):
		return f'<ProcessingLog {self.operation_name}>'

class DeviceActivity(Model):
	"""Model for tracking device activities"""
	
	id = Column(Integer, primary_key=True)
	device_id = Column(String(100), nullable=False)
	device_name = Column(String(200))
	activity_type = Column(String(50), nullable=False)  # 'registration', 'data_received', 'command_sent', 'alert'
	description = Column(Text)
	data_payload = Column(Text)
	severity = Column(String(20), default='info')  # info, warning, error, critical
	resolved = Column(Boolean, default=False)
	created_at = Column(DateTime, default=db.func.current_timestamp())
	
	def __repr__(self):
		return f'<DeviceActivity {self.device_id}:{self.activity_type}>'

class ProcessingLogView(ModelView):
	"""View for processing logs"""
	
	datamodel = None  # Will be set after db creation
	list_columns = ['operation_type', 'operation_name', 'status', 'processing_time_ms', 'created_at']
	show_columns = ['operation_type', 'operation_name', 'input_file', 'output_file', 
					'parameters', 'results', 'processing_time_ms', 'status', 'error_message', 'created_at']
	search_columns = ['operation_type', 'operation_name', 'status']
	
	base_order = ('created_at', 'desc')
	base_permissions = ['can_list', 'can_show']

class DeviceActivityView(ModelView):
	"""View for device activities"""
	
	datamodel = None  # Will be set after db creation
	list_columns = ['device_id', 'device_name', 'activity_type', 'severity', 'resolved', 'created_at']
	show_columns = ['device_id', 'device_name', 'activity_type', 'description', 
					'data_payload', 'severity', 'resolved', 'created_at']
	search_columns = ['device_id', 'device_name', 'activity_type', 'severity']
	
	base_order = ('created_at', 'desc')
	base_permissions = ['can_list', 'can_show', 'can_edit']

class MainDashboardView(BaseView):
	"""Main dashboard combining vision and IoT metrics"""
	
	route_base = "/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard"""
		
		# Get recent processing logs
		recent_processing = db.session.query(ProcessingLog).order_by(
			ProcessingLog.created_at.desc()
		).limit(10).all()
		
		# Get recent device activities
		recent_activities = db.session.query(DeviceActivity).order_by(
			DeviceActivity.created_at.desc()
		).limit(10).all()
		
		# Calculate statistics
		stats = self._calculate_main_stats()
		
		return self.render_template(
			'main_dashboard.html',
			stats=stats,
			recent_processing=recent_processing,
			recent_activities=recent_activities
		)
	
	@expose('/system_status')
	@has_access
	def system_status(self):
		"""System status and health monitoring"""
		
		system_health = self._get_system_health()
		
		return self.render_template(
			'system_status.html',
			health=system_health
		)
	
	def _calculate_main_stats(self):
		"""Calculate main dashboard statistics"""
		
		# Processing statistics
		total_processing = db.session.query(ProcessingLog).count()
		completed_processing = db.session.query(ProcessingLog).filter(
			ProcessingLog.status == 'completed'
		).count()
		
		cv_processing = db.session.query(ProcessingLog).filter(
			ProcessingLog.operation_type == 'computer_vision'
		).count()
		
		iot_operations = db.session.query(ProcessingLog).filter(
			ProcessingLog.operation_type == 'iot_data'
		).count()
		
		# Device statistics
		total_activities = db.session.query(DeviceActivity).count()
		active_alerts = db.session.query(DeviceActivity).filter(
			DeviceActivity.severity.in_(['warning', 'error', 'critical']),
			DeviceActivity.resolved == False
		).count()
		
		# Calculate success rate
		success_rate = (completed_processing / total_processing * 100) if total_processing > 0 else 0
		
		return {
			'total_processing': total_processing,
			'completed_processing': completed_processing,
			'cv_processing': cv_processing,
			'iot_operations': iot_operations,
			'total_activities': total_activities,
			'active_alerts': active_alerts,
			'success_rate': success_rate,
			'system_uptime': '99.8%'
		}
	
	def _get_system_health(self):
		"""Get system health metrics"""
		
		import psutil
		
		# Get system metrics
		cpu_percent = psutil.cpu_percent(interval=1)
		memory = psutil.virtual_memory()
		disk = psutil.disk_usage('/')
		
		# Get recent error count
		recent_errors = db.session.query(ProcessingLog).filter(
			ProcessingLog.status == 'failed',
			ProcessingLog.created_at >= db.func.datetime('now', '-1 hour')
		).count()
		
		# Determine overall health
		health_score = 100
		if cpu_percent > 80:
			health_score -= 20
		if memory.percent > 80:
			health_score -= 20
		if disk.percent > 90:
			health_score -= 30
		if recent_errors > 5:
			health_score -= 30
		
		health_status = 'excellent'
		if health_score < 70:
			health_status = 'poor'
		elif health_score < 85:
			health_status = 'fair'
		elif health_score < 95:
			health_status = 'good'
		
		return {
			'score': health_score,
			'status': health_status,
			'cpu_percent': cpu_percent,
			'memory_percent': memory.percent,
			'disk_percent': disk.percent,
			'recent_errors': recent_errors,
			'timestamp': db.func.current_timestamp()
		}

# Initialize Flask-AppBuilder
appbuilder = AppBuilder(app, db.session)

# Create database tables
with app.app_context():
	db.create_all()
	
	# Set datamodels after db creation
	from flask_appbuilder.models.sqla.interface import SQLAInterface
	ProcessingLogView.datamodel = SQLAInterface(ProcessingLog)
	DeviceActivityView.datamodel = SQLAInterface(DeviceActivity)

# Register views
appbuilder.add_view(
	MainDashboardView,
	"Main Dashboard",
	icon="fa-dashboard",
	category="Dashboard"
)

appbuilder.add_view(
	ComputerVisionView,
	"Computer Vision",
	icon="fa-eye",
	category="AI & Vision"
)

appbuilder.add_view(
	IoTManagementView,
	"IoT Management", 
	icon="fa-microchip",
	category="IoT & Devices"
)

appbuilder.add_view(
	ProcessingLogView,
	"Processing Logs",
	icon="fa-list",
	category="Monitoring"
)

appbuilder.add_view(
	DeviceActivityView,
	"Device Activities",
	icon="fa-history",
	category="Monitoring"
)

# Add separators in menu
appbuilder.add_separator("Dashboard")
appbuilder.add_separator("AI & Vision")
appbuilder.add_separator("IoT & Devices") 
appbuilder.add_separator("Monitoring")

# Custom templates directory
CUSTOM_TEMPLATES = {
	'main_dashboard.html': """
{% extends "appbuilder/base.html" %}

{% block content %}
<div class="container-fluid">
	<div class="row">
		<div class="col-12">
			<h1><i class="fa fa-dashboard"></i> APG Vision & IoT Platform</h1>
			<p class="lead">Integrated computer vision and IoT management platform</p>
		</div>
	</div>
	
	<!-- Statistics Cards -->
	<div class="row">
		<div class="col-md-3">
			<div class="card bg-primary text-white">
				<div class="card-body text-center">
					<h3>{{ stats.total_processing }}</h3>
					<p>Total Operations</p>
				</div>
			</div>
		</div>
		<div class="col-md-3">
			<div class="card bg-success text-white">
				<div class="card-body text-center">
					<h3>{{ stats.cv_processing }}</h3>
					<p>Vision Operations</p>
				</div>
			</div>
		</div>
		<div class="col-md-3">
			<div class="card bg-info text-white">
				<div class="card-body text-center">
					<h3>{{ stats.iot_operations }}</h3>
					<p>IoT Operations</p>
				</div>
			</div>
		</div>
		<div class="col-md-3">
			<div class="card bg-warning text-white">
				<div class="card-body text-center">
					<h3>{{ "%.1f"|format(stats.success_rate) }}%</h3>
					<p>Success Rate</p>
				</div>
			</div>
		</div>
	</div>
	
	<!-- Quick Actions -->
	<div class="row mt-4">
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5><i class="fa fa-bolt"></i> Quick Actions</h5>
				</div>
				<div class="card-body">
					<div class="row">
						<div class="col-md-6">
							<h6>Computer Vision</h6>
							<div class="d-grid gap-2 mb-3">
								<a href="/computer_vision/image_processing" class="btn btn-primary btn-sm">
									<i class="fa fa-image"></i> Process Image
								</a>
								<a href="/computer_vision/video_processing" class="btn btn-primary btn-sm">
									<i class="fa fa-video"></i> Process Video
								</a>
								<a href="/computer_vision/live_stream" class="btn btn-primary btn-sm">
									<i class="fa fa-camera"></i> Live Stream
								</a>
							</div>
						</div>
						<div class="col-md-6">
							<h6>IoT Management</h6>
							<div class="d-grid gap-2 mb-3">
								<a href="/iot_management/register_device" class="btn btn-success btn-sm">
									<i class="fa fa-plus"></i> Add Device
								</a>
								<a href="/iot_management/sensor_data" class="btn btn-success btn-sm">
									<i class="fa fa-thermometer-half"></i> Record Data
								</a>
								<a href="/iot_management/device_commands" class="btn btn-success btn-sm">
									<i class="fa fa-terminal"></i> Send Command
								</a>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
		
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5><i class="fa fa-exclamation-triangle"></i> System Alerts</h5>
				</div>
				<div class="card-body">
					{% if stats.active_alerts > 0 %}
					<div class="alert alert-warning">
						<strong>{{ stats.active_alerts }}</strong> active alerts require attention
					</div>
					{% else %}
					<div class="alert alert-success">
						<i class="fa fa-check"></i> No active alerts - all systems operational
					</div>
					{% endif %}
					
					<p><strong>System Uptime:</strong> {{ stats.system_uptime }}</p>
					<p><strong>Last Check:</strong> <span id="current-time"></span></p>
					
					<a href="{{ url_for('MainDashboardView.system_status') }}" class="btn btn-info btn-sm">
						<i class="fa fa-heartbeat"></i> View System Status
					</a>
				</div>
			</div>
		</div>
	</div>
	
	<!-- Recent Activities -->
	<div class="row mt-4">
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5><i class="fa fa-history"></i> Recent Processing</h5>
				</div>
				<div class="card-body">
					<div class="table-responsive">
						<table class="table table-sm">
							<thead>
								<tr>
									<th>Type</th>
									<th>Operation</th>
									<th>Status</th>
									<th>Time</th>
								</tr>
							</thead>
							<tbody>
								{% for log in recent_processing %}
								<tr>
									<td>
										<span class="badge bg-{{ 'primary' if log.operation_type == 'computer_vision' else 'info' }}">
											{{ log.operation_type.replace('_', ' ').title() }}
										</span>
									</td>
									<td>{{ log.operation_name }}</td>
									<td>
										<span class="badge bg-{{ 'success' if log.status == 'completed' else 'danger' if log.status == 'failed' else 'warning' }}">
											{{ log.status.title() }}
										</span>
									</td>
									<td>{{ log.created_at.strftime('%m/%d %H:%M') if log.created_at else '' }}</td>
								</tr>
								{% endfor %}
							</tbody>
						</table>
					</div>
					<a href="/processinglogview/list/" class="btn btn-outline-primary btn-sm">
						View All Logs
					</a>
				</div>
			</div>
		</div>
		
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5><i class="fa fa-microchip"></i> Device Activities</h5>
				</div>
				<div class="card-body">
					<div class="table-responsive">
						<table class="table table-sm">
							<thead>
								<tr>
									<th>Device</th>
									<th>Activity</th>
									<th>Severity</th>
									<th>Time</th>
								</tr>
							</thead>
							<tbody>
								{% for activity in recent_activities %}
								<tr>
									<td>{{ activity.device_name or activity.device_id }}</td>
									<td>{{ activity.activity_type.replace('_', ' ').title() }}</td>
									<td>
										<span class="badge bg-{{ 'success' if activity.severity == 'info' else 'warning' if activity.severity == 'warning' else 'danger' }}">
											{{ activity.severity.title() }}
										</span>
									</td>
									<td>{{ activity.created_at.strftime('%m/%d %H:%M') if activity.created_at else '' }}</td>
								</tr>
								{% endfor %}
							</tbody>
						</table>
					</div>
					<a href="/deviceactivityview/list/" class="btn btn-outline-primary btn-sm">
						View All Activities
					</a>
				</div>
			</div>
		</div>
	</div>
</div>

<script>
	// Update current time
	function updateTime() {
		document.getElementById('current-time').textContent = new Date().toLocaleString();
	}
	updateTime();
	setInterval(updateTime, 1000);
	
	// Auto-refresh page every 30 seconds
	setTimeout(function() {
		location.reload();
	}, 30000);
</script>
{% endblock %}
""",

	'system_status.html': """
{% extends "appbuilder/base.html" %}

{% block content %}
<div class="container-fluid">
	<div class="row">
		<div class="col-12">
			<h1><i class="fa fa-heartbeat"></i> System Status & Health</h1>
		</div>
	</div>
	
	<!-- Health Overview -->
	<div class="row">
		<div class="col-md-4">
			<div class="card border-{{ 'success' if health.status == 'excellent' else 'warning' if health.status in ['good', 'fair'] else 'danger' }}">
				<div class="card-header bg-{{ 'success' if health.status == 'excellent' else 'warning' if health.status in ['good', 'fair'] else 'danger' }} text-white">
					<h5><i class="fa fa-heartbeat"></i> Overall Health</h5>
				</div>
				<div class="card-body text-center">
					<h2>{{ health.score }}%</h2>
					<p class="lead">{{ health.status.title() }}</p>
					<small class="text-muted">Last updated: {{ health.timestamp }}</small>
				</div>
			</div>
		</div>
		
		<div class="col-md-8">
			<div class="card">
				<div class="card-header">
					<h5><i class="fa fa-tachometer-alt"></i> System Metrics</h5>
				</div>
				<div class="card-body">
					<!-- CPU Usage -->
					<div class="mb-3">
						<div class="d-flex justify-content-between">
							<span>CPU Usage</span>
							<span>{{ "%.1f"|format(health.cpu_percent) }}%</span>
						</div>
						<div class="progress">
							<div class="progress-bar bg-{{ 'success' if health.cpu_percent < 50 else 'warning' if health.cpu_percent < 80 else 'danger' }}" 
								 style="width: {{ health.cpu_percent }}%"></div>
						</div>
					</div>
					
					<!-- Memory Usage -->
					<div class="mb-3">
						<div class="d-flex justify-content-between">
							<span>Memory Usage</span>
							<span>{{ "%.1f"|format(health.memory_percent) }}%</span>
						</div>
						<div class="progress">
							<div class="progress-bar bg-{{ 'success' if health.memory_percent < 50 else 'warning' if health.memory_percent < 80 else 'danger' }}" 
								 style="width: {{ health.memory_percent }}%"></div>
						</div>
					</div>
					
					<!-- Disk Usage -->
					<div class="mb-3">
						<div class="d-flex justify-content-between">
							<span>Disk Usage</span>
							<span>{{ "%.1f"|format(health.disk_percent) }}%</span>
						</div>
						<div class="progress">
							<div class="progress-bar bg-{{ 'success' if health.disk_percent < 50 else 'warning' if health.disk_percent < 80 else 'danger' }}" 
								 style="width: {{ health.disk_percent }}%"></div>
						</div>
					</div>
					
					<!-- Recent Errors -->
					<div class="mb-3">
						<div class="d-flex justify-content-between">
							<span>Recent Errors (last hour)</span>
							<span class="badge bg-{{ 'success' if health.recent_errors == 0 else 'warning' if health.recent_errors < 5 else 'danger' }}">
								{{ health.recent_errors }}
							</span>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
	
	<!-- Service Status -->
	<div class="row mt-4">
		<div class="col-12">
			<div class="card">
				<div class="card-header">
					<h5><i class="fa fa-cogs"></i> Service Status</h5>
				</div>
				<div class="card-body">
					<div class="row">
						<div class="col-md-6">
							<h6>Core Services</h6>
							<ul class="list-group list-group-flush">
								<li class="list-group-item d-flex justify-content-between">
									<span>Web Application</span>
									<span class="badge bg-success">Running</span>
								</li>
								<li class="list-group-item d-flex justify-content-between">
									<span>Database</span>
									<span class="badge bg-success">Connected</span>
								</li>
								<li class="list-group-item d-flex justify-content-between">
									<span>File Storage</span>
									<span class="badge bg-success">Available</span>
								</li>
							</ul>
						</div>
						<div class="col-md-6">
							<h6>AI & IoT Services</h6>
							<ul class="list-group list-group-flush">
								<li class="list-group-item d-flex justify-content-between">
									<span>Computer Vision</span>
									<span class="badge bg-success">Active</span>
								</li>
								<li class="list-group-item d-flex justify-content-between">
									<span>IoT Management</span>
									<span class="badge bg-success">Active</span>
								</li>
								<li class="list-group-item d-flex justify-content-between">
									<span>Alert System</span>
									<span class="badge bg-success">Monitoring</span>
								</li>
							</ul>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<script>
	// Auto-refresh every 10 seconds
	setTimeout(function() {
		location.reload();
	}, 10000);
</script>
{% endblock %}
"""
}

# Helper function to create template files
def create_template_files():
	"""Create template files in the templates directory"""
	
	templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
	os.makedirs(templates_dir, exist_ok=True)
	
	for template_name, content in CUSTOM_TEMPLATES.items():
		template_path = os.path.join(templates_dir, template_name)
		with open(template_path, 'w') as f:
			f.write(content)

# Create directories and files
def setup_application():
	"""Setup application directories and files"""
	
	# Create necessary directories
	os.makedirs('static/uploads/cv', exist_ok=True)
	os.makedirs('static/outputs/cv', exist_ok=True)
	os.makedirs('static/uploads/iot', exist_ok=True)
	os.makedirs('static/img', exist_ok=True)
	os.makedirs('templates', exist_ok=True)
	
	# Create template files
	create_template_files()
	
	print("Application setup completed!")
	print("Directories created:")
	print("  - static/uploads/cv")
	print("  - static/outputs/cv") 
	print("  - static/uploads/iot")
	print("  - static/img")
	print("  - templates")

if __name__ == '__main__':
	# Setup application on first run
	setup_application()
	
	# Run the application
	app.run(host='0.0.0.0', port=8080, debug=True)