#!/usr/bin/env python3
"""
Computer Vision Flask-AppBuilder Blueprint
==========================================

Rich web interface for computer vision capabilities including object detection,
real-time video processing, and image analysis.
"""

import os
import json
import base64
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, render_template, flash, redirect, url_for
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.views import ModelView, SimpleFormView
from flask_appbuilder.forms import DynamicForm
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget, BS3TextAreaFieldWidget
from wtforms import StringField, TextAreaField, SelectField, FileField, IntegerField, BooleanField
from wtforms.validators import DataRequired, Optional as OptionalValidator
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import io

# Import our computer vision capability
from capabilities.computer_vision import (
	ComputerVisionCapability, 
	DetectionType,
	ProcessingMode
)

class ComputerVisionForm(DynamicForm):
	"""Form for computer vision processing"""
	
	image_file = FileField(
		'Image File',
		validators=[DataRequired()],
		description='Upload an image file for analysis'
	)
	
	detection_types = SelectField(
		'Detection Types',
		choices=[
			('face', 'Face Detection'),
			('person', 'Person Detection'), 
			('vehicle', 'Vehicle Detection'),
			('object', 'Object Detection')
		],
		validators=[DataRequired()],
		description='Select what to detect in the image'
	)
	
	enhancement_type = SelectField(
		'Image Enhancement',
		choices=[
			('none', 'No Enhancement'),
			('auto', 'Auto Enhancement'),
			('brightness', 'Brightness Adjustment'),
			('contrast', 'Contrast Adjustment'),
			('denoise', 'Noise Reduction'),
			('sharpen', 'Sharpening')
		],
		default='none',
		description='Apply image enhancement before processing'
	)

class VideoProcessingForm(DynamicForm):
	"""Form for video processing"""
	
	video_file = FileField(
		'Video File',
		validators=[DataRequired()],
		description='Upload a video file for analysis'
	)
	
	detection_types = SelectField(
		'Detection Types',
		choices=[
			('face,person', 'Face & Person Detection'),
			('face', 'Face Detection Only'),
			('person', 'Person Detection Only'),
			('vehicle', 'Vehicle Detection'),
			('object', 'General Object Detection')
		],
		validators=[DataRequired()],
		description='Select what to detect in the video'
	)
	
	save_output = BooleanField(
		'Save Annotated Video',
		default=True,
		description='Save video with detection annotations'
	)

class LiveStreamForm(DynamicForm):
	"""Form for live stream configuration"""
	
	camera_id = IntegerField(
		'Camera ID',
		default=0,
		validators=[DataRequired()],
		description='Camera device ID (usually 0 for default camera)'
	)
	
	detection_types = SelectField(
		'Detection Types',
		choices=[
			('face', 'Face Detection'),
			('person', 'Person Detection'),
			('face,person', 'Face & Person Detection')
		],
		default='face',
		validators=[DataRequired()],
		description='Select what to detect in live stream'
	)

class ComputerVisionView(BaseView):
	"""Main computer vision interface"""
	
	route_base = '/computer_vision'
	default_view = 'dashboard'
	
	def __init__(self):
		super().__init__()
		self.cv_capability = ComputerVisionCapability()
		self.upload_folder = 'static/uploads/cv'
		self.output_folder = 'static/outputs/cv'
		
		# Ensure upload directories exist
		os.makedirs(self.upload_folder, exist_ok=True)
		os.makedirs(self.output_folder, exist_ok=True)
	
	@expose('/')
	@has_access
	def dashboard(self):
		"""Computer vision dashboard"""
		
		# Get recent processing statistics
		stats = self._get_processing_stats()
		
		return self.render_template(
			'computer_vision/dashboard.html',
			stats=stats,
			detection_types=DetectionType
		)
	
	@expose('/image_processing', methods=['GET', 'POST'])
	@has_access
	def image_processing(self):
		"""Image processing interface"""
		
		form = ComputerVisionForm()
		results = None
		
		if form.validate_on_submit():
			try:
				# Handle file upload
				image_file = form.image_file.data
				if image_file:
					filename = secure_filename(image_file.filename)
					timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
					filename = f"{timestamp}_{filename}"
					file_path = os.path.join(self.upload_folder, filename)
					image_file.save(file_path)
					
					# Process image
					detection_types = [form.detection_types.data]
					enhancement_type = form.enhancement_type.data
					
					# Run async processing
					loop = asyncio.new_event_loop()
					asyncio.set_event_loop(loop)
					
					# Enhance image if requested
					if enhancement_type != 'none':
						enhance_result = loop.run_until_complete(
							self.cv_capability.enhance_image_quality(
								file_path, 
								enhancement_type,
								os.path.join(self.output_folder, f"enhanced_{filename}")
							)
						)
					
					# Detect objects
					detection_result = loop.run_until_complete(
						self.cv_capability.detect_objects_in_image(
							file_path,
							detection_types
						)
					)
					
					loop.close()
					
					# Create annotated image
					annotated_path = self._create_annotated_image(
						file_path, 
						detection_result['detections'],
						f"annotated_{filename}"
					)
					
					results = {
						'original_image': f"/static/uploads/cv/{filename}",
						'annotated_image': f"/static/outputs/cv/annotated_{filename}",
						'detections': detection_result['detections'],
						'detection_count': detection_result['detections_count'],
						'enhancement_applied': enhancement_type != 'none'
					}
					
					flash(f"Processed image successfully. Found {detection_result['detections_count']} objects.", 'success')
			
			except Exception as e:
				flash(f"Error processing image: {str(e)}", 'danger')
		
		return self.render_template(
			'computer_vision/image_processing.html',
			form=form,
			results=results
		)
	
	@expose('/video_processing', methods=['GET', 'POST'])
	@has_access
	def video_processing(self):
		"""Video processing interface"""
		
		form = VideoProcessingForm()
		results = None
		
		if form.validate_on_submit():
			try:
				# Handle file upload
				video_file = form.video_file.data
				if video_file:
					filename = secure_filename(video_file.filename)
					timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
					filename = f"{timestamp}_{filename}"
					file_path = os.path.join(self.upload_folder, filename)
					video_file.save(file_path)
					
					# Prepare processing parameters
					detection_types = form.detection_types.data.split(',')
					output_path = None
					
					if form.save_output.data:
						output_filename = f"processed_{filename}"
						output_path = os.path.join(self.output_folder, output_filename)
					
					# Process video
					loop = asyncio.new_event_loop()
					asyncio.set_event_loop(loop)
					
					video_result = loop.run_until_complete(
						self.cv_capability.process_video_file(
							file_path,
							output_path,
							detection_types
						)
					)
					
					loop.close()
					
					results = {
						'original_video': f"/static/uploads/cv/{filename}",
						'processed_video': f"/static/outputs/cv/processed_{filename}" if output_path else None,
						'frames_processed': video_result['frames_processed'],
						'total_detections': video_result['total_detections'],
						'avg_processing_time': video_result['avg_processing_time_ms'],
						'detection_summary': video_result['detection_summary']
					}
					
					flash(f"Processed video successfully. {video_result['frames_processed']} frames, {video_result['total_detections']} total detections.", 'success')
			
			except Exception as e:
				flash(f"Error processing video: {str(e)}", 'danger')
		
		return self.render_template(
			'computer_vision/video_processing.html',
			form=form,
			results=results
		)
	
	@expose('/live_stream', methods=['GET', 'POST'])
	@has_access
	def live_stream(self):
		"""Live stream processing interface"""
		
		form = LiveStreamForm()
		stream_active = False
		
		if form.validate_on_submit():
			try:
				camera_id = form.camera_id.data
				detection_types = form.detection_types.data.split(',')
				
				# Start live detection
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				stream_result = loop.run_until_complete(
					self.cv_capability.start_live_detection(
						camera_id=camera_id,
						detection_types=detection_types
					)
				)
				
				loop.close()
				
				if stream_result['success']:
					stream_active = True
					flash("Live stream started successfully!", 'success')
				else:
					flash("Failed to start live stream. Check camera connection.", 'danger')
			
			except Exception as e:
				flash(f"Error starting live stream: {str(e)}", 'danger')
		
		return self.render_template(
			'computer_vision/live_stream.html',
			form=form,
			stream_active=stream_active
		)
	
	@expose('/stop_stream', methods=['POST'])
	@has_access
	def stop_stream(self):
		"""Stop live stream"""
		try:
			result = self.cv_capability.stop_live_detection()
			if result['success']:
				flash("Live stream stopped successfully.", 'success')
			else:
				flash("Error stopping live stream.", 'danger')
		except Exception as e:
			flash(f"Error stopping stream: {str(e)}", 'danger')
		
		return redirect(url_for('ComputerVisionView.live_stream'))
	
	@expose('/analytics')
	@has_access
	def analytics(self):
		"""Computer vision analytics dashboard"""
		
		# Get processing analytics
		analytics_data = self._get_analytics_data()
		
		return self.render_template(
			'computer_vision/analytics.html',
			analytics=analytics_data
		)
	
	@expose('/api/process_image', methods=['POST'])
	@has_access
	def api_process_image(self):
		"""API endpoint for image processing"""
		try:
			if 'image' not in request.files:
				return jsonify({'error': 'No image file provided'}), 400
			
			image_file = request.files['image']
			detection_types = request.form.get('detection_types', 'face').split(',')
			
			# Save uploaded file
			filename = secure_filename(image_file.filename)
			timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
			filename = f"api_{timestamp}_{filename}"
			file_path = os.path.join(self.upload_folder, filename)
			image_file.save(file_path)
			
			# Process image
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			result = loop.run_until_complete(
				self.cv_capability.detect_objects_in_image(file_path, detection_types)
			)
			
			loop.close()
			
			# Clean up file
			os.remove(file_path)
			
			return jsonify(result)
		
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	def _create_annotated_image(self, image_path: str, detections: List[Dict], output_filename: str) -> str:
		"""Create annotated image with detection results"""
		
		# Load image
		image = cv2.imread(image_path)
		if image is None:
			return None
		
		# Draw detections
		for detection in detections:
			x, y, w, h = detection['bounding_box']
			confidence = detection['confidence']
			object_type = detection['type']
			
			# Color based on object type
			colors = {
				"face": (255, 0, 0),      # Blue
				"person": (0, 255, 0),    # Green
				"vehicle": (0, 0, 255),   # Red
				"object": (255, 255, 0)   # Cyan
			}
			color = colors.get(object_type, (128, 128, 128))
			
			# Draw bounding box
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			
			# Draw label
			label = f"{object_type}: {confidence:.2f}"
			label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
			cv2.rectangle(image, (x, y - label_size[1] - 10), 
						 (x + label_size[0], y), color, -1)
			cv2.putText(image, label, (x, y - 5), 
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		
		# Save annotated image
		output_path = os.path.join(self.output_folder, output_filename)
		cv2.imwrite(output_path, image)
		
		return output_path
	
	def _get_processing_stats(self) -> Dict[str, Any]:
		"""Get processing statistics"""
		
		# In a real implementation, this would query a database
		# For now, return mock data
		return {
			'total_images_processed': 1247,
			'total_videos_processed': 89,
			'total_detections': 5632,
			'avg_processing_time_ms': 145.7,
			'top_detection_types': [
				{'type': 'face', 'count': 2890},
				{'type': 'person', 'count': 1823},
				{'type': 'vehicle', 'count': 651},
				{'type': 'object', 'count': 268}
			],
			'processing_accuracy': 94.2,
			'uptime_hours': 168.5
		}
	
	def _get_analytics_data(self) -> Dict[str, Any]:
		"""Get detailed analytics data"""
		
		# Mock analytics data
		return {
			'daily_processing': [
				{'date': '2024-01-01', 'images': 45, 'videos': 3, 'detections': 234},
				{'date': '2024-01-02', 'images': 67, 'videos': 5, 'detections': 389},
				{'date': '2024-01-03', 'images': 52, 'videos': 2, 'detections': 298},
				{'date': '2024-01-04', 'images': 78, 'videos': 8, 'detections': 456},
				{'date': '2024-01-05', 'images': 91, 'videos': 6, 'detections': 523}
			],
			'detection_accuracy': {
				'face': 96.5,
				'person': 93.2,
				'vehicle': 89.7,
				'object': 87.1
			},
			'processing_times': {
				'image_avg_ms': 145.7,
				'video_avg_ms': 2847.3,
				'peak_processing_time': '14:30',
				'off_peak_processing_time': '03:15'
			},
			'system_performance': {
				'cpu_usage': 45.2,
				'memory_usage': 67.8,
				'gpu_usage': 23.4,
				'disk_usage': 34.6
			}
		}

# Template files would be created separately
# Here's the structure for the templates:

COMPUTER_VISION_TEMPLATES = {
	'computer_vision/dashboard.html': """
{% extends "appbuilder/base.html" %}

{% block content %}
<div class="container-fluid">
	<div class="row">
		<div class="col-12">
			<h1><i class="fa fa-eye"></i> Computer Vision Dashboard</h1>
		</div>
	</div>
	
	<div class="row">
		<div class="col-md-3">
			<div class="card bg-primary text-white">
				<div class="card-body">
					<h4>{{ stats.total_images_processed }}</h4>
					<p>Images Processed</p>
				</div>
			</div>
		</div>
		<div class="col-md-3">
			<div class="card bg-success text-white">
				<div class="card-body">
					<h4>{{ stats.total_videos_processed }}</h4>
					<p>Videos Processed</p>
				</div>
			</div>
		</div>
		<div class="col-md-3">
			<div class="card bg-info text-white">
				<div class="card-body">
					<h4>{{ stats.total_detections }}</h4>
					<p>Total Detections</p>
				</div>
			</div>
		</div>
		<div class="col-md-3">
			<div class="card bg-warning text-white">
				<div class="card-body">
					<h4>{{ "%.1f"|format(stats.avg_processing_time_ms) }}ms</h4>
					<p>Avg Processing Time</p>
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
						<a href="{{ url_for('ComputerVisionView.image_processing') }}" class="btn btn-primary">
							<i class="fa fa-image"></i> Process Image
						</a>
						<a href="{{ url_for('ComputerVisionView.video_processing') }}" class="btn btn-success">
							<i class="fa fa-video"></i> Process Video
						</a>
						<a href="{{ url_for('ComputerVisionView.live_stream') }}" class="btn btn-warning">
							<i class="fa fa-camera"></i> Live Stream
						</a>
						<a href="{{ url_for('ComputerVisionView.analytics') }}" class="btn btn-info">
							<i class="fa fa-chart-bar"></i> Analytics
						</a>
					</div>
				</div>
			</div>
		</div>
		
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5>Detection Types</h5>
				</div>
				<div class="card-body">
					<div class="row">
						{% for det_type in stats.top_detection_types %}
						<div class="col-md-6 mb-2">
							<div class="d-flex justify-content-between">
								<span>{{ det_type.type.title() }}</span>
								<span class="badge bg-primary">{{ det_type.count }}</span>
							</div>
						</div>
						{% endfor %}
					</div>
				</div>
			</div>
		</div>
	</div>
</div>
{% endblock %}
""",

	'computer_vision/image_processing.html': """
{% extends "appbuilder/base.html" %}

{% block content %}
<div class="container-fluid">
	<div class="row">
		<div class="col-12">
			<h1><i class="fa fa-image"></i> Image Processing</h1>
		</div>
	</div>
	
	<div class="row">
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5>Upload & Process Image</h5>
				</div>
				<div class="card-body">
					<form method="POST" enctype="multipart/form-data">
						{{ form.hidden_tag() }}
						
						<div class="mb-3">
							{{ form.image_file.label(class="form-label") }}
							{{ form.image_file(class="form-control") }}
							<div class="form-text">{{ form.image_file.description }}</div>
						</div>
						
						<div class="mb-3">
							{{ form.detection_types.label(class="form-label") }}
							{{ form.detection_types(class="form-select") }}
							<div class="form-text">{{ form.detection_types.description }}</div>
						</div>
						
						<div class="mb-3">
							{{ form.enhancement_type.label(class="form-label") }}
							{{ form.enhancement_type(class="form-select") }}
							<div class="form-text">{{ form.enhancement_type.description }}</div>
						</div>
						
						<button type="submit" class="btn btn-primary">
							<i class="fa fa-cog"></i> Process Image
						</button>
					</form>
				</div>
			</div>
		</div>
		
		{% if results %}
		<div class="col-md-6">
			<div class="card">
				<div class="card-header">
					<h5>Processing Results</h5>
				</div>
				<div class="card-body">
					<p><strong>Detections Found:</strong> {{ results.detection_count }}</p>
					
					<div class="row">
						<div class="col-12 mb-3">
							<h6>Original Image</h6>
							<img src="{{ results.original_image }}" class="img-fluid" alt="Original">
						</div>
						<div class="col-12 mb-3">
							<h6>Annotated Image</h6>
							<img src="{{ results.annotated_image }}" class="img-fluid" alt="Annotated">
						</div>
					</div>
					
					{% if results.detections %}
					<h6>Detection Details</h6>
					<div class="table-responsive">
						<table class="table table-sm">
							<thead>
								<tr>
									<th>Type</th>
									<th>Confidence</th>
									<th>Position</th>
								</tr>
							</thead>
							<tbody>
								{% for detection in results.detections %}
								<tr>
									<td>{{ detection.type.title() }}</td>
									<td>{{ "%.2f"|format(detection.confidence) }}</td>
									<td>{{ detection.center }}</td>
								</tr>
								{% endfor %}
							</tbody>
						</table>
					</div>
					{% endif %}
				</div>
			</div>
		</div>
		{% endif %}
	</div>
</div>
{% endblock %}
"""
}

# Blueprint registration
computer_vision_bp = Blueprint(
	'computer_vision',
	__name__,
	template_folder='templates',
	static_folder='static'
)

# Export the view class for AppBuilder registration
__all__ = ['ComputerVisionView', 'computer_vision_bp']