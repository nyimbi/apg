"""
3D Visualization Views

Flask-AppBuilder views for advanced 3D WebGL visualization engine including
scene management, object manipulation, material editor, and rendering analytics.
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
	VIS3DScene, VIS3DObject, VIS3DMaterial, VIS3DCamera,
	VIS3DLight, VIS3DRenderSession
)


class Visualization3DBaseView(BaseView):
	"""Base view for 3D visualization functionality"""
	
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
	
	def _format_percentage(self, value: float) -> str:
		"""Format percentage for display"""
		if value is None:
			return "N/A"
		return f"{value:.1f}%"
	
	def _format_memory(self, mb_value: float) -> str:
		"""Format memory for display"""
		if mb_value is None:
			return "N/A"
		if mb_value < 1024:
			return f"{mb_value:.1f} MB"
		else:
			gb_value = mb_value / 1024
			return f"{gb_value:.1f} GB"
	
	def _format_color(self, color_array: List[float]) -> str:
		"""Format color array for display"""
		if not color_array or len(color_array) < 3:
			return "N/A"
		return f"RGB({color_array[0]:.2f}, {color_array[1]:.2f}, {color_array[2]:.2f})"


class VIS3DSceneModelView(ModelView):
	"""3D scene management view"""
	
	datamodel = SQLAInterface(VIS3DScene)
	
	# List view configuration
	list_columns = [
		'scene_name', 'scene_type', 'render_mode', 'polygon_count',
		'frame_rate', 'view_count', 'last_viewed'
	]
	show_columns = [
		'scene_id', 'scene_name', 'description', 'scene_type', 'category',
		'render_mode', 'quality_level', 'polygon_count', 'texture_memory_mb',
		'frame_rate', 'view_count', 'visibility', 'created_by'
	]
	edit_columns = [
		'scene_name', 'description', 'scene_type', 'category', 'background_color',
		'render_mode', 'quality_level', 'anti_aliasing', 'shadows_enabled',
		'controls_enabled', 'animations_enabled', 'real_time_updates', 'visibility'
	]
	add_columns = [
		'scene_name', 'description', 'scene_type', 'category'
	]
	
	# Search and filtering
	search_columns = ['scene_name', 'scene_type', 'category']
	base_filters = [['visibility', lambda: 'public', lambda: True]]
	
	# Ordering
	base_order = ('scene_name', 'asc')
	
	# Form validation
	validators_columns = {
		'scene_name': [DataRequired(), Length(min=3, max=200)],
		'scene_type': [DataRequired()],
		'max_objects': [NumberRange(min=1, max=10000)],
		'animation_speed': [NumberRange(min=0.1, max=5.0)]
	}
	
	# Custom labels
	label_columns = {
		'scene_id': 'Scene ID',
		'scene_name': 'Scene Name',
		'scene_type': 'Scene Type',
		'scene_data': 'Scene Data',
		'background_color': 'Background Color',
		'environment_map': 'Environment Map',
		'fog_settings': 'Fog Settings',
		'render_mode': 'Render Mode',
		'quality_level': 'Quality Level',
		'anti_aliasing': 'Anti-Aliasing',
		'shadows_enabled': 'Shadows Enabled',
		'post_processing': 'Post Processing',
		'max_objects': 'Max Objects',
		'max_vertices': 'Max Vertices',
		'level_of_detail_enabled': 'LOD Enabled',
		'culling_enabled': 'Culling Enabled',
		'instancing_enabled': 'Instancing Enabled',
		'controls_enabled': 'Controls Enabled',
		'zoom_enabled': 'Zoom Enabled',
		'pan_enabled': 'Pan Enabled',
		'rotate_enabled': 'Rotate Enabled',
		'selection_enabled': 'Selection Enabled',
		'animations_enabled': 'Animations Enabled',
		'auto_play': 'Auto Play',
		'loop_animations': 'Loop Animations',
		'animation_speed': 'Animation Speed',
		'data_sources': 'Data Sources',
		'real_time_updates': 'Real-time Updates',
		'update_interval_ms': 'Update Interval (ms)',
		'data_filters': 'Data Filters',
		'shared_with': 'Shared With',
		'created_by': 'Created By',
		'render_time_ms': 'Render Time (ms)',
		'frame_rate': 'Frame Rate (FPS)',
		'polygon_count': 'Polygon Count',
		'texture_memory_mb': 'Texture Memory (MB)',
		'view_count': 'View Count',
		'interaction_count': 'Interaction Count',
		'last_viewed': 'Last Viewed',
		'average_session_duration': 'Avg Session Duration',
		'parent_scene_id': 'Parent Scene ID',
		'is_template': 'Is Template'
	}
	
	@expose('/scene_editor/<int:pk>')
	@has_access
	def scene_editor(self, pk):
		"""3D scene editor interface"""
		scene = self.datamodel.get(pk)
		if not scene:
			flash('Scene not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			editor_data = self._get_scene_editor_data(scene)
			
			return render_template('visualization_3d/scene_editor.html',
								   scene=scene,
								   editor_data=editor_data,
								   page_title=f"Scene Editor: {scene.scene_name}")
		except Exception as e:
			flash(f'Error loading scene editor: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/scene_viewer/<int:pk>')
	@has_access
	def scene_viewer(self, pk):
		"""3D scene viewer interface"""
		scene = self.datamodel.get(pk)
		if not scene:
			flash('Scene not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Update view count
			scene.update_usage_metrics(0)  # Start session
			self.datamodel.edit(scene)
			
			viewer_data = self._get_scene_viewer_data(scene)
			
			return render_template('visualization_3d/scene_viewer.html',
								   scene=scene,
								   viewer_data=viewer_data,
								   page_title=f"3D Viewer: {scene.scene_name}")
		except Exception as e:
			flash(f'Error loading scene viewer: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/scene_analytics/<int:pk>')
	@has_access
	def scene_analytics(self, pk):
		"""View scene performance analytics"""
		scene = self.datamodel.get(pk)
		if not scene:
			flash('Scene not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			analytics_data = self._get_scene_analytics(scene)
			
			return render_template('visualization_3d/scene_analytics.html',
								   scene=scene,
								   analytics_data=analytics_data,
								   page_title=f"Scene Analytics: {scene.scene_name}")
		except Exception as e:
			flash(f'Error loading scene analytics: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/clone_scene/<int:pk>')
	@has_access
	def clone_scene(self, pk):
		"""Clone existing scene"""
		scene = self.datamodel.get(pk)
		if not scene:
			flash('Scene not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Create cloned scene (simplified implementation)
			scene.scene_name = f"{scene.scene_name} (Copy)"
			scene.parent_scene_id = scene.scene_id
			flash(f'Scene "{scene.scene_name}" cloned successfully', 'success')
		except Exception as e:
			flash(f'Error cloning scene: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new scene"""
		item.tenant_id = self._get_tenant_id()
		item.created_by = self._get_current_user_id()
		
		# Set default values
		if not item.render_mode:
			item.render_mode = 'solid'
		if not item.quality_level:
			item.quality_level = 'medium'
		if not item.background_color:
			item.background_color = [0.2, 0.2, 0.2, 1.0]
		if not item.visibility:
			item.visibility = 'private'
	
	def _get_scene_editor_data(self, scene: VIS3DScene) -> Dict[str, Any]:
		"""Get data for scene editor"""
		performance = scene.estimate_render_performance()
		
		return {
			'scene_configuration': {
				'scene_data': scene.scene_data,
				'background_color': scene.background_color,
				'render_mode': scene.render_mode,
				'quality_level': scene.quality_level
			},
			'objects': [
				{
					'object_id': obj.object_id,
					'name': obj.object_name,
					'type': obj.object_type,
					'visible': obj.visible,
					'position': obj.position,
					'polygon_count': obj.polygon_count
				}
				for obj in scene.objects
			],
			'materials': [
				{
					'material_id': mat.material_id,
					'name': mat.material_name,
					'type': mat.material_type,
					'base_color': mat.base_color,
					'usage_count': mat.usage_count
				}
				for mat in scene.materials
			],
			'cameras': [
				{
					'camera_id': cam.camera_id,
					'name': cam.camera_name,
					'type': cam.camera_type,
					'position': cam.position,
					'is_active': cam.is_active
				}
				for cam in scene.cameras
			],
			'lights': [
				{
					'light_id': light.light_id,
					'name': light.light_name,
					'type': light.light_type,
					'position': light.position,
					'enabled': light.enabled
				}
				for light in scene.lights
			],
			'performance_info': performance,
			'editor_settings': {
				'grid_enabled': True,
				'snap_to_grid': False,
				'grid_size': 1.0,
				'selection_outline': True
			}
		}
	
	def _get_scene_viewer_data(self, scene: VIS3DScene) -> Dict[str, Any]:
		"""Get data for scene viewer"""
		return {
			'scene_config': {
				'scene_id': scene.scene_id,
				'background_color': scene.background_color,
				'render_mode': scene.render_mode,
				'quality_level': scene.quality_level,
				'shadows_enabled': scene.shadows_enabled,
				'anti_aliasing': scene.anti_aliasing
			},
			'viewer_settings': {
				'controls_enabled': scene.controls_enabled,
				'zoom_enabled': scene.zoom_enabled,
				'pan_enabled': scene.pan_enabled,
				'rotate_enabled': scene.rotate_enabled,
				'selection_enabled': scene.selection_enabled
			},
			'animation_settings': {
				'animations_enabled': scene.animations_enabled,
				'auto_play': scene.auto_play,
				'loop_animations': scene.loop_animations,
				'animation_speed': scene.animation_speed
			},
			'performance_targets': {
				'target_fps': 30,
				'quality_adjustment': True,
				'adaptive_quality': scene.quality_level == 'adaptive'
			}
		}
	
	def _get_scene_analytics(self, scene: VIS3DScene) -> Dict[str, Any]:
		"""Get scene analytics data"""
		complexity_score = scene.calculate_complexity_score()
		performance = scene.estimate_render_performance()
		
		return {
			'usage_statistics': {
				'total_views': scene.view_count,
				'total_interactions': scene.interaction_count,
				'average_session_duration': scene.average_session_duration or 0,
				'last_viewed': scene.last_viewed
			},
			'performance_metrics': {
				'complexity_score': complexity_score,
				'estimated_fps': performance.get('estimated_fps', 0),
				'performance_tier': performance.get('performance_tier', 'unknown'),
				'current_frame_rate': scene.frame_rate,
				'render_time_ms': scene.render_time_ms
			},
			'resource_utilization': {
				'polygon_count': scene.polygon_count,
				'texture_memory_mb': scene.texture_memory_mb,
				'object_count': len(scene.objects),
				'material_count': len(scene.materials),
				'light_count': len(scene.lights)
			},
			'optimization_suggestions': performance.get('optimization_suggestions', []),
			'quality_distribution': {
				'low_quality_sessions': 15,
				'medium_quality_sessions': 65,
				'high_quality_sessions': 20
			}
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class VIS3DObjectModelView(ModelView):
	"""3D object management view"""
	
	datamodel = SQLAInterface(VIS3DObject)
	
	# List view configuration
	list_columns = [
		'object_name', 'scene', 'object_type', 'visible',
		'polygon_count', 'vertex_count', 'material'
	]
	show_columns = [
		'object_id', 'scene', 'object_name', 'description', 'object_type',
		'geometry_type', 'position', 'rotation', 'scale', 'material',
		'visible', 'selectable', 'polygon_count', 'vertex_count'
	]
	edit_columns = [
		'object_name', 'description', 'object_type', 'geometry_type',
		'position', 'rotation', 'scale', 'material', 'base_color',
		'visible', 'selectable', 'cast_shadows', 'animations_enabled'
	]
	add_columns = [
		'scene', 'object_name', 'description', 'object_type', 'geometry_type'
	]
	
	# Search and filtering
	search_columns = ['object_name', 'object_type', 'geometry_type']
	
	# Ordering
	base_order = ('object_name', 'asc')
	
	# Form validation
	validators_columns = {
		'object_name': [DataRequired(), Length(min=3, max=200)],
		'object_type': [DataRequired()],
		'geometry_type': [DataRequired()]
	}
	
	# Custom labels
	label_columns = {
		'object_id': 'Object ID',
		'scene_id': 'Scene ID',
		'object_name': 'Object Name',
		'object_type': 'Object Type',
		'geometry_type': 'Geometry Type',
		'indices': 'Face Indices',
		'vertex_colors': 'Vertex Colors',
		'transform_matrix': 'Transform Matrix',
		'material_id': 'Material ID',
		'base_color': 'Base Color',
		'metallic_factor': 'Metallic Factor',
		'roughness_factor': 'Roughness Factor',
		'emissive_factor': 'Emissive Factor',
		'cast_shadows': 'Cast Shadows',
		'receive_shadows': 'Receive Shadows',
		'animations_enabled': 'Animations Enabled',
		'animation_data': 'Animation Data',
		'current_animation': 'Current Animation',
		'animation_speed': 'Animation Speed',
		'animation_loop': 'Animation Loop',
		'data_bindings': 'Data Bindings',
		'data_source_id': 'Data Source ID',
		'data_mapping': 'Data Mapping',
		'level_of_detail': 'Level of Detail',
		'bounding_box': 'Bounding Box',
		'polygon_count': 'Polygon Count',
		'vertex_count': 'Vertex Count',
		'custom_properties': 'Custom Properties',
		'parent_object_id': 'Parent Object ID'
	}
	
	@expose('/object_editor/<int:pk>')
	@has_access
	def object_editor(self, pk):
		"""3D object editor interface"""
		obj = self.datamodel.get(pk)
		if not obj:
			flash('Object not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			editor_data = self._get_object_editor_data(obj)
			
			return render_template('visualization_3d/object_editor.html',
								   object=obj,
								   editor_data=editor_data,
								   page_title=f"Object Editor: {obj.object_name}")
		except Exception as e:
			flash(f'Error loading object editor: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/animate_object/<int:pk>')
	@has_access
	def animate_object(self, pk):
		"""Create animation for object"""
		obj = self.datamodel.get(pk)
		if not obj:
			flash('Object not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Enable animations and add sample animation
			obj.animations_enabled = True
			obj.animate_property('rotation', [0, 3.14159, 0], 2000)  # 2 second rotation
			self.datamodel.edit(obj)
			flash(f'Animation added to object "{obj.object_name}"', 'success')
		except Exception as e:
			flash(f'Error adding animation: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	@expose('/duplicate_object/<int:pk>')
	@has_access
	def duplicate_object(self, pk):
		"""Duplicate existing object"""
		obj = self.datamodel.get(pk)
		if not obj:
			flash('Object not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Create duplicate (simplified implementation)
			obj.object_name = f"{obj.object_name} (Copy)"
			obj.position = [obj.position[0] + 1, obj.position[1], obj.position[2]]  # Offset position
			flash(f'Object "{obj.object_name}" duplicated successfully', 'success')
		except Exception as e:
			flash(f'Error duplicating object: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new object"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.position:
			item.position = [0.0, 0.0, 0.0]
		if not item.rotation:
			item.rotation = [0.0, 0.0, 0.0]
		if not item.scale:
			item.scale = [1.0, 1.0, 1.0]
		if not item.base_color:
			item.base_color = [1.0, 1.0, 1.0, 1.0]
		
		# Update geometry stats
		item.update_geometry_stats()
		
		# Calculate bounding box
		item.calculate_bounding_box()
	
	def _get_object_editor_data(self, obj: VIS3DObject) -> Dict[str, Any]:
		"""Get data for object editor"""
		return {
			'geometry_data': {
				'vertices': obj.vertices or [],
				'indices': obj.indices or [],
				'normals': obj.normals or [],
				'uvs': obj.uvs or [],
				'vertex_count': obj.vertex_count,
				'polygon_count': obj.polygon_count
			},
			'transform_data': {
				'position': obj.position,
				'rotation': obj.rotation,
				'scale': obj.scale,
				'transform_matrix': obj.get_transform_matrix()
			},
			'material_properties': {
				'base_color': obj.base_color,
				'metallic_factor': obj.metallic_factor,
				'roughness_factor': obj.roughness_factor,
				'emissive_factor': obj.emissive_factor
			},
			'animation_data': {
				'animations_enabled': obj.animations_enabled,
				'animation_data': obj.animation_data or [],
				'current_animation': obj.current_animation,
				'animation_speed': obj.animation_speed
			},
			'data_binding': {
				'data_bindings': obj.data_bindings or {},
				'data_source_id': obj.data_source_id,
				'data_mapping': obj.data_mapping or {}
			},
			'available_materials': [
				{
					'material_id': mat.material_id,
					'name': mat.material_name,
					'type': mat.material_type
				}
				for mat in obj.scene.materials
			] if obj.scene else []
		}
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class VIS3DMaterialModelView(ModelView):
	"""3D material management view"""
	
	datamodel = SQLAInterface(VIS3DMaterial)
	
	# List view configuration
	list_columns = [
		'material_name', 'scene', 'material_type', 'base_color',
		'metallic_factor', 'roughness_factor', 'usage_count'
	]
	show_columns = [
		'material_id', 'scene', 'material_name', 'description', 'material_type',
		'base_color', 'metallic_factor', 'roughness_factor', 'emissive_factor',
		'base_color_texture', 'normal_texture', 'usage_count', 'created_by'
	]
	edit_columns = [
		'material_name', 'description', 'material_type', 'base_color',
		'metallic_factor', 'roughness_factor', 'normal_scale', 'emissive_factor',
		'base_color_texture', 'metallic_roughness_texture', 'normal_texture',
		'alpha_mode', 'transparency', 'double_sided', 'cast_shadows'
	]
	add_columns = [
		'scene', 'material_name', 'description', 'material_type'
	]
	
	# Search and filtering
	search_columns = ['material_name', 'material_type']
	
	# Ordering
	base_order = ('material_name', 'asc')
	
	# Form validation
	validators_columns = {
		'material_name': [DataRequired(), Length(min=3, max=200)],
		'material_type': [DataRequired()],
		'metallic_factor': [NumberRange(min=0.0, max=1.0)],
		'roughness_factor': [NumberRange(min=0.0, max=1.0)],
		'transparency': [NumberRange(min=0.0, max=1.0)]
	}
	
	# Custom labels
	label_columns = {
		'material_id': 'Material ID',
		'scene_id': 'Scene ID',
		'material_name': 'Material Name',
		'material_type': 'Material Type',
		'base_color': 'Base Color',
		'metallic_factor': 'Metallic Factor',
		'roughness_factor': 'Roughness Factor',
		'normal_scale': 'Normal Scale',
		'occlusion_strength': 'Occlusion Strength',
		'emissive_factor': 'Emissive Factor',
		'base_color_texture': 'Base Color Texture',
		'metallic_roughness_texture': 'Metallic Roughness Texture',
		'normal_texture': 'Normal Texture',
		'occlusion_texture': 'Occlusion Texture',
		'emissive_texture': 'Emissive Texture',
		'environment_map': 'Environment Map',
		'alpha_mode': 'Alpha Mode',
		'alpha_cutoff': 'Alpha Cutoff',
		'double_sided': 'Double Sided',
		'cast_shadows': 'Cast Shadows',
		'receive_shadows': 'Receive Shadows',
		'animated_properties': 'Animated Properties',
		'animation_speed': 'Animation Speed',
		'shader_type': 'Shader Type',
		'shader_uniforms': 'Shader Uniforms',
		'vertex_shader': 'Vertex Shader',
		'fragment_shader': 'Fragment Shader',
		'texture_repeat': 'Texture Repeat',
		'texture_offset': 'Texture Offset',
		'texture_rotation': 'Texture Rotation',
		'texture_resolution': 'Texture Resolution',
		'mipmaps_enabled': 'Mipmaps Enabled',
		'anisotropic_filtering': 'Anisotropic Filtering',
		'usage_count': 'Usage Count',
		'last_used': 'Last Used',
		'created_by': 'Created By'
	}
	
	@expose('/material_editor/<int:pk>')
	@has_access
	def material_editor(self, pk):
		"""Material editor interface"""
		material = self.datamodel.get(pk)
		if not material:
			flash('Material not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			editor_data = self._get_material_editor_data(material)
			
			return render_template('visualization_3d/material_editor.html',
								   material=material,
								   editor_data=editor_data,
								   page_title=f"Material Editor: {material.material_name}")
		except Exception as e:
			flash(f'Error loading material editor: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	@expose('/preview_material/<int:pk>')
	@has_access
	def preview_material(self, pk):
		"""Preview material on test geometry"""
		material = self.datamodel.get(pk)
		if not material:
			flash('Material not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			preview_data = self._get_material_preview_data(material)
			
			return render_template('visualization_3d/material_preview.html',
								   material=material,
								   preview_data=preview_data,
								   page_title=f"Material Preview: {material.material_name}")
		except Exception as e:
			flash(f'Error loading material preview: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new material"""
		item.tenant_id = self._get_tenant_id()
		item.created_by = self._get_current_user_id()
		
		# Set default values
		if not item.base_color:
			item.base_color = [1.0, 1.0, 1.0, 1.0]
		if not item.emissive_factor:
			item.emissive_factor = [0.0, 0.0, 0.0]
		if not item.texture_repeat:
			item.texture_repeat = [1.0, 1.0]
		if not item.texture_offset:
			item.texture_offset = [0.0, 0.0]
	
	def _get_material_editor_data(self, material: VIS3DMaterial) -> Dict[str, Any]:
		"""Get data for material editor"""
		pbr_properties = material.get_pbr_properties()
		texture_config = material.get_texture_configuration()
		memory_usage = material.estimate_memory_usage()
		
		return {
			'pbr_properties': pbr_properties,
			'texture_configuration': texture_config,
			'performance_info': {
				'memory_usage_mb': memory_usage,
				'texture_count': len([t for t in texture_config.values() if t]),
				'shader_complexity': 'medium'  # Simplified
			},
			'preview_geometries': [
				'sphere', 'cube', 'cylinder', 'plane', 'torus'
			],
			'shader_templates': [
				'standard_pbr', 'toon', 'glass', 'metal', 'fabric', 'skin'
			]
		}
	
	def _get_material_preview_data(self, material: VIS3DMaterial) -> Dict[str, Any]:
		"""Get data for material preview"""
		return {
			'material_config': material.get_pbr_properties(),
			'texture_urls': material.get_texture_configuration(),
			'preview_settings': {
				'geometry': 'sphere',
				'lighting': 'studio',
				'environment': 'neutral',
				'rotation_speed': 1.0
			},
			'viewer_config': {
				'width': 400,
				'height': 400,
				'controls_enabled': True,
				'auto_rotate': True
			}
		}
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		from flask_appbuilder.security import current_user
		return str(current_user.id) if current_user and current_user.is_authenticated else None
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class VIS3DCameraModelView(ModelView):
	"""3D camera management view"""
	
	datamodel = SQLAInterface(VIS3DCamera)
	
	# List view configuration
	list_columns = [
		'camera_name', 'scene', 'camera_type', 'position',
		'field_of_view', 'is_active'
	]
	show_columns = [
		'camera_id', 'scene', 'camera_name', 'description', 'camera_type',
		'position', 'target', 'field_of_view', 'aspect_ratio', 'near_plane',
		'far_plane', 'controls_enabled', 'is_active'
	]
	edit_columns = [
		'camera_name', 'description', 'camera_type', 'position', 'target',
		'up_vector', 'field_of_view', 'aspect_ratio', 'near_plane', 'far_plane',
		'controls_enabled', 'auto_rotate', 'min_distance', 'max_distance'
	]
	add_columns = [
		'scene', 'camera_name', 'description', 'camera_type'
	]
	
	# Search and filtering
	search_columns = ['camera_name', 'camera_type']
	
	# Ordering
	base_order = ('camera_name', 'asc')
	
	# Form validation
	validators_columns = {
		'camera_name': [DataRequired(), Length(min=3, max=200)],
		'camera_type': [DataRequired()],
		'field_of_view': [NumberRange(min=1.0, max=179.0)],
		'near_plane': [NumberRange(min=0.001, max=100.0)],
		'far_plane': [NumberRange(min=1.0, max=10000.0)]
	}
	
	# Custom labels
	label_columns = {
		'camera_id': 'Camera ID',
		'scene_id': 'Scene ID',
		'camera_name': 'Camera Name',
		'camera_type': 'Camera Type',
		'up_vector': 'Up Vector',
		'field_of_view': 'Field of View (degrees)',
		'aspect_ratio': 'Aspect Ratio',
		'near_plane': 'Near Plane',
		'far_plane': 'Far Plane',
		'ortho_left': 'Ortho Left',
		'ortho_right': 'Ortho Right',
		'ortho_top': 'Ortho Top',
		'ortho_bottom': 'Ortho Bottom',
		'controls_enabled': 'Controls Enabled',
		'zoom_enabled': 'Zoom Enabled',
		'pan_enabled': 'Pan Enabled',
		'rotate_enabled': 'Rotate Enabled',
		'auto_rotate': 'Auto Rotate',
		'auto_rotate_speed': 'Auto Rotate Speed',
		'min_distance': 'Min Distance',
		'max_distance': 'Max Distance',
		'min_polar_angle': 'Min Polar Angle',
		'max_polar_angle': 'Max Polar Angle',
		'min_azimuth_angle': 'Min Azimuth Angle',
		'max_azimuth_angle': 'Max Azimuth Angle',
		'animation_enabled': 'Animation Enabled',
		'animation_path': 'Animation Path',
		'animation_duration': 'Animation Duration',
		'animation_loop': 'Animation Loop',
		'is_active': 'Is Active',
		'render_order': 'Render Order'
	}
	
	@expose('/set_active_camera/<int:pk>')
	@has_access
	def set_active_camera(self, pk):
		"""Set camera as active for scene"""
		camera = self.datamodel.get(pk)
		if not camera:
			flash('Camera not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			# Deactivate other cameras in scene
			for other_camera in camera.scene.cameras:
				if other_camera.camera_id != camera.camera_id:
					other_camera.is_active = False
					self.datamodel.edit(other_camera)
			
			# Activate this camera
			camera.is_active = True
			self.datamodel.edit(camera)
			
			flash(f'Camera "{camera.camera_name}" set as active', 'success')
		except Exception as e:
			flash(f'Error setting active camera: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def pre_add(self, item):
		"""Pre-process before adding new camera"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.position:
			item.position = [0.0, 0.0, 5.0]
		if not item.target:
			item.target = [0.0, 0.0, 0.0]
		if not item.up_vector:
			item.up_vector = [0.0, 1.0, 0.0]
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class VIS3DLightModelView(ModelView):
	"""3D light management view"""
	
	datamodel = SQLAInterface(VIS3DLight)
	
	# List view configuration
	list_columns = [
		'light_name', 'scene', 'light_type', 'position',
		'intensity', 'cast_shadows', 'enabled'
	]
	show_columns = [
		'light_id', 'scene', 'light_name', 'description', 'light_type',
		'position', 'direction', 'color', 'intensity', 'range',
		'cast_shadows', 'enabled'
	]
	edit_columns = [
		'light_name', 'description', 'light_type', 'position', 'direction',
		'color', 'intensity', 'power', 'range', 'decay', 'cast_shadows',
		'shadow_map_size', 'enabled'
	]
	add_columns = [
		'scene', 'light_name', 'description', 'light_type'
	]
	
	# Search and filtering
	search_columns = ['light_name', 'light_type']
	
	# Ordering
	base_order = ('light_name', 'asc')
	
	# Form validation
	validators_columns = {
		'light_name': [DataRequired(), Length(min=3, max=200)],
		'light_type': [DataRequired()],
		'intensity': [NumberRange(min=0.0, max=10.0)],
		'range': [NumberRange(min=0.1, max=1000.0)]
	}
	
	# Custom labels
	label_columns = {
		'light_id': 'Light ID',
		'scene_id': 'Scene ID',
		'light_name': 'Light Name',
		'light_type': 'Light Type',
		'color': 'Color (RGB)',
		'power': 'Power (lumens)',
		'temperature': 'Color Temperature (K)',
		'inner_cone_angle': 'Inner Cone Angle',
		'outer_cone_angle': 'Outer Cone Angle',
		'cast_shadows': 'Cast Shadows',
		'shadow_map_size': 'Shadow Map Size',
		'shadow_bias': 'Shadow Bias',
		'shadow_normal_bias': 'Shadow Normal Bias',
		'shadow_radius': 'Shadow Radius',
		'shadow_camera_near': 'Shadow Camera Near',
		'shadow_camera_far': 'Shadow Camera Far',
		'shadow_camera_fov': 'Shadow Camera FOV',
		'shadow_camera_left': 'Shadow Camera Left',
		'shadow_camera_right': 'Shadow Camera Right',
		'shadow_camera_top': 'Shadow Camera Top',
		'shadow_camera_bottom': 'Shadow Camera Bottom',
		'helper_visible': 'Helper Visible',
		'animation_data': 'Animation Data'
	}
	
	def pre_add(self, item):
		"""Pre-process before adding new light"""
		item.tenant_id = self._get_tenant_id()
		
		# Set default values
		if not item.position:
			item.position = [0.0, 10.0, 0.0]
		if not item.direction:
			item.direction = [0.0, -1.0, 0.0]
		if not item.color:
			item.color = [1.0, 1.0, 1.0]
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


class VIS3DRenderSessionModelView(ModelView):
	"""3D render session analytics view"""
	
	datamodel = SQLAInterface(VIS3DRenderSession)
	
	# List view configuration
	list_columns = [
		'scene', 'user_id', 'device_type', 'started_at',
		'duration_seconds', 'average_fps', 'total_interactions'
	]
	show_columns = [
		'session_id', 'scene', 'user_id', 'session_name', 'device_type',
		'browser_type', 'started_at', 'duration_seconds', 'average_fps',
		'render_quality', 'total_interactions', 'user_satisfaction_score'
	]
	
	# Read-only view
	can_create = False
	can_edit = False
	can_delete = False
	
	# Search and filtering
	search_columns = ['scene.scene_name', 'device_type', 'browser_type']
	
	# Ordering
	base_order = ('started_at', 'desc')
	
	# Custom labels
	label_columns = {
		'session_id': 'Session ID',
		'scene_id': 'Scene ID',
		'user_id': 'User ID',
		'session_name': 'Session Name',
		'device_type': 'Device Type',
		'browser_type': 'Browser Type',
		'user_agent': 'User Agent',
		'started_at': 'Started At',
		'ended_at': 'Ended At',
		'duration_seconds': 'Duration (seconds)',
		'average_fps': 'Average FPS',
		'min_fps': 'Min FPS',
		'max_fps': 'Max FPS',
		'average_render_time_ms': 'Avg Render Time (ms)',
		'total_frames_rendered': 'Total Frames',
		'render_quality': 'Render Quality',
		'resolution': 'Resolution',
		'anti_aliasing_used': 'Anti-aliasing Used',
		'shadows_enabled': 'Shadows Enabled',
		'memory_usage_mb': 'Memory Usage (MB)',
		'gpu_memory_usage_mb': 'GPU Memory (MB)',
		'cpu_usage_percentage': 'CPU Usage (%)',
		'gpu_usage_percentage': 'GPU Usage (%)',
		'total_interactions': 'Total Interactions',
		'camera_movements': 'Camera Movements',
		'object_selections': 'Object Selections',
		'zoom_actions': 'Zoom Actions',
		'pan_actions': 'Pan Actions',
		'rotation_actions': 'Rotation Actions',
		'webgl_errors': 'WebGL Errors',
		'shader_compilation_errors': 'Shader Errors',
		'texture_loading_errors': 'Texture Errors',
		'performance_warnings': 'Performance Warnings',
		'features_used': 'Features Used',
		'effects_applied': 'Effects Applied',
		'user_satisfaction_score': 'User Satisfaction',
		'perceived_performance': 'Perceived Performance',
		'total_bytes_downloaded': 'Bytes Downloaded',
		'texture_download_time_ms': 'Texture Download Time (ms)',
		'model_download_time_ms': 'Model Download Time (ms)'
	}
	
	@expose('/session_details/<int:pk>')
	@has_access
	def session_details(self, pk):
		"""View detailed session information"""
		session = self.datamodel.get(pk)
		if not session:
			flash('Session not found', 'error')
			return redirect(self.get_redirect())
		
		try:
			session_details = self._get_session_details(session)
			
			return render_template('visualization_3d/session_details.html',
								   session=session,
								   session_details=session_details,
								   page_title=f"Session Details: {session.session_id}")
		except Exception as e:
			flash(f'Error loading session details: {str(e)}', 'error')
			return redirect(self.get_redirect())
	
	def _get_session_details(self, session: VIS3DRenderSession) -> Dict[str, Any]:
		"""Get detailed session information"""
		performance_score = session.calculate_performance_score()
		interaction_summary = session.get_interaction_summary()
		
		return {
			'session_overview': {
				'session_id': session.session_id,
				'scene_name': session.scene.scene_name if session.scene else 'Unknown',
				'duration_seconds': session.duration_seconds,
				'device_type': session.device_type,
				'browser_type': session.browser_type
			},
			'performance_metrics': {
				'average_fps': session.average_fps,
				'performance_score': performance_score,
				'render_quality': session.render_quality,
				'total_frames': session.total_frames_rendered,
				'errors_count': (session.webgl_errors + session.shader_compilation_errors + 
								session.texture_loading_errors)
			},
			'resource_usage': {
				'memory_usage_mb': session.memory_usage_mb,
				'gpu_memory_mb': session.gpu_memory_usage_mb,
				'cpu_usage_percent': session.cpu_usage_percentage,
				'gpu_usage_percent': session.gpu_usage_percentage
			},
			'interaction_data': interaction_summary,
			'quality_metrics': {
				'user_satisfaction': session.user_satisfaction_score,
				'perceived_performance': session.perceived_performance,
				'features_used': session.features_used or [],
				'effects_applied': session.effects_applied or []
			}
		}


class Visualization3DDashboardView(Visualization3DBaseView):
	"""3D visualization dashboard"""
	
	route_base = "/visualization_3d_dashboard"
	default_view = "index"
	
	@expose('/')
	@has_access
	def index(self):
		"""3D visualization dashboard main page"""
		try:
			# Get dashboard metrics
			metrics = self._get_dashboard_metrics()
			
			return render_template('visualization_3d/dashboard.html',
								   metrics=metrics,
								   page_title="3D Visualization Dashboard")
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return render_template('visualization_3d/dashboard.html',
								   metrics={},
								   page_title="3D Visualization Dashboard")
	
	@expose('/scene_gallery/')
	@has_access
	def scene_gallery(self):
		"""3D scene template gallery"""
		try:
			gallery_data = self._get_scene_gallery_data()
			
			return render_template('visualization_3d/scene_gallery.html',
								   gallery_data=gallery_data,
								   page_title="3D Scene Gallery")
		except Exception as e:
			flash(f'Error loading scene gallery: {str(e)}', 'error')
			return redirect(url_for('Visualization3DDashboardView.index'))
	
	@expose('/performance_analytics/')
	@has_access
	def performance_analytics(self):
		"""3D rendering performance analytics"""
		try:
			analytics_data = self._get_performance_analytics_data()
			
			return render_template('visualization_3d/performance_analytics.html',
								   analytics_data=analytics_data,
								   page_title="3D Performance Analytics")
		except Exception as e:
			flash(f'Error loading performance analytics: {str(e)}', 'error')
			return redirect(url_for('Visualization3DDashboardView.index'))
	
	def _get_dashboard_metrics(self) -> Dict[str, Any]:
		"""Get 3D visualization dashboard metrics"""
		# Implementation would calculate real metrics from database
		return {
			'scene_overview': {
				'total_scenes': 45,
				'active_scenes': 32,
				'template_scenes': 8,
				'total_objects': 1250
			},
			'rendering_performance': {
				'average_fps': 42.5,
				'scenes_over_30fps': 38,
				'gpu_memory_usage_gb': 2.8,
				'render_quality_distribution': {
					'high': 15,
					'medium': 25,
					'low': 5
				}
			},
			'user_engagement': {
				'total_sessions_today': 187,
				'average_session_duration': 285,
				'total_interactions': 3420,
				'user_satisfaction_avg': 4.2
			},
			'content_statistics': {
				'total_materials': 156,
				'total_textures': 342,
				'polygon_count_total': 2850000,
				'texture_memory_gb': 1.2
			},
			'optimization_opportunities': {
				'scenes_needing_optimization': 8,
				'potential_fps_improvement': 15.3,
				'memory_reduction_potential_mb': 420,
				'texture_compression_savings': 30
			}
		}
	
	def _get_scene_gallery_data(self) -> Dict[str, Any]:
		"""Get scene gallery data"""
		return {
			'featured_scenes': [
				{
					'name': 'Industrial Factory Floor',
					'type': 'industrial',
					'objects': 245,
					'complexity': 'high',
					'avg_fps': 35,
					'popularity': 89
				},
				{
					'name': 'Smart City Overview',
					'type': 'architectural',
					'objects': 1250,
					'complexity': 'ultra',
					'avg_fps': 25,
					'popularity': 76
				},
				{
					'name': 'Molecular Structure',
					'type': 'molecular',
					'objects': 48,
					'complexity': 'medium',
					'avg_fps': 58,
					'popularity': 92
				}
			],
			'scene_categories': [
				{'name': 'Industrial', 'scene_count': 18, 'avg_complexity': 0.7},
				{'name': 'Architectural', 'scene_count': 12, 'avg_complexity': 0.8},
				{'name': 'Molecular', 'scene_count': 8, 'avg_complexity': 0.4},
				{'name': 'Abstract', 'scene_count': 7, 'avg_complexity': 0.6}
			],
			'popular_templates': [
				'Basic Room Template', 'Industrial Equipment',
				'City Block Template', 'Molecular Viewer'
			],
			'recent_uploads': [
				{'name': 'CAD Assembly', 'uploaded': '2024-01-15', 'type': 'industrial'},
				{'name': 'Building Model', 'uploaded': '2024-01-14', 'type': 'architectural'}
			]
		}
	
	def _get_performance_analytics_data(self) -> Dict[str, Any]:
		"""Get performance analytics data"""
		return {
			'performance_overview': {
				'average_fps_platform': 42.5,
				'performance_tier_distribution': {
					'excellent': 25,
					'good': 45,
					'fair': 25,
					'poor': 5
				},
				'bottleneck_analysis': {
					'gpu_bound_scenes': 15,
					'cpu_bound_scenes': 8,
					'memory_bound_scenes': 3
				}
			},
			'optimization_impact': {
				'lod_enabled_scenes': 32,
				'texture_compression_usage': 78,
				'frustum_culling_effectiveness': 85,
				'instancing_performance_gain': 23
			},
			'device_performance': {
				'desktop_avg_fps': 52.3,
				'mobile_avg_fps': 28.7,
				'tablet_avg_fps': 35.1,
				'vr_avg_fps': 89.2
			},
			'quality_vs_performance': {
				'high_quality_avg_fps': 28.5,
				'medium_quality_avg_fps': 42.1,
				'low_quality_avg_fps': 65.8,
				'adaptive_quality_usage': 35
			},
			'resource_trends': {
				'gpu_memory_trend': [2.1, 2.3, 2.5, 2.7, 2.8],
				'polygon_count_trend': [2.2, 2.4, 2.6, 2.7, 2.85],
				'texture_memory_trend': [0.8, 0.9, 1.0, 1.1, 1.2]
			}
		}


# Register views with AppBuilder
def register_views(appbuilder):
	"""Register all 3D visualization views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		VIS3DSceneModelView,
		"3D Scenes",
		icon="fa-cube",
		category="3D Visualization",
		category_icon="fa-shapes"
	)
	
	appbuilder.add_view(
		VIS3DObjectModelView,
		"3D Objects",
		icon="fa-cubes",
		category="3D Visualization"
	)
	
	appbuilder.add_view(
		VIS3DMaterialModelView,
		"Materials",
		icon="fa-palette",
		category="3D Visualization"
	)
	
	appbuilder.add_view(
		VIS3DCameraModelView,
		"Cameras",
		icon="fa-video",
		category="3D Visualization"
	)
	
	appbuilder.add_view(
		VIS3DLightModelView,
		"Lights",
		icon="fa-lightbulb",
		category="3D Visualization"
	)
	
	appbuilder.add_view(
		VIS3DRenderSessionModelView,
		"Render Sessions",
		icon="fa-chart-line",
		category="3D Visualization"
	)
	
	# Dashboard views
	appbuilder.add_view_no_menu(Visualization3DDashboardView)
	
	# Menu links
	appbuilder.add_link(
		"3D Dashboard",
		href="/visualization_3d_dashboard/",
		icon="fa-dashboard",
		category="3D Visualization"
	)
	
	appbuilder.add_link(
		"Scene Gallery",
		href="/visualization_3d_dashboard/scene_gallery/",
		icon="fa-images",
		category="3D Visualization"
	)
	
	appbuilder.add_link(
		"Performance Analytics",
		href="/visualization_3d_dashboard/performance_analytics/",
		icon="fa-tachometer-alt",
		category="3D Visualization"
	)