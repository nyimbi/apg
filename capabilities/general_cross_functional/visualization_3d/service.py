#!/usr/bin/env python3
"""
Advanced 3D WebGL Visualization Engine
=====================================

High-performance 3D visualization system for digital twins with real-time rendering,
interactive controls, and advanced visual effects. Supports complex geometries,
animations, and data overlays.
"""

import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import base64
import gzip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualization_3d")

class RenderMode(Enum):
	"""3D rendering modes"""
	WIREFRAME = "wireframe"
	SOLID = "solid"
	TEXTURED = "textured"
	TRANSPARENT = "transparent"
	POINT_CLOUD = "point_cloud"
	HEAT_MAP = "heat_map"

class GeometryType(Enum):
	"""3D geometry types"""
	MESH = "mesh"
	CAD = "cad"
	POINT_CLOUD = "point_cloud"
	VOXEL = "voxel"
	PRIMITIVE = "primitive"
	PROCEDURAL = "procedural"

class AnimationType(Enum):
	"""Animation types"""
	ROTATION = "rotation"
	TRANSLATION = "translation"
	SCALE = "scale"
	DEFORMATION = "deformation"
	COLOR_CHANGE = "color_change"
	OPACITY_CHANGE = "opacity_change"

@dataclass
class Material:
	"""3D material properties"""
	name: str
	color: List[float]  # RGBA
	metallic: float = 0.0
	roughness: float = 0.5
	opacity: float = 1.0
	emissive: List[float] = None
	texture_url: str = None
	normal_map_url: str = None
	
	def __post_init__(self):
		if self.emissive is None:
			self.emissive = [0.0, 0.0, 0.0]

@dataclass
class Geometry:
	"""3D geometry data"""
	geometry_id: str
	geometry_type: GeometryType
	vertices: List[List[float]]  # 3D coordinates
	indices: List[List[int]] = None  # Face indices
	normals: List[List[float]] = None  # Vertex normals
	uvs: List[List[float]] = None  # Texture coordinates
	colors: List[List[float]] = None  # Vertex colors
	metadata: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.metadata is None:
			self.metadata = {}

@dataclass
class SceneObject:
	"""3D scene object"""
	object_id: str
	name: str
	geometry: Geometry
	material: Material
	transform: Dict[str, List[float]]  # position, rotation, scale
	visible: bool = True
	selectable: bool = True
	animations: List[Dict[str, Any]] = None
	data_bindings: Dict[str, str] = None  # Bind object properties to data
	
	def __post_init__(self):
		if self.animations is None:
			self.animations = []
		if self.data_bindings is None:
			self.data_bindings = {}
		if 'position' not in self.transform:
			self.transform['position'] = [0.0, 0.0, 0.0]
		if 'rotation' not in self.transform:
			self.transform['rotation'] = [0.0, 0.0, 0.0]
		if 'scale' not in self.transform:
			self.transform['scale'] = [1.0, 1.0, 1.0]

@dataclass
class Camera:
	"""3D camera configuration"""
	camera_type: str  # perspective, orthographic
	position: List[float]
	target: List[float]
	up: List[float] = None
	fov: float = 60.0  # Field of view for perspective camera
	near: float = 0.1
	far: float = 1000.0
	
	def __post_init__(self):
		if self.up is None:
			self.up = [0.0, 1.0, 0.0]

@dataclass
class Light:
	"""3D lighting configuration"""
	light_id: str
	light_type: str  # directional, point, spot, ambient
	position: List[float] = None
	direction: List[float] = None
	color: List[float] = None
	intensity: float = 1.0
	shadows: bool = False
	
	def __post_init__(self):
		if self.color is None:
			self.color = [1.0, 1.0, 1.0]

@dataclass
class Scene3D:
	"""Complete 3D scene"""
	scene_id: str
	name: str
	objects: List[SceneObject]
	camera: Camera
	lights: List[Light]
	background_color: List[float] = None
	environment_map: str = None
	fog: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.background_color is None:
			self.background_color = [0.2, 0.2, 0.2, 1.0]

class GeometryGenerator:
	"""Generate procedural geometries"""
	
	@staticmethod
	def create_box(width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> Geometry:
		"""Create a box geometry"""
		w, h, d = width/2, height/2, depth/2
		
		vertices = [
			# Front face
			[-w, -h,  d], [ w, -h,  d], [ w,  h,  d], [-w,  h,  d],
			# Back face
			[-w, -h, -d], [-w,  h, -d], [ w,  h, -d], [ w, -h, -d],
			# Top face
			[-w,  h, -d], [-w,  h,  d], [ w,  h,  d], [ w,  h, -d],
			# Bottom face
			[-w, -h, -d], [ w, -h, -d], [ w, -h,  d], [-w, -h,  d],
			# Right face
			[ w, -h, -d], [ w,  h, -d], [ w,  h,  d], [ w, -h,  d],
			# Left face
			[-w, -h, -d], [-w, -h,  d], [-w,  h,  d], [-w,  h, -d]
		]
		
		indices = [
			# Front face
			[0, 1, 2], [0, 2, 3],
			# Back face
			[4, 5, 6], [4, 6, 7],
			# Top face
			[8, 9, 10], [8, 10, 11],
			# Bottom face
			[12, 13, 14], [12, 14, 15],
			# Right face
			[16, 17, 18], [16, 18, 19],
			# Left face
			[20, 21, 22], [20, 22, 23]
		]
		
		return Geometry(
			geometry_id="box_geometry",
			geometry_type=GeometryType.MESH,
			vertices=vertices,
			indices=indices
		)
	
	@staticmethod
	def create_sphere(radius: float = 1.0, segments: int = 32) -> Geometry:
		"""Create a sphere geometry"""
		vertices = []
		indices = []
		
		for i in range(segments + 1):
			theta = i * np.pi / segments  # 0 to π
			sin_theta = np.sin(theta)
			cos_theta = np.cos(theta)
			
			for j in range(segments + 1):
				phi = j * 2 * np.pi / segments  # 0 to 2π
				sin_phi = np.sin(phi)
				cos_phi = np.cos(phi)
				
				x = radius * sin_theta * cos_phi
				y = radius * cos_theta
				z = radius * sin_theta * sin_phi
				
				vertices.append([x, y, z])
		
		# Generate indices
		for i in range(segments):
			for j in range(segments):
				first = i * (segments + 1) + j
				second = first + segments + 1
				
				# Triangle 1
				indices.append([first, second, first + 1])
				# Triangle 2
				indices.append([second, second + 1, first + 1])
		
		return Geometry(
			geometry_id="sphere_geometry",
			geometry_type=GeometryType.MESH,
			vertices=vertices,
			indices=indices
		)
	
	@staticmethod
	def create_cylinder(radius: float = 1.0, height: float = 2.0, segments: int = 32) -> Geometry:
		"""Create a cylinder geometry"""
		vertices = []
		indices = []
		h = height / 2
		
		# Top and bottom center vertices
		vertices.append([0, h, 0])   # Top center
		vertices.append([0, -h, 0])  # Bottom center
		
		# Side vertices
		for i in range(segments):
			angle = 2 * np.pi * i / segments
			x = radius * np.cos(angle)
			z = radius * np.sin(angle)
			
			vertices.append([x, h, z])   # Top edge
			vertices.append([x, -h, z])  # Bottom edge
		
		# Top face indices
		for i in range(segments):
			next_i = (i + 1) % segments
			indices.append([0, 2 + i * 2, 2 + next_i * 2])
		
		# Bottom face indices
		for i in range(segments):
			next_i = (i + 1) % segments
			indices.append([1, 3 + next_i * 2, 3 + i * 2])
		
		# Side face indices
		for i in range(segments):
			next_i = (i + 1) % segments
			top1 = 2 + i * 2
			bottom1 = 3 + i * 2
			top2 = 2 + next_i * 2
			bottom2 = 3 + next_i * 2
			
			indices.append([top1, bottom1, top2])
			indices.append([bottom1, bottom2, top2])
		
		return Geometry(
			geometry_id="cylinder_geometry",
			geometry_type=GeometryType.MESH,
			vertices=vertices,
			indices=indices
		)

class DataVisualization:
	"""Visualize data on 3D objects"""
	
	@staticmethod
	def create_heat_map_material(data_values: List[float], 
								color_scale: str = "viridis") -> Material:
		"""Create heat map material based on data values"""
		
		# Normalize data values
		min_val = min(data_values)
		max_val = max(data_values)
		normalized = [(v - min_val) / (max_val - min_val) for v in data_values]
		
		# Color scales
		color_scales = {
			"viridis": [(0.267, 0.004, 0.329), (0.229, 0.322, 0.545), 
						(0.127, 0.566, 0.551), (0.369, 0.788, 0.382), (0.993, 0.906, 0.144)],
			"plasma": [(0.050, 0.030, 0.529), (0.493, 0.016, 0.658), 
					   (0.762, 0.094, 0.562), (0.940, 0.437, 0.383), (0.988, 0.809, 0.145)],
			"hot": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]
		}
		
		colors = color_scales.get(color_scale, color_scales["viridis"])
		
		# Interpolate colors based on data
		avg_value = sum(normalized) / len(normalized)
		color_idx = int(avg_value * (len(colors) - 1))
		color_idx = max(0, min(color_idx, len(colors) - 1))
		
		base_color = list(colors[color_idx]) + [1.0]  # Add alpha
		
		return Material(
			name="heat_map_material",
			color=base_color,
			metallic=0.0,
			roughness=0.8,
			opacity=0.8
		)
	
	@staticmethod
	def create_point_cloud_from_data(data_points: List[Dict[str, float]], 
									x_field: str, y_field: str, z_field: str,
									value_field: str = None) -> Geometry:
		"""Create point cloud geometry from data"""
		
		vertices = []
		colors = []
		
		for point in data_points:
			x = point.get(x_field, 0.0)
			y = point.get(y_field, 0.0) 
			z = point.get(z_field, 0.0)
			vertices.append([x, y, z])
			
			if value_field and value_field in point:
				# Color based on value
				value = point[value_field]
				normalized_value = min(1.0, max(0.0, value))
				colors.append([normalized_value, 1.0 - normalized_value, 0.0, 1.0])
			else:
				colors.append([1.0, 1.0, 1.0, 1.0])
		
		return Geometry(
			geometry_id="data_point_cloud",
			geometry_type=GeometryType.POINT_CLOUD,
			vertices=vertices,
			colors=colors
		)

class AnimationController:
	"""Control 3D animations"""
	
	@staticmethod
	def create_rotation_animation(axis: List[float] = [0, 1, 0], 
								 speed: float = 1.0) -> Dict[str, Any]:
		"""Create rotation animation"""
		return {
			"type": AnimationType.ROTATION.value,
			"axis": axis,
			"speed": speed,
			"duration": -1,  # Infinite
			"easing": "linear"
		}
	
	@staticmethod
	def create_pulse_animation(scale_factor: float = 1.2, 
							  duration: float = 2.0) -> Dict[str, Any]:
		"""Create pulsing scale animation"""
		return {
			"type": AnimationType.SCALE.value,
			"keyframes": [
				{"time": 0.0, "value": [1.0, 1.0, 1.0]},
				{"time": 0.5, "value": [scale_factor, scale_factor, scale_factor]},
				{"time": 1.0, "value": [1.0, 1.0, 1.0]}
			],
			"duration": duration,
			"loop": True,
			"easing": "ease-in-out"
		}
	
	@staticmethod
	def create_color_transition_animation(start_color: List[float], 
										 end_color: List[float],
										 duration: float = 3.0) -> Dict[str, Any]:
		"""Create color transition animation"""
		return {
			"type": AnimationType.COLOR_CHANGE.value,
			"keyframes": [
				{"time": 0.0, "value": start_color},
				{"time": 1.0, "value": end_color}
			],
			"duration": duration,
			"loop": False,
			"easing": "ease-in-out"
		}

class Scene3DBuilder:
	"""Build complex 3D scenes"""
	
	def __init__(self):
		self.objects = []
		self.lights = []
		self.camera = None
		
	def add_object(self, scene_object: SceneObject) -> 'Scene3DBuilder':
		"""Add object to scene"""
		self.objects.append(scene_object)
		return self
		
	def add_light(self, light: Light) -> 'Scene3DBuilder':
		"""Add light to scene"""
		self.lights.append(light)
		return self
		
	def set_camera(self, camera: Camera) -> 'Scene3DBuilder':
		"""Set scene camera"""
		self.camera = camera
		return self
		
	def create_industrial_scene(self, twin_data: Dict[str, Any]) -> Scene3D:
		"""Create industrial equipment scene"""
		
		# Main equipment (cylinder for pump/motor)
		equipment_geometry = GeometryGenerator.create_cylinder(radius=2.0, height=4.0)
		equipment_material = Material(
			name="equipment_metal",
			color=[0.7, 0.7, 0.8, 1.0],
			metallic=0.8,
			roughness=0.3
		)
		
		equipment = SceneObject(
			object_id="main_equipment",
			name="Main Equipment",
			geometry=equipment_geometry,
			material=equipment_material,
			transform={
				"position": [0, 0, 0],
				"rotation": [0, 0, 0],
				"scale": [1, 1, 1]
			},
			data_bindings={
				"temperature": "material.color",
				"vibration": "transform.rotation",
				"speed": "animations.rotation.speed"
			}
		)
		
		# Add rotation animation
		equipment.animations.append(
			AnimationController.create_rotation_animation(axis=[0, 1, 0], speed=1.0)
		)
		
		# Base platform
		base_geometry = GeometryGenerator.create_box(width=6.0, height=0.5, depth=6.0)
		base_material = Material(
			name="concrete_base",
			color=[0.6, 0.6, 0.6, 1.0],
			metallic=0.0,
			roughness=0.9
		)
		
		base = SceneObject(
			object_id="base_platform",
			name="Base Platform",
			geometry=base_geometry,
			material=base_material,
			transform={
				"position": [0, -2.5, 0],
				"rotation": [0, 0, 0],
				"scale": [1, 1, 1]
			}
		)
		
		# Piping system
		pipe_geometry = GeometryGenerator.create_cylinder(radius=0.2, height=8.0)
		pipe_material = Material(
			name="steel_pipe",
			color=[0.5, 0.5, 0.5, 1.0],
			metallic=0.9,
			roughness=0.2
		)
		
		inlet_pipe = SceneObject(
			object_id="inlet_pipe",
			name="Inlet Pipe",
			geometry=pipe_geometry,
			material=pipe_material,
			transform={
				"position": [-4, 0, 0],
				"rotation": [0, 0, 90],
				"scale": [1, 1, 1]
			},
			data_bindings={
				"flow_rate": "material.color",
				"pressure": "transform.scale"
			}
		)
		
		outlet_pipe = SceneObject(
			object_id="outlet_pipe",
			name="Outlet Pipe",
			geometry=pipe_geometry,
			material=pipe_material,
			transform={
				"position": [4, 0, 0],
				"rotation": [0, 0, 90],
				"scale": [1, 1, 1]
			},
			data_bindings={
				"flow_rate": "material.color",
				"pressure": "transform.scale"
			}
		)
		
		# Control panel
		panel_geometry = GeometryGenerator.create_box(width=1.0, height=2.0, depth=0.2)
		panel_material = Material(
			name="control_panel",
			color=[0.2, 0.2, 0.2, 1.0],
			metallic=0.1,
			roughness=0.5
		)
		
		control_panel = SceneObject(
			object_id="control_panel",
			name="Control Panel",
			geometry=panel_geometry,
			material=panel_material,
			transform={
				"position": [0, 1, 4],
				"rotation": [0, 0, 0],
				"scale": [1, 1, 1]
			}
		)
		
		# Add objects to builder
		self.add_object(equipment)
		self.add_object(base)
		self.add_object(inlet_pipe)
		self.add_object(outlet_pipe)
		self.add_object(control_panel)
		
		# Set up lighting
		main_light = Light(
			light_id="main_light",
			light_type="directional",
			direction=[-1, -1, -1],
			color=[1.0, 1.0, 1.0],
			intensity=1.0,
			shadows=True
		)
		
		ambient_light = Light(
			light_id="ambient_light",
			light_type="ambient",
			color=[0.4, 0.4, 0.4],
			intensity=0.3
		)
		
		self.add_light(main_light)
		self.add_light(ambient_light)
		
		# Set up camera
		camera = Camera(
			camera_type="perspective",
			position=[10, 5, 10],
			target=[0, 0, 0],
			fov=60.0
		)
		
		self.set_camera(camera)
		
		return self.build("industrial_scene", "Industrial Equipment Scene")
		
	def build(self, scene_id: str, name: str) -> Scene3D:
		"""Build the final scene"""
		return Scene3D(
			scene_id=scene_id,
			name=name,
			objects=self.objects,
			camera=self.camera or Camera(
				camera_type="perspective",
				position=[5, 5, 5],
				target=[0, 0, 0]
			),
			lights=self.lights or [
				Light(
					light_id="default_light",
					light_type="directional",
					direction=[-1, -1, -1],
					color=[1.0, 1.0, 1.0],
					intensity=1.0
				)
			]
		)

class WebGLRenderer:
	"""WebGL rendering interface"""
	
	@staticmethod
	def generate_webgl_viewer_html(scene: Scene3D, 
								  twin_data: Dict[str, Any] = None) -> str:
		"""Generate HTML with WebGL viewer"""
		
		scene_json = json.dumps({
			"scene_id": scene.scene_id,
			"name": scene.name,
			"objects": [WebGLRenderer._serialize_object(obj) for obj in scene.objects],
			"camera": asdict(scene.camera),
			"lights": [asdict(light) for light in scene.lights],
			"background_color": scene.background_color
		}, indent=2)
		
		html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Twin 3D Viewer - {{scene.name}}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: Arial, sans-serif;
            background: #000;
        }}
        
        #viewer-container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        
        #webgl-canvas {{
            display: block;
        }}
        
        #controls-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 8px;
            min-width: 200px;
        }}
        
        #data-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 8px;
            min-width: 250px;
        }}
        
        .control-group {{
            margin-bottom: 10px;
        }}
        
        .control-label {{
            display: block;
            margin-bottom: 5px;
            font-size: 12px;
        }}
        
        .control-input {{
            width: 100%;
            padding: 5px;
            border: none;
            border-radius: 4px;
        }}
        
        .data-item {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        
        .data-value {{
            font-weight: bold;
            color: #00ff00;
        }}
        
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div id="viewer-container">
        <canvas id="webgl-canvas"></canvas>
        
        <div id="loading">Loading 3D Scene...</div>
        
        <div id="controls-panel">
            <h3>3D Controls</h3>
            <div class="control-group">
                <label class="control-label">Render Mode</label>
                <select id="render-mode" class="control-input">
                    <option value="solid">Solid</option>
                    <option value="wireframe">Wireframe</option>
                    <option value="textured">Textured</option>
                    <option value="transparent">Transparent</option>
                </select>
            </div>
            <div class="control-group">
                <label class="control-label">Animation Speed</label>
                <input type="range" id="animation-speed" class="control-input" 
                       min="0" max="2" step="0.1" value="1">
            </div>
            <div class="control-group">
                <label class="control-label">Camera Preset</label>
                <select id="camera-preset" class="control-input">
                    <option value="default">Default View</option>
                    <option value="top">Top View</option>
                    <option value="front">Front View</option>
                    <option value="side">Side View</option>
                    <option value="isometric">Isometric</option>
                </select>
            </div>
        </div>
        
        <div id="data-panel">
            <h3>Real-time Data</h3>
            <div class="data-item">
                <span>Temperature:</span>
                <span class="data-value" id="temperature-value">--°C</span>
            </div>
            <div class="data-item">
                <span>Vibration:</span>
                <span class="data-value" id="vibration-value">-- mm/s</span>
            </div>
            <div class="data-item">
                <span>Pressure:</span>
                <span class="data-value" id="pressure-value">-- bar</span>
            </div>
            <div class="data-item">
                <span>Flow Rate:</span>
                <span class="data-value" id="flow-rate-value">-- L/min</span>
            </div>
            <div class="data-item">
                <span>Status:</span>
                <span class="data-value" id="status-value">Online</span>
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/controls/OrbitControls.js"></script>
    
    <script>
        // Scene data
        const sceneData = {scene_json};
        
        // Global variables
        let scene, camera, renderer, controls;
        let sceneObjects = {{}};
        let animationId;
        let clock = new THREE.Clock();
        
        // Initialize 3D viewer
        function init() {{
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(
                sceneData.background_color[0],
                sceneData.background_color[1], 
                sceneData.background_color[2]
            );
            
            // Create camera
            const cameraData = sceneData.camera;
            camera = new THREE.PerspectiveCamera(
                cameraData.fov,
                window.innerWidth / window.innerHeight,
                cameraData.near,
                cameraData.far
            );
            camera.position.set(...cameraData.position);
            camera.lookAt(...cameraData.target);
            
            // Create renderer
            const canvas = document.getElementById('webgl-canvas');
            renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            
            // Create controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.target.set(...cameraData.target);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Create lights
            sceneData.lights.forEach(lightData => {{
                const light = createLight(lightData);
                if (light) scene.add(light);
            }});
            
            // Create objects
            sceneData.objects.forEach(objectData => {{
                const object = createSceneObject(objectData);
                if (object) {{
                    scene.add(object);
                    sceneObjects[objectData.object_id] = object;
                }}
            }});
            
            // Set up event listeners
            setupEventListeners();
            
            // Hide loading
            document.getElementById('loading').style.display = 'none';
            
            // Start animation loop
            animate();
        }}
        
        function createLight(lightData) {{
            let light;
            
            switch (lightData.light_type) {{
                case 'directional':
                    light = new THREE.DirectionalLight(
                        new THREE.Color(...lightData.color),
                        lightData.intensity
                    );
                    if (lightData.direction) {{
                        light.position.set(...lightData.direction).multiplyScalar(-10);
                        light.target.position.set(0, 0, 0);
                    }}
                    if (lightData.shadows) {{
                        light.castShadow = true;
                        light.shadow.mapSize.width = 2048;
                        light.shadow.mapSize.height = 2048;
                    }}
                    break;
                    
                case 'point':
                    light = new THREE.PointLight(
                        new THREE.Color(...lightData.color),
                        lightData.intensity
                    );
                    if (lightData.position) {{
                        light.position.set(...lightData.position);
                    }}
                    break;
                    
                case 'ambient':
                    light = new THREE.AmbientLight(
                        new THREE.Color(...lightData.color),
                        lightData.intensity
                    );
                    break;
            }}
            
            return light;
        }}
        
        function createSceneObject(objectData) {{
            const geometry = createGeometry(objectData.geometry);
            const material = createMaterial(objectData.material);
            
            if (!geometry || !material) return null;
            
            const mesh = new THREE.Mesh(geometry, material);
            
            // Apply transform
            const transform = objectData.transform;
            mesh.position.set(...transform.position);
            mesh.rotation.set(...transform.rotation);
            mesh.scale.set(...transform.scale);
            
            // Set properties
            mesh.visible = objectData.visible;
            mesh.userData = {{
                object_id: objectData.object_id,
                name: objectData.name,
                data_bindings: objectData.data_bindings,
                animations: objectData.animations
            }};
            
            return mesh;
        }}
        
        function createGeometry(geometryData) {{
            const vertices = new Float32Array(geometryData.vertices.flat());
            
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            
            if (geometryData.indices) {{
                const indices = new Uint16Array(geometryData.indices.flat());
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
            }}
            
            if (geometryData.normals) {{
                const normals = new Float32Array(geometryData.normals.flat());
                geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
            }} else {{
                geometry.computeVertexNormals();
            }}
            
            if (geometryData.colors) {{
                const colors = new Float32Array(geometryData.colors.flat());
                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 4));
            }}
            
            return geometry;
        }}
        
        function createMaterial(materialData) {{
            const material = new THREE.MeshStandardMaterial({{
                color: new THREE.Color(...materialData.color.slice(0, 3)),
                metalness: materialData.metallic,
                roughness: materialData.roughness,
                transparent: materialData.opacity < 1.0,
                opacity: materialData.opacity
            }});
            
            if (materialData.emissive) {{
                material.emissive = new THREE.Color(...materialData.emissive);
            }}
            
            return material;
        }}
        
        function setupEventListeners() {{
            // Render mode control
            document.getElementById('render-mode').addEventListener('change', (e) => {{
                const mode = e.target.value;
                Object.values(sceneObjects).forEach(obj => {{
                    if (obj.material) {{
                        switch (mode) {{
                            case 'wireframe':
                                obj.material.wireframe = true;
                                break;
                            case 'solid':
                                obj.material.wireframe = false;
                                obj.material.transparent = false;
                                break;
                            case 'transparent':
                                obj.material.wireframe = false;
                                obj.material.transparent = true;
                                obj.material.opacity = 0.5;
                                break;
                        }}
                    }}
                }});
            }});
            
            // Camera presets
            document.getElementById('camera-preset').addEventListener('change', (e) => {{
                const preset = e.target.value;
                setCameraPreset(preset);
            }});
            
            // Window resize
            window.addEventListener('resize', onWindowResize);
        }}
        
        function setCameraPreset(preset) {{
            const distance = 15;
            
            switch (preset) {{
                case 'top':
                    camera.position.set(0, distance, 0);
                    camera.lookAt(0, 0, 0);
                    break;
                case 'front':
                    camera.position.set(0, 0, distance);
                    camera.lookAt(0, 0, 0);
                    break;
                case 'side':
                    camera.position.set(distance, 0, 0);
                    camera.lookAt(0, 0, 0);
                    break;
                case 'isometric':
                    camera.position.set(distance * 0.7, distance * 0.7, distance * 0.7);
                    camera.lookAt(0, 0, 0);
                    break;
                default:
                    camera.position.set(...sceneData.camera.position);
                    camera.lookAt(...sceneData.camera.target);
            }}
            
            controls.target.set(0, 0, 0);
            controls.update();
        }}
        
        function animate() {{
            animationId = requestAnimationFrame(animate);
            
            const deltaTime = clock.getDelta();
            
            // Update animations
            Object.values(sceneObjects).forEach(obj => {{
                if (obj.userData.animations) {{
                    obj.userData.animations.forEach(animData => {{
                        if (animData.type === 'rotation') {{
                            const axis = new THREE.Vector3(...animData.axis);
                            obj.rotateOnWorldAxis(axis, animData.speed * deltaTime);
                        }}
                    }});
                }}
            }});
            
            // Update controls
            controls.update();
            
            // Render
            renderer.render(scene, camera);
            
            // Update data panel (simulate real-time data)
            updateDataPanel();
        }}
        
        function updateDataPanel() {{
            // Simulate real-time data updates
            const time = Date.now() * 0.001;
            
            document.getElementById('temperature-value').textContent = 
                (75 + Math.sin(time * 0.5) * 5).toFixed(1) + '°C';
            document.getElementById('vibration-value').textContent = 
                (1.5 + Math.sin(time * 2) * 0.3).toFixed(2) + ' mm/s';
            document.getElementById('pressure-value').textContent = 
                (10 + Math.sin(time * 0.3) * 2).toFixed(1) + ' bar';
            document.getElementById('flow-rate-value').textContent = 
                (100 + Math.sin(time * 0.8) * 10).toFixed(0) + ' L/min';
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        // Initialize when page loads
        window.addEventListener('load', init);
    </script>
</body>
</html>
"""
		
		return html_template
		
	@staticmethod
	def _serialize_object(obj: SceneObject) -> Dict[str, Any]:
		"""Serialize scene object for JSON"""
		return {
			"object_id": obj.object_id,
			"name": obj.name,
			"geometry": {
				"geometry_id": obj.geometry.geometry_id,
				"geometry_type": obj.geometry.geometry_type.value,
				"vertices": obj.geometry.vertices,
				"indices": obj.geometry.indices,
				"normals": obj.geometry.normals,
				"colors": obj.geometry.colors
			},
			"material": {
				"name": obj.material.name,
				"color": obj.material.color,
				"metallic": obj.material.metallic,
				"roughness": obj.material.roughness,
				"opacity": obj.material.opacity,
				"emissive": obj.material.emissive
			},
			"transform": obj.transform,
			"visible": obj.visible,
			"animations": obj.animations,
			"data_bindings": obj.data_bindings
		}

class Visualization3DEngine:
	"""Main 3D visualization engine"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.scenes: Dict[str, Scene3D] = {}
		self.scene_builder = Scene3DBuilder()
		
		logger.info("3D Visualization Engine initialized")
		
	def create_digital_twin_visualization(self, twin_id: str, 
										 twin_data: Dict[str, Any],
										 geometry_data: Dict[str, Any] = None) -> str:
		"""Create 3D visualization for digital twin"""
		
		try:
			# Build scene based on twin type
			twin_type = twin_data.get('twin_type', 'asset')
			
			if twin_type == 'asset' and twin_data.get('category') == 'industrial':
				scene = self.scene_builder.create_industrial_scene(twin_data)
			else:
				# Generic scene
				scene = self._create_generic_scene(twin_id, twin_data, geometry_data)
			
			# Store scene
			self.scenes[twin_id] = scene
			
			# Generate WebGL viewer HTML
			html_content = WebGLRenderer.generate_webgl_viewer_html(scene, twin_data)
			
			logger.info(f"3D visualization created for twin {twin_id}")
			return html_content
			
		except Exception as e:
			logger.error(f"Error creating 3D visualization for twin {twin_id}: {e}")
			return self._create_error_visualization(str(e))
			
	def _create_generic_scene(self, twin_id: str, twin_data: Dict[str, Any], 
							 geometry_data: Dict[str, Any] = None) -> Scene3D:
		"""Create generic 3D scene"""
		
		# Create basic geometry
		geometry = GeometryGenerator.create_box()
		material = Material(
			name="generic_material",
			color=[0.5, 0.7, 1.0, 1.0],
			metallic=0.2,
			roughness=0.6
		)
		
		main_object = SceneObject(
			object_id=f"twin_{twin_id}",
			name=twin_data.get('name', f"Twin {twin_id}"),
			geometry=geometry,
			material=material,
			transform={
				"position": [0, 0, 0],
				"rotation": [0, 0, 0],
				"scale": [1, 1, 1]
			}
		)
		
		# Build scene
		builder = Scene3DBuilder()
		scene = builder.add_object(main_object).build(f"scene_{twin_id}", f"Scene for {twin_id}")
		
		return scene
		
	def _create_error_visualization(self, error_message: str) -> str:
		"""Create error visualization HTML"""
		return f"""
		<html>
		<head><title>3D Visualization Error</title></head>
		<body style="font-family: Arial; padding: 20px;">
			<h2>3D Visualization Error</h2>
			<p>Unable to create 3D visualization:</p>
			<pre style="background: #f0f0f0; padding: 10px; border-radius: 4px;">{error_message}</pre>
		</body>
		</html>
		"""

# Test and example usage
def test_3d_visualization():
	"""Test the 3D visualization system"""
	
	# Sample digital twin data
	twin_data = {
		'twin_id': 'pump_001',
		'name': 'Industrial Pump #1',
		'twin_type': 'asset',
		'category': 'industrial',
		'properties': {
			'temperature': 75.5,
			'vibration': 1.8,
			'pressure': 12.3,
			'flow_rate': 105.2
		}
	}
	
	# Create visualization engine
	viz_engine = Visualization3DEngine()
	
	# Create 3D visualization
	html_content = viz_engine.create_digital_twin_visualization('pump_001', twin_data)
	
	# Save to file for testing
	output_file = Path("pump_3d_visualization.html")
	with open(output_file, 'w') as f:
		f.write(html_content)
	
	print(f"3D visualization saved to: {output_file}")
	print(f"Open the file in a web browser to view the 3D scene")

if __name__ == "__main__":
	test_3d_visualization()