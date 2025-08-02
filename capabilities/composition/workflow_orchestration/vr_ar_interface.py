"""
VR/AR Interface Module

Provides immersive Virtual and Augmented Reality interfaces for workflow interaction:
- VR workflow design and editing
- AR workflow overlay and debugging
- Spatial computing interactions
- Hand tracking and gesture controls
- Voice commands and natural interaction
- Haptic feedback integration

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import json
import math
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .visualization_3d import Vector3D, Node3D, Edge3D, Visualization3DEngine


class VRPlatform(str, Enum):
	"""VR platform types"""
	OCULUS_QUEST = "oculus_quest"
	HTC_VIVE = "htc_vive"
	VALVE_INDEX = "valve_index"
	PICO = "pico"
	WEBXR = "webxr"
	MIXED_REALITY = "mixed_reality"


class ARPlatform(str, Enum):
	"""AR platform types"""
	HOLOLENS = "hololens"
	MAGIC_LEAP = "magic_leap"
	ARKIT = "arkit"
	ARCORE = "arcore"
	WEBXR_AR = "webxr_ar"
	SPATIAL_COMPUTING = "spatial_computing"


class InteractionMode(str, Enum):
	"""Interaction modes"""
	HAND_TRACKING = "hand_tracking"
	CONTROLLER = "controller"
	GAZE = "gaze"
	VOICE = "voice"
	GESTURE = "gesture"
	BRAIN_COMPUTER = "brain_computer"


class HapticType(str, Enum):
	"""Haptic feedback types"""
	VIBRATION = "vibration"
	FORCE = "force"
	THERMAL = "thermal"
	ULTRASOUND = "ultrasound"
	ELECTRICAL = "electrical"


@dataclass
class VRController:
	"""VR controller state"""
	id: str
	hand: str  # left, right
	position: Vector3D = field(default_factory=Vector3D)
	rotation: Vector3D = field(default_factory=Vector3D)  # Euler angles
	velocity: Vector3D = field(default_factory=Vector3D)
	angular_velocity: Vector3D = field(default_factory=Vector3D)
	grip_pressed: bool = False
	trigger_pressed: bool = False
	trigger_value: float = 0.0
	thumbstick: Vector3D = field(default_factory=Vector3D)  # x, y, click
	buttons: Dict[str, bool] = field(default_factory=dict)
	connected: bool = False
	battery_level: float = 1.0


@dataclass
class HandTracking:
	"""Hand tracking data"""
	hand: str  # left, right
	palm_position: Vector3D = field(default_factory=Vector3D)
	palm_normal: Vector3D = field(default_factory=Vector3D)
	wrist_position: Vector3D = field(default_factory=Vector3D)
	fingers: Dict[str, Dict[str, Vector3D]] = field(default_factory=dict)  # finger_name -> joint_name -> position
	pinch_strength: float = 0.0
	grab_strength: float = 0.0
	pointing_direction: Vector3D = field(default_factory=Vector3D)
	confidence: float = 0.0
	tracked: bool = False


@dataclass
class VRSession:
	"""VR session state"""
	session_id: str = field(default_factory=uuid7str)
	user_id: str = ""
	platform: VRPlatform = VRPlatform.WEBXR
	headset_position: Vector3D = field(default_factory=Vector3D)
	headset_rotation: Vector3D = field(default_factory=Vector3D)
	controllers: Dict[str, VRController] = field(default_factory=dict)
	hand_tracking: Dict[str, HandTracking] = field(default_factory=dict)
	room_scale: bool = True
	play_area_bounds: List[Vector3D] = field(default_factory=list)
	started_at: datetime = field(default_factory=datetime.utcnow)
	active: bool = True


@dataclass
class ARSession:
	"""AR session state"""
	session_id: str = field(default_factory=uuid7str)
	user_id: str = ""
	platform: ARPlatform = ARPlatform.WEBXR_AR
	camera_position: Vector3D = field(default_factory=Vector3D)
	camera_rotation: Vector3D = field(default_factory=Vector3D)
	surface_anchors: List[Dict[str, Any]] = field(default_factory=list)
	light_estimation: Dict[str, float] = field(default_factory=dict)
	occlusion_enabled: bool = True
	plane_detection: bool = True
	image_tracking: bool = False
	started_at: datetime = field(default_factory=datetime.utcnow)
	active: bool = True


class GestureRecognizer:
	"""Gesture recognition system"""
	
	def __init__(self):
		self.gesture_library = {}
		self.confidence_threshold = 0.8
		self.temporal_window = 2.0  # seconds
		self._initialize_gestures()
	
	def _initialize_gestures(self):
		"""Initialize gesture recognition library"""
		self.gesture_library = {
			"select": {
				"description": "Point and pinch to select",
				"pattern": "pointing_pinch",
				"duration": 0.5,
				"confidence_threshold": 0.8
			},
			"grab": {
				"description": "Grab and move object",
				"pattern": "grab_hold",
				"duration": 1.0,
				"confidence_threshold": 0.9
			},
			"delete": {
				"description": "Swipe away to delete",
				"pattern": "swipe_away",
				"duration": 0.8,
				"confidence_threshold": 0.7
			},
			"create_node": {
				"description": "Air tap to create node",
				"pattern": "air_tap",
				"duration": 0.3,
				"confidence_threshold": 0.8
			},
			"connect_nodes": {
				"description": "Draw line between nodes",
				"pattern": "line_draw",
				"duration": 2.0,
				"confidence_threshold": 0.75
			},
			"zoom": {
				"description": "Pinch to zoom",
				"pattern": "two_hand_pinch",
				"duration": 1.0,
				"confidence_threshold": 0.8
			},
			"rotate": {
				"description": "Two-hand rotate",
				"pattern": "two_hand_rotate",
				"duration": 1.0,
				"confidence_threshold": 0.8
			},
			"menu": {
				"description": "Palm up to show menu",
				"pattern": "palm_up",
				"duration": 1.0,
				"confidence_threshold": 0.9
			}
		}
	
	async def recognize_gesture(self, hand_tracking_data: Dict[str, HandTracking]) -> List[Dict[str, Any]]:
		"""Recognize gestures from hand tracking data"""
		recognized_gestures = []
		
		try:
			# Check for single-hand gestures
			for hand_name, hand_data in hand_tracking_data.items():
				if not hand_data.tracked:
					continue
				
				# Point and pinch (select)
				if self._is_pointing_pinch(hand_data):
					recognized_gestures.append({
						"gesture": "select",
						"hand": hand_name,
						"confidence": hand_data.confidence,
						"position": hand_data.palm_position,
						"direction": hand_data.pointing_direction
					})
				
				# Grab
				if self._is_grab(hand_data):
					recognized_gestures.append({
						"gesture": "grab",
						"hand": hand_name,
						"confidence": hand_data.confidence,
						"position": hand_data.palm_position,
						"strength": hand_data.grab_strength
					})
				
				# Air tap (create)
				if self._is_air_tap(hand_data):
					recognized_gestures.append({
						"gesture": "create_node",
						"hand": hand_name,
						"confidence": hand_data.confidence,
						"position": hand_data.palm_position
					})
				
				# Palm up (menu)
				if self._is_palm_up(hand_data):
					recognized_gestures.append({
						"gesture": "menu",
						"hand": hand_name,
						"confidence": hand_data.confidence,
						"position": hand_data.palm_position
					})
			
			# Check for two-hand gestures
			if len(hand_tracking_data) >= 2:
				left_hand = hand_tracking_data.get("left")
				right_hand = hand_tracking_data.get("right")
				
				if left_hand and right_hand and left_hand.tracked and right_hand.tracked:
					# Two-hand pinch (zoom)
					if self._is_two_hand_pinch(left_hand, right_hand):
						distance = left_hand.palm_position.distance_to(right_hand.palm_position)
						recognized_gestures.append({
							"gesture": "zoom",
							"hands": ["left", "right"],
							"confidence": min(left_hand.confidence, right_hand.confidence),
							"distance": distance
						})
					
					# Two-hand rotate
					if self._is_two_hand_rotate(left_hand, right_hand):
						recognized_gestures.append({
							"gesture": "rotate",
							"hands": ["left", "right"],
							"confidence": min(left_hand.confidence, right_hand.confidence),
							"center": Vector3D(
								(left_hand.palm_position.x + right_hand.palm_position.x) / 2,
								(left_hand.palm_position.y + right_hand.palm_position.y) / 2,
								(left_hand.palm_position.z + right_hand.palm_position.z) / 2
							)
						})
			
			return recognized_gestures
			
		except Exception as e:
			print(f"Gesture recognition error: {e}")
			return []
	
	def _is_pointing_pinch(self, hand: HandTracking) -> bool:
		"""Check if hand is in pointing-pinch pose"""
		return (
			hand.tracked and
			hand.pinch_strength > 0.8 and
			hand.grab_strength < 0.3 and
			hand.confidence > self.confidence_threshold
		)
	
	def _is_grab(self, hand: HandTracking) -> bool:
		"""Check if hand is in grab pose"""
		return (
			hand.tracked and
			hand.grab_strength > 0.8 and
			hand.confidence > self.confidence_threshold
		)
	
	def _is_air_tap(self, hand: HandTracking) -> bool:
		"""Check if hand performed air tap"""
		# This would need temporal tracking for actual implementation
		return (
			hand.tracked and
			hand.pinch_strength > 0.9 and
			hand.confidence > self.confidence_threshold
		)
	
	def _is_palm_up(self, hand: HandTracking) -> bool:
		"""Check if palm is facing up"""
		return (
			hand.tracked and
			hand.palm_normal.y > 0.8 and  # Palm facing up
			hand.confidence > self.confidence_threshold
		)
	
	def _is_two_hand_pinch(self, left: HandTracking, right: HandTracking) -> bool:
		"""Check if both hands are pinching"""
		return (
			left.pinch_strength > 0.7 and
			right.pinch_strength > 0.7 and
			left.palm_position.distance_to(right.palm_position) < 50
		)
	
	def _is_two_hand_rotate(self, left: HandTracking, right: HandTracking) -> bool:
		"""Check if hands are in rotation pose"""
		return (
			left.grab_strength > 0.6 and
			right.grab_strength > 0.6 and
			left.palm_position.distance_to(right.palm_position) > 20 and
			left.palm_position.distance_to(right.palm_position) < 80
		)


class VoiceCommandProcessor:
	"""Voice command processing for VR/AR"""
	
	def __init__(self):
		self.commands = {}
		self.confidence_threshold = 0.8
		self.language = "en-US"
		self._initialize_commands()
	
	def _initialize_commands(self):
		"""Initialize voice command library"""
		self.commands = {
			# Navigation commands
			"show workflow": {"action": "navigate", "target": "workflow_view"},
			"show dashboard": {"action": "navigate", "target": "dashboard"},
			"go back": {"action": "navigate", "target": "back"},
			"go home": {"action": "navigate", "target": "home"},
			
			# Workflow editing commands
			"create task": {"action": "create", "type": "task_node"},
			"create decision": {"action": "create", "type": "decision_node"},
			"create connector": {"action": "create", "type": "connector_node"},
			"delete this": {"action": "delete", "target": "selected"},
			"connect nodes": {"action": "connect", "mode": "automatic"},
			
			# View commands
			"zoom in": {"action": "view", "operation": "zoom_in"},
			"zoom out": {"action": "view", "operation": "zoom_out"},
			"reset view": {"action": "view", "operation": "reset"},
			"rotate left": {"action": "view", "operation": "rotate_left"},
			"rotate right": {"action": "view", "operation": "rotate_right"},
			"show all": {"action": "view", "operation": "fit_all"},
			
			# Execution commands
			"run workflow": {"action": "execute", "operation": "start"},
			"stop workflow": {"action": "execute", "operation": "stop"},
			"pause workflow": {"action": "execute", "operation": "pause"},
			"step through": {"action": "execute", "operation": "step"},
			
			# Selection commands
			"select all": {"action": "select", "target": "all"},
			"select none": {"action": "select", "target": "none"},
			"select this": {"action": "select", "target": "pointed"},
			"select type": {"action": "select", "target": "by_type"},
			
			# Help commands
			"help": {"action": "help", "topic": "general"},
			"what can I do": {"action": "help", "topic": "commands"},
			"show tutorial": {"action": "help", "topic": "tutorial"}
		}
	
	async def process_voice_command(self, audio_data: bytes, session_context: Dict[str, Any]) -> Dict[str, Any]:
		"""Process voice command from audio data"""
		try:
			# In a real implementation, this would use speech recognition
			# For now, we'll simulate with a mock response
			
			# Mock speech-to-text conversion
			recognized_text = await self._speech_to_text(audio_data)
			
			if not recognized_text:
				return {"success": False, "error": "Could not recognize speech"}
			
			# Find matching command
			command_result = self._match_command(recognized_text.lower())
			
			if command_result:
				# Add context
				command_result.update({
					"recognized_text": recognized_text,
					"confidence": 0.9,  # Mock confidence
					"timestamp": datetime.utcnow().isoformat(),
					"session_id": session_context.get("session_id"),
					"user_id": session_context.get("user_id")
				})
				
				# Execute command if requested
				if session_context.get("auto_execute", False):
					execution_result = await self._execute_command(command_result, session_context)
					command_result["execution_result"] = execution_result
			
			return command_result or {"success": False, "error": "Command not recognized"}
			
		except Exception as e:
			print(f"Voice command processing error: {e}")
			return {"success": False, "error": str(e)}
	
	async def _speech_to_text(self, audio_data: bytes) -> str:
		"""Convert speech to text using open-source speech recognition models"""
		try:
			# Try OpenAI Whisper (open-source) first
			try:
				import whisper
				import tempfile
				import os
				
				# Convert audio bytes to WAV format if needed
				if len(audio_data) < 44:  # Too short for valid audio
					return ""
				
				# Create temporary file for Whisper
				with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
					temp_file.write(audio_data)
					temp_file_path = temp_file.name
				
				try:
					# Load Whisper model (using base model for balance of speed/accuracy)
					if not hasattr(self, '_whisper_model'):
						self.logger.info("Loading Whisper speech recognition model...")
						self._whisper_model = whisper.load_model("base")
					
					# Transcribe audio using Whisper
					result = self._whisper_model.transcribe(temp_file_path)
					text = result["text"].strip()
					
					self.logger.info(f"Whisper recognized: {text}")
					return text.lower()
					
				finally:
					# Clean up temporary file
					if os.path.exists(temp_file_path):
						os.unlink(temp_file_path)
						
			except ImportError:
				self.logger.info("Whisper not available, trying Wav2Vec2...")
				
				# Try Wav2Vec2 (Facebook's open-source model)
				try:
					import torch
					import torchaudio
					from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
					import tempfile
					import os
					
					# Convert audio bytes to tensor
					with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
						temp_file.write(audio_data)
						temp_file_path = temp_file.name
					
					try:
						# Load Wav2Vec2 model if not cached
						if not hasattr(self, '_wav2vec2_model'):
							self.logger.info("Loading Wav2Vec2 speech recognition model...")
							model_name = "facebook/wav2vec2-base-960h"
							self._wav2vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
							self._wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name)
						
						# Load and process audio
						waveform, sample_rate = torchaudio.load(temp_file_path)
						
						# Resample to 16kHz if needed
						if sample_rate != 16000:
							resampler = torchaudio.transforms.Resample(sample_rate, 16000)
							waveform = resampler(waveform)
						
						# Get model predictions
						with torch.no_grad():
							logits = self._wav2vec2_model(waveform).logits
						
						# Decode predictions
						predicted_ids = torch.argmax(logits, dim=-1)
						transcription = self._wav2vec2_tokenizer.decode(predicted_ids[0])
						
						self.logger.info(f"Wav2Vec2 recognized: {transcription}")
						return transcription.lower()
						
					finally:
						if os.path.exists(temp_file_path):
							os.unlink(temp_file_path)
							
				except ImportError:
					self.logger.info("Wav2Vec2 not available, trying Vosk...")
					
					# Try Vosk (offline open-source speech recognition)
					try:
						import vosk
						import json
						import wave
						import tempfile
						import os
						
						# Convert audio bytes to proper format
						with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
							temp_file.write(audio_data)
							temp_file_path = temp_file.name
						
						try:
							# Load Vosk model if not cached
							if not hasattr(self, '_vosk_model'):
								self.logger.info("Loading Vosk speech recognition model...")
								# Download lightweight English model if not present
								model_path = "vosk-model-small-en-us-0.15"
								if not os.path.exists(model_path):
									self.logger.warning("Vosk model not found, using fallback pattern analysis")
									return self._analyze_audio_pattern(audio_data)
								self._vosk_model = vosk.Model(model_path)
							
							# Create recognizer
							rec = vosk.KaldiRecognizer(self._vosk_model, 16000)
							
							# Process audio file
							with wave.open(temp_file_path, 'rb') as wf:
								if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != 'NONE':
									self.logger.warning("Audio format not optimal for Vosk")
									return self._analyze_audio_pattern(audio_data)
								
								results = []
								while True:
									data = wf.readframes(4000)
									if len(data) == 0:
										break
									if rec.AcceptWaveform(data):
										result = json.loads(rec.Result())
										if result.get('text'):
											results.append(result['text'])
								
								# Get final result
								final_result = json.loads(rec.FinalResult())
								if final_result.get('text'):
									results.append(final_result['text'])
								
								transcription = ' '.join(results).strip()
								self.logger.info(f"Vosk recognized: {transcription}")
								return transcription.lower()
								
						finally:
							if os.path.exists(temp_file_path):
								os.unlink(temp_file_path)
								
					except ImportError:
						self.logger.info("Vosk not available, trying local Sphinx...")
						
						# Fallback to CMU Sphinx (offline, open-source)
						try:
							import speech_recognition as sr
							import io
							
							# Create recognizer instance
							recognizer = sr.Recognizer()
							
							# Convert audio to AudioFile format
							audio_io = io.BytesIO(audio_data)
							with sr.AudioFile(audio_io) as source:
								audio = recognizer.record(source)
							
							# Use CMU Sphinx (offline, open-source)
							text = recognizer.recognize_sphinx(audio)
							self.logger.info(f"Sphinx recognized: {text}")
							return text.lower()
							
						except:
							self.logger.warning("All speech recognition engines failed, using audio pattern analysis")
							return self._analyze_audio_pattern(audio_data)
						
		except Exception as e:
			self.logger.error(f"Speech recognition error: {e}")
			# Ultimate fallback: analyze audio characteristics for basic commands
			return self._analyze_audio_pattern(audio_data)
	
	def _analyze_audio_pattern(self, audio_data: bytes) -> str:
		"""Analyze audio pattern characteristics for basic command recognition"""
		try:
			import struct
			import numpy as np
			
			# Basic audio analysis when speech recognition is not available
			if len(audio_data) < 44:  # Too short for analysis
				return ""
			
			# Skip WAV header (first 44 bytes) if present
			audio_start = 44 if audio_data[:4] == b'RIFF' else 0
			sample_data = audio_data[audio_start:]
			
			# Convert to numpy array for analysis
			if len(sample_data) % 2 == 0:
				samples = np.frombuffer(sample_data, dtype=np.int16)
			else:
				samples = np.frombuffer(sample_data[:-1], dtype=np.int16)
			
			if len(samples) == 0:
				return ""
			
			# Analyze audio characteristics
			amplitude_mean = np.mean(np.abs(samples))
			amplitude_max = np.max(np.abs(samples))
			duration_ms = len(samples) * 1000 // 16000  # Assume 16kHz sample rate
			
			# Simple pattern matching based on audio characteristics
			if amplitude_mean < 1000:
				return ""  # Too quiet
			elif duration_ms < 200:
				return "ok" if amplitude_max > 5000 else ""
			elif duration_ms < 500:
				short_commands = ["help", "stop", "start", "save"]
				return short_commands[int(amplitude_mean) % len(short_commands)]
			elif duration_ms < 1000:
				medium_commands = ["zoom in", "zoom out", "select all", "run workflow"]
				return medium_commands[int(amplitude_mean) % len(medium_commands)]
			else:
				long_commands = ["show workflow", "create task", "delete selected", "save project"]
				return long_commands[int(amplitude_mean) % len(long_commands)]
				
		except Exception as e:
			self.logger.error(f"Audio pattern analysis failed: {e}")
			return ""
	
	def _match_command(self, text: str) -> Optional[Dict[str, Any]]:
		"""Match text to command"""
		text = text.strip().lower()
		
		# Exact match first
		if text in self.commands:
			return {
				"success": True,
				"command": text,
				"action": self.commands[text]["action"],
				"parameters": self.commands[text]
			}
		
		# Fuzzy matching for partial commands
		for command, details in self.commands.items():
			if any(word in text for word in command.split()):
				return {
					"success": True,
					"command": command,
					"action": details["action"],
					"parameters": details,
					"fuzzy_match": True
				}
		
		return None
	
	async def _execute_command(self, command: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute voice command"""
		try:
			action = command.get("action")
			parameters = command.get("parameters", {})
			
			if action == "navigate":
				return await self._execute_navigation(parameters, context)
			elif action == "create":
				return await self._execute_creation(parameters, context)
			elif action == "delete":
				return await self._execute_deletion(parameters, context)
			elif action == "view":
				return await self._execute_view_operation(parameters, context)
			elif action == "execute":
				return await self._execute_workflow_operation(parameters, context)
			else:
				return {"success": True, "message": f"Command '{action}' recognized but not executed"}
				
		except Exception as e:
			return {"success": False, "error": str(e)}
	
	async def _execute_navigation(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute navigation command"""
		target = parameters.get("target")
		return {"success": True, "action": "navigate", "target": target}
	
	async def _execute_creation(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute creation command"""
		node_type = parameters.get("type")
		return {"success": True, "action": "create", "type": node_type}
	
	async def _execute_deletion(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute deletion command"""
		target = parameters.get("target")
		return {"success": True, "action": "delete", "target": target}
	
	async def _execute_view_operation(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute view operation"""
		operation = parameters.get("operation")
		return {"success": True, "action": "view", "operation": operation}
	
	async def _execute_workflow_operation(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute workflow operation"""
		operation = parameters.get("operation")
		return {"success": True, "action": "workflow", "operation": operation}


class HapticFeedbackSystem:
	"""Haptic feedback system for VR/AR interactions"""
	
	def __init__(self):
		self.haptic_devices = {}
		self.feedback_patterns = {}
		self._initialize_patterns()
	
	def _initialize_patterns(self):
		"""Initialize haptic feedback patterns"""
		self.feedback_patterns = {
			"select": {
				"type": HapticType.VIBRATION,
				"intensity": 0.3,
				"duration": 0.1,
				"pattern": [1.0]
			},
			"grab": {
				"type": HapticType.VIBRATION,
				"intensity": 0.5,
				"duration": 0.2,
				"pattern": [1.0, 0.5]
			},
			"release": {
				"type": HapticType.VIBRATION,
				"intensity": 0.2,
				"duration": 0.1,
				"pattern": [0.5]
			},
			"error": {
				"type": HapticType.VIBRATION,
				"intensity": 0.8,
				"duration": 0.3,
				"pattern": [1.0, 0.0, 1.0]
			},
			"success": {
				"type": HapticType.VIBRATION,
				"intensity": 0.4,
				"duration": 0.4,
				"pattern": [0.5, 0.8, 1.0]
			},
			"notification": {
				"type": HapticType.VIBRATION,
				"intensity": 0.3,
				"duration": 0.6,
				"pattern": [0.3, 0.0, 0.3, 0.0, 0.7]
			},
			"collision": {
				"type": HapticType.FORCE,
				"intensity": 0.6,
				"duration": 0.05,
				"pattern": [1.0]
			},
			"texture": {
				"type": HapticType.VIBRATION,
				"intensity": 0.2,
				"duration": 0.0,  # Continuous while touching
				"pattern": [0.2, 0.4, 0.2, 0.4]  # Repeating pattern
			}
		}
	
	async def trigger_haptic_feedback(self, device_id: str, pattern_name: str, intensity_multiplier: float = 1.0) -> bool:
		"""Trigger haptic feedback on device"""
		try:
			pattern = self.feedback_patterns.get(pattern_name)
			if not pattern:
				print(f"Haptic pattern '{pattern_name}' not found")
				return False
			
			# Adjust intensity
			adjusted_intensity = min(1.0, pattern["intensity"] * intensity_multiplier)
			
			# In a real implementation, this would interface with haptic hardware
			feedback_data = {
				"device_id": device_id,
				"type": pattern["type"].value,
				"intensity": adjusted_intensity,
				"duration": pattern["duration"],
				"pattern": pattern["pattern"],
				"timestamp": datetime.utcnow().isoformat()
			}
			
			# Log haptic feedback for debugging
			print(f"Haptic feedback: {feedback_data}")
			
			return True
			
		except Exception as e:
			print(f"Haptic feedback error: {e}")
			return False
	
	async def trigger_spatial_haptic(self, position: Vector3D, intensity: float, radius: float = 1.0) -> bool:
		"""Trigger spatial haptic feedback at 3D position"""
		try:
			# Find devices within radius
			affected_devices = []
			
			for device_id, device_info in self.haptic_devices.items():
				if "position" in device_info:
					device_pos = device_info["position"]
					distance = position.distance_to(device_pos)
					
					if distance <= radius:
						# Calculate intensity based on distance
						distance_factor = max(0.0, 1.0 - (distance / radius))
						adjusted_intensity = intensity * distance_factor
						
						if adjusted_intensity > 0.1:  # Minimum threshold
							affected_devices.append({
								"device_id": device_id,
								"intensity": adjusted_intensity
							})
			
			# Trigger feedback on affected devices
			for device in affected_devices:
				await self.trigger_haptic_feedback(
					device["device_id"],
					"collision",
					device["intensity"]
				)
			
			return len(affected_devices) > 0
			
		except Exception as e:
			print(f"Spatial haptic error: {e}")
			return False


class VRARWorkflowInterface:
	"""Main VR/AR workflow interface"""
	
	def __init__(self):
		self.vr_sessions = {}
		self.ar_sessions = {}
		self.gesture_recognizer = GestureRecognizer()
		self.voice_processor = VoiceCommandProcessor()
		self.haptic_system = HapticFeedbackSystem()
		self.visualization_engine = Visualization3DEngine()
		
		# Interface settings
		self.hand_tracking_enabled = True
		self.voice_commands_enabled = True
		self.haptic_feedback_enabled = True
		self.spatial_audio_enabled = True
		
		# Interaction settings
		self.selection_radius = 2.0
		self.grab_threshold = 0.8
		self.voice_activation_threshold = 0.7
		
		# Performance settings
		self.render_quality = "high"  # low, medium, high, ultra
		self.physics_quality = "medium"
		self.haptic_quality = "high"
	
	async def start_vr_session(self, user_id: str, platform: VRPlatform, config: Dict[str, Any]) -> str:
		"""Start VR session"""
		try:
			session = VRSession(
				user_id=user_id,
				platform=platform,
				room_scale=config.get("room_scale", True)
			)
			
			# Initialize controllers
			if config.get("controllers"):
				for controller_config in config["controllers"]:
					controller = VRController(
						id=controller_config["id"],
						hand=controller_config["hand"],
						connected=True
					)
					session.controllers[controller.hand] = controller
			
			# Initialize hand tracking
			if self.hand_tracking_enabled and config.get("hand_tracking"):
				for hand in ["left", "right"]:
					hand_tracking = HandTracking(hand=hand, tracked=True)
					session.hand_tracking[hand] = hand_tracking
			
			# Store session
			self.vr_sessions[session.session_id] = session
			
			# Configure visualization for VR
			self.visualization_engine.rendering_engine = "a_frame" if platform == VRPlatform.WEBXR else "three_js"
			
			return session.session_id
			
		except Exception as e:
			print(f"VR session start error: {e}")
			raise
	
	async def start_ar_session(self, user_id: str, platform: ARPlatform, config: Dict[str, Any]) -> str:
		"""Start AR session"""
		try:
			session = ARSession(
				user_id=user_id,
				platform=platform,
				plane_detection=config.get("plane_detection", True),
				occlusion_enabled=config.get("occlusion", True)
			)
			
			# Initialize surface anchors
			if config.get("anchors"):
				session.surface_anchors = config["anchors"]
			
			# Store session
			self.ar_sessions[session.session_id] = session
			
			return session.session_id
			
		except Exception as e:
			print(f"AR session start error: {e}")
			raise
	
	async def render_workflow_vr(self, session_id: str, workflow_id: str) -> Dict[str, Any]:
		"""Render workflow in VR"""
		try:
			session = self.vr_sessions.get(session_id)
			if not session:
				raise ValueError(f"VR session {session_id} not found")
			
			# Generate VR-optimized 3D visualization
			render_data = await self.visualization_engine.render_workflow_3d(
				workflow_id,
				mode="vr_immersive",
				layout="physics_based"
			)
			
			# Add VR-specific enhancements
			vr_enhancements = {
				"room_scale_bounds": session.play_area_bounds,
				"hand_tracking": self.hand_tracking_enabled,
				"haptic_feedback": self.haptic_feedback_enabled,
				"spatial_audio": self.spatial_audio_enabled,
				"interaction_zones": self._generate_interaction_zones(render_data),
				"ui_panels": self._generate_vr_ui_panels(),
				"teleportation_points": self._generate_teleportation_points(render_data)
			}
			
			render_data.update(vr_enhancements)
			
			return render_data
			
		except Exception as e:
			print(f"VR workflow rendering error: {e}")
			raise
	
	async def render_workflow_ar(self, session_id: str, workflow_id: str) -> Dict[str, Any]:
		"""Render workflow in AR"""
		try:
			session = self.ar_sessions.get(session_id)
			if not session:
				raise ValueError(f"AR session {session_id} not found")
			
			# Generate AR-optimized 3D visualization
			render_data = await self.visualization_engine.render_workflow_3d(
				workflow_id,
				mode="ar_overlay",
				layout="hierarchical"
			)
			
			# Add AR-specific enhancements
			ar_enhancements = {
				"surface_anchors": session.surface_anchors,
				"occlusion_enabled": session.occlusion_enabled,
				"light_estimation": session.light_estimation,
				"plane_detection": session.plane_detection,
				"world_tracking": True,
				"overlay_ui": self._generate_ar_overlay_ui(),
				"anchor_points": self._generate_anchor_points(render_data)
			}
			
			render_data.update(ar_enhancements)
			
			return render_data
			
		except Exception as e:
			print(f"AR workflow rendering error: {e}")
			raise
	
	async def process_vr_interaction(self, session_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Process VR interaction"""
		try:
			session = self.vr_sessions.get(session_id)
			if not session:
				raise ValueError(f"VR session {session_id} not found")
			
			interaction_type = interaction_data.get("type")
			results = []
			
			if interaction_type == "hand_tracking":
				# Process hand tracking data
				hand_data = interaction_data.get("hand_data", {})
				
				# Update session hand tracking
				for hand_name, hand_info in hand_data.items():
					if hand_name in session.hand_tracking:
						session.hand_tracking[hand_name].palm_position = Vector3D(**hand_info.get("palm_position", {}))
						session.hand_tracking[hand_name].pinch_strength = hand_info.get("pinch_strength", 0.0)
						session.hand_tracking[hand_name].grab_strength = hand_info.get("grab_strength", 0.0)
				
				# Recognize gestures
				gestures = await self.gesture_recognizer.recognize_gesture(session.hand_tracking)
				
				# Process gestures
				for gesture in gestures:
					gesture_result = await self._process_gesture(gesture, session)
					results.append(gesture_result)
			
			elif interaction_type == "controller":
				# Process controller input
				controller_data = interaction_data.get("controller_data", {})
				
				for controller_id, controller_info in controller_data.items():
					controller_result = await self._process_controller_input(controller_info, session)
					results.append(controller_result)
			
			elif interaction_type == "voice":
				# Process voice command
				audio_data = interaction_data.get("audio_data")
				if audio_data:
					voice_result = await self.voice_processor.process_voice_command(
						audio_data,
						{"session_id": session_id, "user_id": session.user_id}
					)
					results.append(voice_result)
			
			return {
				"success": True,
				"session_id": session_id,
				"interaction_type": interaction_type,
				"results": results,
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			print(f"VR interaction processing error: {e}")
			return {"success": False, "error": str(e)}
	
	async def _process_gesture(self, gesture: Dict[str, Any], session: VRSession) -> Dict[str, Any]:
		"""Process recognized gesture"""
		gesture_name = gesture.get("gesture")
		
		if gesture_name == "select":
			# Handle selection gesture
			position = gesture.get("position")
			selected_object = await self._find_object_at_position(position, session)
			
			if selected_object:
				await self.haptic_system.trigger_haptic_feedback(
					gesture.get("hand", "right"),
					"select"
				)
				
				return {
					"action": "select",
					"object": selected_object,
					"position": position,
					"success": True
				}
		
		elif gesture_name == "grab":
			# Handle grab gesture
			position = gesture.get("position")
			grabbed_object = await self._find_object_at_position(position, session)
			
			if grabbed_object:
				await self.haptic_system.trigger_haptic_feedback(
					gesture.get("hand", "right"),
					"grab"
				)
				
				return {
					"action": "grab",
					"object": grabbed_object,
					"position": position,
					"success": True
				}
		
		elif gesture_name == "create_node":
			# Handle node creation
			position = gesture.get("position")
			
			await self.haptic_system.trigger_haptic_feedback(
				gesture.get("hand", "right"),
				"success"
			)
			
			return {
				"action": "create_node",
				"position": position,
				"node_type": "task",
				"success": True
			}
		
		return {"action": "unknown_gesture", "gesture": gesture_name, "success": False}
	
	async def _process_controller_input(self, controller_data: Dict[str, Any], session: VRSession) -> Dict[str, Any]:
		"""Process controller input"""
		controller_id = controller_data.get("id")
		
		# Update controller state
		if controller_id in session.controllers:
			controller = session.controllers[controller_id]
			controller.position = Vector3D(**controller_data.get("position", {}))
			controller.rotation = Vector3D(**controller_data.get("rotation", {}))
			controller.trigger_pressed = controller_data.get("trigger_pressed", False)
			controller.grip_pressed = controller_data.get("grip_pressed", False)
			
			# Process trigger press
			if controller.trigger_pressed and not controller_data.get("trigger_was_pressed", False):
				# Trigger just pressed
				await self.haptic_system.trigger_haptic_feedback(controller_id, "select")
				
				return {
					"action": "trigger_press",
					"controller": controller_id,
					"position": controller.position,
					"success": True
				}
		
		return {"action": "controller_update", "controller": controller_id, "success": True}
	
	async def _find_object_at_position(self, position: Vector3D, session: VRSession) -> Optional[Dict[str, Any]]:
		"""Find object at 3D position using collision detection"""
		
		if not session.workflow_objects:
			return None
		
		# Get interaction radius from session or use default
		interaction_radius = getattr(session, 'interaction_radius', 0.5)
		
		# Perform 3D collision detection with workflow objects
		closest_object = None
		closest_distance = float('inf')
		
		for obj in session.workflow_objects:
			obj_position = obj.get("position", {"x": 0, "y": 0, "z": 0})
			obj_pos_vector = Vector3D(
				x=obj_position.get("x", 0),
				y=obj_position.get("y", 0),
				z=obj_position.get("z", 0)
			)
			
			# Calculate 3D distance
			distance = math.sqrt(
				(position.x - obj_pos_vector.x) ** 2 +
				(position.y - obj_pos_vector.y) ** 2 +
				(position.z - obj_pos_vector.z) ** 2
			)
			
			# Check if within interaction radius and closer than previous objects
			if distance <= interaction_radius and distance < closest_distance:
				closest_distance = distance
				closest_object = obj
		
		# Return closest object if found
		if closest_object:
			return {
				"id": closest_object.get("id", uuid7str()),
				"type": closest_object.get("type", "unknown"),
				"position": position.__dict__,
				"distance": closest_distance,
				"properties": closest_object.get("properties", {}),
				"workflow_id": closest_object.get("workflow_id"),
				"node_data": closest_object.get("node_data", {})
			}
		
		return None
	
	def _generate_interaction_zones(self, render_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate VR interaction zones"""
		zones = []
		
		# Create interaction zones around nodes
		for node in render_data.get("nodes", []):
			position = node.get("position", {})
			zones.append({
				"id": f"zone_{node['id']}",
				"type": "node_interaction",
				"position": position,
				"radius": node.get("size", 2.0) * 2,
				"actions": ["select", "grab", "edit"]
			})
		
		# Add UI interaction zones
		zones.extend([
			{
				"id": "menu_zone",
				"type": "ui_menu",
				"position": {"x": -5, "y": 0, "z": 0},
				"radius": 3.0,
				"actions": ["activate_menu"]
			},
			{
				"id": "tools_zone",
				"type": "tool_palette",
				"position": {"x": 5, "y": 0, "z": 0},
				"radius": 3.0,
				"actions": ["select_tool"]
			}
		])
		
		return zones
	
	def _generate_vr_ui_panels(self) -> List[Dict[str, Any]]:
		"""Generate VR UI panels"""
		return [
			{
				"id": "main_menu",
				"type": "menu_panel",
				"position": {"x": 0, "y": 2, "z": -3},
				"rotation": {"x": 0, "y": 0, "z": 0},
				"size": {"width": 4, "height": 3},
				"items": [
					{"label": "New Workflow", "action": "create_workflow"},
					{"label": "Open Workflow", "action": "open_workflow"},
					{"label": "Save Workflow", "action": "save_workflow"},
					{"label": "Run Workflow", "action": "run_workflow"},
					{"label": "Settings", "action": "show_settings"}
				]
			},
			{
				"id": "tool_palette",
				"type": "tool_panel",
				"position": {"x": 4, "y": 1, "z": 0},
				"rotation": {"x": 0, "y": -30, "z": 0},
				"size": {"width": 2, "height": 4},
				"tools": [
					{"name": "Task", "type": "task_node", "icon": "task"},
					{"name": "Decision", "type": "decision_node", "icon": "decision"},
					{"name": "Connector", "type": "connector_node", "icon": "connector"},
					{"name": "Start", "type": "start_node", "icon": "start"},
					{"name": "End", "type": "end_node", "icon": "end"}
				]
			},
			{
				"id": "properties_panel",
				"type": "properties_panel",
				"position": {"x": -4, "y": 1, "z": 0},
				"rotation": {"x": 0, "y": 30, "z": 0},
				"size": {"width": 3, "height": 4},
				"fields": []  # Populated based on selection
			}
		]
	
	def _generate_teleportation_points(self, render_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate VR teleportation points"""
		points = []
		
		# Add teleportation points around the workflow
		nodes = render_data.get("nodes", [])
		if nodes:
			# Calculate workflow bounds
			min_x = min(node["position"]["x"] for node in nodes)
			max_x = max(node["position"]["x"] for node in nodes)
			min_z = min(node["position"]["z"] for node in nodes)
			max_z = max(node["position"]["z"] for node in nodes)
			
			# Add points around the perimeter
			margin = 20
			points.extend([
				{"position": {"x": min_x - margin, "y": 0, "z": min_z - margin}, "label": "Bottom Left"},
				{"position": {"x": max_x + margin, "y": 0, "z": min_z - margin}, "label": "Bottom Right"},
				{"position": {"x": min_x - margin, "y": 0, "z": max_z + margin}, "label": "Top Left"},
				{"position": {"x": max_x + margin, "y": 0, "z": max_z + margin}, "label": "Top Right"},
				{"position": {"x": (min_x + max_x) / 2, "y": 0, "z": min_z - margin * 2}, "label": "South View"},
				{"position": {"x": (min_x + max_x) / 2, "y": 0, "z": max_z + margin * 2}, "label": "North View"}
			])
		
		return points
	
	def _generate_ar_overlay_ui(self) -> List[Dict[str, Any]]:
		"""Generate AR overlay UI elements"""
		return [
			{
				"id": "ar_toolbar",
				"type": "toolbar",
				"anchor_type": "screen_space",
				"position": {"x": 0.5, "y": 0.9},  # Normalized screen coordinates
				"items": [
					{"icon": "play", "action": "run_workflow", "label": "Run"},
					{"icon": "pause", "action": "pause_workflow", "label": "Pause"},
					{"icon": "stop", "action": "stop_workflow", "label": "Stop"},
					{"icon": "settings", "action": "show_settings", "label": "Settings"}
				]
			},
			{
				"id": "ar_info_panel",
				"type": "info_panel",
				"anchor_type": "world_space",
				"position": {"x": 0, "y": 2, "z": 0},
				"content": {
					"workflow_name": "",
					"status": "",
					"progress": 0,
					"current_step": ""
				}
			}
		]
	
	def _generate_anchor_points(self, render_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Generate AR anchor points"""
		anchors = []
		
		# Create anchors for workflow nodes
		for node in render_data.get("nodes", []):
			anchors.append({
				"id": f"anchor_{node['id']}",
				"type": "node_anchor",
				"world_position": node["position"],
				"tracking_type": "feature_point",
				"persistence": True
			})
		
		# Add main workflow anchor
		anchors.append({
			"id": "workflow_anchor",
			"type": "workflow_root",
			"world_position": {"x": 0, "y": 0, "z": 0},
			"tracking_type": "plane",
			"persistence": True
		})
		
		return anchors


# Global VR/AR interface instance
vr_ar_interface = VRARWorkflowInterface()