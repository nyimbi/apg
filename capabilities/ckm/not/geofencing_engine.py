#!/usr/bin/env python3
"""
Geofencing Engine for APG Notification System

This module provides comprehensive location-based notification capabilities,
including geofencing, proximity detection, location analytics, and privacy-aware
location services for contextual messaging.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import geopy.distance
from geopy.geocoders import Nominatim
import hashlib
import secrets

logger = logging.getLogger(__name__)

class LocationEventType(Enum):
	"""Location-based event types"""
	ENTER_GEOFENCE = "enter_geofence"
	EXIT_GEOFENCE = "exit_geofence"
	DWELL_IN_GEOFENCE = "dwell_in_geofence"
	PROXIMITY_ALERT = "proximity_alert"
	LOCATION_UPDATE = "location_update"
	SPEED_THRESHOLD = "speed_threshold"
	ROUTE_DEVIATION = "route_deviation"
	ARRIVAL_PREDICTION = "arrival_prediction"
	BATCH_LOCATION_UPDATE = "batch_location_update"

class GeofenceType(Enum):
	"""Types of geofences"""
	CIRCULAR = "circular"
	POLYGON = "polygon"
	RECTANGULAR = "rectangular"
	ROUTE_CORRIDOR = "route_corridor"
	ADMINISTRATIVE = "administrative"  # City, state, country boundaries
	CUSTOM = "custom"

class LocationAccuracy(Enum):
	"""Location accuracy levels"""
	HIGH = "high"  # GPS precision < 10m
	MEDIUM = "medium"  # GPS precision 10-100m
	LOW = "low"  # GPS precision > 100m
	CELL_TOWER = "cell_tower"  # Cell tower triangulation
	WIFI = "wifi"  # WiFi-based location
	IP = "ip"  # IP geolocation

class GeofenceStatus(Enum):
	"""Geofence status"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	PAUSED = "paused"
	EXPIRED = "expired"

class LocationPrivacyLevel(Enum):
	"""Location privacy levels"""
	FULL = "full"  # Exact coordinates
	APPROXIMATE = "approximate"  # City/neighborhood level
	ANONYMOUS = "anonymous"  # No personal identification
	NONE = "none"  # No location tracking

@dataclass
class Location:
	"""Represents a geographic location"""
	latitude: float
	longitude: float
	accuracy: float = 0.0  # meters
	altitude: Optional[float] = None
	speed: Optional[float] = None  # km/h
	heading: Optional[float] = None  # degrees
	timestamp: datetime = field(default_factory=datetime.utcnow)
	source: LocationAccuracy = LocationAccuracy.GPS
	address: Optional[str] = None
	
	def distance_to(self, other: 'Location') -> float:
		"""Calculate distance to another location in meters"""
		return geopy.distance.geodesic(
			(self.latitude, self.longitude),
			(other.latitude, other.longitude)
		).meters
	
	def bearing_to(self, other: 'Location') -> float:
		"""Calculate bearing to another location in degrees"""
		lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
		lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
		
		dlon = lon2 - lon1
		
		y = math.sin(dlon) * math.cos(lat2)
		x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
		
		bearing = math.atan2(y, x)
		bearing = math.degrees(bearing)
		bearing = (bearing + 360) % 360
		
		return bearing
	
	def __str__(self) -> str:
		return f"Location({self.latitude:.6f}, {self.longitude:.6f})"

@dataclass
class Geofence:
	"""Represents a geofenced area"""
	fence_id: str
	name: str
	fence_type: GeofenceType
	center: Location
	radius: Optional[float] = None  # meters for circular geofences
	vertices: Optional[List[Location]] = None  # for polygon geofences
	tenant_id: str = ""
	user_id: Optional[str] = None  # If user-specific
	status: GeofenceStatus = GeofenceStatus.ACTIVE
	created_at: datetime = field(default_factory=datetime.utcnow)
	expires_at: Optional[datetime] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
	notification_config: Dict[str, Any] = field(default_factory=dict)
	
	def contains_location(self, location: Location) -> bool:
		"""Check if location is within this geofence"""
		if self.fence_type == GeofenceType.CIRCULAR:
			return self._contains_circular(location)
		elif self.fence_type == GeofenceType.POLYGON:
			return self._contains_polygon(location)
		elif self.fence_type == GeofenceType.RECTANGULAR:
			return self._contains_rectangular(location)
		else:
			return False
	
	def _contains_circular(self, location: Location) -> bool:
		"""Check if location is within circular geofence"""
		if not self.radius:
			return False
		return self.center.distance_to(location) <= self.radius
	
	def _contains_polygon(self, location: Location) -> bool:
		"""Check if location is within polygon geofence using ray casting"""
		if not self.vertices or len(self.vertices) < 3:
			return False
		
		x, y = location.longitude, location.latitude
		n = len(self.vertices)
		inside = False
		
		p1x, p1y = self.vertices[0].longitude, self.vertices[0].latitude
		for i in range(1, n + 1):
			p2x, p2y = self.vertices[i % n].longitude, self.vertices[i % n].latitude
			if y > min(p1y, p2y):
				if y <= max(p1y, p2y):
					if x <= max(p1x, p2x):
						if p1y != p2y:
							xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
						if p1x == p2x or x <= xinters:
							inside = not inside
			p1x, p1y = p2x, p2y
		
		return inside
	
	def _contains_rectangular(self, location: Location) -> bool:
		"""Check if location is within rectangular geofence"""
		if not self.vertices or len(self.vertices) != 2:
			return False
		
		# Assuming vertices[0] is bottom-left, vertices[1] is top-right
		min_lat = min(self.vertices[0].latitude, self.vertices[1].latitude)
		max_lat = max(self.vertices[0].latitude, self.vertices[1].latitude)
		min_lon = min(self.vertices[0].longitude, self.vertices[1].longitude)
		max_lon = max(self.vertices[0].longitude, self.vertices[1].longitude)
		
		return (min_lat <= location.latitude <= max_lat and
				min_lon <= location.longitude <= max_lon)

@dataclass
class LocationEvent:
	"""Location-based event"""
	event_id: str
	tenant_id: str
	user_id: str
	event_type: LocationEventType
	location: Location
	geofence: Optional[Geofence] = None
	timestamp: datetime = field(default_factory=datetime.utcnow)
	metadata: Dict[str, Any] = field(default_factory=dict)
	processed: bool = False
	notifications_sent: List[str] = field(default_factory=list)

@dataclass
class LocationTrackingSession:
	"""User location tracking session"""
	session_id: str
	tenant_id: str
	user_id: str
	started_at: datetime
	last_update: datetime
	privacy_level: LocationPrivacyLevel
	accuracy_threshold: float = 100.0  # meters
	update_interval: int = 30  # seconds
	active: bool = True
	geofences: Set[str] = field(default_factory=set)
	location_history: List[Location] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)

class LocationPrivacyManager:
	"""Manage location privacy and anonymization"""
	
	def __init__(self):
		self.anonymization_keys: Dict[str, str] = {}
	
	def apply_privacy_level(
		self,
		location: Location,
		privacy_level: LocationPrivacyLevel,
		user_id: str
	) -> Location:
		"""Apply privacy filtering to location data"""
		
		if privacy_level == LocationPrivacyLevel.FULL:
			return location
		
		elif privacy_level == LocationPrivacyLevel.APPROXIMATE:
			# Round to ~1km precision
			return Location(
				latitude=round(location.latitude, 2),
				longitude=round(location.longitude, 2),
				accuracy=1000.0,
				timestamp=location.timestamp,
				source=location.source
			)
		
		elif privacy_level == LocationPrivacyLevel.ANONYMOUS:
			# Use anonymized coordinates
			anon_key = self._get_anonymization_key(user_id)
			offset_lat = hash(anon_key + "lat") % 1000 / 100000.0  # ~10m offset
			offset_lon = hash(anon_key + "lon") % 1000 / 100000.0
			
			return Location(
				latitude=location.latitude + offset_lat,
				longitude=location.longitude + offset_lon,
				accuracy=100.0,
				timestamp=location.timestamp,
				source=LocationAccuracy.LOW
			)
		
		else:  # NONE
			return None
	
	def _get_anonymization_key(self, user_id: str) -> str:
		"""Get consistent anonymization key for user"""
		if user_id not in self.anonymization_keys:
			self.anonymization_keys[user_id] = hashlib.sha256(
				f"anon_{user_id}_{secrets.token_hex(16)}".encode()
			).hexdigest()
		return self.anonymization_keys[user_id]

class GeocodeService:
	"""Geocoding and reverse geocoding service"""
	
	def __init__(self):
		self.geocoder = Nominatim(user_agent="apg_notification_system")
		self.cache: Dict[str, Any] = {}
	
	async def geocode_address(self, address: str) -> Optional[Location]:
		"""Convert address to coordinates"""
		cache_key = f"geocode_{hashlib.md5(address.encode()).hexdigest()}"
		
		if cache_key in self.cache:
			cached = self.cache[cache_key]
			return Location(
				latitude=cached['lat'],
				longitude=cached['lon'],
				address=address
			)
		
		try:
			result = self.geocoder.geocode(address)
			if result:
				location = Location(
					latitude=result.latitude,
					longitude=result.longitude,
					address=result.address
				)
				
				# Cache result
				self.cache[cache_key] = {
					'lat': result.latitude,
					'lon': result.longitude,
					'address': result.address
				}
				
				return location
		except Exception as e:
			logger.error(f"Geocoding failed for '{address}': {str(e)}")
		
		return None
	
	async def reverse_geocode(self, location: Location) -> Optional[str]:
		"""Convert coordinates to address"""
		cache_key = f"reverse_{location.latitude:.4f}_{location.longitude:.4f}"
		
		if cache_key in self.cache:
			return self.cache[cache_key]
		
		try:
			result = self.geocoder.reverse(f"{location.latitude}, {location.longitude}")
			if result:
				address = result.address
				self.cache[cache_key] = address
				return address
		except Exception as e:
			logger.error(f"Reverse geocoding failed for {location}: {str(e)}")
		
		return None

class GeofenceManager:
	"""Manage geofences and location-based triggers"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.geofences: Dict[str, Geofence] = {}
		self.user_geofences: Dict[str, Set[str]] = {}  # user_id -> fence_ids
		self.privacy_manager = LocationPrivacyManager()
		
	def create_geofence(
		self,
		name: str,
		fence_type: GeofenceType,
		center: Location,
		radius: Optional[float] = None,
		vertices: Optional[List[Location]] = None,
		user_id: Optional[str] = None,
		expires_in_hours: Optional[int] = None,
		notification_config: Optional[Dict[str, Any]] = None
	) -> str:
		"""Create a new geofence"""
		
		fence_id = f"fence_{secrets.token_hex(16)}"
		
		expires_at = None
		if expires_in_hours:
			expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
		
		geofence = Geofence(
			fence_id=fence_id,
			name=name,
			fence_type=fence_type,
			center=center,
			radius=radius,
			vertices=vertices,
			tenant_id=self.tenant_id,
			user_id=user_id,
			expires_at=expires_at,
			notification_config=notification_config or {}
		)
		
		self.geofences[fence_id] = geofence
		
		# Track user-specific geofences
		if user_id:
			if user_id not in self.user_geofences:
				self.user_geofences[user_id] = set()
			self.user_geofences[user_id].add(fence_id)
		
		logger.info(f"Created geofence {fence_id}: {name} for tenant {self.tenant_id}")
		return fence_id
	
	def update_geofence(self, fence_id: str, updates: Dict[str, Any]) -> bool:
		"""Update geofence properties"""
		if fence_id not in self.geofences:
			return False
		
		geofence = self.geofences[fence_id]
		
		for key, value in updates.items():
			if hasattr(geofence, key):
				setattr(geofence, key, value)
		
		logger.info(f"Updated geofence {fence_id}")
		return True
	
	def delete_geofence(self, fence_id: str) -> bool:
		"""Delete a geofence"""
		if fence_id not in self.geofences:
			return False
		
		geofence = self.geofences[fence_id]
		
		# Remove from user tracking
		if geofence.user_id and geofence.user_id in self.user_geofences:
			self.user_geofences[geofence.user_id].discard(fence_id)
		
		del self.geofences[fence_id]
		logger.info(f"Deleted geofence {fence_id}")
		return True
	
	def get_active_geofences(self, user_id: Optional[str] = None) -> List[Geofence]:
		"""Get active geofences, optionally filtered by user"""
		active_fences = []
		
		for geofence in self.geofences.values():
			# Check if expired
			if geofence.expires_at and datetime.utcnow() > geofence.expires_at:
				geofence.status = GeofenceStatus.EXPIRED
				continue
			
			# Check if active
			if geofence.status != GeofenceStatus.ACTIVE:
				continue
			
			# Filter by user if specified
			if user_id and geofence.user_id and geofence.user_id != user_id:
				continue
			
			active_fences.append(geofence)
		
		return active_fences
	
	def check_geofence_events(
		self,
		user_id: str,
		current_location: Location,
		previous_location: Optional[Location] = None,
		privacy_level: LocationPrivacyLevel = LocationPrivacyLevel.FULL
	) -> List[LocationEvent]:
		"""Check for geofence events (enter/exit/dwell)"""
		
		# Apply privacy filtering
		filtered_location = self.privacy_manager.apply_privacy_level(
			current_location, privacy_level, user_id
		)
		
		if not filtered_location:
			return []
		
		events = []
		active_fences = self.get_active_geofences(user_id)
		
		for geofence in active_fences:
			# Check current containment
			is_inside = geofence.contains_location(filtered_location)
			
			# Check previous containment if available
			was_inside = False
			if previous_location:
				filtered_previous = self.privacy_manager.apply_privacy_level(
					previous_location, privacy_level, user_id
				)
				if filtered_previous:
					was_inside = geofence.contains_location(filtered_previous)
			
			# Generate events based on state changes
			if is_inside and not was_inside:
				# Enter event
				event = LocationEvent(
					event_id=f"event_{secrets.token_hex(16)}",
					tenant_id=self.tenant_id,
					user_id=user_id,
					event_type=LocationEventType.ENTER_GEOFENCE,
					location=filtered_location,
					geofence=geofence,
					metadata={
						'fence_name': geofence.name,
						'fence_type': geofence.fence_type.value
					}
				)
				events.append(event)
				
			elif not is_inside and was_inside:
				# Exit event 
				event = LocationEvent(
					event_id=f"event_{secrets.token_hex(16)}",
					tenant_id=self.tenant_id,
					user_id=user_id,
					event_type=LocationEventType.EXIT_GEOFENCE,
					location=filtered_location,
					geofence=geofence,
					metadata={
						'fence_name': geofence.name,
						'fence_type': geofence.fence_type.value
					}
				)
				events.append(event)
		
		return events

class LocationAnalytics:
	"""Location analytics and insights"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.location_history: Dict[str, List[Location]] = {}
		self.dwell_times: Dict[str, Dict[str, List[Tuple[datetime, datetime]]]] = {}
		
	def add_location_history(self, user_id: str, location: Location):
		"""Add location to user's history"""
		if user_id not in self.location_history:
			self.location_history[user_id] = []
		
		self.location_history[user_id].append(location)
		
		# Keep only last 1000 locations per user
		if len(self.location_history[user_id]) > 1000:
			self.location_history[user_id] = self.location_history[user_id][-1000:]
	
	def calculate_dwell_time(
		self,
		user_id: str,
		geofence: Geofence,
		start_time: datetime,
		end_time: datetime
	) -> float:
		"""Calculate dwell time in geofence (in minutes)"""
		if user_id not in self.dwell_times:
			self.dwell_times[user_id] = {}
		
		if geofence.fence_id not in self.dwell_times[user_id]:
			self.dwell_times[user_id][geofence.fence_id] = []
		
		# Add dwell period
		self.dwell_times[user_id][geofence.fence_id].append((start_time, end_time))
		
		# Calculate total time
		total_minutes = (end_time - start_time).total_seconds() / 60.0
		return total_minutes
	
	def get_frequent_locations(
		self,
		user_id: str,
		radius_meters: float = 100.0,
		min_visits: int = 3
	) -> List[Dict[str, Any]]:
		"""Identify frequently visited locations"""
		if user_id not in self.location_history:
			return []
		
		locations = self.location_history[user_id]
		clusters = []
		
		for location in locations:
			# Find existing cluster within radius
			matched_cluster = None
			for cluster in clusters:
				if location.distance_to(cluster['center']) <= radius_meters:
					matched_cluster = cluster
					break
			
			if matched_cluster:
				# Add to existing cluster
				matched_cluster['locations'].append(location)
				matched_cluster['visits'] += 1
				# Update centroid
				matched_cluster['center'] = self._calculate_centroid(matched_cluster['locations'])
			else:
				# Create new cluster
				clusters.append({
					'center': location,
					'locations': [location],
					'visits': 1
				})
		
		# Filter by minimum visits and sort by frequency
		frequent_locations = [
			cluster for cluster in clusters
			if cluster['visits'] >= min_visits
		]
		frequent_locations.sort(key=lambda x: x['visits'], reverse=True)
		
		return frequent_locations
	
	def _calculate_centroid(self, locations: List[Location]) -> Location:
		"""Calculate centroid of location cluster"""
		if not locations:
			return Location(0, 0)
		
		total_lat = sum(loc.latitude for loc in locations)
		total_lon = sum(loc.longitude for loc in locations)
		count = len(locations)
		
		return Location(
			latitude=total_lat / count,
			longitude=total_lon / count
		)
	
	def analyze_movement_patterns(self, user_id: str) -> Dict[str, Any]:
		"""Analyze user movement patterns"""
		if user_id not in self.location_history:
			return {}
		
		locations = self.location_history[user_id]
		if len(locations) < 2:
			return {}
		
		# Calculate movement statistics
		total_distance = 0.0
		max_speed = 0.0
		speeds = []
		
		for i in range(1, len(locations)):
			prev_loc = locations[i-1]
			curr_loc = locations[i]
			
			distance = prev_loc.distance_to(curr_loc)
			time_diff = (curr_loc.timestamp - prev_loc.timestamp).total_seconds()
			
			if time_diff > 0:
				speed = (distance / time_diff) * 3.6  # km/h
				speeds.append(speed)
				max_speed = max(max_speed, speed)
				total_distance += distance
		
		avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
		
		# Identify home and work locations (most frequent during night/day)
		day_locations = []
		night_locations = []
		
		for location in locations:
			hour = location.timestamp.hour
			if 9 <= hour <= 17:  # Work hours
				day_locations.append(location)
			elif 22 <= hour or hour <= 6:  # Sleep hours
				night_locations.append(location)
		
		home_candidates = self.get_frequent_locations(user_id)
		work_candidates = self.get_frequent_locations(user_id)
		
		return {
			'total_distance_km': total_distance / 1000.0,
			'avg_speed_kmh': avg_speed,
			'max_speed_kmh': max_speed,
			'total_locations': len(locations),
			'time_span_hours': (locations[-1].timestamp - locations[0].timestamp).total_seconds() / 3600.0,
			'frequent_locations': self.get_frequent_locations(user_id),
			'estimated_home': home_candidates[0] if home_candidates else None,
			'movement_summary': 'active' if avg_speed > 5 else 'stationary'
		}

class LocationNotificationEngine:
	"""Engine for location-based notifications"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.notification_queue: List[Dict[str, Any]] = []
		self.processed_events: Set[str] = set()
		
	async def process_location_event(
		self,
		event: LocationEvent,
		notification_service: Any = None  # Integration point with main notification system
	) -> List[str]:
		"""Process location event and trigger notifications"""
		
		if event.event_id in self.processed_events:
			return []
		
		notifications_sent = []
		
		# Get notification configuration from geofence
		if event.geofence and event.geofence.notification_config:
			config = event.geofence.notification_config
			
			# Create notification based on event type
			notification_data = await self._create_location_notification(event, config)
			
			if notification_data:
				# Queue notification for sending
				self.notification_queue.append(notification_data)
				
				# If notification service is available, send immediately
				if notification_service:
					notification_id = await self._send_notification(
						notification_service, notification_data
					)
					if notification_id:
						notifications_sent.append(notification_id)
				
				logger.info(f"Queued location notification for event {event.event_id}")
		
		# Mark event as processed
		self.processed_events.add(event.event_id)
		event.processed = True
		event.notifications_sent = notifications_sent
		
		return notifications_sent
	
	async def _create_location_notification(
		self,
		event: LocationEvent,
		config: Dict[str, Any]
	) -> Optional[Dict[str, Any]]:
		"""Create notification data from location event"""
		
		if event.event_type == LocationEventType.ENTER_GEOFENCE:
			return await self._create_enter_notification(event, config)
		elif event.event_type == LocationEventType.EXIT_GEOFENCE:
			return await self._create_exit_notification(event, config)
		elif event.event_type == LocationEventType.PROXIMITY_ALERT:
			return await self._create_proximity_notification(event, config)
		else:
			return None
	
	async def _create_enter_notification(
		self,
		event: LocationEvent,
		config: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Create notification for geofence entry"""
		
		template = config.get('enter_template', {})
		
		return {
			'user_id': event.user_id,
			'template_id': template.get('template_id', 'default_enter'),
			'subject': template.get('subject', f"Welcome to {event.geofence.name}!"),
			'message': template.get('message', f"You've entered {event.geofence.name}"),
			'channels': config.get('channels', ['push']),
			'priority': config.get('priority', 'normal'),
			'context': {
				'location_name': event.geofence.name,
				'event_type': 'enter',
				'timestamp': event.timestamp.isoformat(),
				'latitude': event.location.latitude,
				'longitude': event.location.longitude
			},
			'metadata': {
				'location_based': True,
				'geofence_id': event.geofence.fence_id,
				'event_id': event.event_id
			}
		}
	
	async def _create_exit_notification(
		self,
		event: LocationEvent,
		config: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Create notification for geofence exit"""
		
		template = config.get('exit_template', {})
		
		return {
			'user_id': event.user_id,
			'template_id': template.get('template_id', 'default_exit'),
			'subject': template.get('subject', f"Thanks for visiting {event.geofence.name}!"),
			'message': template.get('message', f"You've left {event.geofence.name}"),
			'channels': config.get('channels', ['push']),
			'priority': config.get('priority', 'normal'),
			'context': {
				'location_name': event.geofence.name,
				'event_type': 'exit',
				'timestamp': event.timestamp.isoformat(),
				'latitude': event.location.latitude,
				'longitude': event.location.longitude
			},
			'metadata': {
				'location_based': True,
				'geofence_id': event.geofence.fence_id,
				'event_id': event.event_id
			}
		}
	
	async def _create_proximity_notification(
		self,
		event: LocationEvent,
		config: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Create notification for proximity alert"""
		
		template = config.get('proximity_template', {})
		
		return {
			'user_id': event.user_id,
			'template_id': template.get('template_id', 'default_proximity'),
			'subject': template.get('subject', f"You're near {event.geofence.name}!"),
			'message': template.get('message', f"You're approaching {event.geofence.name}"),
			'channels': config.get('channels', ['push']),
			'priority': config.get('priority', 'normal'),
			'context': {
				'location_name': event.geofence.name,
				'event_type': 'proximity',
				'timestamp': event.timestamp.isoformat(),
				'latitude': event.location.latitude,
				'longitude': event.location.longitude
			},
			'metadata': {
				'location_based': True,
				'geofence_id': event.geofence.fence_id,
				'event_id': event.event_id
			}
		}
	
	async def _send_notification(
		self,
		notification_service: Any,
		notification_data: Dict[str, Any]
	) -> Optional[str]:
		"""Send notification through main notification service"""
		try:
			# This would integrate with the main notification system
			# notification_id = await notification_service.send_notification(notification_data)
			# return notification_id
			
			# For now, just simulate
			notification_id = f"notif_{secrets.token_hex(8)}"
			logger.info(f"Sent location notification {notification_id}")
			return notification_id
			
		except Exception as e:
			logger.error(f"Failed to send location notification: {str(e)}")
			return None

class GeofencingEngine:
	"""Main geofencing engine orchestrating all location-based functionality"""
	
	def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Initialize components
		self.geofence_manager = GeofenceManager(tenant_id)
		self.analytics = LocationAnalytics(tenant_id)
		self.notification_engine = LocationNotificationEngine(tenant_id)
		self.geocode_service = GeocodeService()
		
		# Tracking sessions
		self.tracking_sessions: Dict[str, LocationTrackingSession] = {}
		
		# Event processing
		self.event_queue: List[LocationEvent] = []
		self.event_processors: List[Any] = []
		
		logger.info(f"Geofencing engine initialized for tenant {tenant_id}")
	
	async def start_location_tracking(
		self,
		user_id: str,
		privacy_level: LocationPrivacyLevel = LocationPrivacyLevel.FULL,
		accuracy_threshold: float = 100.0,
		update_interval: int = 30
	) -> str:
		"""Start location tracking session for user"""
		
		session_id = f"session_{secrets.token_hex(16)}"
		
		session = LocationTrackingSession(
			session_id=session_id,
			tenant_id=self.tenant_id,
			user_id=user_id,
			started_at=datetime.utcnow(),
			last_update=datetime.utcnow(),
			privacy_level=privacy_level,
			accuracy_threshold=accuracy_threshold,
			update_interval=update_interval
		)
		
		self.tracking_sessions[session_id] = session
		
		logger.info(f"Started location tracking session {session_id} for user {user_id}")
		return session_id
	
	async def update_user_location(
		self,
		user_id: str,
		location: Location,
		session_id: Optional[str] = None
	) -> List[LocationEvent]:
		"""Update user location and check for geofence events"""
		
		# Find active tracking session
		active_session = None
		if session_id and session_id in self.tracking_sessions:
			active_session = self.tracking_sessions[session_id]
		else:
			# Find any active session for user
			for session in self.tracking_sessions.values():
				if session.user_id == user_id and session.active:
					active_session = session
					break
		
		if not active_session:
			logger.warning(f"No active tracking session found for user {user_id}")
			return []
		
		# Update session
		previous_location = None
		if active_session.location_history:
			previous_location = active_session.location_history[-1]
		
		active_session.location_history.append(location)
		active_session.last_update = datetime.utcnow()
		
		# Keep location history manageable
		if len(active_session.location_history) > 100:
			active_session.location_history = active_session.location_history[-100:]
		
		# Add to analytics
		self.analytics.add_location_history(user_id, location)
		
		# Check for geofence events
		events = self.geofence_manager.check_geofence_events(
			user_id=user_id,
			current_location=location,
			previous_location=previous_location,
			privacy_level=active_session.privacy_level
		)
		
		# Process events
		for event in events:
			self.event_queue.append(event)
			
			# Process notification immediately if configured
			if event.geofence and event.geofence.notification_config:
				await self.notification_engine.process_location_event(event)
		
		logger.debug(f"Updated location for user {user_id}, generated {len(events)} events")
		return events
	
	async def create_location_geofence(
		self,
		name: str,
		address: str,
		radius: float,
		user_id: Optional[str] = None,
		notification_config: Optional[Dict[str, Any]] = None
	) -> Optional[str]:
		"""Create geofence from address"""
		
		# Geocode address to location
		location = await self.geocode_service.geocode_address(address)
		if not location:
			logger.error(f"Failed to geocode address: {address}")
			return None
		
		# Create circular geofence
		fence_id = self.geofence_manager.create_geofence(
			name=name,
			fence_type=GeofenceType.CIRCULAR,
			center=location,
			radius=radius,
			user_id=user_id,
			notification_config=notification_config
		)
		
		return fence_id
	
	async def create_polygon_geofence(
		self,
		name: str,
		coordinates: List[Tuple[float, float]],  # [(lat, lon), ...]
		user_id: Optional[str] = None,
		notification_config: Optional[Dict[str, Any]] = None
	) -> str:
		"""Create polygon geofence from coordinates"""
		
		# Convert coordinates to Location objects
		vertices = [Location(lat, lon) for lat, lon in coordinates]
		
		# Calculate centroid as center
		center_lat = sum(v.latitude for v in vertices) / len(vertices)
		center_lon = sum(v.longitude for v in vertices) / len(vertices)
		center = Location(center_lat, center_lon)
		
		# Create polygon geofence
		fence_id = self.geofence_manager.create_geofence(
			name=name,
			fence_type=GeofenceType.POLYGON,
			center=center,
			vertices=vertices,
			user_id=user_id,
			notification_config=notification_config
		)
		
		return fence_id
	
	async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
		"""Get comprehensive location analytics for user"""
		
		movement_patterns = self.analytics.analyze_movement_patterns(user_id)
		frequent_locations = self.analytics.get_frequent_locations(user_id)
		
		# Get active tracking session info
		active_session = None
		for session in self.tracking_sessions.values():
			if session.user_id == user_id and session.active:
				active_session = session
				break
		
		return {
			'user_id': user_id,
			'movement_patterns': movement_patterns,
			'frequent_locations': frequent_locations,
			'tracking_session': {
				'active': active_session is not None,
				'session_id': active_session.session_id if active_session else None,
				'privacy_level': active_session.privacy_level.value if active_session else None,
				'last_update': active_session.last_update.isoformat() if active_session else None
			},
			'geofences': len(self.geofence_manager.user_geofences.get(user_id, [])),
			'total_events': len([e for e in self.event_queue if e.user_id == user_id])
		}
	
	async def process_event_queue(self, notification_service: Any = None) -> int:
		"""Process queued location events"""
		
		processed_count = 0
		
		for event in self.event_queue:
			if not event.processed:
				await self.notification_engine.process_location_event(
					event, notification_service
				)
				processed_count += 1
		
		logger.info(f"Processed {processed_count} location events")
		return processed_count
	
	async def stop_location_tracking(self, user_id: str, session_id: Optional[str] = None) -> bool:
		"""Stop location tracking for user"""
		
		sessions_stopped = 0
		
		if session_id and session_id in self.tracking_sessions:
			session = self.tracking_sessions[session_id]
			if session.user_id == user_id:
				session.active = False
				sessions_stopped += 1
		else:
			# Stop all sessions for user
			for session in self.tracking_sessions.values():
				if session.user_id == user_id and session.active:
					session.active = False
					sessions_stopped += 1
		
		logger.info(f"Stopped {sessions_stopped} tracking sessions for user {user_id}")
		return sessions_stopped > 0
	
	async def get_engine_status(self) -> Dict[str, Any]:
		"""Get geofencing engine status"""
		
		active_sessions = sum(1 for s in self.tracking_sessions.values() if s.active)
		total_geofences = len(self.geofence_manager.geofences)
		pending_events = len([e for e in self.event_queue if not e.processed])
		
		return {
			'tenant_id': self.tenant_id,
			'active_tracking_sessions': active_sessions,
			'total_geofences': total_geofences,
			'pending_events': pending_events,
			'total_events_processed': len([e for e in self.event_queue if e.processed]),
			'engine_status': 'operational',
			'components': {
				'geofence_manager': 'active',
				'analytics': 'active',
				'notification_engine': 'active',
				'geocode_service': 'active'
			},
			'timestamp': datetime.utcnow().isoformat()
		}

# Factory function for easy instantiation
def create_geofencing_engine(tenant_id: str, config: Optional[Dict[str, Any]] = None) -> GeofencingEngine:
	"""Create a new geofencing engine instance"""
	return GeofencingEngine(tenant_id=tenant_id, config=config)

# Example usage
if __name__ == "__main__":
	import asyncio
	
	async def demo_geofencing():
		"""Demonstrate geofencing capabilities"""
		
		# Create geofencing engine
		engine = create_geofencing_engine("demo-tenant")
		
		# Start location tracking for a user
		session_id = await engine.start_location_tracking(
			user_id="user123",
			privacy_level=LocationPrivacyLevel.FULL,
			accuracy_threshold=50.0,
			update_interval=30
		)
		
		print(f"✅ Started tracking session: {session_id}")
		
		# Create a geofence around a location
		fence_id = await engine.create_location_geofence(
			name="Office Building",
			address="123 Main Street, San Francisco, CA",
			radius=100.0,  # 100 meter radius
			user_id="user123",
			notification_config={
				'channels': ['push', 'email'],
				'enter_template': {
					'subject': 'Welcome to the office!',
					'message': 'You\'ve arrived at work. Have a productive day!'
				},
				'exit_template': {
					'subject': 'Thanks for your hard work!',
					'message': 'You\'ve left the office. Safe travels home!'
				}
			}
		)
		
		if fence_id:
			print(f"✅ Created geofence: {fence_id}")
			
			# Simulate location updates
			test_locations = [
				Location(37.7749, -122.4194),  # Outside geofence
				Location(37.7849, -122.4094),  # Moving closer
				Location(37.7849, -122.4084),  # Inside geofence (enter event)
				Location(37.7849, -122.4074),  # Still inside
				Location(37.7749, -122.4194),  # Outside again (exit event)
			]
			
			for i, location in enumerate(test_locations):
				events = await engine.update_user_location("user123", location, session_id)
				print(f"📍 Location {i+1}: {len(events)} events generated")
				
				for event in events:
					print(f"   🎯 {event.event_type.value}: {event.geofence.name if event.geofence else 'N/A'}")
			
			# Process event queue
			processed = await engine.process_event_queue()
			print(f"✅ Processed {processed} events")
			
			# Get user analytics
			analytics = await engine.get_user_analytics("user123")
			print(f"📊 User analytics: {analytics['movement_patterns']['total_locations']} locations tracked")
			
			# Get engine status
			status = await engine.get_engine_status()
			print(f"🚀 Engine status: {status['engine_status']}")
		
		else:
			print("❌ Failed to create geofence")
	
	# Run demo
	asyncio.run(demo_geofencing())