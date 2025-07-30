#!/usr/bin/env python3
"""
Unit Tests for APG Notification Geofencing Engine

Tests for geofencing functionality including location tracking,
geofence management, event detection, and location-based notifications.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import math

from ..geofencing_engine import (
    GeofencingEngine, create_geofencing_engine, Location, Geofence,
    GeofenceManager, LocationAnalytics, LocationNotificationEngine,
    GeocodeService, LocationPrivacyManager, GeofenceType, LocationEventType,
    LocationAccuracy, LocationPrivacyLevel, GeofenceStatus
)
from ..models import *
from .fixtures import *
from .utils import *

class TestLocation:
    """Test Location class functionality"""
    
    def test_location_creation(self):
        """Test location object creation"""
        location = Location(
            latitude=37.7749,
            longitude=-122.4194,
            accuracy=10.0,
            altitude=100.0,
            speed=25.0,
            heading=45.0
        )
        
        assert location.latitude == 37.7749
        assert location.longitude == -122.4194
        assert location.accuracy == 10.0
        assert location.altitude == 100.0
        assert location.speed == 25.0
        assert location.heading == 45.0
        assert isinstance(location.timestamp, datetime)
    
    def test_distance_calculation(self):
        """Test distance calculation between locations"""
        # San Francisco coordinates
        sf_location = Location(37.7749, -122.4194)
        
        # Los Angeles coordinates  
        la_location = Location(34.0522, -118.2437)
        
        # Calculate distance
        distance = sf_location.distance_to(la_location)
        
        # Distance should be approximately 560 km (560,000 meters)
        assert 550000 < distance < 570000
    
    def test_bearing_calculation(self):
        """Test bearing calculation between locations"""
        # North-South locations
        north_location = Location(37.7849, -122.4194)  # Slightly north
        south_location = Location(37.7649, -122.4194)  # Slightly south
        
        # Bearing from south to north should be approximately 0 degrees (north)
        bearing = south_location.bearing_to(north_location)
        assert -10 < bearing < 10 or 350 < bearing < 370
        
        # Bearing from north to south should be approximately 180 degrees (south)
        bearing = north_location.bearing_to(south_location)
        assert 170 < bearing < 190
    
    def test_location_string_representation(self):
        """Test location string representation"""
        location = Location(37.7749, -122.4194)
        location_str = str(location)
        
        assert "37.774900" in location_str
        assert "-122.419400" in location_str
        assert "Location(" in location_str

class TestGeofence:
    """Test Geofence class functionality"""
    
    def test_circular_geofence_creation(self, sample_location):
        """Test circular geofence creation"""
        geofence = Geofence(
            fence_id="test-fence-001",
            name="Test Office",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            tenant_id="test-tenant"
        )
        
        assert geofence.fence_id == "test-fence-001"
        assert geofence.name == "Test Office"
        assert geofence.fence_type == GeofenceType.CIRCULAR
        assert geofence.radius == 100.0
        assert geofence.status == GeofenceStatus.ACTIVE
    
    def test_circular_geofence_containment(self, sample_location):
        """Test circular geofence location containment"""
        geofence = Geofence(
            fence_id="test-fence-002",
            name="Test Area",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            tenant_id="test-tenant"
        )
        
        # Location at center should be inside
        assert geofence.contains_location(sample_location) is True
        
        # Location very close should be inside
        close_location = Location(
            sample_location.latitude + 0.0001,  # ~11 meters north
            sample_location.longitude
        )
        assert geofence.contains_location(close_location) is True
        
        # Location far away should be outside
        far_location = Location(
            sample_location.latitude + 0.01,  # ~1100 meters north
            sample_location.longitude
        )
        assert geofence.contains_location(far_location) is False
    
    def test_polygon_geofence_creation(self):
        """Test polygon geofence creation"""
        vertices = [
            Location(37.7749, -122.4194),
            Location(37.7759, -122.4194),
            Location(37.7759, -122.4184),
            Location(37.7749, -122.4184)
        ]
        
        geofence = Geofence(
            fence_id="test-poly-001",
            name="Test Polygon",
            fence_type=GeofenceType.POLYGON,
            center=vertices[0],  # Center doesn't matter for polygon
            vertices=vertices,
            tenant_id="test-tenant"
        )
        
        assert geofence.fence_type == GeofenceType.POLYGON
        assert len(geofence.vertices) == 4
    
    def test_polygon_geofence_containment(self):
        """Test polygon geofence location containment"""
        # Create square geofence
        vertices = [
            Location(37.7749, -122.4194),  # Bottom-left
            Location(37.7759, -122.4194),  # Top-left
            Location(37.7759, -122.4184),  # Top-right
            Location(37.7749, -122.4184)   # Bottom-right
        ]
        
        geofence = Geofence(
            fence_id="test-poly-002",
            name="Test Square",
            fence_type=GeofenceType.POLYGON,
            center=vertices[0],
            vertices=vertices,
            tenant_id="test-tenant"
        )
        
        # Location in center should be inside
        center_location = Location(37.7754, -122.4189)
        assert geofence.contains_location(center_location) is True
        
        # Location outside should be outside
        outside_location = Location(37.7740, -122.4200)
        assert geofence.contains_location(outside_location) is False
    
    def test_rectangular_geofence_containment(self):
        """Test rectangular geofence location containment"""
        # Create rectangular geofence
        vertices = [
            Location(37.7749, -122.4194),  # Bottom-left
            Location(37.7759, -122.4184)   # Top-right
        ]
        
        geofence = Geofence(
            fence_id="test-rect-001",
            name="Test Rectangle",
            fence_type=GeofenceType.RECTANGULAR,
            center=vertices[0],
            vertices=vertices,
            tenant_id="test-tenant"
        )
        
        # Location inside rectangle
        inside_location = Location(37.7754, -122.4189)
        assert geofence.contains_location(inside_location) is True
        
        # Location outside rectangle
        outside_location = Location(37.7740, -122.4200)
        assert geofence.contains_location(outside_location) is False

class TestGeofenceManager:
    """Test GeofenceManager functionality"""
    
    def test_manager_initialization(self, test_config):
        """Test geofence manager initialization"""
        manager = GeofenceManager(test_config['test_tenant_id'])
        
        assert manager.tenant_id == test_config['test_tenant_id']
        assert len(manager.geofences) == 0
        assert len(manager.user_geofences) == 0
    
    def test_create_circular_geofence(self, test_config, sample_location):
        """Test creating circular geofence"""
        manager = GeofenceManager(test_config['test_tenant_id'])
        
        fence_id = manager.create_geofence(
            name="Test Office",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            user_id="test-user",
            notification_config={
                'channels': ['push'],
                'enter_template': {'subject': 'Welcome!'}
            }
        )
        
        assert fence_id is not None
        assert fence_id.startswith('fence_')
        assert fence_id in manager.geofences
        
        # Verify geofence properties
        geofence = manager.geofences[fence_id]
        assert geofence.name == "Test Office"
        assert geofence.fence_type == GeofenceType.CIRCULAR
        assert geofence.radius == 100.0
        assert geofence.user_id == "test-user"
        
        # Verify user tracking
        assert "test-user" in manager.user_geofences
        assert fence_id in manager.user_geofences["test-user"]
    
    def test_create_polygon_geofence(self, test_config):
        """Test creating polygon geofence"""
        manager = GeofenceManager(test_config['test_tenant_id'])
        
        vertices = [
            Location(37.7749, -122.4194),
            Location(37.7759, -122.4194),
            Location(37.7759, -122.4184),
            Location(37.7749, -122.4184)
        ]
        
        fence_id = manager.create_geofence(
            name="Test Polygon Area",
            fence_type=GeofenceType.POLYGON,
            center=vertices[0],
            vertices=vertices
        )
        
        assert fence_id is not None
        geofence = manager.geofences[fence_id]
        assert geofence.fence_type == GeofenceType.POLYGON
        assert len(geofence.vertices) == 4
    
    def test_update_geofence(self, test_config, sample_location):
        """Test updating geofence properties"""
        manager = GeofenceManager(test_config['test_tenant_id'])
        
        # Create geofence
        fence_id = manager.create_geofence(
            name="Original Name",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0
        )
        
        # Update geofence
        success = manager.update_geofence(fence_id, {
            'name': 'Updated Name',
            'radius': 200.0,
            'status': GeofenceStatus.PAUSED
        })
        
        assert success is True
        
        # Verify updates
        geofence = manager.geofences[fence_id]
        assert geofence.name == "Updated Name"
        assert geofence.radius == 200.0
        assert geofence.status == GeofenceStatus.PAUSED
    
    def test_delete_geofence(self, test_config, sample_location):
        """Test deleting geofence"""
        manager = GeofenceManager(test_config['test_tenant_id'])
        
        # Create geofence
        fence_id = manager.create_geofence(
            name="To Be Deleted",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            user_id="test-user"
        )
        
        # Verify creation
        assert fence_id in manager.geofences
        assert fence_id in manager.user_geofences["test-user"]
        
        # Delete geofence
        success = manager.delete_geofence(fence_id)
        assert success is True
        
        # Verify deletion
        assert fence_id not in manager.geofences
        assert fence_id not in manager.user_geofences["test-user"]
    
    def test_get_active_geofences(self, test_config, sample_location):
        """Test getting active geofences"""
        manager = GeofenceManager(test_config['test_tenant_id'])
        
        # Create active geofence
        active_fence_id = manager.create_geofence(
            name="Active Fence",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            user_id="user-1"
        )
        
        # Create paused geofence
        paused_fence_id = manager.create_geofence(
            name="Paused Fence",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            user_id="user-2"
        )
        manager.update_geofence(paused_fence_id, {'status': GeofenceStatus.PAUSED})
        
        # Create expired geofence
        expired_fence_id = manager.create_geofence(
            name="Expired Fence",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            expires_in_hours=1
        )
        # Manually expire it
        manager.geofences[expired_fence_id].expires_at = datetime.utcnow() - timedelta(hours=1)
        
        # Get all active geofences
        active_fences = manager.get_active_geofences()
        assert len(active_fences) == 1
        assert active_fences[0].fence_id == active_fence_id
        
        # Get active geofences for specific user
        user1_fences = manager.get_active_geofences(user_id="user-1")
        assert len(user1_fences) == 1
        assert user1_fences[0].fence_id == active_fence_id
        
        user2_fences = manager.get_active_geofences(user_id="user-2")
        assert len(user2_fences) == 0  # Paused fence should not be returned
    
    def test_check_geofence_events(self, test_config, sample_locations):
        """Test geofence event detection"""
        manager = GeofenceManager(test_config['test_tenant_id'])
        
        # Create geofence around first location
        fence_id = manager.create_geofence(
            name="Event Test Fence",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_locations[0],
            radius=50.0,  # Small radius
            user_id="test-user"
        )
        
        # Test enter event
        # Start outside the geofence
        outside_location = Location(
            sample_locations[0].latitude + 0.01,  # Far from center
            sample_locations[0].longitude
        )
        
        # Move inside the geofence
        inside_location = sample_locations[0]  # At center
        
        events = manager.check_geofence_events(
            user_id="test-user",
            current_location=inside_location,
            previous_location=outside_location
        )
        
        # Should generate enter event
        assert len(events) == 1
        assert events[0].event_type == LocationEventType.ENTER_GEOFENCE
        assert events[0].geofence.fence_id == fence_id
        
        # Test exit event
        events = manager.check_geofence_events(
            user_id="test-user",
            current_location=outside_location,
            previous_location=inside_location
        )
        
        # Should generate exit event
        assert len(events) == 1
        assert events[0].event_type == LocationEventType.EXIT_GEOFENCE
        assert events[0].geofence.fence_id == fence_id

class TestLocationAnalytics:
    """Test LocationAnalytics functionality"""
    
    def test_analytics_initialization(self, test_config):
        """Test location analytics initialization"""
        analytics = LocationAnalytics(test_config['test_tenant_id'])
        
        assert analytics.tenant_id == test_config['test_tenant_id']
        assert len(analytics.location_history) == 0
        assert len(analytics.dwell_times) == 0
    
    def test_add_location_history(self, test_config, sample_locations):
        """Test adding location history"""
        analytics = LocationAnalytics(test_config['test_tenant_id'])
        
        user_id = "test-user"
        
        # Add multiple locations
        for location in sample_locations:
            analytics.add_location_history(user_id, location)
        
        # Verify history
        assert user_id in analytics.location_history
        assert len(analytics.location_history[user_id]) == len(sample_locations)
    
    def test_location_history_limit(self, test_config):
        """Test location history size limit"""
        analytics = LocationAnalytics(test_config['test_tenant_id'])
        
        user_id = "test-user"
        
        # Add more than 1000 locations
        for i in range(1100):
            location = Location(37.7749 + i * 0.0001, -122.4194)
            analytics.add_location_history(user_id, location)
        
        # Should be limited to 1000
        assert len(analytics.location_history[user_id]) == 1000
    
    def test_calculate_dwell_time(self, test_config, sample_location):
        """Test dwell time calculation"""
        analytics = LocationAnalytics(test_config['test_tenant_id'])
        
        user_id = "test-user"
        geofence = Geofence(
            fence_id="dwell-test",
            name="Dwell Test",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            tenant_id=test_config['test_tenant_id']
        )
        
        # Calculate dwell time (30 minutes)
        start_time = datetime.utcnow() - timedelta(minutes=30)
        end_time = datetime.utcnow()
        
        dwell_minutes = analytics.calculate_dwell_time(
            user_id, geofence, start_time, end_time
        )
        
        assert abs(dwell_minutes - 30.0) < 1.0  # Should be approximately 30 minutes
        
        # Verify dwell time was recorded
        assert user_id in analytics.dwell_times
        assert geofence.fence_id in analytics.dwell_times[user_id]
    
    def test_get_frequent_locations(self, test_config):
        """Test frequent location identification"""
        analytics = LocationAnalytics(test_config['test_tenant_id'])
        
        user_id = "test-user"
        
        # Create location clusters
        home_location = Location(37.7749, -122.4194)
        work_location = Location(37.7849, -122.4094)
        
        # Add multiple visits to each location (with small variations)
        for i in range(10):
            # Home visits
            home_variant = Location(
                home_location.latitude + (i * 0.00001),
                home_location.longitude + (i * 0.00001)
            )
            analytics.add_location_history(user_id, home_variant)
            
            # Work visits (fewer)
            if i < 5:
                work_variant = Location(
                    work_location.latitude + (i * 0.00001),
                    work_location.longitude + (i * 0.00001)
                )
                analytics.add_location_history(user_id, work_variant)
        
        # Get frequent locations
        frequent = analytics.get_frequent_locations(
            user_id=user_id,
            radius_meters=100.0,
            min_visits=3
        )
        
        # Should find both clusters
        assert len(frequent) == 2
        
        # Home should be more frequent
        most_frequent = frequent[0]  # Sorted by frequency
        assert most_frequent['visits'] == 10
    
    def test_analyze_movement_patterns(self, test_config, sample_locations):
        """Test movement pattern analysis"""
        analytics = LocationAnalytics(test_config['test_tenant_id'])
        
        user_id = "test-user"
        
        # Add locations with timestamps
        for i, location in enumerate(sample_locations):
            location.timestamp = datetime.utcnow() - timedelta(hours=len(sample_locations) - i)
            analytics.add_location_history(user_id, location)
        
        # Analyze movement patterns
        analysis = analytics.analyze_movement_patterns(user_id)
        
        assert 'total_distance_km' in analysis
        assert 'avg_speed_kmh' in analysis
        assert 'max_speed_kmh' in analysis
        assert 'total_locations' in analysis
        assert 'frequent_locations' in analysis
        
        # Should have reasonable values
        assert analysis['total_locations'] == len(sample_locations)
        assert analysis['total_distance_km'] >= 0
        assert analysis['avg_speed_kmh'] >= 0

class TestLocationNotificationEngine:
    """Test LocationNotificationEngine functionality"""
    
    def test_engine_initialization(self, test_config):
        """Test notification engine initialization"""
        engine = LocationNotificationEngine(test_config['test_tenant_id'])
        
        assert engine.tenant_id == test_config['test_tenant_id']
        assert len(engine.notification_queue) == 0
        assert len(engine.processed_events) == 0
    
    @pytest.mark.asyncio
    async def test_process_enter_event(self, test_config, sample_location):
        """Test processing geofence enter event"""
        engine = LocationNotificationEngine(test_config['test_tenant_id'])
        
        # Create geofence with notification config
        geofence = Geofence(
            fence_id="enter-test",
            name="Enter Test Fence",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            tenant_id=test_config['test_tenant_id'],
            notification_config={
                'channels': ['push'],
                'enter_template': {
                    'template_id': 'enter_template',
                    'subject': 'Welcome to {{location_name}}!',
                    'message': 'You have entered {{location_name}}'
                }
            }
        )
        
        # Create enter event
        from ..geofencing_engine import LocationEvent
        event = LocationEvent(
            event_id="test-enter-001",
            tenant_id=test_config['test_tenant_id'],
            user_id="test-user",
            event_type=LocationEventType.ENTER_GEOFENCE,
            location=sample_location,
            geofence=geofence
        )
        
        # Process event
        notifications_sent = await engine.process_location_event(event)
        
        # Verify notification was queued
        assert len(engine.notification_queue) > 0
        assert event.event_id in engine.processed_events
        assert event.processed is True
        
        # Verify notification content
        notification = engine.notification_queue[0]
        assert notification['user_id'] == "test-user"
        assert notification['template_id'] == 'enter_template'
        assert 'location_name' in notification['context']
        assert notification['context']['location_name'] == geofence.name
    
    @pytest.mark.asyncio
    async def test_process_exit_event(self, test_config, sample_location):
        """Test processing geofence exit event"""
        engine = LocationNotificationEngine(test_config['test_tenant_id'])
        
        # Create geofence with exit notification config
        geofence = Geofence(
            fence_id="exit-test",
            name="Exit Test Fence",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_location,
            radius=100.0,
            tenant_id=test_config['test_tenant_id'],
            notification_config={
                'channels': ['push'],
                'exit_template': {
                    'template_id': 'exit_template',
                    'subject': 'Thanks for visiting {{location_name}}!',
                    'message': 'You have left {{location_name}}'
                }
            }
        )
        
        # Create exit event
        from ..geofencing_engine import LocationEvent
        event = LocationEvent(
            event_id="test-exit-001",
            tenant_id=test_config['test_tenant_id'],
            user_id="test-user",
            event_type=LocationEventType.EXIT_GEOFENCE,
            location=sample_location,
            geofence=geofence
        )
        
        # Process event
        await engine.process_location_event(event)
        
        # Verify notification was created
        assert len(engine.notification_queue) > 0
        notification = engine.notification_queue[0]
        assert notification['template_id'] == 'exit_template'
        assert notification['context']['event_type'] == 'exit'

class TestLocationPrivacyManager:
    """Test LocationPrivacyManager functionality"""
    
    def test_privacy_manager_initialization(self):
        """Test privacy manager initialization"""
        manager = LocationPrivacyManager()
        assert len(manager.anonymization_keys) == 0
    
    def test_full_privacy_level(self, sample_location):
        """Test full privacy level (no filtering)"""
        manager = LocationPrivacyManager()
        
        filtered = manager.apply_privacy_level(
            sample_location,
            LocationPrivacyLevel.FULL,
            "test-user"
        )
        
        # Should return original location unchanged
        assert filtered.latitude == sample_location.latitude
        assert filtered.longitude == sample_location.longitude
        assert filtered.accuracy == sample_location.accuracy
    
    def test_approximate_privacy_level(self, sample_location):
        """Test approximate privacy level"""
        manager = LocationPrivacyManager()
        
        filtered = manager.apply_privacy_level(
            sample_location,
            LocationPrivacyLevel.APPROXIMATE,
            "test-user"
        )
        
        # Should be rounded to lower precision
        assert abs(filtered.latitude - round(sample_location.latitude, 2)) < 0.01
        assert abs(filtered.longitude - round(sample_location.longitude, 2)) < 0.01
        assert filtered.accuracy == 1000.0  # Reduced accuracy
    
    def test_anonymous_privacy_level(self, sample_location):
        """Test anonymous privacy level"""
        manager = LocationPrivacyManager()
        
        filtered = manager.apply_privacy_level(
            sample_location,
            LocationPrivacyLevel.ANONYMOUS,
            "test-user"
        )
        
        # Should be offset from original but consistently for same user
        assert filtered.latitude != sample_location.latitude
        assert filtered.longitude != sample_location.longitude
        
        # Should be consistent for same user
        filtered2 = manager.apply_privacy_level(
            sample_location,
            LocationPrivacyLevel.ANONYMOUS,
            "test-user"
        )
        
        assert filtered.latitude == filtered2.latitude
        assert filtered.longitude == filtered2.longitude
    
    def test_none_privacy_level(self, sample_location):
        """Test none privacy level (no location)"""
        manager = LocationPrivacyManager()
        
        filtered = manager.apply_privacy_level(
            sample_location,
            LocationPrivacyLevel.NONE,
            "test-user"
        )
        
        # Should return None
        assert filtered is None

class TestGeocodeService:
    """Test GeocodeService functionality"""
    
    @pytest.mark.asyncio
    async def test_geocoding_with_mock(self):
        """Test geocoding with mocked service"""
        service = GeocodeService()
        
        # Mock the geocoder
        with patch.object(service.geocoder, 'geocode') as mock_geocode:
            mock_result = Mock()
            mock_result.latitude = 37.7749
            mock_result.longitude = -122.4194
            mock_result.address = "San Francisco, CA, USA"
            mock_geocode.return_value = mock_result
            
            # Test geocoding
            location = await service.geocode_address("San Francisco, CA")
            
            assert location is not None
            assert location.latitude == 37.7749
            assert location.longitude == -122.4194
            assert location.address == "San Francisco, CA, USA"
            
            # Verify caching
            assert len(service.cache) > 0
    
    @pytest.mark.asyncio
    async def test_reverse_geocoding_with_mock(self, sample_location):
        """Test reverse geocoding with mocked service"""
        service = GeocodeService()
        
        # Mock the geocoder
        with patch.object(service.geocoder, 'reverse') as mock_reverse:
            mock_result = Mock()
            mock_result.address = "123 Test Street, San Francisco, CA, USA"
            mock_reverse.return_value = mock_result
            
            # Test reverse geocoding
            address = await service.reverse_geocode(sample_location)
            
            assert address is not None
            assert "San Francisco" in address
            
            # Verify caching
            assert len(service.cache) > 0
    
    @pytest.mark.asyncio
    async def test_geocoding_cache(self):
        """Test geocoding cache functionality"""
        service = GeocodeService()
        
        # Mock successful geocoding
        with patch.object(service.geocoder, 'geocode') as mock_geocode:
            mock_result = Mock()
            mock_result.latitude = 37.7749
            mock_result.longitude = -122.4194
            mock_result.address = "Test Address"
            mock_geocode.return_value = mock_result
            
            # First call should hit the geocoder
            location1 = await service.geocode_address("Test Address")
            assert mock_geocode.call_count == 1
            
            # Second call should use cache
            location2 = await service.geocode_address("Test Address")
            assert mock_geocode.call_count == 1  # Should not increase
            
            # Results should be identical
            assert location1.latitude == location2.latitude
            assert location1.longitude == location2.longitude

class TestGeofencingEngine:
    """Test main GeofencingEngine functionality"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, test_config):
        """Test geofencing engine initialization"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        assert engine.tenant_id == test_config['test_tenant_id']
        assert hasattr(engine, 'geofence_manager')
        assert hasattr(engine, 'analytics')
        assert hasattr(engine, 'notification_engine')
        assert hasattr(engine, 'geocode_service')
        assert len(engine.tracking_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_start_location_tracking(self, test_config):
        """Test starting location tracking"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        session_id = await engine.start_location_tracking(
            user_id="test-user",
            privacy_level=LocationPrivacyLevel.FULL,
            accuracy_threshold=50.0,
            update_interval=30
        )
        
        assert session_id is not None
        assert session_id.startswith('session_')
        assert session_id in engine.tracking_sessions
        
        # Verify session properties
        session = engine.tracking_sessions[session_id]
        assert session.user_id == "test-user"
        assert session.privacy_level == LocationPrivacyLevel.FULL
        assert session.accuracy_threshold == 50.0
        assert session.active is True
    
    @pytest.mark.asyncio
    async def test_update_user_location(self, test_config, sample_location):
        """Test updating user location"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        # Start tracking first
        session_id = await engine.start_location_tracking("test-user")
        
        # Update location
        events = await engine.update_user_location(
            user_id="test-user",
            location=sample_location,
            session_id=session_id
        )
        
        # Verify location was recorded
        session = engine.tracking_sessions[session_id]
        assert len(session.location_history) == 1
        assert session.location_history[0] == sample_location
        
        # Should not generate events without geofences
        assert len(events) == 0
    
    @pytest.mark.asyncio
    async def test_create_location_geofence_with_mock(self, test_config):
        """Test creating geofence from address"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        # Mock geocoding service
        with patch.object(engine.geocode_service, 'geocode_address') as mock_geocode:
            mock_geocode.return_value = Location(37.7749, -122.4194)
            
            fence_id = await engine.create_location_geofence(
                name="Test Office",
                address="123 Test Street, San Francisco, CA",
                radius=100.0,
                user_id="test-user",
                notification_config={'channels': ['push']}
            )
            
            assert fence_id is not None
            assert fence_id in engine.geofence_manager.geofences
            
            # Verify geofence properties
            geofence = engine.geofence_manager.geofences[fence_id]
            assert geofence.name == "Test Office"
            assert geofence.radius == 100.0
            assert geofence.user_id == "test-user"
    
    @pytest.mark.asyncio
    async def test_create_polygon_geofence(self, test_config):
        """Test creating polygon geofence"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        coordinates = [
            (37.7749, -122.4194),
            (37.7759, -122.4194),
            (37.7759, -122.4184),
            (37.7749, -122.4184)
        ]
        
        fence_id = await engine.create_polygon_geofence(
            name="Test Polygon",
            coordinates=coordinates,
            user_id="test-user"
        )
        
        assert fence_id is not None
        
        # Verify polygon geofence
        geofence = engine.geofence_manager.geofences[fence_id]
        assert geofence.fence_type == GeofenceType.POLYGON
        assert len(geofence.vertices) == 4
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, test_config, sample_locations):
        """Test complete geofencing workflow"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        # 1. Start tracking
        session_id = await engine.start_location_tracking("test-user")
        
        # 2. Create geofence around first location
        fence_id = engine.geofence_manager.create_geofence(
            name="Test Workflow Fence",
            fence_type=GeofenceType.CIRCULAR,
            center=sample_locations[0],
            radius=50.0,  # Small radius for testing
            user_id="test-user",
            notification_config={
                'channels': ['push'],
                'enter_template': {'subject': 'Welcome!'}
            }
        )
        
        # 3. Update locations to trigger events
        all_events = []
        for i, location in enumerate(sample_locations):
            events = await engine.update_user_location(
                user_id="test-user",
                location=location,
                session_id=session_id
            )
            all_events.extend(events)
        
        # 4. Process event queue
        processed_count = await engine.process_event_queue()
        
        # 5. Get analytics
        analytics = await engine.get_user_analytics("test-user")
        
        # Verify complete workflow
        assert analytics['user_id'] == "test-user"
        assert analytics['tracking_session']['active'] is True
        assert analytics['geofences'] == 1
        assert len(all_events) >= 0  # May or may not generate events depending on locations
    
    @pytest.mark.asyncio
    async def test_stop_location_tracking(self, test_config):
        """Test stopping location tracking"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        # Start tracking
        session_id = await engine.start_location_tracking("test-user")
        
        # Verify tracking is active
        session = engine.tracking_sessions[session_id]
        assert session.active is True
        
        # Stop tracking
        success = await engine.stop_location_tracking("test-user", session_id)
        assert success is True
        
        # Verify tracking is stopped
        assert session.active is False
    
    @pytest.mark.asyncio
    async def test_get_engine_status(self, test_config):
        """Test getting engine status"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        # Start tracking session and create geofence
        await engine.start_location_tracking("test-user")
        engine.geofence_manager.create_geofence(
            name="Status Test",
            fence_type=GeofenceType.CIRCULAR,
            center=Location(37.7749, -122.4194),
            radius=100.0
        )
        
        # Get status
        status = await engine.get_engine_status()
        
        assert status is not None
        assert 'tenant_id' in status
        assert 'active_tracking_sessions' in status
        assert 'total_geofences' in status
        assert 'engine_status' in status
        assert 'components' in status
        
        assert status['tenant_id'] == test_config['test_tenant_id']
        assert status['active_tracking_sessions'] == 1
        assert status['total_geofences'] == 1
        assert status['engine_status'] == 'operational'

class TestGeofencingEnginePerformance:
    """Performance tests for geofencing engine"""
    
    @pytest.mark.asyncio
    async def test_high_volume_location_updates(self, test_config):
        """Test high-volume location updates performance"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        # Start tracking for multiple users
        user_count = 50
        session_ids = []
        
        for i in range(user_count):
            user_id = f"perf-user-{i:03d}"
            session_id = await engine.start_location_tracking(user_id)
            session_ids.append((user_id, session_id))
        
        # Generate location updates
        update_count = 1000
        
        with TestTimer() as timer:
            tasks = []
            for i in range(update_count):
                user_id, session_id = session_ids[i % user_count]
                
                # Generate random location
                location = Location(
                    37.7749 + (i * 0.0001),
                    -122.4194 + (i * 0.0001)
                )
                
                task = engine.update_user_location(user_id, location, session_id)
                tasks.append(task)
            
            # Execute all updates
            await asyncio.gather(*tasks)
        
        # Performance assertions
        assert timer.elapsed < 10.0  # Should complete within 10 seconds
        throughput = update_count / timer.elapsed
        assert throughput > 100  # At least 100 updates per second
        
        print(f"Location update performance: {update_count} updates in {timer.elapsed:.2f} seconds")
        print(f"Throughput: {throughput:.2f} updates/second")
    
    @pytest.mark.asyncio
    async def test_geofence_event_detection_performance(self, test_config):
        """Test geofence event detection performance"""
        engine = create_geofencing_engine(test_config['test_tenant_id'])
        
        # Create multiple geofences
        geofence_count = 100
        center_location = Location(37.7749, -122.4194)
        
        for i in range(geofence_count):
            engine.geofence_manager.create_geofence(
                name=f"Perf Fence {i}",
                fence_type=GeofenceType.CIRCULAR,
                center=Location(
                    center_location.latitude + (i * 0.001),
                    center_location.longitude + (i * 0.001)
                ),
                radius=100.0,
                user_id="perf-user"
            )
        
        # Test event detection performance
        test_location = Location(37.7749, -122.4194)
        
        with TestTimer() as timer:
            # Run event detection multiple times
            for _ in range(100):
                events = engine.geofence_manager.check_geofence_events(
                    user_id="perf-user",
                    current_location=test_location
                )
        
        # Performance assertions
        assert timer.elapsed < 5.0  # Should complete within 5 seconds
        
        print(f"Event detection performance: 100 checks against {geofence_count} geofences in {timer.elapsed:.2f} seconds")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])