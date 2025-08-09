#!/usr/bin/env python3
"""
Unit Tests for APG Notification Service

Tests for the core notification service functionality including
sending notifications, template management, user preferences, and integrations.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from ..service import NotificationService, create_notification_service
from ..models import *
from .fixtures import *
from .utils import *

class TestNotificationService:
    """Test notification service core functionality"""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, test_config):
        """Test service initialization"""
        service = create_notification_service(
            tenant_id=test_config['test_tenant_id']
        )
        
        assert service.tenant_id == test_config['test_tenant_id']
        assert service.config is not None
        assert hasattr(service, 'templates')
        assert hasattr(service, 'user_profiles')
        assert hasattr(service, 'channel_providers')
    
    @pytest.mark.asyncio 
    async def test_send_notification_success(
        self,
        notification_service,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test successful notification sending"""
        
        # Setup mock data
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Send notification
        result = await notification_service.send_notification(sample_notification_request)
        
        # Assertions
        assert result is not None
        assert 'notification_id' in result
        assert 'delivery_results' in result
        assert len(result['delivery_results']) > 0
        
        # Verify delivery results
        for delivery in result['delivery_results']:
            assert 'channel' in delivery
            assert 'status' in delivery
            assert delivery['status'] in ['delivered', 'pending', 'failed']
    
    @pytest.mark.asyncio
    async def test_send_notification_invalid_template(
        self,
        notification_service,
        sample_user_profile
    ):
        """Test notification sending with invalid template"""
        
        # Setup user profile
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        
        # Create request with invalid template
        request = NotificationRequest(
            template_id='invalid-template',
            user_id=sample_user_profile.user_id,
            channels=[DeliveryChannel.EMAIL],
            tenant_id=TEST_CONFIG['test_tenant_id']
        )
        
        # Should raise exception
        with pytest.raises(ValueError, match="Template not found"):
            await notification_service.send_notification(request)
    
    @pytest.mark.asyncio
    async def test_send_notification_invalid_user(
        self,
        notification_service,
        sample_notification_template
    ):
        """Test notification sending with invalid user"""
        
        # Setup template
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Create request with invalid user
        request = NotificationRequest(
            template_id=sample_notification_template.id,
            user_id='invalid-user',
            channels=[DeliveryChannel.EMAIL],
            tenant_id=TEST_CONFIG['test_tenant_id']
        )
        
        # Should raise exception
        with pytest.raises(ValueError, match="User profile not found"):
            await notification_service.send_notification(request)
    
    @pytest.mark.asyncio
    async def test_bulk_notification_success(
        self,
        notification_service,
        sample_bulk_request,
        sample_notification_template
    ):
        """Test successful bulk notification sending"""
        
        # Setup template
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Setup user profiles for all users in bulk request
        for user_id in sample_bulk_request.user_ids:
            profile = UserProfile(
                user_id=user_id,
                email=f"{user_id}@test.com",
                name=f"User {user_id}",
                tenant_id=TEST_CONFIG['test_tenant_id']
            )
            notification_service.user_profiles[user_id] = profile
        
        # Send bulk notifications
        result = await notification_service.send_bulk_notifications(sample_bulk_request)
        
        # Assertions
        assert result is not None
        assert 'batch_id' in result
        assert 'results' in result
        assert len(result['results']) == len(sample_bulk_request.user_ids)
        
        # Verify individual results
        for user_result in result['results']:
            assert 'user_id' in user_result
            assert 'notification_id' in user_result
            assert user_result['user_id'] in sample_bulk_request.user_ids
    
    @pytest.mark.asyncio
    async def test_template_management(self, notification_service):
        """Test template creation, update, and deletion"""
        
        # Create template
        template_data = {
            'name': 'Test Template',
            'subject_template': 'Hello {{user_name}}',
            'text_template': 'Welcome {{user_name}} to {{company_name}}!'
        }
        
        template_id = await notification_service.create_template(**template_data)
        assert template_id is not None
        assert template_id in notification_service.templates
        
        # Get template
        template = await notification_service.get_template(template_id)
        assert template is not None
        assert template.name == template_data['name']
        
        # Update template
        updates = {'name': 'Updated Template Name'}
        success = await notification_service.update_template(template_id, updates)
        assert success
        
        updated_template = await notification_service.get_template(template_id)
        assert updated_template.name == updates['name']
        
        # Delete template
        success = await notification_service.delete_template(template_id)
        assert success
        assert template_id not in notification_service.templates
    
    @pytest.mark.asyncio
    async def test_user_profile_management(self, notification_service):
        """Test user profile creation, update, and deletion"""
        
        # Create user profile
        profile_data = {
            'user_id': 'test-user-profile',
            'email': 'test@example.com',
            'name': 'Test User',
            'preferences': {'email_enabled': True}
        }
        
        success = await notification_service.create_user_profile(**profile_data)
        assert success
        assert profile_data['user_id'] in notification_service.user_profiles
        
        # Get profile
        profile = await notification_service.get_user_profile(profile_data['user_id'])
        assert profile is not None
        assert profile.email == profile_data['email']
        
        # Update profile
        updates = {'name': 'Updated User Name'}
        success = await notification_service.update_user_profile(profile_data['user_id'], updates)
        assert success
        
        updated_profile = await notification_service.get_user_profile(profile_data['user_id'])
        assert updated_profile.name == updates['name']
        
        # Update preferences
        new_prefs = {'sms_enabled': True, 'push_enabled': False}
        success = await notification_service.update_user_preferences(
            profile_data['user_id'], new_prefs
        )
        assert success
        
        # Delete profile  
        success = await notification_service.delete_user_profile(profile_data['user_id'])
        assert success
        assert profile_data['user_id'] not in notification_service.user_profiles
    
    @pytest.mark.asyncio
    async def test_campaign_management(self, notification_service, sample_campaign):
        """Test campaign creation and management"""
        
        # Create campaign
        campaign_id = await notification_service.create_campaign(sample_campaign)
        assert campaign_id is not None
        assert campaign_id in notification_service.campaigns
        
        # Get campaign
        campaign = await notification_service.get_campaign(campaign_id)
        assert campaign is not None
        assert campaign.name == sample_campaign.name
        
        # Update campaign
        updates = {'status': CampaignStatus.PAUSED}
        success = await notification_service.update_campaign(campaign_id, updates)
        assert success
        
        updated_campaign = await notification_service.get_campaign(campaign_id)
        assert updated_campaign.status == CampaignStatus.PAUSED
        
        # Delete campaign
        success = await notification_service.delete_campaign(campaign_id)
        assert success
        assert campaign_id not in notification_service.campaigns
    
    @pytest.mark.asyncio
    async def test_notification_status_tracking(
        self,
        notification_service,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test notification status tracking"""
        
        # Setup data
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Send notification
        result = await notification_service.send_notification(sample_notification_request)
        notification_id = result['notification_id']
        
        # Get notification status
        status = await notification_service.get_notification_status(notification_id)
        assert status is not None
        assert 'status' in status
        assert 'delivery_results' in status
        
        # Update delivery status (simulate external update)
        await notification_service.update_delivery_status(
            notification_id,
            DeliveryChannel.EMAIL,
            DeliveryStatus.DELIVERED,
            {'external_id': 'test-external-id'}
        )
        
        # Verify status update
        updated_status = await notification_service.get_notification_status(notification_id)
        email_delivery = next(
            (d for d in updated_status['delivery_results'] if d['channel'] == 'email'),
            None
        )
        assert email_delivery is not None
        assert email_delivery['status'] == 'delivered'
    
    @pytest.mark.asyncio
    async def test_personalization_integration(
        self,
        notification_service,
        mock_personalization_engine,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test personalization engine integration"""
        
        # Setup service with personalization
        notification_service.personalization_engine = mock_personalization_engine
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Send notification with personalization
        sample_notification_request.enable_personalization = True
        result = await notification_service.send_notification(sample_notification_request)
        
        # Verify personalization was called
        mock_personalization_engine.personalize_message.assert_called_once()
        
        # Verify result includes personalization data
        assert 'personalization_applied' in result
        assert result['personalization_applied'] is True
    
    @pytest.mark.asyncio
    async def test_analytics_integration(
        self,
        notification_service,
        mock_analytics_engine,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test analytics engine integration"""
        
        # Setup service with analytics
        notification_service.analytics_engine = mock_analytics_engine
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Send notification
        result = await notification_service.send_notification(sample_notification_request)
        
        # Verify analytics tracking was called
        mock_analytics_engine.track_delivery.assert_called()
    
    @pytest.mark.asyncio
    async def test_security_integration(
        self,
        notification_service,
        mock_security_engine,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test security engine integration"""
        
        # Setup service with security
        notification_service.security_engine = mock_security_engine
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Send notification
        result = await notification_service.send_notification(sample_notification_request)
        
        # Verify security validation was called
        mock_security_engine.validate_and_secure_data.assert_called()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, notification_service):
        """Test rate limiting functionality"""
        
        # Enable rate limiting
        notification_service.config.rate_limiting_enabled = True
        notification_service.config.rate_limit_per_minute = 10
        
        # Create test data
        template = UltimateNotificationTemplate(
            id='rate-test-template',
            name='Rate Test',
            subject_template='Test',
            text_template='Test message',
            tenant_id=TEST_CONFIG['test_tenant_id']
        )
        notification_service.templates[template.id] = template
        
        profile = UserProfile(
            user_id='rate-test-user',
            email='rate@test.com',
            tenant_id=TEST_CONFIG['test_tenant_id']
        )
        notification_service.user_profiles[profile.user_id] = profile
        
        # Send notifications up to rate limit
        requests_sent = 0
        for i in range(15):  # Exceed rate limit
            request = NotificationRequest(
                template_id=template.id,
                user_id=profile.user_id,
                channels=[DeliveryChannel.EMAIL],
                tenant_id=TEST_CONFIG['test_tenant_id']
            )
            
            try:
                await notification_service.send_notification(request)
                requests_sent += 1
            except Exception as e:
                if "rate limit" in str(e).lower():
                    break
        
        # Should have hit rate limit before sending all 15
        assert requests_sent <= 10
    
    @pytest.mark.asyncio
    async def test_notification_scheduling(
        self,
        notification_service,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test notification scheduling"""
        
        # Setup data
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Schedule notification for future delivery
        future_time = datetime.utcnow() + timedelta(hours=1)
        sample_notification_request.scheduled_at = future_time
        
        # Send scheduled notification
        result = await notification_service.send_notification(sample_notification_request)
        
        # Should be scheduled, not immediately delivered
        assert result['status'] == 'scheduled'
        assert 'scheduled_at' in result
        
        # Verify it's in the scheduled notifications queue
        scheduled_notifications = await notification_service.get_scheduled_notifications()
        assert len(scheduled_notifications) > 0
        assert any(n['notification_id'] == result['notification_id'] for n in scheduled_notifications)
    
    @pytest.mark.asyncio
    async def test_notification_retry_logic(
        self,
        notification_service,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test notification retry logic for failed deliveries"""
        
        # Setup data
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Configure channel provider to fail initially
        email_provider = notification_service.channel_providers[DeliveryChannel.EMAIL]
        email_provider.send.side_effect = [
            Exception("Temporary failure"),  # First attempt fails
            DeliveryResult(  # Second attempt succeeds
                channel=DeliveryChannel.EMAIL,
                status=DeliveryStatus.DELIVERED,
                external_id='retry-success'
            )
        ]
        
        # Enable retry logic
        notification_service.config.retry_enabled = True
        notification_service.config.max_retries = 2
        
        # Send notification
        result = await notification_service.send_notification(sample_notification_request)
        
        # Should eventually succeed after retry
        assert any(d['status'] == 'delivered' for d in result['delivery_results'])
        
        # Verify retry was attempted
        assert email_provider.send.call_count == 2
    
    @pytest.mark.asyncio
    async def test_service_health_check(self, notification_service):
        """Test service health check"""
        
        health_status = await notification_service.get_health_status()
        
        assert health_status is not None
        assert 'status' in health_status
        assert 'components' in health_status
        assert 'timestamp' in health_status
        
        # Check component statuses
        components = health_status['components']
        assert 'templates' in components
        assert 'user_profiles' in components
        assert 'channel_providers' in components
    
    @pytest.mark.asyncio
    async def test_service_metrics(self, notification_service):
        """Test service metrics collection"""
        
        metrics = await notification_service.get_service_metrics()
        
        assert metrics is not None
        assert 'notifications_sent' in metrics
        assert 'delivery_rates' in metrics
        assert 'channel_performance' in metrics
        assert 'error_rates' in metrics

class TestNotificationServicePerformance:
    """Performance tests for notification service"""
    
    @pytest.mark.asyncio
    async def test_bulk_notification_performance(self, notification_service):
        """Test bulk notification performance"""
        
        # Create template
        template = UltimateNotificationTemplate(
            id='perf-template',
            name='Performance Test',
            subject_template='Test {{user_name}}',
            text_template='Performance test for {{user_name}}',
            tenant_id=TEST_CONFIG['test_tenant_id']
        )
        notification_service.templates[template.id] = template
        
        # Create user profiles
        user_count = 100
        user_ids = []
        for i in range(user_count):
            user_id = f'perf-user-{i:04d}'
            profile = UserProfile(
                user_id=user_id,
                email=f'user{i}@perf.test',
                name=f'Performance User {i}',
                tenant_id=TEST_CONFIG['test_tenant_id']
            )
            notification_service.user_profiles[user_id] = profile
            user_ids.append(user_id)
        
        # Create bulk request
        bulk_request = BulkNotificationRequest(
            template_id=template.id,
            user_ids=user_ids,
            channels=[DeliveryChannel.EMAIL],
            context={'company_name': 'Performance Test Co'},
            tenant_id=TEST_CONFIG['test_tenant_id']
        )
        
        # Measure performance
        with TestTimer() as timer:
            result = await notification_service.send_bulk_notifications(bulk_request)
        
        # Performance assertions
        assert timer.elapsed < 10.0  # Should complete within 10 seconds
        assert len(result['results']) == user_count
        assert result['summary']['success_rate'] > 0.9  # At least 90% success rate
        
        print(f"Bulk notification performance: {user_count} notifications in {timer.elapsed:.2f} seconds")
        print(f"Throughput: {user_count / timer.elapsed:.2f} notifications/second")
    
    @pytest.mark.asyncio
    async def test_concurrent_notification_handling(self, notification_service):
        """Test concurrent notification handling"""
        
        # Setup test data
        template = UltimateNotificationTemplate(
            id='concurrent-template',
            name='Concurrent Test',
            subject_template='Concurrent {{user_name}}',
            text_template='Concurrent test for {{user_name}}',
            tenant_id=TEST_CONFIG['test_tenant_id']
        )
        notification_service.templates[template.id] = template
        
        # Create user profiles and requests
        concurrent_count = 50
        tasks = []
        
        for i in range(concurrent_count):
            user_id = f'concurrent-user-{i:04d}'
            profile = UserProfile(
                user_id=user_id,
                email=f'concurrent{i}@test.com',
                name=f'Concurrent User {i}',
                tenant_id=TEST_CONFIG['test_tenant_id']
            )
            notification_service.user_profiles[user_id] = profile
            
            request = NotificationRequest(
                template_id=template.id,
                user_id=user_id,
                channels=[DeliveryChannel.EMAIL],
                context={'user_name': profile.name},
                tenant_id=TEST_CONFIG['test_tenant_id']
            )
            
            # Create async task
            task = notification_service.send_notification(request)
            tasks.append(task)
        
        # Execute all tasks concurrently
        with TestTimer() as timer:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        # Performance assertions
        assert timer.elapsed < 5.0  # Should handle concurrent requests quickly
        assert len(successful_results) >= concurrent_count * 0.9  # At least 90% success
        
        print(f"Concurrent handling: {len(successful_results)} successful, {len(failed_results)} failed")
        print(f"Total time: {timer.elapsed:.2f} seconds")

class TestNotificationServiceError:
    """Error handling tests for notification service"""
    
    @pytest.mark.asyncio
    async def test_graceful_channel_failure(
        self,
        notification_service,
        sample_notification_request,
        sample_user_profile, 
        sample_notification_template
    ):
        """Test graceful handling of channel provider failures"""
        
        # Setup data
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Configure one channel to fail
        email_provider = notification_service.channel_providers[DeliveryChannel.EMAIL]
        email_provider.send.side_effect = Exception("Channel failure")
        
        # Add multiple channels to request
        sample_notification_request.channels = [DeliveryChannel.EMAIL, DeliveryChannel.SMS]
        
        # Send notification
        result = await notification_service.send_notification(sample_notification_request)
        
        # Should have results for both channels
        assert len(result['delivery_results']) == 2
        
        # Email should have failed, SMS should have succeeded
        email_result = next(d for d in result['delivery_results'] if d['channel'] == 'email')
        sms_result = next(d for d in result['delivery_results'] if d['channel'] == 'sms')
        
        assert email_result['status'] == 'failed'
        assert sms_result['status'] == 'delivered'
        
        # Overall notification should still be considered successful
        assert result['status'] == 'partial_success'
    
    @pytest.mark.asyncio
    async def test_template_rendering_error(
        self,
        notification_service,
        sample_user_profile
    ):
        """Test handling of template rendering errors"""
        
        # Create template with invalid syntax
        template = UltimateNotificationTemplate(
            id='invalid-template',
            name='Invalid Template',
            subject_template='Hello {{user_name}',  # Missing closing brace
            text_template='Welcome {{invalid_variable}}',  # Variable not in context
            tenant_id=TEST_CONFIG['test_tenant_id']
        )
        notification_service.templates[template.id] = template
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        
        # Create request
        request = NotificationRequest(
            template_id=template.id,
            user_id=sample_user_profile.user_id,
            channels=[DeliveryChannel.EMAIL],
            context={'user_name': 'Test User'},  # Missing variables for template
            tenant_id=TEST_CONFIG['test_tenant_id']
        )
        
        # Should handle rendering error gracefully
        with pytest.raises(ValueError, match="Template rendering failed"):
            await notification_service.send_notification(request)
    
    @pytest.mark.asyncio
    async def test_database_connection_error(self, notification_service):
        """Test handling of database connection errors"""
        
        # Simulate database connection failure
        if hasattr(notification_service, 'database'):
            notification_service.database.execute.side_effect = Exception("Database connection failed")
        
        # Operations that require database should handle error gracefully
        with pytest.raises(Exception, match="Database"):
            await notification_service.create_template(
                name='Test Template',
                subject_template='Test',
                text_template='Test message'
            )
    
    @pytest.mark.asyncio
    async def test_cache_failure_fallback(
        self,
        notification_service,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test fallback when cache is unavailable"""
        
        # Setup data
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Simulate cache failure
        if hasattr(notification_service, 'cache'):
            notification_service.cache.get.side_effect = Exception("Cache unavailable")
            notification_service.cache.set.side_effect = Exception("Cache unavailable")
        
        # Should still work without cache
        result = await notification_service.send_notification(sample_notification_request)
        
        # Notification should still be sent successfully
        assert result is not None
        assert 'notification_id' in result
        assert 'delivery_results' in result

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])