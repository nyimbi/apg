#!/usr/bin/env python3
"""
Unit Tests for APG Notification Analytics Engine

Tests for analytics functionality including delivery tracking,
engagement metrics, reporting, and performance insights.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from ..analytics_engine import AnalyticsEngine, create_analytics_engine
from ..models import *
from .fixtures import *
from .utils import *

class TestAnalyticsEngine:
    """Test analytics engine core functionality"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, test_config):
        """Test analytics engine initialization"""
        engine = create_analytics_engine(
            tenant_id=test_config['test_tenant_id']
        )
        
        assert engine.tenant_id == test_config['test_tenant_id']
        assert hasattr(engine, 'delivery_metrics')
        assert hasattr(engine, 'engagement_metrics')
        assert hasattr(engine, 'channel_performance')
    
    @pytest.mark.asyncio
    async def test_track_delivery_event(self, test_config):
        """Test delivery event tracking"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Track delivery event
        await engine.track_delivery(
            notification_id='test-notification-001',
            user_id='test-user-001',
            channel=DeliveryChannel.EMAIL,
            status=DeliveryStatus.DELIVERED,
            template_id='test-template-001',
            metadata={
                'external_id': 'email-123',
                'delivery_time_ms': 250
            }
        )
        
        # Verify metrics were updated
        assert engine.delivery_metrics.total_sent > 0
        assert engine.delivery_metrics.total_delivered > 0
        assert DeliveryChannel.EMAIL in engine.channel_performance
        
        # Verify event was stored
        events = await engine.get_delivery_events(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow()
        )
        assert len(events) > 0
        assert events[0]['notification_id'] == 'test-notification-001'
    
    @pytest.mark.asyncio
    async def test_track_engagement_event(self, test_config):
        """Test engagement event tracking"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Track engagement event
        await engine.track_engagement(
            notification_id='test-notification-002',
            user_id='test-user-002',
            event_type='open',
            channel=DeliveryChannel.EMAIL,
            timestamp=datetime.utcnow(),
            metadata={
                'user_agent': 'Test Browser',
                'ip_address': '192.168.1.1'
            }
        )
        
        # Verify engagement metrics were updated
        assert engine.engagement_metrics.total_opens > 0
        
        # Verify event was stored
        events = await engine.get_engagement_events(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow()
        )
        assert len(events) > 0
        assert events[0]['event_type'] == 'open'
    
    @pytest.mark.asyncio
    async def test_delivery_rate_calculation(self, test_config):
        """Test delivery rate calculation"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Track multiple delivery events with different statuses
        notifications = [
            ('notif-001', DeliveryStatus.DELIVERED),
            ('notif-002', DeliveryStatus.DELIVERED),
            ('notif-003', DeliveryStatus.FAILED),
            ('notif-004', DeliveryStatus.DELIVERED),
            ('notif-005', DeliveryStatus.BOUNCED)
        ]
        
        for notif_id, status in notifications:
            await engine.track_delivery(
                notification_id=notif_id,
                user_id='test-user',
                channel=DeliveryChannel.EMAIL,
                status=status,
                template_id='test-template'
            )
        
        # Calculate delivery rate
        delivery_rate = await engine.calculate_delivery_rate(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow(),
            channel=DeliveryChannel.EMAIL
        )
        
        # Should be 3 delivered out of 5 total = 0.6
        assert abs(delivery_rate - 0.6) < 0.01
    
    @pytest.mark.asyncio
    async def test_engagement_rate_calculation(self, test_config):
        """Test engagement rate calculation"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Track deliveries first
        for i in range(10):
            await engine.track_delivery(
                notification_id=f'notif-{i:03d}',
                user_id=f'user-{i:03d}',
                channel=DeliveryChannel.EMAIL,
                status=DeliveryStatus.DELIVERED,
                template_id='test-template'
            )
        
        # Track engagement events for some notifications
        engagement_events = [
            ('notif-001', 'open'),
            ('notif-002', 'open'),
            ('notif-003', 'open'),
            ('notif-001', 'click'),  # Same notification can have multiple events
            ('notif-004', 'click')
        ]
        
        for notif_id, event_type in engagement_events:
            await engine.track_engagement(
                notification_id=notif_id,
                user_id='test-user',
                event_type=event_type,
                channel=DeliveryChannel.EMAIL
            )
        
        # Calculate open rate
        open_rate = await engine.calculate_open_rate(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow(),
            channel=DeliveryChannel.EMAIL
        )
        
        # 3 unique opens out of 10 delivered = 0.3
        assert abs(open_rate - 0.3) < 0.01
        
        # Calculate click rate
        click_rate = await engine.calculate_click_rate(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow(),
            channel=DeliveryChannel.EMAIL
        )
        
        # 2 unique clicks out of 10 delivered = 0.2
        assert abs(click_rate - 0.2) < 0.01
    
    @pytest.mark.asyncio
    async def test_channel_performance_analysis(self, test_config):
        """Test channel performance analysis"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Track events for different channels
        channels_data = [
            (DeliveryChannel.EMAIL, 100, 95, 60, 25),    # sent, delivered, opened, clicked
            (DeliveryChannel.SMS, 50, 48, 30, 8),
            (DeliveryChannel.PUSH, 75, 70, 45, 15)
        ]
        
        for channel, sent, delivered, opened, clicked in channels_data:
            # Track deliveries
            for i in range(sent):
                status = DeliveryStatus.DELIVERED if i < delivered else DeliveryStatus.FAILED
                await engine.track_delivery(
                    notification_id=f'{channel.value}-{i:03d}',
                    user_id=f'user-{i:03d}',
                    channel=channel,
                    status=status,
                    template_id='test-template'
                )
            
            # Track engagement
            for i in range(opened):
                await engine.track_engagement(
                    notification_id=f'{channel.value}-{i:03d}',
                    user_id=f'user-{i:03d}',
                    event_type='open',
                    channel=channel
                )
            
            for i in range(clicked):
                await engine.track_engagement(
                    notification_id=f'{channel.value}-{i:03d}',
                    user_id=f'user-{i:03d}',
                    event_type='click',
                    channel=channel
                )
        
        # Get channel performance comparison
        performance = await engine.get_channel_performance(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow()
        )
        
        # Verify performance data
        assert len(performance) == 3
        assert DeliveryChannel.EMAIL.value in performance
        assert DeliveryChannel.SMS.value in performance
        assert DeliveryChannel.PUSH.value in performance
        
        # Check email channel metrics
        email_perf = performance[DeliveryChannel.EMAIL.value]
        assert email_perf['total_sent'] == 100
        assert email_perf['total_delivered'] == 95
        assert abs(email_perf['delivery_rate'] - 0.95) < 0.01
        assert abs(email_perf['open_rate'] - 0.632) < 0.01  # 60/95 delivered
    
    @pytest.mark.asyncio
    async def test_user_engagement_analysis(self, test_config):
        """Test individual user engagement analysis"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        user_id = 'engagement-test-user'
        
        # Track multiple notifications for the user
        notifications = [
            ('notif-a', True, True, False),   # delivered, opened, clicked
            ('notif-b', True, True, True),
            ('notif-c', True, False, False),
            ('notif-d', False, False, False),  # not delivered
            ('notif-e', True, True, True)
        ]
        
        for notif_id, delivered, opened, clicked in notifications:
            # Track delivery
            status = DeliveryStatus.DELIVERED if delivered else DeliveryStatus.FAILED
            await engine.track_delivery(
                notification_id=notif_id,
                user_id=user_id,
                channel=DeliveryChannel.EMAIL,
                status=status,
                template_id='test-template'
            )
            
            # Track engagement
            if opened:
                await engine.track_engagement(
                    notification_id=notif_id,
                    user_id=user_id,
                    event_type='open',
                    channel=DeliveryChannel.EMAIL
                )
            
            if clicked:
                await engine.track_engagement(
                    notification_id=notif_id,
                    user_id=user_id,
                    event_type='click',
                    channel=DeliveryChannel.EMAIL
                )
        
        # Get user engagement analysis
        user_engagement = await engine.get_user_engagement_analysis(
            user_id=user_id,
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow()
        )
        
        # Verify analysis
        assert user_engagement['user_id'] == user_id
        assert user_engagement['total_notifications_sent'] == 5
        assert user_engagement['total_delivered'] == 4
        assert user_engagement['total_opened'] == 4
        assert user_engagement['total_clicked'] == 2
        assert abs(user_engagement['open_rate'] - 1.0) < 0.01  # 4/4 delivered
        assert abs(user_engagement['click_rate'] - 0.5) < 0.01  # 2/4 delivered
    
    @pytest.mark.asyncio
    async def test_template_performance_analysis(self, test_config):
        """Test template performance analysis"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Track events for different templates
        templates_data = [
            ('template-welcome', 200, 190, 120, 40),
            ('template-promo', 150, 140, 70, 25),
            ('template-reminder', 100, 95, 30, 5)
        ]
        
        for template_id, sent, delivered, opened, clicked in templates_data:
            # Track deliveries
            for i in range(sent):
                status = DeliveryStatus.DELIVERED if i < delivered else DeliveryStatus.FAILED
                await engine.track_delivery(
                    notification_id=f'{template_id}-{i:03d}',
                    user_id=f'user-{i:03d}',
                    channel=DeliveryChannel.EMAIL,
                    status=status,
                    template_id=template_id
                )
            
            # Track engagement
            for i in range(opened):
                await engine.track_engagement(
                    notification_id=f'{template_id}-{i:03d}',
                    user_id=f'user-{i:03d}',
                    event_type='open',
                    channel=DeliveryChannel.EMAIL
                )
            
            for i in range(clicked):
                await engine.track_engagement(
                    notification_id=f'{template_id}-{i:03d}',
                    user_id=f'user-{i:03d}',
                    event_type='click',
                    channel=DeliveryChannel.EMAIL
                )
        
        # Get template performance analysis
        template_performance = await engine.get_template_performance(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow()
        )
        
        # Verify analysis
        assert len(template_performance) == 3
        
        # Check welcome template (should have best performance)
        welcome_perf = next(t for t in template_performance if t['template_id'] == 'template-welcome')
        assert welcome_perf['total_sent'] == 200
        assert welcome_perf['total_delivered'] == 190
        assert abs(welcome_perf['open_rate'] - 0.632) < 0.01  # 120/190
    
    @pytest.mark.asyncio
    async def test_time_series_analysis(self, test_config):
        """Test time-series analytics"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Generate time-series data over 24 hours
        base_time = datetime.utcnow() - timedelta(hours=24)
        
        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)
            
            # Simulate varying activity throughout the day
            activity_multiplier = 1.0 + 0.5 * (hour % 12) / 12  # Peak mid-day
            notifications_count = int(10 * activity_multiplier)
            
            for i in range(notifications_count):
                await engine.track_delivery(
                    notification_id=f'ts-{hour:02d}-{i:03d}',
                    user_id=f'user-{i:03d}',
                    channel=DeliveryChannel.EMAIL,
                    status=DeliveryStatus.DELIVERED,
                    template_id='test-template',
                    timestamp=timestamp
                )
                
                # Some engagement
                if i % 3 == 0:  # 1/3 open rate
                    await engine.track_engagement(
                        notification_id=f'ts-{hour:02d}-{i:03d}',
                        user_id=f'user-{i:03d}',
                        event_type='open',
                        channel=DeliveryChannel.EMAIL,
                        timestamp=timestamp + timedelta(minutes=5)
                    )
        
        # Get time-series data
        time_series = await engine.get_time_series_data(
            start_date=base_time,
            end_date=datetime.utcnow(),
            interval='hour'
        )
        
        # Verify time series
        assert len(time_series) == 24
        assert all('timestamp' in point for point in time_series)
        assert all('total_sent' in point for point in time_series)
        assert all('total_delivered' in point for point in time_series)
        assert all('total_opened' in point for point in time_series)
    
    @pytest.mark.asyncio
    async def test_cohort_analysis(self, test_config):
        """Test cohort analysis functionality"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Create user cohorts based on registration date
        cohorts = {
            'cohort_week1': [f'w1_user_{i:03d}' for i in range(50)],
            'cohort_week2': [f'w2_user_{i:03d}' for i in range(40)],
            'cohort_week3': [f'w3_user_{i:03d}' for i in range(30)]
        }
        
        # Track notifications for each cohort with different engagement patterns
        for cohort_name, users in cohorts.items():
            engagement_rate = 0.8 if 'week1' in cohort_name else 0.6 if 'week2' in cohort_name else 0.4
            
            for i, user_id in enumerate(users):
                await engine.track_delivery(
                    notification_id=f'{cohort_name}_{i:03d}',
                    user_id=user_id,
                    channel=DeliveryChannel.EMAIL,
                    status=DeliveryStatus.DELIVERED,
                    template_id='test-template'
                )
                
                # Engagement based on cohort rate
                if i < len(users) * engagement_rate:
                    await engine.track_engagement(
                        notification_id=f'{cohort_name}_{i:03d}',
                        user_id=user_id,
                        event_type='open',
                        channel=DeliveryChannel.EMAIL
                    )
        
        # Get cohort analysis
        cohort_analysis = await engine.get_cohort_analysis(
            cohort_definitions=cohorts,
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow()
        )
        
        # Verify cohort analysis
        assert len(cohort_analysis) == 3
        
        # Week 1 cohort should have highest engagement
        week1_cohort = next(c for c in cohort_analysis if c['cohort_name'] == 'cohort_week1')
        assert abs(week1_cohort['open_rate'] - 0.8) < 0.05
    
    @pytest.mark.asyncio
    async def test_a_b_test_analysis(self, test_config):
        """Test A/B test analysis functionality"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Create A/B test data
        test_variants = {
            'variant_a': {
                'template_id': 'template-a',
                'users': [f'a_user_{i:03d}' for i in range(100)],
                'expected_performance': 0.3  # 30% open rate
            },
            'variant_b': {
                'template_id': 'template-b', 
                'users': [f'b_user_{i:03d}' for i in range(100)],
                'expected_performance': 0.4  # 40% open rate (better variant)
            }
        }
        
        # Track notifications for each variant
        for variant_name, variant_data in test_variants.items():
            template_id = variant_data['template_id']
            users = variant_data['users']
            performance = variant_data['expected_performance']
            
            for i, user_id in enumerate(users):
                await engine.track_delivery(
                    notification_id=f'{variant_name}_{i:03d}',
                    user_id=user_id,
                    channel=DeliveryChannel.EMAIL,
                    status=DeliveryStatus.DELIVERED,
                    template_id=template_id,
                    metadata={'ab_test_variant': variant_name}
                )
                
                # Engagement based on expected performance
                if i < len(users) * performance:
                    await engine.track_engagement(
                        notification_id=f'{variant_name}_{i:03d}',
                        user_id=user_id,
                        event_type='open',
                        channel=DeliveryChannel.EMAIL
                    )
        
        # Analyze A/B test results
        ab_results = await engine.analyze_ab_test(
            test_name='template_comparison_test',
            variants=['variant_a', 'variant_b'],
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow(),
            metric='open_rate'
        )
        
        # Verify A/B test results
        assert 'variants' in ab_results
        assert len(ab_results['variants']) == 2
        assert 'statistical_significance' in ab_results
        
        # Variant B should perform better
        variant_a_result = next(v for v in ab_results['variants'] if v['variant'] == 'variant_a')
        variant_b_result = next(v for v in ab_results['variants'] if v['variant'] == 'variant_b')
        
        assert variant_b_result['open_rate'] > variant_a_result['open_rate']
    
    @pytest.mark.asyncio
    async def test_comprehensive_reporting(self, test_config, sample_analytics_data):
        """Test comprehensive report generation"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Simulate comprehensive data by populating engine with sample data
        # In a real implementation, this would come from tracked events
        
        # Generate comprehensive report
        report = await engine.generate_comprehensive_report(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            include_user_segments=True,
            include_channel_comparison=True,
            include_template_analysis=True,
            include_time_series=True
        )
        
        # Verify report structure
        assert 'summary' in report
        assert 'delivery_metrics' in report
        assert 'engagement_metrics' in report
        assert 'channel_performance' in report
        assert 'template_performance' in report
        assert 'time_series' in report
        assert 'recommendations' in report
        
        # Verify summary contains key metrics
        summary = report['summary']
        assert 'total_notifications_sent' in summary
        assert 'overall_delivery_rate' in summary
        assert 'overall_open_rate' in summary
        assert 'overall_click_rate' in summary
    
    @pytest.mark.asyncio
    async def test_real_time_metrics(self, test_config):
        """Test real-time metrics functionality"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Track some recent events
        for i in range(10):
            await engine.track_delivery(
                notification_id=f'realtime-{i:03d}',
                user_id=f'user-{i:03d}',
                channel=DeliveryChannel.EMAIL,
                status=DeliveryStatus.DELIVERED,
                template_id='test-template'
            )
        
        # Get real-time metrics
        realtime_metrics = await engine.get_realtime_metrics()
        
        # Verify real-time metrics
        assert 'current_hour' in realtime_metrics
        assert 'last_24_hours' in realtime_metrics
        assert 'active_campaigns' in realtime_metrics
        assert 'live_delivery_rate' in realtime_metrics
        
        # Current hour should show recent activity
        current_hour = realtime_metrics['current_hour']
        assert current_hour['notifications_sent'] >= 10
    
    @pytest.mark.asyncio
    async def test_custom_event_tracking(self, test_config):
        """Test custom event tracking"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Track custom events
        custom_events = [
            ('purchase_completed', {'amount': 99.99, 'product_id': 'prod-123'}),
            ('form_submitted', {'form_type': 'contact', 'fields_completed': 5}),
            ('video_watched', {'duration_seconds': 120, 'completion_rate': 0.75})
        ]
        
        for event_type, event_data in custom_events:
            await engine.track_custom_event(
                notification_id='custom-test-001',
                user_id='custom-user-001',
                event_type=event_type,
                event_data=event_data,
                timestamp=datetime.utcnow()
            )
        
        # Get custom events
        custom_event_data = await engine.get_custom_events(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow(),
            event_types=['purchase_completed', 'form_submitted', 'video_watched']
        )
        
        # Verify custom events
        assert len(custom_event_data) == 3
        assert any(e['event_type'] == 'purchase_completed' for e in custom_event_data)
        assert any(e['event_type'] == 'form_submitted' for e in custom_event_data)
        assert any(e['event_type'] == 'video_watched' for e in custom_event_data)

class TestAnalyticsEnginePerformance:
    """Performance tests for analytics engine"""
    
    @pytest.mark.asyncio
    async def test_high_volume_event_ingestion(self, test_config):
        """Test high-volume event ingestion performance"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Generate high volume of events
        event_count = 1000
        
        with TestTimer() as timer:
            # Track delivery events
            tasks = []
            for i in range(event_count):
                task = engine.track_delivery(
                    notification_id=f'perf-{i:06d}',
                    user_id=f'user-{i % 100:03d}',  # 100 unique users
                    channel=DeliveryChannel.EMAIL,
                    status=DeliveryStatus.DELIVERED,
                    template_id=f'template-{i % 10:02d}'  # 10 unique templates
                )
                tasks.append(task)
            
            # Execute all tracking tasks
            await asyncio.gather(*tasks)
        
        # Performance assertions
        assert timer.elapsed < 5.0  # Should process 1000 events in under 5 seconds
        throughput = event_count / timer.elapsed
        assert throughput > 200  # At least 200 events per second
        
        print(f"Event ingestion performance: {event_count} events in {timer.elapsed:.2f} seconds")
        print(f"Throughput: {throughput:.2f} events/second")
    
    @pytest.mark.asyncio
    async def test_concurrent_report_generation(self, test_config):
        """Test concurrent report generation performance"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Populate with some data first
        for i in range(100):
            await engine.track_delivery(
                notification_id=f'concurrent-{i:03d}',
                user_id=f'user-{i:03d}',
                channel=DeliveryChannel.EMAIL,
                status=DeliveryStatus.DELIVERED,
                template_id='test-template'
            )
        
        # Generate multiple reports concurrently
        report_tasks = []
        for i in range(5):
            task = engine.generate_comprehensive_report(
                start_date=datetime.utcnow() - timedelta(hours=1),
                end_date=datetime.utcnow()
            )
            report_tasks.append(task)
        
        # Measure concurrent execution time
        with TestTimer() as timer:
            reports = await asyncio.gather(*report_tasks)
        
        # All reports should be generated successfully
        assert len(reports) == 5
        assert all(report is not None for report in reports)
        assert timer.elapsed < 10.0  # Should complete within 10 seconds
        
        print(f"Concurrent report generation: 5 reports in {timer.elapsed:.2f} seconds")

class TestAnalyticsEngineIntegration:
    """Integration tests for analytics engine"""
    
    @pytest.mark.asyncio
    async def test_notification_service_integration(
        self,
        notification_service,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test integration with notification service"""
        
        # Setup analytics engine in notification service
        analytics_engine = create_analytics_engine(TEST_CONFIG['test_tenant_id'])
        notification_service.analytics_engine = analytics_engine
        
        # Setup test data
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Send notification (should trigger analytics)
        result = await notification_service.send_notification(sample_notification_request)
        
        # Verify analytics were tracked
        delivery_events = await analytics_engine.get_delivery_events(
            start_date=datetime.utcnow() - timedelta(minutes=1),
            end_date=datetime.utcnow()
        )
        
        assert len(delivery_events) > 0
        assert delivery_events[0]['notification_id'] == result['notification_id']
    
    @pytest.mark.asyncio
    async def test_external_analytics_export(self, test_config):
        """Test exporting analytics data to external systems"""
        engine = create_analytics_engine(test_config['test_tenant_id'])
        
        # Track some events
        for i in range(50):
            await engine.track_delivery(
                notification_id=f'export-{i:03d}',
                user_id=f'user-{i:03d}',
                channel=DeliveryChannel.EMAIL,
                status=DeliveryStatus.DELIVERED,
                template_id='test-template'
            )
        
        # Export data in different formats
        export_formats = ['json', 'csv', 'parquet']
        
        for format_type in export_formats:
            export_data = await engine.export_analytics_data(
                start_date=datetime.utcnow() - timedelta(hours=1),
                end_date=datetime.utcnow(),
                format=format_type
            )
            
            assert export_data is not None
            assert len(export_data) > 0
            
            if format_type == 'json':
                # Verify JSON structure
                import json
                parsed_data = json.loads(export_data)
                assert isinstance(parsed_data, (list, dict))

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])