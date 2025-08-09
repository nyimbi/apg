#!/usr/bin/env python3
"""
Basic Personalization Example

This example demonstrates how to get started with APG Deep Personalization
using simple templates and basic personalization strategies.

Requirements:
- APG notification system with personalization subcapability
- Valid tenant ID and user data
- Python 3.8+
"""

import asyncio
import logging
from typing import Dict, Any, List

# Import APG personalization components
from apg.capabilities.common.notification.personalization import (
    create_personalization_service,
    PersonalizationConfig,
    PersonalizationServiceLevel,
    PersonalizationStrategy
)
from apg.capabilities.common.notification.api_models import (
    UltimateNotificationTemplate,
    DeliveryChannel,
    NotificationPriority
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_personalization_example():
    """Demonstrate basic personalization functionality"""
    
    # Step 1: Configure personalization service
    logger.info("🚀 Setting up personalization service...")
    
    config = PersonalizationConfig(
        service_level=PersonalizationServiceLevel.STANDARD,
        enable_real_time=True,
        content_generation_enabled=True,
        behavioral_analysis_enabled=True,
        min_quality_score=0.6  # Lower threshold for basic example
    )
    
    service = create_personalization_service(
        tenant_id="demo-tenant",
        config=config
    )
    
    # Step 2: Create a basic notification template
    logger.info("📝 Creating notification template...")
    
    template = UltimateNotificationTemplate(
        id="welcome-basic-001",
        name="Basic Welcome Message",
        subject_template="Welcome to {{company_name}}, {{user_name}}! 🎉",
        text_template="""
Hi {{user_name}},

Welcome to {{company_name}}! We're thrilled to have you join our community.

Here's what you can expect:
• Personalized recommendations based on your interests
• Regular updates about {{primary_interest}}
• Exclusive offers tailored just for you

Ready to get started? Click the link below:
{{action_url}}

Best regards,
The {{company_name}} Team
        """.strip(),
        html_template="""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <h1 style="color: #007bff;">Welcome {{user_name}}! 🎉</h1>
    
    <p>We're thrilled to have you join <strong>{{company_name}}</strong>!</p>
    
    <h3>What's Next?</h3>
    <ul>
        <li>Personalized recommendations based on your interests</li>
        <li>Regular updates about <em>{{primary_interest}}</em></li>
        <li>Exclusive offers tailored just for you</li>
    </ul>
    
    <div style="margin: 20px 0;">
        <a href="{{action_url}}" 
           style="background: #007bff; color: white; padding: 12px 24px; 
                  text-decoration: none; border-radius: 5px; display: inline-block;">
            Get Started Now
        </a>
    </div>
    
    <p>Best regards,<br>The {{company_name}} Team</p>
</body>
</html>
        """.strip(),
        tenant_id="demo-tenant"
    )
    
    # Step 3: Define sample users with different profiles
    sample_users = [
        {
            "user_id": "user_001",
            "context": {
                "user_name": "Alice Johnson",
                "first_name": "Alice",
                "company_name": "TechCorp",
                "primary_interest": "artificial intelligence",
                "action_url": "https://techcorp.com/onboarding?user=alice",
                "user_segment": "tech_enthusiast",
                "previous_interactions": ["visited_ai_blog", "downloaded_whitepaper"],
                "preferred_tone": "professional"
            }
        },
        {
            "user_id": "user_002", 
            "context": {
                "user_name": "Bob Smith",
                "first_name": "Bob",
                "company_name": "TechCorp",
                "primary_interest": "web development",
                "action_url": "https://techcorp.com/onboarding?user=bob",
                "user_segment": "developer",
                "previous_interactions": ["viewed_tutorials", "joined_webinar"],
                "preferred_tone": "casual"
            }
        },
        {
            "user_id": "user_003",
            "context": {
                "user_name": "Carol Davis",
                "first_name": "Carol", 
                "company_name": "TechCorp",
                "primary_interest": "data science",
                "action_url": "https://techcorp.com/onboarding?user=carol",
                "user_segment": "data_scientist",
                "previous_interactions": ["downloaded_dataset", "read_research_paper"],
                "preferred_tone": "friendly"
            }
        }
    ]
    
    # Step 4: Personalize messages for each user
    logger.info("🎯 Personalizing messages for sample users...")
    
    results = []
    for user_data in sample_users:
        logger.info(f"Personalizing for {user_data['context']['user_name']}...")
        
        # Personalize the notification
        result = await service.personalize_notification(
            notification_template=template,
            user_id=user_data["user_id"],
            context=user_data["context"],
            channels=[DeliveryChannel.EMAIL]
        )
        
        results.append({
            "user": user_data["context"]["user_name"],
            "result": result
        })
        
        # Display personalization results
        print(f"\n{'='*60}")
        print(f"👤 USER: {user_data['context']['user_name']}")
        print(f"{'='*60}")
        print(f"📊 Quality Score: {result.get('quality_score', 0):.2f}")
        print(f"⚡ Processing Time: {result.get('processing_time_ms', 0)}ms")
        print(f"🧠 Strategies: {', '.join(result.get('strategies_applied', []))}")
        print(f"\n📧 SUBJECT: {result['personalized_content']['subject']}")
        print(f"\n📝 MESSAGE:\n{result['personalized_content']['text']}")
        
        if result.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in result['recommendations']:
                print(f"   • {rec}")
    
    # Step 5: Demonstrate batch personalization
    logger.info("📦 Demonstrating batch personalization...")
    
    batch_result = await service.personalize_campaign(
        campaign={
            "id": "welcome-campaign-001",
            "name": "Welcome Campaign",
            "campaign_type": "transactional"
        },
        target_users=[user["user_id"] for user in sample_users],
        context={
            "campaign_type": "welcome_series",
            "company_name": "TechCorp",
            "batch_id": "batch_001"
        }
    )
    
    print(f"\n{'='*60}")
    print(f"📦 BATCH PERSONALIZATION RESULTS")
    print(f"{'='*60}")
    print(f"👥 Total Users: {batch_result['campaign_stats']['total_users']}")
    print(f"✅ Successful: {batch_result['campaign_stats']['successful_personalizations']}")
    print(f"📊 Success Rate: {batch_result['campaign_stats']['success_rate']:.1%}")
    print(f"⭐ Avg Quality: {batch_result['campaign_stats']['avg_quality_score']:.2f}")
    
    # Step 6: Get service statistics
    logger.info("📈 Retrieving service statistics...")
    
    stats = service.get_service_stats()
    
    print(f"\n{'='*60}")
    print(f"📈 SERVICE STATISTICS")
    print(f"{'='*60}")
    print(f"🎯 Total Personalizations: {stats['service_stats']['total_personalizations']}")
    print(f"✅ Successful: {stats['service_stats']['successful_personalizations']}")
    print(f"⚡ Avg Response Time: {stats['service_stats']['avg_response_time_ms']:.1f}ms")
    print(f"⭐ Avg Quality Score: {stats['service_stats']['avg_quality_score']:.2f}")
    print(f"🤖 AI Model Calls: {stats['service_stats']['ai_model_calls']}")
    
    return results


async def demonstrate_user_insights():
    """Demonstrate user insights and preference management"""
    
    logger.info("🔍 Demonstrating user insights...")
    
    config = PersonalizationConfig(
        service_level=PersonalizationServiceLevel.PREMIUM,
        enable_predictive=True,
        behavioral_analysis_enabled=True
    )
    
    service = create_personalization_service(
        tenant_id="demo-tenant",
        config=config
    )
    
    # Update user preferences
    await service.update_user_preferences(
        user_id="user_001",
        preferences={
            "content_preferences": {
                "topics": ["artificial_intelligence", "machine_learning", "automation"],
                "tone": "professional",
                "length": "medium"
            },
            "channel_preferences": {
                "email": 0.9,
                "sms": 0.6,
                "push": 0.8
            },
            "timing_preferences": {
                "preferred_hours": [9, 10, 14, 15],
                "timezone": "America/New_York"
            }
        }
    )
    
    # Get comprehensive insights
    insights = await service.get_personalization_insights(
        user_id="user_001",
        include_predictions=True
    )
    
    print(f"\n{'='*60}")
    print(f"🔍 USER INSIGHTS - Alice Johnson")
    print(f"{'='*60}")
    print(f"📊 Profile Completeness: {insights['profile_summary']['completeness']:.1%}")
    print(f"🎯 Model Confidence: {insights['profile_summary']['confidence']:.1%}")
    print(f"📈 Engagement Prediction: {insights['predictive_scores']['engagement_prediction']:.2f}")
    print(f"⚠️  Churn Risk: {insights['predictive_scores']['churn_risk']:.2f}")
    
    if insights.get('predictions'):
        print(f"\n🔮 PREDICTIONS:")
        for key, value in insights['predictions'].items():
            print(f"   • {key}: {value}")


async def main():
    """Run all examples"""
    
    print("🎯 APG Deep Personalization - Basic Examples")
    print("=" * 60)
    
    try:
        # Run basic personalization example
        await basic_personalization_example()
        
        # Demonstrate user insights
        await demonstrate_user_insights()
        
        print(f"\n{'='*60}")
        print("✅ All examples completed successfully!")
        print("💡 Try modifying the templates and user data to see different results.")
        print("📚 Check the documentation for more advanced features.")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())