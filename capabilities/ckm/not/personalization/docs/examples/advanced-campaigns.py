#!/usr/bin/env python3
"""
Advanced Campaign Personalization Example

This example demonstrates sophisticated campaign personalization including:
- Multi-step drip campaigns
- Behavioral trigger automation
- Cross-channel personalization
- Advanced A/B testing
- Real-time optimization

Requirements:
- APG notification system with Enterprise+ personalization
- Valid tenant ID and comprehensive user data
- Python 3.8+
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random

# Import APG personalization components
from apg.capabilities.common.notification.personalization import (
    create_personalization_service,
    PersonalizationConfig,
    PersonalizationServiceLevel,
    PersonalizationStrategy,
    PersonalizationTrigger
)
from apg.capabilities.common.notification.api_models import (
    UltimateNotificationTemplate,
    DeliveryChannel,
    NotificationPriority,
    CampaignType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedCampaignOrchestrator:
    """Advanced campaign orchestration with AI-powered personalization"""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        
        # Initialize Enterprise-level personalization
        self.config = PersonalizationConfig(
            service_level=PersonalizationServiceLevel.ENTERPRISE,
            enable_real_time=True,
            enable_predictive=True,
            enable_emotional_intelligence=True,
            enable_cross_channel_sync=True,
            content_generation_enabled=True,
            behavioral_analysis_enabled=True,
            min_quality_score=0.8
        )
        
        self.personalization_service = create_personalization_service(
            tenant_id=tenant_id,
            config=self.config
        )
        
        # Campaign templates
        self.templates = {}
        self.user_journeys = {}
        
        logger.info(f"🚀 Advanced Campaign Orchestrator initialized for {tenant_id}")
    
    async def setup_drip_campaign_templates(self):
        """Create sophisticated drip campaign templates"""
        
        self.templates["onboarding_day_0"] = UltimateNotificationTemplate(
            id="onboard_0",
            name="Welcome & First Steps",
            subject_template="Welcome to {{company_name}}, {{user_name}}! Let's get started 🚀",
            text_template="""
Hi {{user_name}},

Welcome to {{company_name}}! I'm thrilled you've joined our community of {{user_segment}} professionals.

Based on your interest in {{primary_interest}}, I've prepared a personalized getting-started guide just for you.

🎯 Your Next Steps:
1. Complete your profile (2 minutes)
2. Explore {{recommended_feature}} 
3. Join the {{relevant_community}} community

Ready to dive in?
{{action_url}}

To your success,
{{sender_name}}
P.S. Questions? Just reply to this email - I read every message personally.
            """.strip(),
            tenant_id=self.tenant_id
        )
        
        self.templates["onboarding_day_3"] = UltimateNotificationTemplate(
            id="onboard_3",
            name="Progress Check & Value Delivery",
            subject_template="{{user_name}}, how's your {{company_name}} journey going? 📈",
            text_template="""
Hi {{user_name}},

I wanted to check in on your progress with {{company_name}}.

{{#if has_completed_setup}}
Great job completing your setup! I noticed you're particularly interested in {{engaged_features}}.
{{else}}
I see you haven't had a chance to complete your setup yet - no worries! Life gets busy.
{{/if}}

{{#if user_segment_is_advanced}}
Since you're an experienced {{user_segment}}, you might be interested in our advanced features:
• {{advanced_feature_1}}
• {{advanced_feature_2}}
{{else}}
Here's what other {{user_segment}}s are loving most:
• {{popular_feature_1}} ({{usage_stat_1}})
• {{popular_feature_2}} ({{usage_stat_2}})
{{/if}}

{{personalized_insight}}

Continue your journey:
{{action_url}}

Cheering you on,
{{sender_name}}
            """.strip(),
            tenant_id=self.tenant_id
        )
        
        self.templates["value_demonstration"] = UltimateNotificationTemplate(
            id="value_demo",
            name="Success Stories & Social Proof",
            subject_template="{{user_name}}, see what {{similar_user}} achieved with {{company_name}}",
            text_template="""
Hi {{user_name}},

I wanted to share an inspiring story that reminded me of you.

{{similar_user}}, a {{similar_role}} from {{similar_industry}}, was facing similar challenges with {{pain_point}}.

Here's what happened after they started using {{company_name}}:

📊 Results in {{timeframe}}:
• {{metric_1}}: {{improvement_1}}
• {{metric_2}}: {{improvement_2}}
• {{metric_3}}: {{improvement_3}}

"{{testimonial_quote}}" - {{similar_user}}

{{#if behavioral_indicator_positive}}
Based on your activity, I can see you're on a similar path to success!
{{else}}
You have all the same opportunities available to you.
{{/if}}

Ready to accelerate your results?
{{action_url}}

Here to help,
{{sender_name}}

P.S. {{similar_user}} started exactly where you are now. The only difference? They took action.
            """.strip(),
            tenant_id=self.tenant_id
        )
        
        logger.info("📝 Drip campaign templates created")
    
    async def create_behavioral_user_profiles(self) -> List[Dict[str, Any]]:
        """Create diverse user profiles for demonstration"""
        
        profiles = [
            {
                "user_id": "enterprise_exec_001",
                "segment": "enterprise_executive",
                "context": {
                    "user_name": "Sarah Williams",
                    "first_name": "Sarah",
                    "role": "VP of Marketing",
                    "company_size": "Enterprise (5000+ employees)",
                    "industry": "Technology",
                    "primary_interest": "marketing automation",
                    "pain_point": "lead qualification inefficiency",
                    "behavioral_indicators": {
                        "engagement_level": "high",
                        "feature_adoption": "advanced",
                        "time_spent_daily": 45,
                        "has_completed_setup": True,
                        "engaged_features": ["analytics_dashboard", "automation_builder"]
                    },
                    "psychographic": {
                        "personality": "analytical",
                        "communication_style": "direct",
                        "decision_making": "data_driven",
                        "preferred_tone": "professional"
                    }
                }
            },
            {
                "user_id": "startup_founder_002",
                "segment": "startup_founder",
                "context": {
                    "user_name": "Mike Chen",
                    "first_name": "Mike", 
                    "role": "Founder & CEO",
                    "company_size": "Startup (10-50 employees)",
                    "industry": "SaaS",
                    "primary_interest": "growth hacking",
                    "pain_point": "limited marketing budget efficiency",
                    "behavioral_indicators": {
                        "engagement_level": "medium",
                        "feature_adoption": "basic",
                        "time_spent_daily": 15,
                        "has_completed_setup": False,
                        "engaged_features": ["email_campaigns"]
                    },
                    "psychographic": {
                        "personality": "innovative",
                        "communication_style": "casual",
                        "decision_making": "intuitive",
                        "preferred_tone": "friendly"
                    }
                }
            },
            {
                "user_id": "marketing_manager_003",
                "segment": "marketing_manager", 
                "context": {
                    "user_name": "Jennifer Rodriguez",
                    "first_name": "Jennifer",
                    "role": "Marketing Manager",
                    "company_size": "Mid-market (200-1000 employees)",
                    "industry": "E-commerce", 
                    "primary_interest": "customer segmentation",
                    "pain_point": "personalization at scale",
                    "behavioral_indicators": {
                        "engagement_level": "high",
                        "feature_adoption": "intermediate",
                        "time_spent_daily": 30,
                        "has_completed_setup": True,
                        "engaged_features": ["segmentation_tools", "a_b_testing"]
                    },
                    "psychographic": {
                        "personality": "creative",
                        "communication_style": "enthusiastic",
                        "decision_making": "collaborative",
                        "preferred_tone": "inspiring"
                    }
                }
            }
        ]
        
        # Enrich profiles with AI-generated insights
        for profile in profiles:
            await self._enrich_user_profile(profile)
        
        return profiles
    
    async def _enrich_user_profile(self, profile: Dict[str, Any]):
        """Enrich user profile with AI-generated behavioral insights"""
        
        # Update user preferences based on profile
        await self.personalization_service.update_user_preferences(
            user_id=profile["user_id"],
            preferences={
                "content_preferences": {
                    "topics": [profile["context"]["primary_interest"], "best_practices", "case_studies"],
                    "tone": profile["context"]["psychographic"]["preferred_tone"],
                    "length": "medium" if profile["context"]["behavioral_indicators"]["time_spent_daily"] > 20 else "short"
                },
                "channel_preferences": {
                    "email": 0.9,
                    "in_app": 0.8 if profile["context"]["behavioral_indicators"]["engagement_level"] == "high" else 0.5,
                    "sms": 0.3 if profile["segment"] == "enterprise_executive" else 0.7
                },
                "timing_preferences": {
                    "preferred_hours": [9, 10, 14, 15] if profile["segment"] == "enterprise_executive" else [8, 12, 16, 18],
                    "timezone": "America/New_York"
                }
            },
            trigger=PersonalizationTrigger.BEHAVIORAL_PATTERN
        )
        
        # Generate AI insights
        insights = await self.personalization_service.get_personalization_insights(
            user_id=profile["user_id"],
            include_predictions=True
        )
        
        profile["ai_insights"] = insights
        
        logger.info(f"🧠 Enriched profile for {profile['context']['user_name']}")
    
    async def execute_intelligent_drip_campaign(self, user_profiles: List[Dict[str, Any]]):
        """Execute a sophisticated drip campaign with AI optimization"""
        
        logger.info("🎯 Starting intelligent drip campaign execution...")
        
        campaign_results = []
        
        for day, template_key in enumerate(["onboarding_day_0", "onboarding_day_3", "value_demonstration"]):
            logger.info(f"📅 Day {day * 3}: Executing {template_key} campaign step...")
            
            template = self.templates[template_key]
            
            # Personalize for each user with advanced strategies
            for profile in user_profiles:
                
                # Build rich context for personalization
                context = await self._build_campaign_context(profile, day)
                
                # Apply advanced personalization strategies
                strategies = self._select_optimal_strategies(profile, day)
                
                result = await self.personalization_service.personalize_notification(
                    notification_template=template,
                    user_id=profile["user_id"],
                    context=context,
                    channels=self._select_optimal_channels(profile)
                )
                
                campaign_results.append({
                    "day": day,
                    "template": template_key,
                    "user": profile["context"]["user_name"],
                    "segment": profile["segment"],
                    "quality_score": result.get("quality_score", 0),
                    "strategies": result.get("strategies_applied", []),
                    "predicted_engagement": result.get("predicted_engagement", {}),
                    "personalized_content": result["personalized_content"]
                })
                
                # Display results
                self._display_campaign_result(profile, result, day, template_key)
        
        return campaign_results
    
    async def _build_campaign_context(self, profile: Dict[str, Any], day: int) -> Dict[str, Any]:
        """Build rich context for campaign personalization"""
        
        base_context = profile["context"].copy()
        
        # Add campaign-specific context
        base_context.update({
            "company_name": "MarketingPro",
            "sender_name": "Alex Thompson",
            "campaign_day": day,
            "user_segment": profile["segment"],
            "action_url": f"https://marketingpro.com/continue?user={profile['user_id']}&day={day}"
        })
        
        # Add behavioral context
        behavioral = profile["context"]["behavioral_indicators"]
        base_context.update({
            "has_completed_setup": behavioral["has_completed_setup"],
            "engaged_features": ", ".join(behavioral["engaged_features"]),
            "user_segment_is_advanced": behavioral["feature_adoption"] == "advanced"
        })
        
        # Add segment-specific context
        if profile["segment"] == "enterprise_executive":
            base_context.update({
                "recommended_feature": "Executive Dashboard",
                "relevant_community": "Executive Leadership",
                "advanced_feature_1": "Advanced Analytics Suite",
                "advanced_feature_2": "Custom Report Builder"
            })
        elif profile["segment"] == "startup_founder":
            base_context.update({
                "recommended_feature": "Growth Tracker",
                "relevant_community": "Startup Founders",
                "popular_feature_1": "Viral Coefficient Calculator",
                "popular_feature_2": "Referral Program Builder",
                "usage_stat_1": "94% see growth in 30 days",
                "usage_stat_2": "2.3x average referral rate"
            })
        
        # Add success story context (for value demonstration)
        if day == 2:  # Value demonstration day
            base_context.update({
                "similar_user": "David Kim" if profile["segment"] == "startup_founder" else "Lisa Zhang",
                "similar_role": profile["context"]["role"],
                "similar_industry": profile["context"]["industry"],
                "timeframe": "3 months",
                "metric_1": "Lead Quality Score",
                "improvement_1": "+47%",
                "metric_2": "Conversion Rate", 
                "improvement_2": "+23%",
                "metric_3": "Time to Close",
                "improvement_3": "-31%",
                "testimonial_quote": "MarketingPro transformed how we approach lead qualification. The AI insights are game-changing.",
                "behavioral_indicator_positive": behavioral["engagement_level"] == "high"
            })
        
        # Add AI-generated personalization insights
        if profile.get("ai_insights"):
            base_context["personalized_insight"] = self._generate_personalized_insight(profile)
        
        return base_context
    
    def _select_optimal_strategies(self, profile: Dict[str, Any], day: int) -> List[PersonalizationStrategy]:
        """Select optimal personalization strategies based on user profile and campaign day"""
        
        base_strategies = [
            PersonalizationStrategy.NEURAL_CONTENT,
            PersonalizationStrategy.BEHAVIORAL_ADAPTIVE,
            PersonalizationStrategy.CONTEXTUAL_INTELLIGENCE
        ]
        
        # Add strategies based on user segment
        if profile["segment"] == "enterprise_executive":
            base_strategies.extend([
                PersonalizationStrategy.PREDICTIVE_OPTIMIZATION,
                PersonalizationStrategy.CROSS_CHANNEL_SYNC
            ])
        
        # Add strategies based on campaign day
        if day == 0:  # Welcome message
            base_strategies.append(PersonalizationStrategy.EMOTIONAL_RESONANCE)
        elif day == 1:  # Progress check
            base_strategies.extend([
                PersonalizationStrategy.REAL_TIME_ADAPTATION,
                PersonalizationStrategy.BEHAVIORAL_ADAPTIVE
            ])
        elif day == 2:  # Value demonstration
            base_strategies.extend([
                PersonalizationStrategy.EMPATHY_DRIVEN,
                PersonalizationStrategy.PREDICTIVE_OPTIMIZATION
            ])
        
        return base_strategies
    
    def _select_optimal_channels(self, profile: Dict[str, Any]) -> List[DeliveryChannel]:
        """Select optimal delivery channels based on user profile"""
        
        channels = [DeliveryChannel.EMAIL]  # Always include email
        
        # Add channels based on segment and preferences
        if profile["segment"] == "startup_founder":
            channels.extend([DeliveryChannel.SLACK, DeliveryChannel.IN_APP])
        elif profile["segment"] == "enterprise_executive":
            channels.extend([DeliveryChannel.IN_APP])
        else:
            channels.extend([DeliveryChannel.IN_APP, DeliveryChannel.WEB_PUSH])
        
        return channels
    
    def _generate_personalized_insight(self, profile: Dict[str, Any]) -> str:
        """Generate personalized insight based on AI analysis"""
        
        insights = [
            f"Based on your {profile['context']['role']} background, I think you'll particularly value our advanced segmentation capabilities.",
            f"Users in {profile['context']['industry']} typically see the biggest impact from our automation features.",
            f"Given your interest in {profile['context']['primary_interest']}, you're positioned to see significant ROI quickly."
        ]
        
        return random.choice(insights)
    
    def _display_campaign_result(self, profile: Dict[str, Any], result: Dict[str, Any], day: int, template_key: str):
        """Display formatted campaign result"""
        
        print(f"\n{'='*80}")
        print(f"📅 DAY {day * 3} - {template_key.upper()}")
        print(f"👤 {profile['context']['user_name']} ({profile['segment']})")
        print(f"{'='*80}")
        print(f"📊 Quality Score: {result.get('quality_score', 0):.2f}")
        print(f"🧠 Strategies: {', '.join(result.get('strategies_applied', []))}")
        print(f"⚡ Processing: {result.get('processing_time_ms', 0)}ms")
        
        if result.get('predicted_engagement'):
            engagement = result['predicted_engagement']
            print(f"🎯 Predicted Engagement:")
            print(f"   📖 Open: {engagement.get('open_probability', 0):.1%}")
            print(f"   👆 Click: {engagement.get('click_probability', 0):.1%}")
            print(f"   💰 Convert: {engagement.get('convert_probability', 0):.1%}")
        
        print(f"\n📧 SUBJECT: {result['personalized_content'].get('subject', 'N/A')}")
        print(f"\n📝 MESSAGE:\n{result['personalized_content'].get('text', 'N/A')[:500]}...")
        
        if result.get('recommendations'):
            print(f"\n💡 AI RECOMMENDATIONS:")
            for rec in result['recommendations'][:3]:
                print(f"   • {rec}")


async def demonstrate_cross_channel_orchestration():
    """Demonstrate sophisticated cross-channel campaign orchestration"""
    
    logger.info("🔄 Demonstrating cross-channel orchestration...")
    
    orchestrator = AdvancedCampaignOrchestrator("enterprise-demo")
    
    # Setup cross-channel templates
    channels_config = {
        DeliveryChannel.EMAIL: {
            "template": "detailed_content",
            "timing": "morning",
            "followup_delay": 24
        },
        DeliveryChannel.IN_APP: {
            "template": "concise_prompt", 
            "timing": "immediate",
            "followup_delay": 2
        },
        DeliveryChannel.SMS: {
            "template": "urgent_reminder",
            "timing": "afternoon",
            "followup_delay": 6
        }
    }
    
    print(f"\n{'='*80}")
    print("🔄 CROSS-CHANNEL ORCHESTRATION DEMO")
    print(f"{'='*80}")
    
    for channel, config in channels_config.items():
        print(f"📱 {channel.value.upper()}: {config['template']} → {config['timing']} → {config['followup_delay']}h followup")
    
    print("\n✅ Cross-channel orchestration configured")
    print("💡 In production, this would coordinate timing, frequency, and content across all channels")


async def main():
    """Run advanced campaign examples"""
    
    print("🚀 APG Deep Personalization - Advanced Campaign Examples")
    print("=" * 80)
    
    try:
        # Initialize campaign orchestrator
        orchestrator = AdvancedCampaignOrchestrator("enterprise-demo")
        
        # Setup templates
        await orchestrator.setup_drip_campaign_templates()
        
        # Create behavioral user profiles
        user_profiles = await orchestrator.create_behavioral_user_profiles()
        
        # Execute intelligent drip campaign
        campaign_results = await orchestrator.execute_intelligent_drip_campaign(user_profiles)
        
        # Demonstrate cross-channel orchestration
        await demonstrate_cross_channel_orchestration()
        
        # Display campaign summary
        print(f"\n{'='*80}")
        print("📊 CAMPAIGN SUMMARY")
        print(f"{'='*80}")
        
        total_messages = len(campaign_results)
        avg_quality = sum(r["quality_score"] for r in campaign_results) / total_messages
        high_quality_count = sum(1 for r in campaign_results if r["quality_score"] > 0.8)
        
        print(f"📨 Total Messages: {total_messages}")
        print(f"⭐ Average Quality: {avg_quality:.2f}")
        print(f"🎯 High Quality (>0.8): {high_quality_count}/{total_messages} ({high_quality_count/total_messages:.1%})")
        
        # Top strategies used
        all_strategies = []
        for result in campaign_results:
            all_strategies.extend(result["strategies"])
        
        from collections import Counter
        strategy_counts = Counter(all_strategies)
        
        print(f"\n🧠 TOP STRATEGIES USED:")
        for strategy, count in strategy_counts.most_common(5):
            print(f"   • {strategy}: {count} times")
        
        print(f"\n{'='*80}")
        print("✅ Advanced campaign demonstration completed!")
        print("💡 This showcases enterprise-level personalization capabilities")
        print("🚀 Ready to revolutionize your marketing campaigns!")
        
    except Exception as e:
        logger.error(f"❌ Advanced campaign example failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the advanced examples
    asyncio.run(main())