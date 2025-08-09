#!/usr/bin/env python3
"""
APG API Service Mesh - Natural Language Policies Demo

Interactive demonstration of revolutionary natural language policy creation
that converts plain English into service mesh configurations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PolicyExample:
	"""Example natural language policy."""
	description: str
	category: str
	complexity: str
	expected_output: Dict[str, Any]

class NaturalLanguagePolicyDemo:
	"""Demonstration of natural language policy processing."""
	
	def __init__(self):
		self.policy_examples = self._load_policy_examples()
		self.policy_history = []
	
	def _load_policy_examples(self) -> List[PolicyExample]:
		"""Load example policies for demonstration."""
		return [
			PolicyExample(
				description="Rate limit the payment service to 1000 requests per minute",
				category="Traffic Management",
				complexity="Simple",
				expected_output={
					"type": "rate_limiting",
					"target": {"service": "payment-service"},
					"limits": {"requests_per_minute": 1000},
					"enforcement": "strict"
				}
			),
			
			PolicyExample(
				description="Only allow authenticated users from the mobile app to access the user profile API",
				category="Security & Access Control", 
				complexity="Medium",
				expected_output={
					"type": "access_control",
					"target": {"service": "user-profile-api"},
					"authentication": {"required": True},
					"authorization": {"source_app": "mobile"},
					"enforcement": "strict"
				}
			),
			
			PolicyExample(
				description="Route 20% of traffic to version 2 of the recommendation engine for A/B testing, but only for premium users during business hours",
				category="Traffic Routing",
				complexity="Complex",
				expected_output={
					"type": "traffic_routing",
					"target": {"service": "recommendation-engine"},
					"routing_rules": {
						"version_split": {"v1": 80, "v2": 20},
						"conditions": {
							"user_tier": "premium",
							"time_range": "business_hours"
						}
					},
					"purpose": "ab_testing"
				}
			),
			
			PolicyExample(
				description="Enable circuit breaker for the external payment gateway with 5 failure threshold and 30 second timeout",
				category="Resilience",
				complexity="Medium",
				expected_output={
					"type": "circuit_breaker",
					"target": {"service": "external-payment-gateway"},
					"configuration": {
						"failure_threshold": 5,
						"timeout_seconds": 30,
						"half_open_requests": 3
					}
				}
			),
			
			PolicyExample(
				description="Automatically scale the analytics service between 2 and 10 replicas based on CPU usage above 70% and queue depth over 100 items",
				category="Auto-scaling",
				complexity="Complex",
				expected_output={
					"type": "auto_scaling",
					"target": {"service": "analytics-service"},
					"scaling_policy": {
						"min_replicas": 2,
						"max_replicas": 10,
						"triggers": [
							{"metric": "cpu_usage", "threshold": 70, "operator": "gt"},
							{"metric": "queue_depth", "threshold": 100, "operator": "gt"}
						],
						"scaling_behavior": "gradual"
					}
				}
			),
			
			PolicyExample(
				description="Block all requests from IP addresses in suspicious countries and log security events to the compliance dashboard",
				category="Security & Compliance",
				complexity="Complex",
				expected_output={
					"type": "security_policy",
					"target": {"scope": "global"},
					"rules": [
						{
							"action": "block",
							"condition": {"source_ip": {"geo_filter": "suspicious_countries"}},
							"logging": {"destination": "compliance_dashboard", "level": "security_event"}
						}
					]
				}
			),
			
			PolicyExample(
				description="Implement blue-green deployment for the search service with automatic rollback if error rate exceeds 2% within 10 minutes",
				category="Deployment Strategy",
				complexity="Complex",
				expected_output={
					"type": "deployment_strategy",
					"target": {"service": "search-service"},
					"strategy": {
						"type": "blue_green",
						"health_checks": {
							"error_rate_threshold": 2.0,
							"monitoring_window_minutes": 10,
							"auto_rollback": True
						}
					}
				}
			),
			
			PolicyExample(
				description="Encrypt all communication between financial services using mTLS and require certificate validation",
				category="Security",
				complexity="Medium",
				expected_output={
					"type": "mtls_policy",
					"target": {"service_group": "financial-services"},
					"encryption": {
						"protocol": "mTLS",
						"certificate_validation": "required",
						"cipher_suites": "strong_only"
					}
				}
			)
		]
	
	async def start_demo(self):
		"""Start the interactive natural language policy demo."""
		print("\n" + "="*80)
		print("🧠 APG SERVICE MESH - NATURAL LANGUAGE POLICIES DEMO")
		print("="*80)
		print("\n🗣️  Revolutionary Policy Creation with Plain English")
		print("🤖 AI-Powered: Natural Language → Service Mesh Configuration")
		print("⚡ Zero YAML: No configuration files needed!")
		print("\n" + "-"*80)
		
		while True:
			await self._show_main_menu()
			choice = input("\nEnter your choice [1-4]: ").strip()
			
			if choice == "1":
				await self._demo_example_policies()
			elif choice == "2":
				await self._interactive_policy_creation()
			elif choice == "3":
				await self._show_policy_categories()
			elif choice == "4":
				print("\n👋 Thank you for exploring APG Service Mesh Natural Language Policies!")
				break
			else:
				print("❌ Invalid choice. Please try again.")
	
	async def _show_main_menu(self):
		"""Show the main demo menu."""
		print(f"\n{'='*50}")
		print("🎯 NATURAL LANGUAGE POLICY DEMO MENU")
		print("="*50)
		print("1. 📋 View Example Policies")
		print("2. ✨ Create Custom Policy")
		print("3. 📊 Policy Categories Overview")
		print("4. 🚪 Exit Demo")
	
	async def _demo_example_policies(self):
		"""Demonstrate example policies."""
		print(f"\n{'='*60}")
		print("📋 EXAMPLE NATURAL LANGUAGE POLICIES")
		print("="*60)
		
		for i, example in enumerate(self.policy_examples, 1):
			print(f"\n🔹 Example {i}: {example.category} ({example.complexity})")
			print("-" * 50)
			print(f"📝 Natural Language:")
			print(f'   "{example.description}"')
			
			print(f"\n🤖 AI Processing...")
			await asyncio.sleep(1)  # Simulate processing
			
			print(f"✅ Generated Configuration:")
			print(f"   Category: {example.category}")
			print(f"   Complexity: {example.complexity}")
			print(f"   Output: {json.dumps(example.expected_output, indent=6)}")
			
			if i < len(self.policy_examples):
				input("\n[Press Enter for next example...]")
	
	async def _interactive_policy_creation(self):
		"""Interactive policy creation."""
		print(f"\n{'='*60}")
		print("✨ INTERACTIVE POLICY CREATION")
		print("="*60)
		print("\n🗣️  Describe your policy in plain English!")
		print("💡 Examples:")
		print('   • "Rate limit the API to 500 requests per minute"')
		print('   • "Scale the web service when CPU exceeds 80%"')
		print('   • "Block traffic from suspicious IP addresses"')
		print('   • "Route 10% of traffic to the new version"')
		
		while True:
			print("\n" + "-"*50)
			user_policy = input("🎤 Describe your policy (or 'back' to return): ").strip()
			
			if user_policy.lower() in ['back', 'exit', 'quit']:
				break
			
			if not user_policy:
				print("❌ Please enter a policy description.")
				continue
			
			await self._process_user_policy(user_policy)
	
	async def _process_user_policy(self, policy_text: str):
		"""Process user's natural language policy."""
		print(f"\n🧠 Processing: \"{policy_text}\"")
		print("🤖 AI Analysis in progress...")
		await asyncio.sleep(2)  # Simulate AI processing
		
		# Analyze the policy text
		analysis = await self._analyze_policy_text(policy_text)
		
		print(f"\n📊 ANALYSIS RESULTS:")
		print("-" * 30)
		print(f"🎯 Intent: {analysis['intent']}")
		print(f"📈 Confidence: {analysis['confidence']:.1%}")
		print(f"🏷️  Category: {analysis['category']}")
		print(f"⚙️  Parameters: {json.dumps(analysis['parameters'], indent=4)}")
		
		# Generate mesh configuration
		print(f"\n⚙️  Generating mesh configuration...")
		await asyncio.sleep(1)
		
		mesh_config = await self._generate_mesh_configuration(analysis)
		
		print(f"\n✅ GENERATED CONFIGURATION:")
		print("-" * 30)
		print(json.dumps(mesh_config, indent=2))
		
		# Simulate deployment
		print(f"\n🚀 Deployment simulation...")
		await asyncio.sleep(1.5)
		
		deployment_result = await self._simulate_deployment(mesh_config)
		
		print(f"\n📈 DEPLOYMENT RESULT:")
		print("-" * 30)
		print(f"Status: {deployment_result['status']}")
		print(f"Message: {deployment_result['message']}")
		
		if deployment_result['warnings']:
			print(f"⚠️  Warnings: {', '.join(deployment_result['warnings'])}")
		
		# Add to history
		self.policy_history.append({
			'timestamp': datetime.now().isoformat(),
			'input': policy_text,
			'analysis': analysis,
			'configuration': mesh_config,
			'deployment': deployment_result
		})
		
		print(f"\n✨ Policy processed successfully!")
	
	async def _analyze_policy_text(self, policy_text: str) -> Dict[str, Any]:
		"""Analyze natural language policy text."""
		text_lower = policy_text.lower()
		
		# Intent classification
		if "rate limit" in text_lower or "throttle" in text_lower:
			intent = "rate_limiting"
			category = "Traffic Management"
			
			# Extract rate limit values
			rate_match = re.search(r'(\d+)\s+requests?\s+per\s+(minute|second|hour)', text_lower)
			if rate_match:
				rate = int(rate_match.group(1))
				unit = rate_match.group(2)
			else:
				rate = 100
				unit = "minute"
			
			parameters = {
				"rate": rate,
				"unit": unit,
				"service": self._extract_service_name(text_lower)
			}
			confidence = 0.92
		
		elif "scale" in text_lower or "replicas" in text_lower:
			intent = "auto_scaling"
			category = "Resource Management"
			
			# Extract scaling parameters
			replica_match = re.search(r'(\d+)\s+replicas?', text_lower)
			cpu_match = re.search(r'cpu.*?(\d+)%', text_lower)
			
			parameters = {
				"service": self._extract_service_name(text_lower),
				"replicas": int(replica_match.group(1)) if replica_match else None,
				"cpu_threshold": int(cpu_match.group(1)) if cpu_match else None
			}
			confidence = 0.88
		
		elif "route" in text_lower and ("traffic" in text_lower or "%" in text_lower):
			intent = "traffic_routing"
			category = "Traffic Management"
			
			# Extract routing percentages
			percent_match = re.search(r'(\d+)%', text_lower)
			version_match = re.search(r'version\s+(\w+)', text_lower)
			
			parameters = {
				"service": self._extract_service_name(text_lower),
				"percentage": int(percent_match.group(1)) if percent_match else 50,
				"target_version": version_match.group(1) if version_match else "v2"
			}
			confidence = 0.90
		
		elif "block" in text_lower or "deny" in text_lower:
			intent = "security_policy"
			category = "Security"
			
			parameters = {
				"action": "block",
				"condition": self._extract_security_conditions(text_lower),
				"scope": "global" if "all" in text_lower else "service"
			}
			confidence = 0.85
		
		elif "circuit breaker" in text_lower or "circuit-breaker" in text_lower:
			intent = "circuit_breaker"
			category = "Resilience"
			
			# Extract circuit breaker parameters
			threshold_match = re.search(r'(\d+)\s+failure', text_lower)
			timeout_match = re.search(r'(\d+)\s+second', text_lower)
			
			parameters = {
				"service": self._extract_service_name(text_lower),
				"failure_threshold": int(threshold_match.group(1)) if threshold_match else 5,
				"timeout_seconds": int(timeout_match.group(1)) if timeout_match else 30
			}
			confidence = 0.87
		
		elif "encrypt" in text_lower or "tls" in text_lower or "mtls" in text_lower:
			intent = "encryption_policy"
			category = "Security"
			
			parameters = {
				"encryption_type": "mTLS" if "mtls" in text_lower else "TLS",
				"target": self._extract_service_name(text_lower) or "all_services",
				"certificate_validation": "required"
			}
			confidence = 0.89
		
		else:
			intent = "general_policy"
			category = "General"
			parameters = {"description": policy_text}
			confidence = 0.60
		
		return {
			"intent": intent,
			"category": category,
			"confidence": confidence,
			"parameters": parameters,
			"original_text": policy_text
		}
	
	def _extract_service_name(self, text: str) -> str:
		"""Extract service name from text."""
		# Common service name patterns
		service_patterns = [
			r'the\s+(\w+[-_]?\w*)\s+service',
			r'(\w+[-_]?\w*)\s+service',
			r'(\w+[-_]?\w*)\s+api',
			r'(\w+[-_]?\w*)\s+gateway',
			r'(\w+[-_]?\w*)\s+server'
		]
		
		for pattern in service_patterns:
			match = re.search(pattern, text)
			if match:
				return match.group(1).replace(' ', '-')
		
		# Default service names based on context
		if 'payment' in text:
			return 'payment-service'
		elif 'user' in text:
			return 'user-service'
		elif 'auth' in text:
			return 'auth-service'
		elif 'api' in text:
			return 'api-service'
		
		return 'unknown-service'
	
	def _extract_security_conditions(self, text: str) -> Dict[str, Any]:
		"""Extract security conditions from text."""
		conditions = {}
		
		if 'ip' in text:
			if 'suspicious' in text:
				conditions['source_ip'] = {'type': 'geo_filter', 'filter': 'suspicious_countries'}
			else:
				conditions['source_ip'] = {'type': 'whitelist'}
		
		if 'country' in text or 'countries' in text:
			conditions['geo_location'] = {'type': 'country_filter'}
		
		if 'user' in text:
			conditions['user_authentication'] = {'required': True}
		
		return conditions
	
	async def _generate_mesh_configuration(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate service mesh configuration from analysis."""
		intent = analysis['intent']
		params = analysis['parameters']
		
		base_config = {
			"apiVersion": "apg.mesh.io/v1",
			"kind": "MeshPolicy", 
			"metadata": {
				"name": f"{intent}-policy-{int(datetime.now().timestamp())}",
				"namespace": "default",
				"labels": {
					"created-by": "natural-language-ai",
					"category": analysis['category'].lower().replace(' ', '-')
				}
			}
		}
		
		if intent == "rate_limiting":
			base_config["spec"] = {
				"type": "RateLimitPolicy",
				"target": {"service": params.get('service', 'unknown-service')},
				"limits": {
					f"requests_per_{params.get('unit', 'minute')}": params.get('rate', 100)
				},
				"enforcement": "strict"
			}
		
		elif intent == "auto_scaling":
			base_config["spec"] = {
				"type": "AutoScalingPolicy",
				"target": {"service": params.get('service', 'unknown-service')},
				"scaling": {
					"minReplicas": 1,
					"maxReplicas": params.get('replicas', 10),
					"triggers": []
				}
			}
			
			if params.get('cpu_threshold'):
				base_config["spec"]["scaling"]["triggers"].append({
					"type": "cpu",
					"threshold": f"{params['cpu_threshold']}%"
				})
		
		elif intent == "traffic_routing":
			base_config["spec"] = {
				"type": "TrafficRoutingPolicy",
				"target": {"service": params.get('service', 'unknown-service')},
				"routing": {
					"strategy": "weighted",
					"weights": {
						"v1": 100 - params.get('percentage', 50),
						params.get('target_version', 'v2'): params.get('percentage', 50)
					}
				}
			}
		
		elif intent == "circuit_breaker":
			base_config["spec"] = {
				"type": "CircuitBreakerPolicy",
				"target": {"service": params.get('service', 'unknown-service')},
				"circuitBreaker": {
					"failureThreshold": params.get('failure_threshold', 5),
					"timeoutSeconds": params.get('timeout_seconds', 30),
					"halfOpenRequests": 3
				}
			}
		
		elif intent == "security_policy":
			base_config["spec"] = {
				"type": "SecurityPolicy",
				"target": {"scope": params.get('scope', 'service')},
				"rules": [{
					"action": params.get('action', 'block'),
					"conditions": params.get('condition', {})
				}]
			}
		
		elif intent == "encryption_policy":
			base_config["spec"] = {
				"type": "EncryptionPolicy",
				"target": {"service": params.get('target', 'all-services')},
				"encryption": {
					"protocol": params.get('encryption_type', 'TLS'),
					"certificateValidation": params.get('certificate_validation', 'required')
				}
			}
		
		else:
			base_config["spec"] = {
				"type": "GeneralPolicy",
				"description": params.get('description', 'Custom policy'),
				"rules": []
			}
		
		return base_config
	
	async def _simulate_deployment(self, config: Dict[str, Any]) -> Dict[str, Any]:
		"""Simulate policy deployment."""
		policy_type = config.get('spec', {}).get('type', 'Unknown')
		target_service = config.get('spec', {}).get('target', {}).get('service', 'unknown')
		
		# Simulate validation warnings
		warnings = []
		
		if target_service == 'unknown-service':
			warnings.append("Service name could not be determined - please verify target service")
		
		if policy_type == "RateLimitPolicy":
			limits = config.get('spec', {}).get('limits', {})
			rate_key = list(limits.keys())[0] if limits else None
			if rate_key and limits.get(rate_key, 0) > 10000:
				warnings.append("Very high rate limit - consider if this is intentional")
		
		# Simulate successful deployment
		return {
			"status": "success",
			"message": f"Successfully deployed {policy_type} for {target_service}",
			"warnings": warnings,
			"deployment_time": "2.3 seconds",
			"policy_id": config['metadata']['name']
		}
	
	async def _show_policy_categories(self):
		"""Show overview of policy categories."""
		print(f"\n{'='*60}")
		print("📊 POLICY CATEGORIES OVERVIEW")
		print("="*60)
		
		categories = {}
		for example in self.policy_examples:
			if example.category not in categories:
				categories[example.category] = []
			categories[example.category].append(example)
		
		for category, examples in categories.items():
			print(f"\n🏷️  {category}")
			print("-" * 40)
			for example in examples:
				complexity_icon = {"Simple": "🟢", "Medium": "🟡", "Complex": "🔴"}
				icon = complexity_icon.get(example.complexity, "⚪")
				print(f"  {icon} {example.complexity}: {example.description[:60]}...")
		
		print(f"\n📈 Total Categories: {len(categories)}")
		print(f"📋 Total Examples: {len(self.policy_examples)}")
		
		input("\n[Press Enter to continue...]")


async def main():
	"""Run the natural language policies demo."""
	
	print("🧠 APG Service Mesh - Natural Language Policies Demonstration")
	print("🗣️  Experience the future of policy management!")
	
	demo = NaturalLanguagePolicyDemo()
	
	try:
		await demo.start_demo()
	except KeyboardInterrupt:
		print("\n\n👋 Demo interrupted. Thank you for exploring APG Service Mesh!")
	except Exception as e:
		print(f"\n❌ Demo error: {e}")
	
	print("\n🌟 This was just a preview of APG Service Mesh capabilities!")
	print("🚀 The full implementation includes:")
	print("   • Real AI processing with Ollama models")
	print("   • Advanced policy validation and optimization")
	print("   • Integration with service mesh infrastructure")
	print("   • Voice command support for policy creation")
	print("   • 3D visualization of policy effects")
	print("   • And much more!")
	
	print(f"\n📖 Learn more: https://docs.apg-mesh.io")
	print("🎯 Get started: ./scripts/quick-start.sh")


if __name__ == "__main__":
	asyncio.run(main())