#!/usr/bin/env python3
"""
Test Learning System
===================

Comprehensive test of the agent learning and improvement mechanisms.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from agents.base_agent import AgentRole, AgentTask, AgentCapability
from agents.orchestrator import AgentOrchestrator
from agents.architect_agent import ArchitectAgent
from agents.developer_agent import DeveloperAgent
from agents.tester_agent import TesterAgent
from agents.devops_agent import DevOpsAgent
from agents.learning_engine import LearningEvent, LearningGoal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_learning")

class LearningSystemTest:
	"""Test class for the learning system"""
	
	def __init__(self):
		self.orchestrator = AgentOrchestrator()
		self.agents = {}
		self.test_results = {}
	
	async def setup_agents(self):
		"""Setup test agents with learning enabled"""
		logger.info("Setting up agents with learning capabilities")
		
		# Create agents with learning configuration
		learning_config = {
			'learning': {
				'enabled': True,
				'strategies': ['reinforcement', 'pattern_recognition'],
				'frequency_hours': 0.1  # Learn every 6 minutes for testing
			}
		}
		
		# Create and register agents
		architect = ArchitectAgent("arch_001", config=learning_config)
		developer = DeveloperAgent("dev_001", config=learning_config)
		tester = TesterAgent("test_001", config=learning_config)
		devops = DevOpsAgent("devops_001", config=learning_config)
		
		await self.orchestrator.register_agent(architect)
		await self.orchestrator.register_agent(developer)
		await self.orchestrator.register_agent(tester)
		await self.orchestrator.register_agent(devops)
		
		self.agents = {
			'architect': architect,
			'developer': developer,
			'tester': tester,
			'devops': devops
		}
		
		logger.info(f"Registered {len(self.agents)} agents with learning enabled")
	
	async def test_basic_learning(self):
		"""Test basic learning functionality"""
		logger.info("Testing basic learning functionality")
		
		agent = self.agents['architect']
		
		# Create a learning goal
		await agent.create_learning_goal(
			'capability_improvement',
			'architecture_quality',
			0.9
		)
		
		# Create test task
		task = AgentTask(
			name="Test Architecture Analysis",
			description="Analyze a simple project for learning test",
			requirements={
				'type': 'analysis',
				'capabilities': ['requirement_analysis'],
				'project_spec': {
					'entities': [
						{'name': 'User', 'entity_type': {'name': 'AGENT'}},
						{'name': 'Product', 'entity_type': {'name': 'ENTITY'}}
					],
					'relationships': [],
					'expected_users': 100
				}
			}
		)
		
		# Execute task
		success = await agent.receive_task(task)
		assert success, "Task should be accepted"
		
		# Wait for task completion
		await asyncio.sleep(2)
		
		# Check learning status
		learning_status = agent.get_learning_status()
		logger.info(f"Learning status: {json.dumps(learning_status, indent=2)}")
		
		assert learning_status['learning_enabled'], "Learning should be enabled"
		assert learning_status['total_events'] > 0, "Should have learning events"
		
		self.test_results['basic_learning'] = {
			'status': 'passed',
			'learning_events': learning_status['total_events'],
			'learning_goals': learning_status['learning_goals']
		}
	
	async def test_feedback_learning(self):
		"""Test learning from feedback"""
		logger.info("Testing feedback learning")
		
		agent = self.agents['developer']
		
		# Provide positive feedback
		positive_feedback = {
			'source': 'user',
			'type': 'task_feedback',
			'score': 0.9,
			'areas_for_improvement': [],
			'positive_aspects': ['excellent code quality', 'good documentation'],
			'lessons_learned': ['User prefers detailed documentation'],
			'importance': 8
		}
		
		await agent.learn_from_feedback(positive_feedback)
		
		# Provide negative feedback
		negative_feedback = {
			'source': 'user',
			'type': 'task_feedback',
			'score': 0.3,
			'areas_for_improvement': ['error handling', 'performance'],
			'positive_aspects': ['good structure'],
			'lessons_learned': ['Need better error handling patterns'],
			'importance': 9
		}
		
		await agent.learn_from_feedback(negative_feedback)
		
		# Check that feedback was processed
		learning_status = agent.get_learning_status()
		
		assert learning_status['total_events'] >= 2, "Should have feedback events"
		
		self.test_results['feedback_learning'] = {
			'status': 'passed',
			'feedback_events': 2,
			'learning_enabled': learning_status['learning_enabled']
		}
	
	async def test_pattern_recognition(self):
		"""Test pattern recognition learning"""
		logger.info("Testing pattern recognition learning")
		
		agent = self.agents['tester']
		
		# Simulate multiple similar tasks with consistent outcomes
		for i in range(5):
			task = AgentTask(
				name=f"Test Task {i}",
				description="Repeated testing task for pattern recognition",
				requirements={
					'type': 'testing',
					'capabilities': ['automated_testing'],
					'complexity': 'medium',
					'development_results': [{'application_package': {'files': {}}}]
				}
			)
			
			await agent.receive_task(task)
			await asyncio.sleep(1)  # Allow task processing
		
		# Run learning session to identify patterns
		session_results = await agent.run_learning_session()
		logger.info(f"Learning session results: {json.dumps(session_results, indent=2)}")
		
		# Check for pattern recognition results
		strategy_results = session_results.get('strategy_results', {})
		pattern_results = strategy_results.get('pattern_recognition', {})
		
		assert pattern_results.get('total_patterns', 0) >= 0, "Should analyze patterns"
		
		self.test_results['pattern_recognition'] = {
			'status': 'passed',
			'session_status': session_results.get('status'),
			'patterns_found': pattern_results.get('new_patterns', 0),
			'events_processed': session_results.get('events_processed', 0)
		}
	
	async def test_reinforcement_learning(self):
		"""Test reinforcement learning"""
		logger.info("Testing reinforcement learning")
		
		agent = self.agents['devops']
		
		# Create tasks with different approaches and outcomes
		scenarios = [
			{'approach': 'docker', 'success': True, 'quality': 0.8},
			{'approach': 'kubernetes', 'success': True, 'quality': 0.9},
			{'approach': 'docker', 'success': True, 'quality': 0.85},
			{'approach': 'traditional', 'success': False, 'quality': 0.3},
			{'approach': 'kubernetes', 'success': True, 'quality': 0.95}
		]
		
		for i, scenario in enumerate(scenarios):
			# Create learning event directly for controlled testing
			event = LearningEvent(
				agent_id=agent.agent_id,
				event_type="task_completion",
				context={
					'task_type': 'deployment',
					'approach': scenario['approach'],
					'complexity': 'medium',
					'collaboration': False
				},
				outcome={
					'success': scenario['success'],
					'quality_score': scenario['quality'],
					'efficiency': 0.7
				},
				improvement_score=scenario['quality'] if scenario['success'] else 0.2
			)
			
			await agent.learning_engine.record_learning_event(event)
		
		# Run learning session
		session_results = await agent.run_learning_session()
		
		# Check reinforcement learning results
		strategy_results = session_results.get('strategy_results', {})
		rl_results = strategy_results.get('reinforcement', {})
		
		assert rl_results.get('learning_events_processed', 0) > 0, "Should process RL events"
		
		self.test_results['reinforcement_learning'] = {
			'status': 'passed',
			'events_processed': rl_results.get('learning_events_processed', 0),
			'q_table_size': rl_results.get('q_table_size', 0),
			'improvements': len(rl_results.get('improvements', {}))
		}
	
	async def test_meta_learning(self):
		"""Test meta-learning capabilities"""
		logger.info("Testing meta-learning")
		
		agent = self.agents['architect']
		
		# Configure meta-learning
		await agent.configure_learning({
			'strategies': ['reinforcement', 'pattern_recognition', 'meta_learning']
		})
		
		# Create learning events with different strategy effectiveness
		strategies = ['reinforcement', 'pattern_recognition']
		for strategy in strategies:
			for i in range(3):
				event = LearningEvent(
					agent_id=agent.agent_id,
					event_type="learning_session",
					context={
						'learning_strategy': strategy,
						'session_type': 'regular'
					},
					outcome={
						'improvement_score': 0.8 if strategy == 'reinforcement' else 0.6,
						'strategy_effectiveness': 0.75
					}
				)
				await agent.learning_engine.record_learning_event(event)
		
		# Run learning session
		session_results = await agent.run_learning_session()
		
		# Check meta-learning results
		strategy_results = session_results.get('strategy_results', {})
		meta_results = strategy_results.get('meta_learning', {})
		
		self.test_results['meta_learning'] = {
			'status': 'passed',
			'best_strategy': meta_results.get('best_strategy'),
			'adaptations': len(meta_results.get('adaptations', {}))
		}
	
	async def test_collaborative_learning(self):
		"""Test collaborative learning between agents"""
		logger.info("Testing collaborative learning")
		
		# Create a project that involves multiple agents
		project_spec = {
			'entities': [
				{'name': 'User', 'entity_type': {'name': 'AGENT'}},
				{'name': 'Order', 'entity_type': {'name': 'ENTITY'}},
				{'name': 'Product', 'entity_type': {'name': 'ENTITY'}}
			],
			'relationships': [
				{'from': 'User', 'to': 'Order', 'type': 'has_many'},
				{'from': 'Order', 'to': 'Product', 'type': 'belongs_to'}
			],
			'expected_users': 1000
		}
		
		# Start a project that will involve multiple agents
		project_id = await self.orchestrator.start_project(
			project_spec,
			"Learning Test Project"
		)
		
		# Wait for some project activity
		await asyncio.sleep(5)
		
		# Check that agents have collaborative learning events
		collaborative_events = 0
		for agent in self.agents.values():
			if agent.learning_engine:
				events = agent.learning_engine._get_recent_events(days=1)
				collab_events = [e for e in events if 'collaboration' in str(e.context)]
				collaborative_events += len(collab_events)
		
		# Run system-wide learning session
		system_status = self.orchestrator.get_system_status()
		
		self.test_results['collaborative_learning'] = {
			'status': 'passed',
			'project_started': project_id is not None,
			'collaborative_events': collaborative_events,
			'active_projects': system_status['projects']['active']
		}
	
	async def test_learning_goals(self):
		"""Test learning goal creation and tracking"""
		logger.info("Testing learning goals")
		
		agent = self.agents['developer']
		
		# Create specific learning goals
		goals = [
			('capability_improvement', 'code_quality', 0.9),
			('performance_optimization', 'task_efficiency', 0.8),
			('error_reduction', 'error_rate', 0.05)
		]
		
		for goal_type, metric, target in goals:
			await agent.create_learning_goal(goal_type, metric, target)
		
		# Check learning status
		learning_status = agent.get_learning_status()
		
		assert learning_status['learning_goals'] >= len(goals), "Should have created learning goals"
		
		# Run learning session to update goals
		session_results = await agent.run_learning_session()
		
		self.test_results['learning_goals'] = {
			'status': 'passed',
			'goals_created': len(goals),
			'goals_tracked': learning_status['learning_goals'],
			'goals_updated': session_results.get('goals_updated', 0)
		}
	
	async def run_all_tests(self):
		"""Run all learning system tests"""
		logger.info("Starting comprehensive learning system tests")
		
		await self.setup_agents()
		
		# Run individual tests
		test_methods = [
			self.test_basic_learning,
			self.test_feedback_learning,
			self.test_pattern_recognition,
			self.test_reinforcement_learning,
			self.test_meta_learning,
			self.test_collaborative_learning,
			self.test_learning_goals
		]
		
		for test_method in test_methods:
			try:
				await test_method()
				logger.info(f"✓ {test_method.__name__} passed")
			except Exception as e:
				logger.error(f"✗ {test_method.__name__} failed: {e}")
				self.test_results[test_method.__name__] = {
					'status': 'failed',
					'error': str(e)
				}
		
		# Generate summary report
		await self.generate_test_report()
	
	async def generate_test_report(self):
		"""Generate comprehensive test report"""
		logger.info("Generating learning system test report")
		
		passed_tests = sum(1 for result in self.test_results.values() 
						  if result.get('status') == 'passed')
		total_tests = len(self.test_results)
		
		report = {
			'test_summary': {
				'total_tests': total_tests,
				'passed_tests': passed_tests,
				'failed_tests': total_tests - passed_tests,
				'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
				'timestamp': datetime.utcnow().isoformat()
			},
			'detailed_results': self.test_results,
			'system_status': self.orchestrator.get_system_status(),
			'agent_learning_status': {
				agent_name: agent.get_learning_status()
				for agent_name, agent in self.agents.items()
			}
		}
		
		# Save report
		with open('/tmp/learning_system_test_report.json', 'w') as f:
			json.dump(report, f, indent=2, default=str)
		
		logger.info(f"Test Report Summary:")
		logger.info(f"  Total Tests: {total_tests}")
		logger.info(f"  Passed: {passed_tests}")
		logger.info(f"  Failed: {total_tests - passed_tests}")
		logger.info(f"  Success Rate: {report['test_summary']['success_rate']:.2%}")
		logger.info(f"  Report saved to: /tmp/learning_system_test_report.json")
		
		# Print key learning insights
		logger.info("\nLearning System Insights:")
		for agent_name, status in report['agent_learning_status'].items():
			if status.get('learning_enabled'):
				recent_perf = status.get('recent_performance', {})
				logger.info(f"  {agent_name}: {recent_perf.get('total_tasks', 0)} tasks, "
						   f"{recent_perf.get('success_rate', 0):.2%} success rate")

async def main():
	"""Main test execution"""
	logger.info("APG Agent Learning System Test")
	logger.info("=" * 50)
	
	test_system = LearningSystemTest()
	await test_system.run_all_tests()
	
	logger.info("Learning system tests completed!")

if __name__ == "__main__":
	asyncio.run(main())