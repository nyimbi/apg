"""
APG Employee Data Management - Immersive UI Components

Revolutionary UI components with AI-powered interfaces and
natural language interactions for 10x better user experience.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import json
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, render_template_string
from flask_login import current_user


# ============================================================================
# REVOLUTIONARY UI COMPONENT TEMPLATES
# ============================================================================

IMMERSIVE_LIST_TEMPLATE = """
<div class="immersive-employee-list">
	<div class="ai-powered-header">
		<h2>🚀 {{ title }}</h2>
		<div class="ai-insights-summary">
			<div class="metric">
				<span class="label">Total Employees:</span>
				<span class="value">{{ total_count }}</span>
			</div>
			<div class="metric">
				<span class="label">High Performers:</span>
				<span class="value">{{ high_performers_count }}</span>
			</div>
			<div class="metric">
				<span class="label">Retention Risk:</span>
				<span class="value risk-{{ risk_level }}">{{ retention_risk_count }}</span>
			</div>
		</div>
		<div class="ai-actions">
			<button onclick="bulkAIAnalysis()" class="btn btn-ai">🧠 Bulk AI Analysis</button>
			<button onclick="conversationalSearch()" class="btn btn-chat">💬 Ask AI</button>
			<button onclick="generateInsights()" class="btn btn-insights">📊 Generate Insights</button>
		</div>
	</div>
	
	<div class="intelligent-search-bar">
		<input type="text" id="ai-search" placeholder="Ask me anything... 'Show engineers with Python skills' or 'Who's at risk of leaving?'" />
		<button onclick="processNaturalLanguageQuery()" class="btn btn-primary">🔍 AI Search</button>
	</div>
	
	<div class="employee-grid">
		{% for employee in employees %}
		<div class="employee-card" data-employee-id="{{ employee.employee_id }}">
			<div class="employee-header">
				<img src="{{ employee.photo_url or '/static/img/default_avatar.png' }}" class="employee-avatar" />
				<div class="employee-basic-info">
					<h3>{{ employee.full_name }}</h3>
					<p>{{ employee.position.position_title }} • {{ employee.department.department_name }}</p>
					<span class="status-badge status-{{ employee.employment_status.lower() }}">
						{{ employee.employment_status }}
					</span>
				</div>
				<div class="ai-score-container">
					{% if employee.ai_profile %}
					<div class="ai-confidence-score">
						<div class="score-circle" data-score="{{ employee.ai_profile.confidence_score }}">
							{{ "%.0f"|format(employee.ai_profile.confidence_score * 100) }}%
						</div>
						<span class="score-label">AI Confidence</span>
					</div>
					{% endif %}
				</div>
			</div>
			
			<div class="employee-insights">
				{% if employee.ai_profile %}
				<div class="insight-item">
					<span class="insight-label">Retention Risk:</span>
					<div class="risk-indicator risk-{{ get_risk_level(employee.ai_profile.retention_risk_score) }}">
						{{ "%.0f"|format(employee.ai_profile.retention_risk_score * 100) }}%
					</div>
				</div>
				<div class="insight-item">
					<span class="insight-label">Engagement:</span>
					<div class="engagement-indicator engagement-{{ employee.ai_profile.engagement_level }}">
						{{ employee.ai_profile.engagement_level }}
					</div>
				</div>
				<div class="insight-item">
					<span class="insight-label">Performance:</span>
					<div class="performance-indicator">
						{{ "%.0f"|format(employee.ai_profile.performance_prediction * 100) }}%
					</div>
				</div>
				{% endif %}
			</div>
			
			<div class="quick-actions">
				<button onclick="aiAnalyze('{{ employee.employee_id }}')" class="btn btn-sm btn-ai" title="AI Analysis">
					🧠
				</button>
				<button onclick="startChat('{{ employee.employee_id }}')" class="btn btn-sm btn-chat" title="Chat">
					💬
				</button>
				<button onclick="showInsights('{{ employee.employee_id }}')" class="btn btn-sm btn-insights" title="Insights">
					📊
				</button>
				<button onclick="quickEdit('{{ employee.employee_id }}')" class="btn btn-sm btn-edit" title="Quick Edit">
					✏️
				</button>
			</div>
		</div>
		{% endfor %}
	</div>
</div>

<style>
.immersive-employee-list {
	padding: 20px;
	background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
	border-radius: 12px;
	color: white;
}

.ai-powered-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 20px;
	padding: 20px;
	background: rgba(255, 255, 255, 0.1);
	border-radius: 8px;
	backdrop-filter: blur(10px);
}

.ai-insights-summary {
	display: flex;
	gap: 30px;
}

.metric {
	text-align: center;
}

.metric .label {
	display: block;
	font-size: 0.9em;
	opacity: 0.8;
}

.metric .value {
	display: block;
	font-size: 1.5em;
	font-weight: bold;
}

.intelligent-search-bar {
	display: flex;
	gap: 10px;
	margin-bottom: 20px;
	padding: 15px;
	background: rgba(255, 255, 255, 0.95);
	border-radius: 8px;
	box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.intelligent-search-bar input {
	flex: 1;
	padding: 12px;
	border: none;
	border-radius: 6px;
	font-size: 16px;
	color: #333;
}

.employee-grid {
	display: grid;
	grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
	gap: 20px;
}

.employee-card {
	background: rgba(255, 255, 255, 0.95);
	color: #333;
	border-radius: 12px;
	padding: 20px;
	box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
	transition: transform 0.3s ease, box-shadow 0.3s ease;
	backdrop-filter: blur(10px);
}

.employee-card:hover {
	transform: translateY(-5px);
	box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2);
}

.employee-header {
	display: flex;
	align-items: center;
	gap: 15px;
	margin-bottom: 15px;
}

.employee-avatar {
	width: 60px;
	height: 60px;
	border-radius: 50%;
	border: 3px solid #667eea;
}

.employee-basic-info h3 {
	margin: 0;
	font-size: 1.2em;
	font-weight: 600;
}

.employee-basic-info p {
	margin: 5px 0;
	color: #666;
	font-size: 0.9em;
}

.status-badge {
	padding: 4px 8px;
	border-radius: 12px;
	font-size: 0.8em;
	font-weight: 500;
}

.status-active { background: #e8f5e8; color: #2e7d32; }
.status-inactive { background: #fff3e0; color: #f57c00; }
.status-terminated { background: #ffebee; color: #c62828; }

.ai-score-container {
	margin-left: auto;
}

.score-circle {
	width: 50px;
	height: 50px;
	border-radius: 50%;
	background: conic-gradient(#4caf50 0deg, #4caf50 var(--percentage), #e0e0e0 var(--percentage));
	display: flex;
	align-items: center;
	justify-content: center;
	font-weight: bold;
	font-size: 0.9em;
}

.employee-insights {
	display: grid;
	grid-template-columns: repeat(3, 1fr);
	gap: 10px;
	margin-bottom: 15px;
}

.insight-item {
	text-align: center;
}

.insight-label {
	display: block;
	font-size: 0.8em;
	color: #666;
	margin-bottom: 5px;
}

.risk-indicator, .engagement-indicator, .performance-indicator {
	padding: 6px;
	border-radius: 6px;
	font-weight: 600;
	font-size: 0.9em;
}

.risk-low { background: #e8f5e8; color: #2e7d32; }
.risk-medium { background: #fff3e0; color: #f57c00; }
.risk-high { background: #ffebee; color: #c62828; }
.risk-critical { background: #ffebee; color: #b71c1c; }

.engagement-champion { background: #e8f5e8; color: #2e7d32; }
.engagement-highly_engaged { background: #e3f2fd; color: #1976d2; }
.engagement-engaged { background: #f3e5f5; color: #7b1fa2; }
.engagement-somewhat_engaged { background: #fff3e0; color: #f57c00; }
.engagement-disengaged { background: #ffebee; color: #c62828; }

.performance-indicator {
	background: #e3f2fd;
	color: #1976d2;
}

.quick-actions {
	display: flex;
	gap: 8px;
	justify-content: center;
}

.btn {
	padding: 8px 16px;
	border: none;
	border-radius: 6px;
	cursor: pointer;
	font-weight: 500;
	transition: all 0.2s ease;
}

.btn-ai { background: #667eea; color: white; }
.btn-chat { background: #42a5f5; color: white; }
.btn-insights { background: #66bb6a; color: white; }
.btn-edit { background: #ffa726; color: white; }
.btn-primary { background: #1976d2; color: white; }

.btn:hover {
	transform: translateY(-2px);
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.btn-sm {
	padding: 6px 12px;
	font-size: 0.9em;
}
</style>
"""

CONVERSATIONAL_VIEW_TEMPLATE = """
<div class="conversational-employee-view">
	<div class="employee-header-section">
		<div class="employee-photo-section">
			<img src="{{ employee.photo_url or '/static/img/default_avatar.png' }}" class="large-employee-avatar" />
			<div class="photo-actions">
				<button onclick="updatePhoto('{{ employee.employee_id }}')" class="btn btn-sm">Update Photo</button>
			</div>
		</div>
		
		<div class="employee-basic-details">
			<h1>{{ employee.full_name }}</h1>
			<h2>{{ employee.position.position_title }}</h2>
			<p class="department">{{ employee.department.department_name }}</p>
			
			<div class="contact-info">
				<div class="contact-item">
					<span class="icon">📧</span>
					<span class="value">{{ employee.work_email }}</span>
				</div>
				{% if employee.phone_mobile %}
				<div class="contact-item">
					<span class="icon">📱</span>
					<span class="value">{{ employee.phone_mobile }}</span>
				</div>
				{% endif %}
				<div class="contact-item">
					<span class="icon">📅</span>
					<span class="value">Hired {{ employee.hire_date.strftime('%B %d, %Y') }}</span>
				</div>
			</div>
		</div>
		
		<div class="ai-insights-summary-card">
			{% if employee.ai_profile %}
			<h3>🧠 AI Insights</h3>
			<div class="insight-metrics">
				<div class="metric">
					<span class="label">Retention Risk</span>
					<div class="progress-bar">
						<div class="progress-fill" style="width: {{ employee.ai_profile.retention_risk_score * 100 }}%"></div>
					</div>
					<span class="percentage">{{ "%.0f"|format(employee.ai_profile.retention_risk_score * 100) }}%</span>
				</div>
				<div class="metric">
					<span class="label">Engagement</span>
					<span class="engagement-badge engagement-{{ employee.ai_profile.engagement_level }}">
						{{ employee.ai_profile.engagement_level.replace('_', ' ').title() }}
					</span>
				</div>
				<div class="metric">
					<span class="label">Performance</span>
					<div class="progress-bar">
						<div class="progress-fill success" style="width: {{ employee.ai_profile.performance_prediction * 100 }}%"></div>
					</div>
					<span class="percentage">{{ "%.0f"|format(employee.ai_profile.performance_prediction * 100) }}%</span>
				</div>
			</div>
			{% endif %}
			
			<div class="quick-actions-section">
				<button onclick="aiAnalyze('{{ employee.employee_id }}')" class="btn btn-ai">
					🧠 Full AI Analysis
				</button>
				<button onclick="generateReport('{{ employee.employee_id }}')" class="btn btn-insights">
					📊 Generate Report
				</button>
			</div>
		</div>
	</div>
	
	<div class="conversational-interface-section">
		<div class="chat-container" id="employee-chat-{{ employee.employee_id }}">
			<div class="chat-header">
				<h3>💬 Conversational HR Assistant</h3>
				<div class="chat-status">
					<span class="status-indicator online"></span>
					<span>AI Assistant is ready</span>
				</div>
			</div>
			
			<div class="chat-messages" id="chat-messages-{{ employee.employee_id }}">
				<div class="message assistant-message">
					<div class="message-avatar">🤖</div>
					<div class="message-content">
						<p>Hello! I'm your AI HR assistant. I can help you with information about {{ employee.first_name }}, answer questions about their performance, generate reports, or suggest development plans. What would you like to know?</p>
					</div>
					<div class="message-time">{{ datetime.now().strftime('%H:%M') }}</div>
				</div>
			</div>
			
			<div class="chat-input-area">
				<div class="quick-prompts">
					<button onclick="sendQuickPrompt('{{ employee.employee_id }}', 'analyze_performance')" class="quick-prompt">
						📈 Analyze Performance
					</button>
					<button onclick="sendQuickPrompt('{{ employee.employee_id }}', 'retention_analysis')" class="quick-prompt">
						🔒 Retention Analysis
					</button>
					<button onclick="sendQuickPrompt('{{ employee.employee_id }}', 'development_plan')" class="quick-prompt">
						🎯 Development Plan
					</button>
					<button onclick="sendQuickPrompt('{{ employee.employee_id }}', 'skill_assessment')" class="quick-prompt">
						🎓 Skills Assessment
					</button>
				</div>
				
				<div class="chat-input-container">
					<input 
						type="text" 
						id="chat-input-{{ employee.employee_id }}" 
						placeholder="Ask anything about {{ employee.first_name }}... 'How is their performance?' or 'What development opportunities exist?'"
						onkeypress="handleChatKeypress(event, '{{ employee.employee_id }}')"
					/>
					<button onclick="sendChatMessage('{{ employee.employee_id }}')" class="btn btn-primary">
						Send
					</button>
					<button onclick="startVoiceInput('{{ employee.employee_id }}')" class="btn btn-voice" title="Voice Input">
						🎤
					</button>
				</div>
			</div>
		</div>
	</div>
	
	<div class="employee-details-tabs">
		<div class="tab-navigation">
			<button class="tab-btn active" onclick="showTab('overview')">Overview</button>
			<button class="tab-btn" onclick="showTab('ai-insights')">AI Insights</button>
			<button class="tab-btn" onclick="showTab('performance')">Performance</button>
			<button class="tab-btn" onclick="showTab('development')">Development</button>
			<button class="tab-btn" onclick="showTab('history')">History</button>
		</div>
		
		<div class="tab-content">
			<div id="overview-tab" class="tab-panel active">
				{{ render_overview_panel(employee) }}
			</div>
			
			<div id="ai-insights-tab" class="tab-panel">
				{{ render_ai_insights_panel(employee) }}
			</div>
			
			<div id="performance-tab" class="tab-panel">
				{{ render_performance_panel(employee) }}
			</div>
			
			<div id="development-tab" class="tab-panel">
				{{ render_development_panel(employee) }}
			</div>
			
			<div id="history-tab" class="tab-panel">
				{{ render_history_panel(employee) }}
			</div>
		</div>
	</div>
</div>

<style>
.conversational-employee-view {
	padding: 20px;
	max-width: 1400px;
	margin: 0 auto;
}

.employee-header-section {
	display: grid;
	grid-template-columns: 200px 1fr 350px;
	gap: 30px;
	margin-bottom: 30px;
	padding: 30px;
	background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
	border-radius: 12px;
	color: white;
}

.large-employee-avatar {
	width: 150px;
	height: 150px;
	border-radius: 50%;
	border: 4px solid white;
	box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.employee-basic-details h1 {
	margin: 0 0 10px 0;
	font-size: 2.2em;
	font-weight: 600;
}

.employee-basic-details h2 {
	margin: 0 0 5px 0;
	font-size: 1.3em;
	font-weight: 400;
	opacity: 0.9;
}

.department {
	font-size: 1.1em;
	opacity: 0.8;
	margin-bottom: 20px;
}

.contact-info {
	display: flex;
	flex-direction: column;
	gap: 10px;
}

.contact-item {
	display: flex;
	align-items: center;
	gap: 10px;
}

.contact-item .icon {
	font-size: 1.2em;
}

.ai-insights-summary-card {
	background: rgba(255, 255, 255, 0.1);
	padding: 20px;
	border-radius: 8px;
	backdrop-filter: blur(10px);
}

.ai-insights-summary-card h3 {
	margin: 0 0 15px 0;
	font-size: 1.2em;
}

.insight-metrics {
	display: flex;
	flex-direction: column;
	gap: 15px;
	margin-bottom: 20px;
}

.metric {
	display: flex;
	flex-direction: column;
	gap: 5px;
}

.metric .label {
	font-size: 0.9em;
	opacity: 0.8;
}

.progress-bar {
	height: 8px;
	background: rgba(255, 255, 255, 0.3);
	border-radius: 4px;
	overflow: hidden;
}

.progress-fill {
	height: 100%;
	background: #ff6b6b;
	border-radius: 4px;
	transition: width 0.3s ease;
}

.progress-fill.success {
	background: #4ecdc4;
}

.percentage {
	font-size: 0.9em;
	font-weight: 600;
}

.conversational-interface-section {
	margin-bottom: 30px;
}

.chat-container {
	background: white;
	border-radius: 12px;
	box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
	overflow: hidden;
}

.chat-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	padding: 20px;
	background: #f8f9fa;
	border-bottom: 1px solid #e9ecef;
}

.chat-status {
	display: flex;
	align-items: center;
	gap: 8px;
	font-size: 0.9em;
	color: #666;
}

.status-indicator {
	width: 8px;
	height: 8px;
	border-radius: 50%;
}

.status-indicator.online {
	background: #4caf50;
}

.chat-messages {
	height: 400px;
	overflow-y: auto;
	padding: 20px;
}

.message {
	display: flex;
	gap: 12px;
	margin-bottom: 20px;
}

.message-avatar {
	width: 35px;
	height: 35px;
	border-radius: 50%;
	background: #667eea;
	display: flex;
	align-items: center;
	justify-content: center;
	font-size: 1.2em;
	flex-shrink: 0;
}

.message-content {
	flex: 1;
	background: #f8f9fa;
	padding: 12px 16px;
	border-radius: 12px;
}

.assistant-message .message-content {
	background: #e3f2fd;
}

.message-time {
	font-size: 0.8em;
	color: #666;
	align-self: flex-end;
}

.chat-input-area {
	padding: 20px;
	background: #f8f9fa;
	border-top: 1px solid #e9ecef;
}

.quick-prompts {
	display: flex;
	gap: 10px;
	margin-bottom: 15px;
	flex-wrap: wrap;
}

.quick-prompt {
	padding: 8px 12px;
	background: white;
	border: 1px solid #ddd;
	border-radius: 20px;
	cursor: pointer;
	font-size: 0.9em;
	transition: all 0.2s ease;
}

.quick-prompt:hover {
	background: #667eea;
	color: white;
	border-color: #667eea;
}

.chat-input-container {
	display: flex;
	gap: 10px;
}

.chat-input-container input {
	flex: 1;
	padding: 12px;
	border: 1px solid #ddd;
	border-radius: 6px;
	font-size: 16px;
}

.employee-details-tabs {
	background: white;
	border-radius: 12px;
	box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
	overflow: hidden;
}

.tab-navigation {
	display: flex;
	background: #f8f9fa;
	border-bottom: 1px solid #e9ecef;
}

.tab-btn {
	flex: 1;
	padding: 15px 20px;
	border: none;
	background: transparent;
	cursor: pointer;
	font-weight: 500;
	transition: all 0.2s ease;
}

.tab-btn.active {
	background: white;
	color: #667eea;
	border-bottom: 2px solid #667eea;
}

.tab-btn:hover:not(.active) {
	background: #e9ecef;
}

.tab-content {
	padding: 30px;
}

.tab-panel {
	display: none;
}

.tab-panel.active {
	display: block;
}
</style>

<script>
function showTab(tabName) {
	// Hide all tabs
	document.querySelectorAll('.tab-panel').forEach(panel => {
		panel.classList.remove('active');
	});
	document.querySelectorAll('.tab-btn').forEach(btn => {
		btn.classList.remove('active');
	});
	
	// Show selected tab
	document.getElementById(tabName + '-tab').classList.add('active');
	event.target.classList.add('active');
}

function handleChatKeypress(event, employeeId) {
	if (event.key === 'Enter') {
		sendChatMessage(employeeId);
	}
}

function sendChatMessage(employeeId) {
	const input = document.getElementById(`chat-input-${employeeId}`);
	const message = input.value.trim();
	
	if (!message) return;
	
	// Add user message to chat
	addMessageToChat(employeeId, message, 'user');
	
	// Clear input
	input.value = '';
	
	// Send to backend
	fetch('/employee_management/process_conversation', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			employee_id: employeeId,
			message: message,
			session_id: getOrCreateSessionId(employeeId)
		})
	})
	.then(response => response.json())
	.then(data => {
		addMessageToChat(employeeId, data.text_content, 'assistant');
	})
	.catch(error => {
		console.error('Error:', error);
		addMessageToChat(employeeId, 'Sorry, I encountered an error. Please try again.', 'assistant');
	});
}

function addMessageToChat(employeeId, message, sender) {
	const chatMessages = document.getElementById(`chat-messages-${employeeId}`);
	const messageDiv = document.createElement('div');
	messageDiv.className = `message ${sender}-message`;
	
	const avatar = sender === 'user' ? '👤' : '🤖';
	const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
	
	messageDiv.innerHTML = `
		<div class="message-avatar">${avatar}</div>
		<div class="message-content">
			<p>${message}</p>
		</div>
		<div class="message-time">${time}</div>
	`;
	
	chatMessages.appendChild(messageDiv);
	chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendQuickPrompt(employeeId, promptType) {
	const prompts = {
		'analyze_performance': 'Analyze this employee\'s performance trends and provide insights.',
		'retention_analysis': 'Assess this employee\'s retention risk and suggest interventions.',
		'development_plan': 'Create a personalized development plan for this employee.',
		'skill_assessment': 'Evaluate this employee\'s skills and suggest improvements.'
	};
	
	const message = prompts[promptType];
	if (message) {
		document.getElementById(`chat-input-${employeeId}`).value = message;
		sendChatMessage(employeeId);
	}
}

function getOrCreateSessionId(employeeId) {
	let sessionId = sessionStorage.getItem(`chat-session-${employeeId}`);
	if (!sessionId) {
		sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
		sessionStorage.setItem(`chat-session-${employeeId}`, sessionId);
	}
	return sessionId;
}

// Initialize chat interface
document.addEventListener('DOMContentLoaded', function() {
	// Auto-focus chat input
	const chatInput = document.querySelector('[id^="chat-input-"]');
	if (chatInput) {
		chatInput.focus();
	}
});
</script>
"""

INTELLIGENT_EDIT_TEMPLATE = """
<div class="intelligent-edit-form">
	<div class="form-header">
		<h2>✏️ Intelligent Employee Editor</h2>
		<div class="ai-assistance-indicator">
			<span class="ai-status active">🧠 AI Assistant Active</span>
			<button onclick="toggleAIAssistance()" class="btn btn-sm">Configure AI</button>
		</div>
	</div>
	
	<div class="edit-form-container">
		<div class="form-sections">
			<div class="form-section">
				<h3>👤 Personal Information</h3>
				<div class="form-grid">
					{{ render_field('employee_number', 'Employee Number', 'text', ai_suggestions.employee_number) }}
					{{ render_field('first_name', 'First Name', 'text', ai_suggestions.first_name) }}
					{{ render_field('middle_name', 'Middle Name', 'text', ai_suggestions.middle_name) }}
					{{ render_field('last_name', 'Last Name', 'text', ai_suggestions.last_name) }}
					{{ render_field('preferred_name', 'Preferred Name', 'text', ai_suggestions.preferred_name) }}
				</div>
			</div>
			
			<div class="form-section">
				<h3>📧 Contact Information</h3>
				<div class="form-grid">
					{{ render_field('work_email', 'Work Email', 'email', ai_suggestions.work_email) }}
					{{ render_field('personal_email', 'Personal Email', 'email', ai_suggestions.personal_email) }}
					{{ render_field('phone_mobile', 'Mobile Phone', 'tel', ai_suggestions.phone_mobile) }}
					{{ render_field('phone_work', 'Work Phone', 'tel', ai_suggestions.phone_work) }}
				</div>
			</div>
			
			<div class="form-section">
				<h3>🏢 Employment Details</h3>
				<div class="form-grid">
					{{ render_select_field('department', 'Department', departments, ai_suggestions.department) }}
					{{ render_select_field('position', 'Position', positions, ai_suggestions.position) }}
					{{ render_select_field('manager', 'Manager', managers, ai_suggestions.manager) }}
					{{ render_select_field('employment_status', 'Status', employment_statuses, ai_suggestions.employment_status) }}
					{{ render_select_field('employment_type', 'Type', employment_types, ai_suggestions.employment_type) }}
					{{ render_date_field('hire_date', 'Hire Date', ai_suggestions.hire_date) }}
				</div>
			</div>
			
			<div class="form-section">
				<h3>💰 Compensation</h3>
				<div class="form-grid">
					{{ render_field('base_salary', 'Base Salary', 'number', ai_suggestions.base_salary) }}
					{{ render_field('hourly_rate', 'Hourly Rate', 'number', ai_suggestions.hourly_rate) }}
					{{ render_select_field('currency_code', 'Currency', currencies, ai_suggestions.currency_code) }}
				</div>
			</div>
		</div>
		
		<div class="ai-assistant-panel">
			<div class="ai-panel-header">
				<h3>🧠 AI Assistant</h3>
				<div class="ai-confidence-score">
					<span class="label">Confidence:</span>
					<span class="score">{{ ai_confidence }}%</span>
				</div>
			</div>
			
			<div class="ai-suggestions-list">
				<h4>💡 Smart Suggestions</h4>
				<div id="ai-suggestions">
					{% for suggestion in ai_suggestions.suggestions %}
					<div class="suggestion-item">
						<div class="suggestion-icon">{{ suggestion.icon }}</div>
						<div class="suggestion-content">
							<div class="suggestion-title">{{ suggestion.title }}</div>
							<div class="suggestion-description">{{ suggestion.description }}</div>
						</div>
						<button onclick="applySuggestion('{{ suggestion.field }}', '{{ suggestion.value }}')" class="btn btn-sm btn-apply">
							Apply
						</button>
					</div>
					{% endfor %}
				</div>
			</div>
			
			<div class="data-quality-check">
				<h4>📊 Data Quality</h4>
				<div class="quality-metrics">
					<div class="quality-metric">
						<span class="label">Completeness:</span>
						<div class="progress-bar">
							<div class="progress-fill" style="width: {{ data_quality.completeness }}%"></div>
						</div>
						<span class="percentage">{{ data_quality.completeness }}%</span>
					</div>
					<div class="quality-metric">
						<span class="label">Accuracy:</span>
						<div class="progress-bar">
							<div class="progress-fill" style="width: {{ data_quality.accuracy }}%"></div>
						</div>
						<span class="percentage">{{ data_quality.accuracy }}%</span>
					</div>
					<div class="quality-metric">
						<span class="label">Consistency:</span>
						<div class="progress-bar">
							<div class="progress-fill" style="width: {{ data_quality.consistency }}%"></div>
						</div>
						<span class="percentage">{{ data_quality.consistency }}%</span>
					</div>
				</div>
				
				<button onclick="runQualityCheck()" class="btn btn-quality">
					🔍 Run Quality Check
				</button>
			</div>
			
			<div class="ai-chat-widget">
				<h4>💬 Ask AI Assistant</h4>
				<div class="chat-input-container">
					<input type="text" id="ai-chat-input" placeholder="Ask for help with this form..." />
					<button onclick="sendAIQuestion()" class="btn btn-chat">Send</button>
				</div>
				<div class="ai-response" id="ai-response"></div>
			</div>
		</div>
	</div>
	
	<div class="form-actions">
		<button onclick="saveWithAI()" class="btn btn-primary">💾 Save with AI Validation</button>
		<button onclick="saveDraft()" class="btn btn-secondary">📝 Save Draft</button>
		<button onclick="previewChanges()" class="btn btn-preview">👁️ Preview Changes</button>
		<button onclick="cancelEdit()" class="btn btn-cancel">❌ Cancel</button>
	</div>
</div>

<style>
.intelligent-edit-form {
	max-width: 1400px;
	margin: 0 auto;
	padding: 20px;
}

.form-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 30px;
	padding: 20px;
	background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
	color: white;
	border-radius: 12px;
}

.ai-assistance-indicator {
	display: flex;
	align-items: center;
	gap: 15px;
}

.ai-status {
	padding: 8px 12px;
	background: rgba(255, 255, 255, 0.2);
	border-radius: 20px;
	font-size: 0.9em;
}

.ai-status.active {
	background: rgba(76, 175, 80, 0.3);
}

.edit-form-container {
	display: grid;
	grid-template-columns: 1fr 400px;
	gap: 30px;
}

.form-sections {
	display: flex;
	flex-direction: column;
	gap: 30px;
}

.form-section {
	background: white;
	padding: 25px;
	border-radius: 12px;
	box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.form-section h3 {
	margin: 0 0 20px 0;
	color: #333;
	font-size: 1.2em;
	font-weight: 600;
	border-bottom: 2px solid #667eea;
	padding-bottom: 10px;
}

.form-grid {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
	gap: 20px;
}

.ai-assistant-panel {
	background: white;
	border-radius: 12px;
	box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
	height: fit-content;
	position: sticky;
	top: 20px;
}

.ai-panel-header {
	padding: 20px;
	background: #f8f9fa;
	border-radius: 12px 12px 0 0;
	border-bottom: 1px solid #e9ecef;
	display: flex;
	justify-content: space-between;
	align-items: center;
}

.ai-confidence-score {
	display: flex;
	align-items: center;
	gap: 8px;
	font-size: 0.9em;
}

.ai-suggestions-list {
	padding: 20px;
	border-bottom: 1px solid #e9ecef;
}

.ai-suggestions-list h4 {
	margin: 0 0 15px 0;
	color: #333;
	font-size: 1em;
}

.suggestion-item {
	display: flex;
	align-items: center;
	gap: 12px;
	padding: 12px;
	background: #f8f9fa;
	border-radius: 8px;
	margin-bottom: 10px;
}

.suggestion-icon {
	font-size: 1.2em;
}

.suggestion-content {
	flex: 1;
}

.suggestion-title {
	font-weight: 600;
	font-size: 0.9em;
}

.suggestion-description {
	font-size: 0.8em;
	color: #666;
}

.btn-apply {
	background: #4caf50;
	color: white;
	border: none;
	padding: 6px 12px;
	border-radius: 4px;
	cursor: pointer;
	font-size: 0.8em;
}

.data-quality-check {
	padding: 20px;
	border-bottom: 1px solid #e9ecef;
}

.quality-metrics {
	display: flex;
	flex-direction: column;
	gap: 12px;
	margin-bottom: 15px;
}

.quality-metric {
	display: flex;
	align-items: center;
	gap: 10px;
	font-size: 0.9em;
}

.quality-metric .label {
	min-width: 80px;
	font-size: 0.8em;
	color: #666;
}

.progress-bar {
	flex: 1;
	height: 8px;
	background: #e0e0e0;
	border-radius: 4px;
	overflow: hidden;
}

.progress-fill {
	height: 100%;
	background: #4caf50;
	border-radius: 4px;
}

.percentage {
	min-width: 40px;
	text-align: right;
	font-size: 0.8em;
	font-weight: 600;
}

.ai-chat-widget {
	padding: 20px;
}

.chat-input-container {
	display: flex;
	gap: 8px;
	margin-bottom: 10px;
}

.chat-input-container input {
	flex: 1;
	padding: 8px;
	border: 1px solid #ddd;
	border-radius: 4px;
	font-size: 0.9em;
}

.ai-response {
	padding: 10px;
	background: #f0f8ff;
	border-radius: 6px;
	font-size: 0.9em;
	line-height: 1.4;
	display: none;
}

.form-actions {
	display: flex;
	gap: 15px;
	justify-content: center;
	margin-top: 30px;
	padding: 20px;
	background: #f8f9fa;
	border-radius: 12px;
}

.btn {
	padding: 12px 24px;
	border: none;
	border-radius: 6px;
	cursor: pointer;
	font-weight: 500;
	transition: all 0.2s ease;
}

.btn-primary { background: #667eea; color: white; }
.btn-secondary { background: #6c757d; color: white; }
.btn-preview { background: #17a2b8; color: white; }
.btn-cancel { background: #dc3545; color: white; }
.btn-quality { background: #ffc107; color: #333; }
.btn-chat { background: #007bff; color: white; }

.btn:hover {
	transform: translateY(-2px);
	box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
</style>
"""


# ============================================================================
# UI COMPONENT RENDERER FUNCTIONS
# ============================================================================

class ImmersiveUIRenderer:
	"""Renderer for immersive UI components."""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(f"UIRenderer.{tenant_id}")

	def render_immersive_list(self, employees: List[Any], context: Dict[str, Any]) -> str:
		"""Render immersive employee list."""
		return render_template_string(IMMERSIVE_LIST_TEMPLATE, 
									employees=employees, **context)

	def render_conversational_view(self, employee: Any, context: Dict[str, Any]) -> str:
		"""Render conversational employee view."""
		return render_template_string(CONVERSATIONAL_VIEW_TEMPLATE, 
									employee=employee, **context)

	def render_intelligent_edit_form(self, employee: Any, context: Dict[str, Any]) -> str:
		"""Render intelligent edit form."""
		return render_template_string(INTELLIGENT_EDIT_TEMPLATE, 
									employee=employee, **context)

	def render_ai_insights_widget(self, employee: Any) -> str:
		"""Render AI insights widget."""
		if not employee.ai_profile:
			return '<div class="no-ai-insights">No AI insights available</div>'
		
		return f'''
		<div class="ai-insights-widget">
			<h4>🧠 AI Insights</h4>
			<div class="insights-grid">
				<div class="insight-card retention-risk">
					<div class="insight-icon">🔒</div>
					<div class="insight-content">
						<div class="insight-value">{employee.ai_profile.retention_risk_score:.1%}</div>
						<div class="insight-label">Retention Risk</div>
					</div>
				</div>
				<div class="insight-card engagement">
					<div class="insight-icon">⚡</div>
					<div class="insight-content">
						<div class="insight-value">{employee.ai_profile.engagement_level}</div>
						<div class="insight-label">Engagement</div>
					</div>
				</div>
				<div class="insight-card performance">
					<div class="insight-icon">📈</div>
					<div class="insight-content">
						<div class="insight-value">{employee.ai_profile.performance_prediction:.1%}</div>
						<div class="insight-label">Performance</div>
					</div>
				</div>
			</div>
		</div>
		'''

	def render_quick_actions_widget(self, employee: Any) -> str:
		"""Render quick actions widget."""
		return f'''
		<div class="quick-actions-widget">
			<button onclick="aiAnalyze('{employee.employee_id}')" class="action-btn ai-action">
				<span class="action-icon">🧠</span>
				<span class="action-label">AI Analysis</span>
			</button>
			<button onclick="startChat('{employee.employee_id}')" class="action-btn chat-action">
				<span class="action-icon">💬</span>
				<span class="action-label">Chat</span>
			</button>
			<button onclick="showInsights('{employee.employee_id}')" class="action-btn insights-action">
				<span class="action-icon">📊</span>
				<span class="action-label">Insights</span>
			</button>
			<button onclick="quickEdit('{employee.employee_id}')" class="action-btn edit-action">
				<span class="action-icon">✏️</span>
				<span class="action-label">Quick Edit</span>
			</button>
		</div>
		'''