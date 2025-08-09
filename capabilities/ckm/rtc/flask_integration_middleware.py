"""
Flask-AppBuilder Integration Middleware for Real-Time Collaboration

This middleware automatically enables page-level collaboration on any Flask-AppBuilder page
with presence tracking, contextual chat, form delegation, and assistance requests.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs
import logging

from flask import Flask, request, session, g, current_app, jsonify
from flask_appbuilder import AppBuilder
from werkzeug.wrappers import Response
from uuid_extensions import uuid7str

from .websocket_manager import websocket_manager
from .service import CollaborationService, CollaborationContext
from .models import RTCPageCollaboration

logger = logging.getLogger(__name__)


class FlaskAppBuilderCollaborationMiddleware:
	"""
	Middleware that automatically enables collaboration on Flask-AppBuilder pages.
	
	Features:
	- Automatic page context detection and room creation
	- Real-time presence tracking on all pages
	- Contextual chat overlay without disrupting existing UI
	- Form field delegation with visual indicators
	- Assistance request integration with page content
	- Mobile-responsive collaboration interface
	"""
	
	def __init__(self, app: Flask = None, appbuilder: AppBuilder = None):
		self.app = app
		self.appbuilder = appbuilder
		self.collaboration_pages: Dict[str, RTCPageCollaboration] = {}
		self.page_contexts: Dict[str, Dict[str, Any]] = {}
		
		if app is not None:
			self.init_app(app, appbuilder)
	
	def init_app(self, app: Flask, appbuilder: AppBuilder):
		"""Initialize middleware with Flask app and AppBuilder"""
		self.app = app
		self.appbuilder = appbuilder
		
		# Register middleware hooks
		app.before_request(self._before_request)
		app.after_request(self._after_request)
		app.teardown_request(self._teardown_request)
		
		# Register collaboration routes
		self._register_collaboration_routes()
		
		# Add collaboration assets
		self._register_assets()
		
		logger.info("Flask-AppBuilder collaboration middleware initialized")
	
	def _before_request(self):
		"""Before request hook to set up collaboration context"""
		# Skip for static files and API endpoints
		if (request.endpoint and 
			(request.endpoint.startswith('static') or 
			 request.endpoint.startswith('api') or
			 request.endpoint.startswith('rtc-ajax'))):
			return
		
		# Extract page context
		page_context = self._extract_page_context()
		if page_context:
			g.collaboration_context = page_context
			g.collaboration_enabled = True
			
			# Track page access
			self._track_page_access(page_context)
		else:
			g.collaboration_enabled = False
	
	def _after_request(self, response: Response) -> Response:
		"""After request hook to inject collaboration UI"""
		# Only inject for HTML responses
		if (not g.get('collaboration_enabled', False) or 
			not response.content_type.startswith('text/html')):
			return response
		
		# Inject collaboration widget into page
		if hasattr(g, 'collaboration_context'):
			modified_content = self._inject_collaboration_widget(
				response.get_data(as_text=True),
				g.collaboration_context
			)
			response.set_data(modified_content)
		
		return response
	
	def _teardown_request(self, error):
		"""Cleanup collaboration context after request"""
		if hasattr(g, 'collaboration_context'):
			# Update last activity timestamp
			context = g.collaboration_context
			if context['page_url'] in self.page_contexts:
				self.page_contexts[context['page_url']]['last_activity'] = datetime.utcnow()
	
	def _extract_page_context(self) -> Optional[Dict[str, Any]]:
		"""Extract collaboration context from current Flask-AppBuilder page"""
		try:
			# Get current user from session
			user_id = session.get('user_id', 'anonymous')
			tenant_id = session.get('tenant_id', 'default')
			
			# Extract page information
			page_url = request.path
			page_title = self._extract_page_title()
			blueprint_name = self._extract_blueprint_name()
			view_name = self._extract_view_name()
			
			# Detect page type
			page_type = self._detect_page_type()
			
			# Extract form context if present
			form_fields = self._extract_form_fields()
			
			return {
				'user_id': user_id,
				'tenant_id': tenant_id,
				'page_url': page_url,
				'page_title': page_title,
				'blueprint_name': blueprint_name,
				'view_name': view_name,
				'page_type': page_type,
				'form_fields': form_fields,
				'timestamp': datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Error extracting page context: {e}")
			return None
	
	def _extract_page_title(self) -> str:
		"""Extract page title from Flask-AppBuilder context"""
		# Try to get from AppBuilder view
		if hasattr(g, 'appbuilder'):
			view = getattr(g.appbuilder, 'current_view', None)
			if view and hasattr(view, 'page_title'):
				return view.page_title
		
		# Fallback to URL-based title
		path_parts = request.path.strip('/').split('/')
		return ' '.join(part.replace('_', ' ').title() for part in path_parts[-2:])
	
	def _extract_blueprint_name(self) -> str:
		"""Extract Flask-AppBuilder blueprint name"""
		if request.blueprint:
			return request.blueprint
		
		# Extract from URL pattern
		path_parts = request.path.strip('/').split('/')
		return path_parts[0] if path_parts else 'default'
	
	def _extract_view_name(self) -> str:
		"""Extract Flask-AppBuilder view name"""
		if request.endpoint:
			# Remove blueprint prefix
			endpoint_parts = request.endpoint.split('.')
			return endpoint_parts[-1] if endpoint_parts else 'index'
		
		# Extract from URL
		path_parts = request.path.strip('/').split('/')
		return path_parts[1] if len(path_parts) > 1 else 'index'
	
	def _detect_page_type(self) -> str:
		"""Detect the type of Flask-AppBuilder page"""
		path = request.path.lower()
		
		if '/add' in path or '/create' in path:
			return 'form_create'
		elif '/edit' in path or '/update' in path:
			return 'form_edit'
		elif '/show' in path or '/view' in path:
			return 'detail_view'
		elif '/list' in path or path.endswith('/'):
			return 'list_view'
		elif '/dashboard' in path:
			return 'dashboard'
		else:
			return 'general'
	
	def _extract_form_fields(self) -> List[Dict[str, Any]]:
		"""Extract form field information for delegation"""
		# This would analyze the response to extract form fields
		# For now, return mock data structure
		return [
			{
				'field_name': 'name',
				'field_type': 'text',
				'label': 'Name',
				'required': True,
				'delegatable': True
			},
			{
				'field_name': 'email',
				'field_type': 'email',
				'label': 'Email Address',
				'required': True,
				'delegatable': True
			}
		]
	
	def _track_page_access(self, context: Dict[str, Any]):
		"""Track page access for collaboration analytics"""
		page_url = context['page_url']
		
		# Update page context cache
		self.page_contexts[page_url] = {
			**context,
			'current_users': self.page_contexts.get(page_url, {}).get('current_users', []),
			'last_activity': datetime.utcnow()
		}
		
		# Add user to current users if not already present
		user_id = context['user_id']
		current_users = self.page_contexts[page_url]['current_users']
		if user_id not in current_users:
			current_users.append(user_id)
		
		logger.debug(f"Page access tracked: {page_url} by {user_id}")
	
	def _inject_collaboration_widget(self, html_content: str, context: Dict[str, Any]) -> str:
		"""Inject collaboration widget into HTML page"""
		# Create collaboration widget HTML
		widget_html = self._generate_collaboration_widget_html(context)
		
		# Inject before closing body tag
		if '</body>' in html_content:
			html_content = html_content.replace('</body>', f"{widget_html}</body>")
		else:
			# Fallback: append at end
			html_content += widget_html
		
		return html_content
	
	def _generate_collaboration_widget_html(self, context: Dict[str, Any]) -> str:
		"""Generate HTML for collaboration widget"""
		page_url = context['page_url']
		current_users = self.page_contexts.get(page_url, {}).get('current_users', [])
		user_count = len(current_users)
		
		return f"""
		<!-- APG Real-Time Collaboration Widget -->
		<div id="rtc-collaboration-widget" class="rtc-widget-container" data-page-url="{page_url}">
			<!-- Presence Indicator -->
			<div class="rtc-presence-indicator" title="{user_count} users collaborating">
				<i class="fa fa-users"></i>
				<span class="rtc-user-count">{user_count}</span>
			</div>
			
			<!-- Chat Toggle -->
			<div class="rtc-chat-toggle" onclick="toggleCollaborationChat()" title="Open collaborative chat">
				<i class="fa fa-comments"></i>
			</div>
			
			<!-- Assistance Request -->
			<div class="rtc-assistance-toggle" onclick="requestAssistance()" title="Request assistance">
				<i class="fa fa-question-circle"></i>
			</div>
			
			<!-- Collaboration Panel (Hidden by default) -->
			<div id="rtc-collaboration-panel" class="rtc-panel hidden">
				<!-- Chat Interface -->
				<div class="rtc-chat-container">
					<div class="rtc-chat-header">
						<h4>Page Collaboration</h4>
						<button onclick="closeCollaborationPanel()" class="rtc-close-btn">&times;</button>
					</div>
					<div class="rtc-chat-messages" id="rtc-chat-messages"></div>
					<div class="rtc-chat-input-container">
						<input type="text" id="rtc-chat-input" placeholder="Type a message..." onkeypress="handleChatKeypress(event)">
						<button onclick="sendChatMessage()" class="rtc-send-btn">
							<i class="fa fa-paper-plane"></i>
						</button>
					</div>
				</div>
				
				<!-- Form Delegation Interface -->
				<div class="rtc-delegation-container">
					<h5>Form Field Delegation</h5>
					<div id="rtc-form-fields"></div>
				</div>
			</div>
		</div>
		
		<!-- Collaboration CSS -->
		<style>
		.rtc-widget-container {{
			position: fixed;
			top: 20px;
			right: 20px;
			z-index: 9999;
			display: flex;
			flex-direction: column;
			gap: 10px;
		}}
		
		.rtc-presence-indicator, .rtc-chat-toggle, .rtc-assistance-toggle {{
			background: #007bff;
			color: white;
			border-radius: 50%;
			width: 48px;
			height: 48px;
			display: flex;
			align-items: center;
			justify-content: center;
			cursor: pointer;
			box-shadow: 0 2px 10px rgba(0,0,0,0.2);
			transition: all 0.3s ease;
		}}
		
		.rtc-presence-indicator:hover, .rtc-chat-toggle:hover, .rtc-assistance-toggle:hover {{
			background: #0056b3;
			transform: scale(1.1);
		}}
		
		.rtc-user-count {{
			font-size: 12px;
			margin-left: 4px;
		}}
		
		.rtc-panel {{
			position: fixed;
			top: 80px;
			right: 20px;
			width: 350px;
			max-height: 600px;
			background: white;
			border-radius: 8px;
			box-shadow: 0 4px 20px rgba(0,0,0,0.3);
			overflow: hidden;
		}}
		
		.rtc-panel.hidden {{
			display: none;
		}}
		
		.rtc-chat-header {{
			background: #007bff;
			color: white;
			padding: 15px;
			display: flex;
			justify-content: space-between;
			align-items: center;
		}}
		
		.rtc-chat-messages {{
			height: 300px;
			overflow-y: auto;
			padding: 15px;
			border-bottom: 1px solid #eee;
		}}
		
		.rtc-chat-input-container {{
			display: flex;
			padding: 15px;
		}}
		
		.rtc-chat-input-container input {{
			flex: 1;
			padding: 8px 12px;
			border: 1px solid #ddd;
			border-radius: 4px;
			margin-right: 10px;
		}}
		
		.rtc-send-btn {{
			background: #007bff;
			color: white;
			border: none;
			border-radius: 4px;
			padding: 8px 12px;
			cursor: pointer;
		}}
		
		.rtc-delegation-container {{
			padding: 15px;
			background: #f8f9fa;
		}}
		
		@media (max-width: 768px) {{
			.rtc-widget-container {{
				top: 10px;
				right: 10px;
			}}
			
			.rtc-panel {{
				width: calc(100vw - 20px);
				right: 10px;
			}}
		}}
		</style>
		
		<!-- Collaboration JavaScript -->
		<script>
		let rtcWebSocket = null;
		let rtcPageUrl = '{page_url}';
		
		// Initialize WebSocket connection
		function initCollaborationWebSocket() {{
			const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
			const host = window.location.host;
			rtcWebSocket = new WebSocket(`${{protocol}}//${{host}}/api/v1/rtc/ws/default/user123?page_url=${{encodeURIComponent(rtcPageUrl)}}`);
			
			rtcWebSocket.onmessage = function(event) {{
				const message = JSON.parse(event.data);
				handleCollaborationMessage(message);
			}};
		}}
		
		// Handle collaboration messages
		function handleCollaborationMessage(message) {{
			if (message.type === 'chat_message') {{
				displayChatMessage(message);
			}} else if (message.type === 'user_join' || message.type === 'user_leave') {{
				updatePresenceCount();
			}}
		}}
		
		// Toggle chat panel
		function toggleCollaborationChat() {{
			const panel = document.getElementById('rtc-collaboration-panel');
			panel.classList.toggle('hidden');
			
			if (!panel.classList.contains('hidden') && !rtcWebSocket) {{
				initCollaborationWebSocket();
			}}
		}}
		
		// Close collaboration panel
		function closeCollaborationPanel() {{
			document.getElementById('rtc-collaboration-panel').classList.add('hidden');
		}}
		
		// Send chat message
		function sendChatMessage() {{
			const input = document.getElementById('rtc-chat-input');
			const message = input.value.trim();
			
			if (message && rtcWebSocket) {{
				rtcWebSocket.send(JSON.stringify({{
					type: 'chat_message',
					message: message,
					page_url: rtcPageUrl,
					timestamp: new Date().toISOString()
				}}));
				
				input.value = '';
			}}
		}}
		
		// Handle chat input keypress
		function handleChatKeypress(event) {{
			if (event.key === 'Enter') {{
				sendChatMessage();
			}}
		}}
		
		// Display chat message
		function displayChatMessage(message) {{
			const chatMessages = document.getElementById('rtc-chat-messages');
			const messageDiv = document.createElement('div');
			messageDiv.className = 'rtc-chat-message';
			messageDiv.innerHTML = `
				<div class="rtc-message-header">
					<strong>${{message.username || message.user_id}}</strong>
					<small>${{new Date(message.timestamp).toLocaleTimeString()}}</small>
				</div>
				<div class="rtc-message-content">${{message.message}}</div>
			`;
			
			chatMessages.appendChild(messageDiv);
			chatMessages.scrollTop = chatMessages.scrollHeight;
		}}
		
		// Request assistance
		function requestAssistance() {{
			const description = prompt('Describe what you need help with:');
			if (description) {{
				fetch('/rtc-ajax/request-assistance/', {{
					method: 'POST',
					headers: {{
						'Content-Type': 'application/json'
					}},
					body: JSON.stringify({{
						page_url: rtcPageUrl,
						description: description
					}})
				}})
				.then(response => response.json())
				.then(data => {{
					if (data.success) {{
						alert('Assistance request sent successfully!');
					}} else {{
						alert('Error sending assistance request: ' + data.error);
					}}
				}});
			}}
		}}
		
		// Update presence count
		function updatePresenceCount() {{
			fetch(`/rtc-ajax/presence/${{encodeURIComponent(rtcPageUrl)}}`)
			.then(response => response.json())
			.then(data => {{
				const countElement = document.querySelector('.rtc-user-count');
				if (countElement) {{
					countElement.textContent = data.users.length;
				}}
			}});
		}}
		
		// Initialize on page load
		document.addEventListener('DOMContentLoaded', function() {{
			updatePresenceCount();
			setInterval(updatePresenceCount, 30000); // Update every 30 seconds
		}});
		</script>
		"""
	
	def _register_collaboration_routes(self):
		"""Register additional collaboration routes"""
		@self.app.route('/rtc/page-context/<path:page_url>')
		def get_page_context(page_url):
			"""Get collaboration context for a specific page"""
			if page_url in self.page_contexts:
				return jsonify(self.page_contexts[page_url])
			else:
				return jsonify({'error': 'Page not found'}), 404
		
		@self.app.route('/rtc/enable-collaboration', methods=['POST'])
		def enable_page_collaboration():
			"""Enable collaboration for current page"""
			data = request.get_json()
			page_url = data.get('page_url', request.referrer)
			
			if page_url:
				# Enable collaboration for page
				context = self._extract_page_context()
				if context:
					self._track_page_access(context)
					return jsonify({'success': True, 'message': 'Collaboration enabled'})
			
			return jsonify({'error': 'Could not enable collaboration'}), 400
	
	def _register_assets(self):
		"""Register collaboration CSS and JavaScript assets"""
		# This would register assets with Flask-AppBuilder's asset management
		# For now, assets are injected inline
		pass
	
	def get_page_statistics(self) -> Dict[str, Any]:
		"""Get collaboration statistics for all pages"""
		return {
			'total_pages': len(self.page_contexts),
			'active_collaborations': len([
				p for p in self.page_contexts.values() 
				if len(p.get('current_users', [])) > 1
			]),
			'total_users': len(set(
				user for page in self.page_contexts.values() 
				for user in page.get('current_users', [])
			)),
			'pages': {
				url: {
					'title': context.get('page_title', 'Unknown'),
					'type': context.get('page_type', 'general'),
					'users': len(context.get('current_users', [])),
					'last_activity': context.get('last_activity', '').isoformat() if context.get('last_activity') else None
				}
				for url, context in self.page_contexts.items()
			}
		}


# Global middleware instance
collaboration_middleware = FlaskAppBuilderCollaborationMiddleware()


def init_collaboration_middleware(app: Flask, appbuilder: AppBuilder):
	"""Initialize collaboration middleware with Flask app"""
	collaboration_middleware.init_app(app, appbuilder)
	return collaboration_middleware