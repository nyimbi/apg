"""
WebRTC Client Manager for APG Real-Time Collaboration

Provides server-side utilities and client-side JavaScript generation for WebRTC functionality.
Handles peer connection management, media handling, and browser compatibility.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class WebRTCClientManager:
	"""Manages WebRTC client-side functionality and browser compatibility"""
	
	def __init__(self):
		self.supported_codecs = {
			"video": ["VP8", "VP9", "H264", "AV1"],
			"audio": ["OPUS", "G722", "PCMU", "PCMA"]
		}
		self.quality_profiles = {
			"sd": {"width": 640, "height": 480, "framerate": 15, "bitrate": 500000},
			"hd": {"width": 1280, "height": 720, "framerate": 30, "bitrate": 1500000},
			"4k": {"width": 3840, "height": 2160, "framerate": 30, "bitrate": 8000000}
		}
	
	def generate_client_javascript(self, config: Dict[str, Any] = None) -> str:
		"""Generate client-side JavaScript for WebRTC functionality"""
		
		if config is None:
			config = {}
			
		ice_servers = config.get("ice_servers", [
			{"urls": "stun:stun.l.google.com:19302"}
		])
		
		quality_profiles = json.dumps(self.quality_profiles)
		ice_servers_json = json.dumps(ice_servers)
		
		return f"""
/**
 * APG Real-Time Collaboration WebRTC Client
 * Handles peer-to-peer video/audio communication and screen sharing
 */

class APGWebRTCClient {{
	constructor(websocketManager, options = {{}}) {{
		this.websocketManager = websocketManager;
		this.options = {{
			autoStartAudio: true,
			autoStartVideo: true,
			qualityProfile: 'hd',
			...options
		}};
		
		// WebRTC configuration
		this.iceServers = {ice_servers_json};
		this.qualityProfiles = {quality_profiles};
		
		// Connection state
		this.localStream = null;
		this.localScreenStream = null;
		this.peerConnections = new Map();
		this.remoteStreams = new Map();
		this.currentCall = null;
		this.isScreenSharing = false;
		this.audioEnabled = this.options.autoStartAudio;
		this.videoEnabled = this.options.autoStartVideo;
		
		// UI elements (will be set by initialization)
		this.localVideoElement = null;
		this.remoteVideoContainer = null;
		this.controlsContainer = null;
		
		// Bind methods
		this.handleWebRTCMessage = this.handleWebRTCMessage.bind(this);
		this.handleIceCandidate = this.handleIceCandidate.bind(this);
		this.handleRemoteStream = this.handleRemoteStream.bind(this);
		
		// Initialize
		this.initialize();
	}}
	
	async initialize() {{
		console.log('Initializing APG WebRTC Client...');
		
		// Check browser compatibility
		if (!this.checkBrowserSupport()) {{
			console.error('WebRTC not supported in this browser');
			return;
		}}
		
		// Register WebSocket message handler
		if (this.websocketManager) {{
			this.websocketManager.addMessageHandler('webrtc_', this.handleWebRTCMessage);
		}}
		
		// Set up UI if elements exist
		this.setupUI();
		
		console.log('APG WebRTC Client initialized');
	}}
	
	checkBrowserSupport() {{
		return !!(navigator.mediaDevices && 
				navigator.mediaDevices.getUserMedia && 
				window.RTCPeerConnection);
	}}
	
	setupUI() {{
		// Find video elements
		this.localVideoElement = document.getElementById('local-video') || 
								document.querySelector('.webrtc-local-video');
		this.remoteVideoContainer = document.getElementById('remote-videos') || 
								   document.querySelector('.webrtc-remote-videos');
		this.controlsContainer = document.getElementById('webrtc-controls') || 
								document.querySelector('.webrtc-controls');
		
		// Create UI if it doesn't exist
		if (!this.localVideoElement || !this.remoteVideoContainer) {{
			this.createDefaultUI();
		}}
		
		// Set up control event listeners
		this.setupControlListeners();
	}}
	
	createDefaultUI() {{
		const container = document.createElement('div');
		container.className = 'apg-webrtc-container';
		container.innerHTML = `
			<div class="webrtc-video-grid">
				<div class="local-video-container">
					<video id="local-video" class="webrtc-local-video" autoplay muted playsinline></video>
					<div class="video-label">You</div>
				</div>
				<div id="remote-videos" class="webrtc-remote-videos"></div>
			</div>
			<div id="webrtc-controls" class="webrtc-controls">
				<button id="toggle-audio" class="webrtc-btn audio-enabled">
					<i class="fas fa-microphone"></i>
				</button>
				<button id="toggle-video" class="webrtc-btn video-enabled">
					<i class="fas fa-video"></i>
				</button>
				<button id="toggle-screen-share" class="webrtc-btn">
					<i class="fas fa-desktop"></i>
				</button>
				<button id="end-call" class="webrtc-btn end-call">
					<i class="fas fa-phone-slash"></i>
				</button>
			</div>
		`;
		
		// Add to page
		const targetContainer = document.querySelector('.collaboration-widget') || 
							   document.body;
		targetContainer.appendChild(container);
		
		// Update references
		this.localVideoElement = document.getElementById('local-video');
		this.remoteVideoContainer = document.getElementById('remote-videos');
		this.controlsContainer = document.getElementById('webrtc-controls');
	}}
	
	setupControlListeners() {{
		if (!this.controlsContainer) return;
		
		// Audio toggle
		const audioBtn = this.controlsContainer.querySelector('#toggle-audio');
		if (audioBtn) {{
			audioBtn.addEventListener('click', () => this.toggleAudio());
		}}
		
		// Video toggle
		const videoBtn = this.controlsContainer.querySelector('#toggle-video');
		if (videoBtn) {{
			videoBtn.addEventListener('click', () => this.toggleVideo());
		}}
		
		// Screen share toggle
		const screenBtn = this.controlsContainer.querySelector('#toggle-screen-share');
		if (screenBtn) {{
			screenBtn.addEventListener('click', () => this.toggleScreenShare());
		}}
		
		// End call
		const endBtn = this.controlsContainer.querySelector('#end-call');
		if (endBtn) {{
			endBtn.addEventListener('click', () => this.endCall());
		}}
	}}
	
	async startCall(callId, sessionId) {{
		try {{
			console.log(`Starting WebRTC call: ${{callId}}`);
			
			// Get user media
			await this.getUserMedia();
			
			// Send call start message
			this.websocketManager.send({{
				type: 'webrtc_call_start',
				call_id: callId,
				session_id: sessionId
			}});
			
			this.currentCall = {{ callId, sessionId }};
			
		}} catch (error) {{
			console.error('Error starting call:', error);
			this.handleError('Failed to start call', error);
		}}
	}}
	
	async joinCall(callId, sessionId) {{
		try {{
			console.log(`Joining WebRTC call: ${{callId}}`);
			
			// Get user media
			await this.getUserMedia();
			
			// Send call join message
			this.websocketManager.send({{
				type: 'webrtc_call_join',
				call_id: callId,
				session_id: sessionId
			}});
			
			this.currentCall = {{ callId, sessionId }};
			
		}} catch (error) {{
			console.error('Error joining call:', error);
			this.handleError('Failed to join call', error);
		}}
	}}
	
	async endCall() {{
		if (!this.currentCall) return;
		
		try {{
			console.log(`Ending WebRTC call: ${{this.currentCall.callId}}`);
			
			// Send call end message
			this.websocketManager.send({{
				type: 'webrtc_call_end',
				call_id: this.currentCall.callId
			}});
			
			// Clean up local resources
			await this.cleanup();
			
		}} catch (error) {{
			console.error('Error ending call:', error);
		}}
	}}
	
	async getUserMedia() {{
		try {{
			const quality = this.qualityProfiles[this.options.qualityProfile];
			
			const constraints = {{
				video: this.videoEnabled ? {{
					width: {{ ideal: quality.width }},
					height: {{ ideal: quality.height }},
					frameRate: {{ ideal: quality.framerate }}
				}} : false,
				audio: this.audioEnabled ? {{
					echoCancellation: true,
					noiseSuppression: true,
					autoGainControl: true
				}} : false
			}};
			
			this.localStream = await navigator.mediaDevices.getUserMedia(constraints);
			
			// Display local video
			if (this.localVideoElement) {{
				this.localVideoElement.srcObject = this.localStream;
			}}
			
			console.log('Got user media successfully');
			
		}} catch (error) {{
			console.error('Error getting user media:', error);
			throw error;
		}}
	}}
	
	async getScreenShare() {{
		try {{
			this.localScreenStream = await navigator.mediaDevices.getDisplayMedia({{
				video: {{
					cursor: 'always'
				}},
				audio: true
			}});
			
			// Handle screen share end
			this.localScreenStream.getVideoTracks()[0].onended = () => {{
				this.stopScreenShare();
			}};
			
			console.log('Got screen share successfully');
			
		}} catch (error) {{
			console.error('Error getting screen share:', error);
			throw error;
		}}
	}}
	
	async createPeerConnection(userId) {{
		const config = {{
			iceServers: this.iceServers,
			iceCandidatePoolSize: 10
		}};
		
		const peerConnection = new RTCPeerConnection(config);
		
		// Add local stream tracks
		if (this.localStream) {{
			this.localStream.getTracks().forEach(track => {{
				peerConnection.addTrack(track, this.localStream);
			}});
		}}
		
		// Handle ICE candidates
		peerConnection.onicecandidate = (event) => {{
			if (event.candidate) {{
				this.websocketManager.send({{
					type: 'webrtc_ice_candidate',
					call_id: this.currentCall?.callId,
					target_user_id: userId,
					candidate: event.candidate
				}});
			}}
		}};
		
		// Handle remote streams
		peerConnection.ontrack = (event) => {{
			console.log('Received remote stream from:', userId);
			this.handleRemoteStream(userId, event.streams[0]);
		}};
		
		// Handle connection state changes
		peerConnection.onconnectionstatechange = () => {{
			console.log(`Peer connection state with ${{userId}}:`, peerConnection.connectionState);
		}};
		
		this.peerConnections.set(userId, peerConnection);
		return peerConnection;
	}}
	
	async createOffer(userId) {{
		try {{
			const peerConnection = await this.createPeerConnection(userId);
			const offer = await peerConnection.createOffer();
			await peerConnection.setLocalDescription(offer);
			
			this.websocketManager.send({{
				type: 'webrtc_offer',
				call_id: this.currentCall?.callId,
				target_user_id: userId,
				offer: offer
			}});
			
		}} catch (error) {{
			console.error('Error creating offer:', error);
		}}
	}}
	
	async handleOffer(fromUserId, offer) {{
		try {{
			const peerConnection = await this.createPeerConnection(fromUserId);
			await peerConnection.setRemoteDescription(offer);
			
			const answer = await peerConnection.createAnswer();
			await peerConnection.setLocalDescription(answer);
			
			this.websocketManager.send({{
				type: 'webrtc_answer',
				call_id: this.currentCall?.callId,
				target_user_id: fromUserId,
				answer: answer
			}});
			
		}} catch (error) {{
			console.error('Error handling offer:', error);
		}}
	}}
	
	async handleAnswer(fromUserId, answer) {{
		try {{
			const peerConnection = this.peerConnections.get(fromUserId);
			if (peerConnection) {{
				await peerConnection.setRemoteDescription(answer);
			}}
			
		}} catch (error) {{
			console.error('Error handling answer:', error);
		}}
	}}
	
	async handleIceCandidate(fromUserId, candidate) {{
		try {{
			const peerConnection = this.peerConnections.get(fromUserId);
			if (peerConnection) {{
				await peerConnection.addIceCandidate(candidate);
			}}
			
		}} catch (error) {{
			console.error('Error handling ICE candidate:', error);
		}}
	}}
	
	handleRemoteStream(userId, stream) {{
		this.remoteStreams.set(userId, stream);
		
		// Create video element for remote stream
		if (this.remoteVideoContainer) {{
			let videoElement = this.remoteVideoContainer.querySelector(`[data-user-id="${{userId}}"]`);
			
			if (!videoElement) {{
				videoElement = document.createElement('div');
				videoElement.className = 'remote-video-container';
				videoElement.setAttribute('data-user-id', userId);
				videoElement.innerHTML = `
					<video class="remote-video" autoplay playsinline></video>
					<div class="video-label">${{userId.substring(0, 8)}}...</div>
				`;
				this.remoteVideoContainer.appendChild(videoElement);
			}}
			
			const video = videoElement.querySelector('video');
			video.srcObject = stream;
		}}
	}}
	
	removeRemoteStream(userId) {{
		this.remoteStreams.delete(userId);
		
		// Remove video element
		if (this.remoteVideoContainer) {{
			const videoElement = this.remoteVideoContainer.querySelector(`[data-user-id="${{userId}}"]`);
			if (videoElement) {{
				videoElement.remove();
			}}
		}}
	}}
	
	async toggleAudio() {{
		if (this.localStream) {{
			const audioTracks = this.localStream.getAudioTracks();
			audioTracks.forEach(track => {{
				track.enabled = !track.enabled;
			}});
			
			this.audioEnabled = audioTracks[0]?.enabled || false;
			
			// Update UI
			const audioBtn = this.controlsContainer?.querySelector('#toggle-audio');
			if (audioBtn) {{
				audioBtn.className = `webrtc-btn ${{this.audioEnabled ? 'audio-enabled' : 'audio-disabled'}}`;
				audioBtn.querySelector('i').className = `fas ${{this.audioEnabled ? 'fa-microphone' : 'fa-microphone-slash'}}`;
			}}
			
			// Notify other participants
			if (this.currentCall) {{
				this.websocketManager.send({{
					type: 'webrtc_media_toggle',
					call_id: this.currentCall.callId,
					media_type: 'audio',
					enabled: this.audioEnabled
				}});
			}}
		}}
	}}
	
	async toggleVideo() {{
		if (this.localStream) {{
			const videoTracks = this.localStream.getVideoTracks();
			videoTracks.forEach(track => {{
				track.enabled = !track.enabled;
			}});
			
			this.videoEnabled = videoTracks[0]?.enabled || false;
			
			// Update UI
			const videoBtn = this.controlsContainer?.querySelector('#toggle-video');
			if (videoBtn) {{
				videoBtn.className = `webrtc-btn ${{this.videoEnabled ? 'video-enabled' : 'video-disabled'}}`;
				videoBtn.querySelector('i').className = `fas ${{this.videoEnabled ? 'fa-video' : 'fa-video-slash'}}`;
			}}
			
			// Notify other participants
			if (this.currentCall) {{
				this.websocketManager.send({{
					type: 'webrtc_media_toggle',
					call_id: this.currentCall.callId,
					media_type: 'video',
					enabled: this.videoEnabled
				}});
			}}
		}}
	}}
	
	async toggleScreenShare() {{
		if (this.isScreenSharing) {{
			await this.stopScreenShare();
		}} else {{
			await this.startScreenShare();
		}}
	}}
	
	async startScreenShare() {{
		try {{
			await this.getScreenShare();
			
			// Replace video track in all peer connections
			const videoTrack = this.localScreenStream.getVideoTracks()[0];
			
			for (const [userId, peerConnection] of this.peerConnections) {{
				const sender = peerConnection.getSenders().find(s => 
					s.track && s.track.kind === 'video'
				);
				
				if (sender) {{
					await sender.replaceTrack(videoTrack);
				}}
			}}
			
			// Update local video display
			if (this.localVideoElement) {{
				this.localVideoElement.srcObject = this.localScreenStream;
			}}
			
			this.isScreenSharing = true;
			
			// Update UI
			const screenBtn = this.controlsContainer?.querySelector('#toggle-screen-share');
			if (screenBtn) {{
				screenBtn.className = 'webrtc-btn screen-sharing';
				screenBtn.querySelector('i').className = 'fas fa-stop';
			}}
			
			// Notify other participants
			if (this.currentCall) {{
				this.websocketManager.send({{
					type: 'webrtc_screen_share_start',
					call_id: this.currentCall.callId,
					share_type: 'screen'
				}});
			}}
			
		}} catch (error) {{
			console.error('Error starting screen share:', error);
		}}
	}}
	
	async stopScreenShare() {{
		if (this.localScreenStream) {{
			this.localScreenStream.getTracks().forEach(track => track.stop());
			this.localScreenStream = null;
		}}
		
		// Restore camera video
		if (this.localStream) {{
			const videoTrack = this.localStream.getVideoTracks()[0];
			
			for (const [userId, peerConnection] of this.peerConnections) {{
				const sender = peerConnection.getSenders().find(s => 
					s.track && s.track.kind === 'video'
				);
				
				if (sender && videoTrack) {{
					await sender.replaceTrack(videoTrack);
				}}
			}}
			
			// Update local video display
			if (this.localVideoElement) {{
				this.localVideoElement.srcObject = this.localStream;
			}}
		}}
		
		this.isScreenSharing = false;
		
		// Update UI
		const screenBtn = this.controlsContainer?.querySelector('#toggle-screen-share');
		if (screenBtn) {{
			screenBtn.className = 'webrtc-btn';
			screenBtn.querySelector('i').className = 'fas fa-desktop';
		}}
		
		// Notify other participants
		if (this.currentCall) {{
			this.websocketManager.send({{
				type: 'webrtc_screen_share_stop',
				call_id: this.currentCall.callId
			}});
		}}
	}}
	
	async handleWebRTCMessage(message) {{
		try {{
			switch (message.type) {{
				case 'webrtc_call_started':
					console.log('Call started by:', message.host_user_id);
					break;
					
				case 'webrtc_peer_joined':
					console.log('Peer joined:', message.user_id);
					await this.createOffer(message.user_id);
					break;
					
				case 'webrtc_peer_left':
					console.log('Peer left:', message.user_id);
					this.removePeer(message.user_id);
					break;
					
				case 'webrtc_call_ended':
					console.log('Call ended by:', message.ended_by);
					await this.cleanup();
					break;
					
				case 'webrtc_offer':
					await this.handleOffer(message.from_user_id, message.offer);
					break;
					
				case 'webrtc_answer':
					await this.handleAnswer(message.from_user_id, message.answer);
					break;
					
				case 'webrtc_ice_candidate':
					await this.handleIceCandidate(message.from_user_id, message.candidate);
					break;
					
				case 'webrtc_media_toggle':
					this.handleRemoteMediaToggle(message.user_id, message.media_type, message.enabled);
					break;
					
				case 'webrtc_screen_share_started':
					this.handleRemoteScreenShareStart(message.user_id);
					break;
					
				case 'webrtc_screen_share_stopped':
					this.handleRemoteScreenShareStop(message.user_id);
					break;
					
				default:
					console.log('Unknown WebRTC message type:', message.type);
			}}
		}} catch (error) {{
			console.error('Error handling WebRTC message:', error);
		}}
	}}
	
	handleRemoteMediaToggle(userId, mediaType, enabled) {{
		console.log(`Remote ${{mediaType}} toggle:`, userId, enabled);
		
		// Update UI indicators if needed
		const videoElement = this.remoteVideoContainer?.querySelector(`[data-user-id="${{userId}}"]`);
		if (videoElement) {{
			if (mediaType === 'video') {{
				videoElement.classList.toggle('video-disabled', !enabled);
			}} else if (mediaType === 'audio') {{
				videoElement.classList.toggle('audio-disabled', !enabled);
			}}
		}}
	}}
	
	handleRemoteScreenShareStart(userId) {{
		console.log('Remote screen share started by:', userId);
		// Could show notification or update UI
	}}
	
	handleRemoteScreenShareStop(userId) {{
		console.log('Remote screen share stopped by:', userId);
		// Could show notification or update UI
	}}
	
	removePeer(userId) {{
		// Close peer connection
		const peerConnection = this.peerConnections.get(userId);
		if (peerConnection) {{
			peerConnection.close();
			this.peerConnections.delete(userId);
		}}
		
		// Remove remote stream
		this.removeRemoteStream(userId);
	}}
	
	async cleanup() {{
		// Stop local streams
		if (this.localStream) {{
			this.localStream.getTracks().forEach(track => track.stop());
			this.localStream = null;
		}}
		
		if (this.localScreenStream) {{
			this.localScreenStream.getTracks().forEach(track => track.stop());
			this.localScreenStream = null;
		}}
		
		// Close all peer connections
		for (const [userId, peerConnection] of this.peerConnections) {{
			peerConnection.close();
		}}
		
		this.peerConnections.clear();
		this.remoteStreams.clear();
		this.currentCall = null;
		this.isScreenSharing = false;
		
		// Clear UI
		if (this.localVideoElement) {{
			this.localVideoElement.srcObject = null;
		}}
		
		if (this.remoteVideoContainer) {{
			this.remoteVideoContainer.innerHTML = '';
		}}
		
		console.log('WebRTC cleanup completed');
	}}
	
	handleError(message, error) {{
		console.error(message, error);
		
		// Could show user-friendly error notification
		if (typeof showNotification === 'function') {{
			showNotification(message, 'error');
		}}
	}}
	
	// Public API methods
	getConnectionStats() {{
		const stats = {{
			currentCall: this.currentCall,
			localStream: !!this.localStream,
			screenSharing: this.isScreenSharing,
			audioEnabled: this.audioEnabled,
			videoEnabled: this.videoEnabled,
			peerCount: this.peerConnections.size,
			remoteStreamCount: this.remoteStreams.size
		}};
		
		return stats;
	}}
	
	setQualityProfile(profile) {{
		if (this.qualityProfiles[profile]) {{
			this.options.qualityProfile = profile;
			console.log('Quality profile set to:', profile);
		}}
	}}
}}

// Export for global use
window.APGWebRTCClient = APGWebRTCClient;

// Auto-initialize if websocket manager is available
if (typeof window.websocketManager !== 'undefined') {{
	window.webrtcClient = new APGWebRTCClient(window.websocketManager);
}}
"""
	
	def generate_webrtc_css(self) -> str:
		"""Generate CSS styles for WebRTC UI components"""
		return """
/* APG WebRTC Client Styles */
.apg-webrtc-container {
	position: relative;
	width: 100%;
	height: 500px;
	background: #1a1a1a;
	border-radius: 8px;
	overflow: hidden;
	font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}

.webrtc-video-grid {
	display: grid;
	grid-template-columns: 1fr 2fr;
	height: calc(100% - 60px);
	gap: 8px;
	padding: 8px;
}

.local-video-container {
	position: relative;
	background: #2a2a2a;
	border-radius: 8px;
	overflow: hidden;
}

.webrtc-local-video {
	width: 100%;
	height: 100%;
	object-fit: cover;
}

.webrtc-remote-videos {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
	gap: 8px;
	overflow-y: auto;
}

.remote-video-container {
	position: relative;
	background: #2a2a2a;
	border-radius: 8px;
	overflow: hidden;
	aspect-ratio: 16/9;
}

.remote-video {
	width: 100%;
	height: 100%;
	object-fit: cover;
}

.video-label {
	position: absolute;
	bottom: 8px;
	left: 8px;
	background: rgba(0, 0, 0, 0.7);
	color: white;
	padding: 4px 8px;
	border-radius: 4px;
	font-size: 12px;
	font-weight: 500;
}

.webrtc-controls {
	display: flex;
	justify-content: center;
	align-items: center;
	height: 60px;
	background: #333;
	gap: 12px;
	padding: 0 20px;
}

.webrtc-btn {
	width: 44px;
	height: 44px;
	border: none;
	border-radius: 50%;
	background: #4a4a4a;
	color: white;
	cursor: pointer;
	display: flex;
	align-items: center;
	justify-content: center;
	transition: all 0.2s ease;
	font-size: 16px;
}

.webrtc-btn:hover {
	background: #5a5a5a;
	transform: scale(1.05);
}

.webrtc-btn.audio-enabled {
	background: #28a745;
}

.webrtc-btn.audio-disabled {
	background: #dc3545;
}

.webrtc-btn.video-enabled {
	background: #28a745;
}

.webrtc-btn.video-disabled {
	background: #dc3545;
}

.webrtc-btn.screen-sharing {
	background: #007bff;
}

.webrtc-btn.end-call {
	background: #dc3545;
}

.webrtc-btn.end-call:hover {
	background: #c82333;
}

/* Mobile responsive */
@media (max-width: 768px) {
	.webrtc-video-grid {
		grid-template-columns: 1fr;
		grid-template-rows: 200px 1fr;
	}
	
	.webrtc-remote-videos {
		grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
	}
	
	.webrtc-controls {
		padding: 0 10px;
		gap: 8px;
	}
	
	.webrtc-btn {
		width: 40px;
		height: 40px;
		font-size: 14px;
	}
}

/* Accessibility */
.webrtc-btn:focus {
	outline: 2px solid #007bff;
	outline-offset: 2px;
}

/* Connection state indicators */
.remote-video-container.video-disabled::after {
	content: "📹";
	position: absolute;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);
	font-size: 24px;
	opacity: 0.7;
}

.remote-video-container.audio-disabled::before {
	content: "🔇";
	position: absolute;
	top: 8px;
	right: 8px;
	font-size: 16px;
	z-index: 1;
}

/* Loading states */
.webrtc-loading {
	display: flex;
	align-items: center;
	justify-content: center;
	color: white;
	font-size: 14px;
}

.webrtc-loading::after {
	content: "";
	width: 20px;
	height: 20px;
	border: 2px solid #333;
	border-top: 2px solid #007bff;
	border-radius: 50%;
	animation: spin 1s linear infinite;
	margin-left: 8px;
}

@keyframes spin {
	0% { transform: rotate(0deg); }
	100% { transform: rotate(360deg); }
}
"""

	def get_browser_compatibility_info(self) -> Dict[str, Any]:
		"""Get browser compatibility information"""
		return {
			"supported_browsers": {
				"chrome": {"min_version": 60, "webrtc_support": "full"},
				"firefox": {"min_version": 60, "webrtc_support": "full"},
				"safari": {"min_version": 11, "webrtc_support": "partial"},
				"edge": {"min_version": 79, "webrtc_support": "full"},
				"opera": {"min_version": 47, "webrtc_support": "full"}
			},
			"features": {
				"getUserMedia": "Required for camera/microphone access",
				"getDisplayMedia": "Required for screen sharing",
				"RTCPeerConnection": "Required for peer-to-peer connections",
				"RTCDataChannel": "Required for file transfer"
			},
			"limitations": {
				"mobile_browsers": "Limited codec support on some mobile browsers",
				"ios_safari": "Screen sharing not supported on iOS Safari",
				"old_browsers": "Polyfills may be required for older browser versions"
			}
		}


# Global client manager instance
webrtc_client_manager = WebRTCClientManager()


if __name__ == "__main__":
	# Test the client manager
	print("Testing WebRTC client manager...")
	
	# Generate JavaScript
	js_code = webrtc_client_manager.generate_client_javascript()
	print(f"Generated JavaScript code: {len(js_code)} characters")
	
	# Generate CSS
	css_code = webrtc_client_manager.generate_webrtc_css()
	print(f"Generated CSS code: {len(css_code)} characters")
	
	# Get compatibility info
	compat_info = webrtc_client_manager.get_browser_compatibility_info()
	print(f"Browser compatibility info: {len(compat_info['supported_browsers'])} browsers")
	
	print("✅ WebRTC client manager test completed")