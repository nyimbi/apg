"""
API Service for APG Workflow Mobile

Handles all HTTP communication with APG backend services.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Union, Callable
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from pathlib import Path

import httpx
from httpx import AsyncClient, Response, TimeoutException, ConnectError

from ..models.api_response import APIResponse, PaginationInfo
from ..models.user import User, AuthToken
from ..utils.constants import API_BASE_URL, API_TIMEOUT, API_RETRY_ATTEMPTS
from ..utils.exceptions import (
	APIException, 
	AuthenticationException, 
	NetworkException,
	ValidationException
)


@dataclass
class RequestConfig:
	"""HTTP request configuration"""
	method: str
	url: str
	headers: Optional[Dict[str, str]] = None
	params: Optional[Dict[str, Any]] = None
	json_data: Optional[Dict[str, Any]] = None
	files: Optional[Dict[str, Any]] = None
	timeout: float = API_TIMEOUT
	retry_attempts: int = API_RETRY_ATTEMPTS
	retry_delay: float = 1.0


class APIService:
	"""Core API service for backend communication"""
	
	def __init__(self, app=None):
		self.app = app
		self.logger = logging.getLogger(__name__)
		
		# HTTP client configuration
		self.base_url = API_BASE_URL
		self.timeout = API_TIMEOUT
		self.retry_attempts = API_RETRY_ATTEMPTS
		
		# Client instance
		self._client: Optional[AsyncClient] = None
		
		# Authentication state
		self._auth_token: Optional[str] = None
		self._refresh_token: Optional[str] = None
		self._token_expires_at: Optional[float] = None
		
		# Request interceptors
		self._request_interceptors: List[Callable] = []
		self._response_interceptors: List[Callable] = []
		
		# Network state
		self._is_online = True
		
		self.logger.info("API Service initialized")
	
	@property
	def client(self) -> AsyncClient:
		"""Get or create HTTP client instance"""
		if self._client is None:
			self._client = AsyncClient(
				base_url=self.base_url,
				timeout=self.timeout,
				headers={
					"Content-Type": "application/json",
					"Accept": "application/json",
					"User-Agent": "APG-Workflow-Mobile/1.0.0",
				}
			)
		return self._client
	
	async def close(self):
		"""Close HTTP client"""
		if self._client:
			await self._client.aclose()
			self._client = None
			self.logger.info("API client closed")
	
	def set_auth_token(self, token: str, refresh_token: Optional[str] = None, 
					   expires_at: Optional[float] = None):
		"""Set authentication token"""
		self._auth_token = token
		self._refresh_token = refresh_token
		self._token_expires_at = expires_at
		self.logger.info("Authentication token set")
	
	def clear_auth_token(self):
		"""Clear authentication token"""
		self._auth_token = None
		self._refresh_token = None
		self._token_expires_at = None
		self.logger.info("Authentication token cleared")
	
	def add_request_interceptor(self, interceptor: Callable):
		"""Add request interceptor"""
		self._request_interceptors.append(interceptor)
	
	def add_response_interceptor(self, interceptor: Callable):
		"""Add response interceptor"""
		self._response_interceptors.append(interceptor)
	
	def set_network_status(self, is_online: bool):
		"""Set network connectivity status"""
		self._is_online = is_online
		if is_online:
			self.logger.info("Network connectivity restored")
		else:
			self.logger.warning("Network connectivity lost")
	
	async def _prepare_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
		"""Prepare request headers"""
		request_headers = {
			"Content-Type": "application/json",
			"Accept": "application/json",
			"User-Agent": "APG-Workflow-Mobile/1.0.0",
		}
		
		# Add custom headers
		if headers:
			request_headers.update(headers)
		
		# Add authentication token
		if self._auth_token:
			request_headers["Authorization"] = f"Bearer {self._auth_token}"
		
		# Add tenant context if available
		if self.app and hasattr(self.app, 'app_state'):
			user = self.app.app_state.current_user
			if user and user.tenant_id:
				request_headers["X-Tenant-ID"] = user.tenant_id
		
		# Add device information
		request_headers["X-Device-Platform"] = "mobile"
		request_headers["X-App-Version"] = "1.0.0"
		
		return request_headers
	
	async def _check_token_expiry(self) -> bool:
		"""Check if token needs refresh"""
		if not self._token_expires_at or not self._refresh_token:
			return False
		
		# Check if token expires in next 5 minutes
		import time
		return time.time() >= (self._token_expires_at - 300)
	
	async def _refresh_auth_token(self) -> bool:
		"""Refresh authentication token"""
		if not self._refresh_token:
			return False
		
		try:
			self.logger.info("Refreshing authentication token")
			
			response = await self.client.post(
				"/auth/refresh",
				json={"refresh_token": self._refresh_token}
			)
			
			if response.status_code == 200:
				data = response.json()
				self.set_auth_token(
					token=data["access_token"],
					refresh_token=data.get("refresh_token", self._refresh_token),
					expires_at=data.get("expires_at")
				)
				self.logger.info("Token refreshed successfully")
				return True
			else:
				self.logger.error(f"Token refresh failed: {response.status_code}")
				self.clear_auth_token()
				return False
				
		except Exception as e:
			self.logger.error(f"Token refresh error: {e}")
			self.clear_auth_token()
			return False
	
	async def _make_request(self, config: RequestConfig) -> Response:
		"""Make HTTP request with retry logic"""
		if not self._is_online:
			raise NetworkException("No network connection available")
		
		# Check token expiry
		if await self._check_token_expiry():
			await self._refresh_auth_token()
		
		# Prepare headers
		headers = await self._prepare_headers(config.headers)
		
		# Apply request interceptors
		for interceptor in self._request_interceptors:
			config = await interceptor(config)
		
		last_exception = None
		
		for attempt in range(config.retry_attempts):
			try:
				self.logger.debug(f"Making {config.method} request to {config.url} (attempt {attempt + 1})")
				
				response = await self.client.request(
					method=config.method,
					url=config.url,
					headers=headers,
					params=config.params,
					json=config.json_data,
					files=config.files,
					timeout=config.timeout
				)
				
				# Apply response interceptors
				for interceptor in self._response_interceptors:
					response = await interceptor(response)
				
				# Handle authentication errors
				if response.status_code == 401:
					self.logger.warning("Authentication failed, clearing token")
					self.clear_auth_token()
					if self.app:
						await self.app.handle_logout()
					raise AuthenticationException("Authentication required")
				
				# Handle other HTTP errors
				if response.status_code >= 400:
					error_data = {}
					try:
						error_data = response.json()
					except:
						pass
					
					error_message = error_data.get("message", f"HTTP {response.status_code}")
					
					if response.status_code == 422:
						raise ValidationException(error_message, error_data)
					else:
						raise APIException(error_message, response.status_code, error_data)
				
				return response
				
			except (TimeoutException, ConnectError) as e:
				last_exception = NetworkException(f"Network error: {e}")
				self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
				
				if attempt < config.retry_attempts - 1:
					await asyncio.sleep(config.retry_delay * (2 ** attempt))
				
			except (AuthenticationException, ValidationException, APIException):
				# Don't retry these exceptions
				raise
				
			except Exception as e:
				last_exception = APIException(f"Unexpected error: {e}")
				self.logger.error(f"Unexpected request error (attempt {attempt + 1}): {e}")
				
				if attempt < config.retry_attempts - 1:
					await asyncio.sleep(config.retry_delay * (2 ** attempt))
		
		raise last_exception or APIException("Max retry attempts exceeded")
	
	async def get(self, url: str, params: Optional[Dict[str, Any]] = None,
				  headers: Optional[Dict[str, str]] = None, 
				  timeout: Optional[float] = None) -> APIResponse:
		"""Make GET request"""
		config = RequestConfig(
			method="GET",
			url=url,
			params=params,
			headers=headers,
			timeout=timeout or self.timeout
		)
		
		response = await self._make_request(config)
		return self._parse_response(response)
	
	async def post(self, url: str, data: Optional[Dict[str, Any]] = None,
				   headers: Optional[Dict[str, str]] = None,
				   timeout: Optional[float] = None) -> APIResponse:
		"""Make POST request"""
		config = RequestConfig(
			method="POST",
			url=url,
			json_data=data,
			headers=headers,
			timeout=timeout or self.timeout
		)
		
		response = await self._make_request(config)
		return self._parse_response(response)
	
	async def put(self, url: str, data: Optional[Dict[str, Any]] = None,
				  headers: Optional[Dict[str, str]] = None,
				  timeout: Optional[float] = None) -> APIResponse:
		"""Make PUT request"""
		config = RequestConfig(
			method="PUT",
			url=url,
			json_data=data,
			headers=headers,
			timeout=timeout or self.timeout
		)
		
		response = await self._make_request(config)
		return self._parse_response(response)
	
	async def patch(self, url: str, data: Optional[Dict[str, Any]] = None,
					headers: Optional[Dict[str, str]] = None,
					timeout: Optional[float] = None) -> APIResponse:
		"""Make PATCH request"""
		config = RequestConfig(
			method="PATCH",
			url=url,
			json_data=data,
			headers=headers,
			timeout=timeout or self.timeout
		)
		
		response = await self._make_request(config)
		return self._parse_response(response)
	
	async def delete(self, url: str, headers: Optional[Dict[str, str]] = None,
					 timeout: Optional[float] = None) -> APIResponse:
		"""Make DELETE request"""
		config = RequestConfig(
			method="DELETE",
			url=url,
			headers=headers,
			timeout=timeout or self.timeout
		)
		
		response = await self._make_request(config)
		return self._parse_response(response)
	
	async def upload_file(self, url: str, file_path: Path, field_name: str = "file",
						  data: Optional[Dict[str, Any]] = None,
						  progress_callback: Optional[Callable[[int], None]] = None) -> APIResponse:
		"""Upload file with progress tracking"""
		if not file_path.exists():
			raise FileNotFoundError(f"File not found: {file_path}")
		
		files = {field_name: open(file_path, 'rb')}
		
		try:
			config = RequestConfig(
				method="POST",
				url=url,
				json_data=data,
				files=files,
				timeout=300  # Longer timeout for file uploads
			)
			
			# TODO: Implement progress callback with httpx
			response = await self._make_request(config)
			return self._parse_response(response)
			
		finally:
			files[field_name].close()
	
	async def download_file(self, url: str, file_path: Path,
						   progress_callback: Optional[Callable[[int], None]] = None) -> bool:
		"""Download file with progress tracking"""
		try:
			async with self.client.stream("GET", url) as response:
				response.raise_for_status()
				
				total_size = int(response.headers.get("content-length", 0))
				downloaded = 0
				
				with open(file_path, 'wb') as f:
					async for chunk in response.aiter_bytes(chunk_size=8192):
						f.write(chunk)
						downloaded += len(chunk)
						
						if progress_callback and total_size > 0:
							progress = int((downloaded / total_size) * 100)
							progress_callback(progress)
				
				return True
				
		except Exception as e:
			self.logger.error(f"File download failed: {e}")
			if file_path.exists():
				file_path.unlink()
			return False
	
	def _parse_response(self, response: Response) -> APIResponse:
		"""Parse HTTP response into APIResponse"""
		try:
			data = response.json()
		except:
			data = {"message": response.text}
		
		# Extract pagination info if present
		pagination = None
		if "pagination" in data:
			pagination_data = data["pagination"]
			pagination = PaginationInfo(
				page=pagination_data.get("page", 1),
				limit=pagination_data.get("limit", 20),
				total=pagination_data.get("total", 0),
				total_pages=pagination_data.get("total_pages", 0)
			)
		
		return APIResponse(
			success=response.status_code < 400,
			status_code=response.status_code,
			message=data.get("message"),
			data=data.get("data", data),
			pagination=pagination,
			headers=dict(response.headers)
		)
	
	async def health_check(self) -> bool:
		"""Check API health"""
		try:
			response = await self.get("/health")
			return response.success
		except Exception as e:
			self.logger.error(f"Health check failed: {e}")
			return False
	
	def get_websocket_url(self, path: str = "") -> str:
		"""Get WebSocket URL for real-time communication"""
		ws_base = self.base_url.replace("http", "ws")
		ws_url = urljoin(ws_base, f"/ws{path}")
		
		# Add authentication token as query parameter
		if self._auth_token:
			separator = "&" if "?" in ws_url else "?"
			ws_url += f"{separator}token={self._auth_token}"
		
		return ws_url