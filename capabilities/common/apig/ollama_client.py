#!/usr/bin/env python3
"""
Ollama AI Integration Client

Production-grade integration with Ollama for local LLM inference.
Provides natural language processing capabilities for policy generation,
traffic analysis, and threat detection.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from urllib.parse import urljoin

# Configure logging
logger = logging.getLogger(__name__)


class OllamaModelStatus(str, Enum):
    """Ollama model status."""
    AVAILABLE = "available"
    LOADING = "loading"
    ERROR = "error"
    NOT_FOUND = "not_found"


@dataclass
class OllamaConfig:
    """Configuration for Ollama service."""
    base_url: str = "http://localhost:11434"
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    model_cache_size: int = 5
    default_model: str = "llama3.2:latest"


@dataclass
class ModelInfo:
    """Information about an available Ollama model."""
    name: str
    size: int
    digest: str
    modified_at: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    template: Optional[str] = None
    system: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    """Request for text generation."""
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = False
    raw: bool = False
    format: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    keep_alive: Optional[str] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""
    model: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class EmbeddingRequest:
    """Request for text embeddings."""
    model: str
    prompt: str
    options: Dict[str, Any] = field(default_factory=dict)
    keep_alive: Optional[str] = None


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""
    embedding: List[float]
    model: str
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None


class OllamaError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Connection-related errors."""
    pass


class OllamaModelError(OllamaError):
    """Model-related errors."""
    pass


class ProductionOllamaClient:
    """
    Production-grade Ollama client for local LLM inference.
    
    Provides robust integration with Ollama service including model management,
    text generation, embeddings, and comprehensive error handling with retries
    and circuit breaker patterns.
    """
    
    def __init__(self, config: OllamaConfig, tenant_id: str):
        """
        Initialize Ollama client.
        
        Args:
            config: Ollama service configuration
            tenant_id: APG tenant identifier
        """
        self.config = config
        self.tenant_id = tenant_id
        self.session: Optional[aiohttp.ClientSession] = None
        self.available_models: Dict[str, ModelInfo] = {}
        self.model_cache: Dict[str, datetime] = {}  # Track loaded models
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        
        # Circuit breaker state
        self.circuit_failures = 0
        self.circuit_opened_at: Optional[datetime] = None
        self.circuit_threshold = 5
        self.circuit_timeout = 60
        
        logger.info(f"Ollama client initialized for tenant {tenant_id}")
    
    async def initialize(self) -> None:
        """Initialize client connection and load model information."""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=30)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            # Test connection and load models
            await self._test_connection()
            await self.refresh_models()
            
            logger.info("Ollama client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {str(e)}")
            raise OllamaConnectionError(f"Initialization failed: {str(e)}")
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using specified model.
        
        Args:
            request: Generation request parameters
            
        Returns:
            GenerationResponse: Generated text and metadata
            
        Raises:
            OllamaModelError: If model is not available
            OllamaConnectionError: If service is unavailable
        """
        if self._is_circuit_open():
            raise OllamaConnectionError("Circuit breaker is open")
        
        start_time = time.perf_counter()
        
        try:
            # Ensure model is loaded
            await self._ensure_model_loaded(request.model)
            
            # Prepare request payload
            payload = {
                'model': request.model,
                'prompt': request.prompt,
                'stream': request.stream
            }
            
            if request.system:
                payload['system'] = request.system
            if request.template:
                payload['template'] = request.template
            if request.context:
                payload['context'] = request.context
            if request.raw:
                payload['raw'] = request.raw
            if request.format:
                payload['format'] = request.format
            if request.options:
                payload['options'] = request.options
            if request.keep_alive:
                payload['keep_alive'] = request.keep_alive
            
            # Make request with retries
            response_data = await self._make_request_with_retries(
                'POST', '/api/generate', payload
            )
            
            # Parse response
            generation_response = GenerationResponse(
                model=response_data['model'],
                response=response_data['response'],
                done=response_data['done'],
                context=response_data.get('context'),
                total_duration=response_data.get('total_duration'),
                load_duration=response_data.get('load_duration'),
                prompt_eval_count=response_data.get('prompt_eval_count'),
                prompt_eval_duration=response_data.get('prompt_eval_duration'),
                eval_count=response_data.get('eval_count'),
                eval_duration=response_data.get('eval_duration'),
                created_at=datetime.now(timezone.utc)
            )
            
            # Update performance metrics
            response_time = time.perf_counter() - start_time
            self._update_metrics(response_time, True)
            
            return generation_response
            
        except Exception as e:
            response_time = time.perf_counter() - start_time
            self._update_metrics(response_time, False)
            
            logger.error(f"Text generation failed: {str(e)}")
            raise OllamaModelError(f"Generation failed: {str(e)}")
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """
        Generate text with streaming response.
        
        Args:
            request: Generation request (stream=True will be set)
            
        Yields:
            str: Streaming text chunks
        """
        if self._is_circuit_open():
            raise OllamaConnectionError("Circuit breaker is open")
        
        request.stream = True
        
        try:
            await self._ensure_model_loaded(request.model)
            
            payload = {
                'model': request.model,
                'prompt': request.prompt,
                'stream': True
            }
            
            if request.system:
                payload['system'] = request.system
            if request.options:
                payload['options'] = request.options
            
            url = urljoin(self.config.base_url, '/api/generate')
            
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    raise OllamaError(f"Request failed with status {response.status}")
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode().strip())
                            if 'response' in data:
                                yield data['response']
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            raise OllamaModelError(f"Streaming failed: {str(e)}")
    
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings for text.
        
        Args:
            request: Embedding request parameters
            
        Returns:
            EmbeddingResponse: Text embeddings and metadata
        """
        if self._is_circuit_open():
            raise OllamaConnectionError("Circuit breaker is open")
        
        try:
            await self._ensure_model_loaded(request.model)
            
            payload = {
                'model': request.model,
                'prompt': request.prompt
            }
            
            if request.options:
                payload['options'] = request.options
            if request.keep_alive:
                payload['keep_alive'] = request.keep_alive
            
            response_data = await self._make_request_with_retries(
                'POST', '/api/embeddings', payload
            )
            
            return EmbeddingResponse(
                embedding=response_data['embedding'],
                model=response_data['model'],
                total_duration=response_data.get('total_duration'),
                load_duration=response_data.get('load_duration'),
                prompt_eval_count=response_data.get('prompt_eval_count')
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise OllamaModelError(f"Embedding failed: {str(e)}")
    
    async def list_models(self) -> List[ModelInfo]:
        """
        Get list of available models.
        
        Returns:
            List of ModelInfo objects
        """
        try:
            response_data = await self._make_request_with_retries('GET', '/api/tags')
            
            models = []
            for model_data in response_data.get('models', []):
                model_info = ModelInfo(
                    name=model_data['name'],
                    size=model_data.get('size', 0),
                    digest=model_data.get('digest', ''),
                    modified_at=datetime.fromisoformat(
                        model_data['modified_at'].replace('Z', '+00:00')
                    ) if model_data.get('modified_at') else datetime.now(timezone.utc),
                    parameters=model_data.get('parameters', {}),
                    template=model_data.get('template'),
                    system=model_data.get('system'),
                    details=model_data.get('details', {})
                )
                models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise OllamaConnectionError(f"Model listing failed: {str(e)}")
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            bool: True if model was pulled successfully
        """
        try:
            payload = {'name': model_name}
            
            # Use longer timeout for model pulling
            url = urljoin(self.config.base_url, '/api/pull')
            timeout = aiohttp.ClientTimeout(total=600)  # 10 minutes
            
            async with self.session.post(url, json=payload, timeout=timeout) as response:
                if response.status == 200:
                    # Stream the pull progress
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode().strip())
                                if data.get('status') == 'success':
                                    logger.info(f"Model {model_name} pulled successfully")
                                    await self.refresh_models()
                                    return True
                                elif 'error' in data:
                                    raise OllamaError(data['error'])
                            except json.JSONDecodeError:
                                continue
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {str(e)}")
            raise OllamaModelError(f"Model pull failed: {str(e)}")
    
    async def show_model(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of model to show
            
        Returns:
            ModelInfo if model exists, None otherwise
        """
        try:
            payload = {'name': model_name}
            response_data = await self._make_request_with_retries(
                'POST', '/api/show', payload
            )
            
            return ModelInfo(
                name=response_data.get('modelfile', model_name),
                size=0,  # Not provided in show response
                digest=response_data.get('digest', ''),
                modified_at=datetime.now(timezone.utc),
                parameters=response_data.get('parameters', {}),
                template=response_data.get('template'),
                system=response_data.get('system'),
                details=response_data.get('details', {})
            )
            
        except Exception as e:
            logger.error(f"Failed to show model {model_name}: {str(e)}")
            return None
    
    async def refresh_models(self) -> None:
        """Refresh cached model information."""
        try:
            models = await self.list_models()
            self.available_models = {model.name: model for model in models}
            
            logger.info(f"Refreshed {len(models)} available models")
            
        except Exception as e:
            logger.error(f"Failed to refresh models: {str(e)}")
    
    async def get_model_status(self, model_name: str) -> OllamaModelStatus:
        """
        Get status of a specific model.
        
        Args:
            model_name: Name of model to check
            
        Returns:
            OllamaModelStatus: Current model status
        """
        try:
            if model_name in self.available_models:
                return OllamaModelStatus.AVAILABLE
            
            # Try to show the model to check if it exists remotely
            model_info = await self.show_model(model_name)
            if model_info:
                return OllamaModelStatus.LOADING
            else:
                return OllamaModelStatus.NOT_FOUND
                
        except Exception as e:
            logger.error(f"Failed to get model status: {str(e)}")
            return OllamaModelStatus.ERROR
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get client performance statistics.
        
        Returns:
            Performance statistics dictionary
        """
        success_rate = (
            self.successful_requests / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        average_response_time = (
            self.total_response_time / self.successful_requests 
            if self.successful_requests > 0 else 0.0
        )
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'average_response_time_ms': average_response_time * 1000,
            'available_models': len(self.available_models),
            'circuit_failures': self.circuit_failures,
            'circuit_open': self._is_circuit_open()
        }
    
    async def close(self) -> None:
        """Close client connection and cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Ollama client connection closed")
    
    # Private helper methods
    
    async def _test_connection(self) -> None:
        """Test connection to Ollama service."""
        try:
            url = urljoin(self.config.base_url, '/api/tags')
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise OllamaConnectionError(f"Connection test failed: {response.status}")
        except Exception as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {str(e)}")
    
    async def _ensure_model_loaded(self, model_name: str) -> None:
        """Ensure model is loaded and available."""
        if model_name not in self.available_models:
            await self.refresh_models()
            
            if model_name not in self.available_models:
                # Try to pull the model
                logger.info(f"Model {model_name} not available, attempting to pull...")
                if not await self.pull_model(model_name):
                    raise OllamaModelError(f"Model {model_name} is not available")
    
    async def _make_request_with_retries(
        self, 
        method: str, 
        endpoint: str, 
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        url = urljoin(self.config.base_url, endpoint)
        
        for attempt in range(self.config.max_retries):
            try:
                if method == 'GET':
                    async with self.session.get(url, params=payload) as response:
                        return await self._handle_response(response)
                else:
                    async with self.session.request(method, url, json=payload) as response:
                        return await self._handle_response(response)
                        
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                else:
                    self.circuit_failures += 1
                    if self.circuit_failures >= self.circuit_threshold:
                        self.circuit_opened_at = datetime.now(timezone.utc)
                    
                    raise OllamaConnectionError(f"Request failed after {self.config.max_retries} attempts: {str(e)}")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response."""
        if response.status == 200:
            self.circuit_failures = 0  # Reset on success
            return await response.json()
        else:
            error_text = await response.text()
            raise OllamaError(f"Request failed with status {response.status}: {error_text}")
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_opened_at is None:
            return False
        
        time_since_opened = (datetime.now(timezone.utc) - self.circuit_opened_at).total_seconds()
        return time_since_opened < self.circuit_timeout
    
    def _update_metrics(self, response_time: float, success: bool) -> None:
        """Update performance metrics."""
        self.total_requests += 1
        self.total_response_time += response_time
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1


# Export main classes
__all__ = [
    'ProductionOllamaClient',
    'OllamaConfig',
    'ModelInfo',
    'GenerationRequest',
    'GenerationResponse',
    'EmbeddingRequest', 
    'EmbeddingResponse',
    'OllamaModelStatus',
    'OllamaError',
    'OllamaConnectionError',
    'OllamaModelError'
]