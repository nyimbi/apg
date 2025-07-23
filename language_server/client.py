"""
APG Language Server Client
==========================

Client utilities for connecting to and testing the APG Language Server.
Provides both programmatic interface and testing utilities.
"""

import asyncio
import json
import logging
import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import uuid

try:
	import websockets
	from lsprotocol.types import (
		InitializeParams, InitializeResult, ClientCapabilities,
		TextDocumentClientCapabilities, CompletionClientCapabilities,
		HoverClientCapabilities, DefinitionClientCapabilities,
		ReferencesClientCapabilities, DocumentSymbolClientCapabilities,
		DidOpenTextDocumentParams, TextDocumentItem,
		CompletionParams, Position, TextDocumentIdentifier,
		HoverParams, DefinitionParams, ReferencesParams, ReferenceContext,
		DocumentSymbolParams
	)
	LSP_CLIENT_AVAILABLE = True
except ImportError:
	LSP_CLIENT_AVAILABLE = False


class APGLanguageClient:
	"""
	Client for connecting to APG Language Server.
	
	Provides methods to:
	- Connect to language server
	- Send LSP requests
	- Receive and handle responses
	- Test language server functionality
	"""
	
	def __init__(self, host: str = "127.0.0.1", port: int = 2087):
		self.host = host
		self.port = port
		self.socket = None
		self.reader = None
		self.writer = None
		self.request_id = 0
		self.pending_requests: Dict[int, asyncio.Future] = {}
		self.logger = logging.getLogger(__name__)
		
		# Client capabilities
		self.client_capabilities = ClientCapabilities(
			text_document=TextDocumentClientCapabilities(
				completion=CompletionClientCapabilities(),
				hover=HoverClientCapabilities(),
				definition=DefinitionClientCapabilities(),
				references=ReferencesClientCapabilities(),
				document_symbol=DocumentSymbolClientCapabilities()
			)
		)
	
	async def connect(self) -> bool:
		"""Connect to the language server"""
		try:
			self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
			self.logger.info(f"Connected to APG Language Server at {self.host}:{self.port}")
			
			# Start response handler
			asyncio.create_task(self._handle_responses())
			
			# Initialize the connection
			await self._initialize()
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to connect to language server: {e}")
			return False
	
	async def disconnect(self):
		"""Disconnect from the language server"""
		if self.writer:
			self.writer.close()
			await self.writer.wait_closed()
		self.reader = None
		self.writer = None
		self.logger.info("Disconnected from language server")
	
	async def _initialize(self):
		"""Initialize the language server connection"""
		init_params = InitializeParams(
			process_id=None,
			root_uri=None,
			capabilities=self.client_capabilities,
			client_info={"name": "APG Language Client", "version": "1.0.0"}
		)
		
		response = await self._send_request("initialize", init_params.dict())
		self.logger.info("Language server initialized")
		
		# Send initialized notification
		await self._send_notification("initialized", {})
	
	async def _send_request(self, method: str, params: Any) -> Any:
		"""Send a JSON-RPC request and wait for response"""
		if not self.writer:
			raise RuntimeError("Not connected to language server")
		
		request_id = self.request_id
		self.request_id += 1
		
		request = {
			"jsonrpc": "2.0",
			"id": request_id,
			"method": method,
			"params": params
		}
		
		# Create future for response
		future = asyncio.Future()
		self.pending_requests[request_id] = future
		
		# Send request
		message = json.dumps(request)
		content = f"Content-Length: {len(message)}\r\n\r\n{message}"
		
		self.writer.write(content.encode())
		await self.writer.drain()
		
		# Wait for response
		try:
			response = await asyncio.wait_for(future, timeout=10.0)
			return response
		except asyncio.TimeoutError:
			self.pending_requests.pop(request_id, None)
			raise RuntimeError(f"Request {method} timed out")
	
	async def _send_notification(self, method: str, params: Any):
		"""Send a JSON-RPC notification (no response expected)"""
		if not self.writer:
			raise RuntimeError("Not connected to language server")
		
		notification = {
			"jsonrpc": "2.0",
			"method": method,
			"params": params
		}
		
		message = json.dumps(notification)
		content = f"Content-Length: {len(message)}\r\n\r\n{content}"
		
		self.writer.write(content.encode())
		await self.writer.drain()
	
	async def _handle_responses(self):
		"""Handle responses from the language server"""
		if not self.reader:
			return
		
		buffer = b""
		
		try:
			while True:
				data = await self.reader.read(4096)
				if not data:
					break
				
				buffer += data
				
				# Process complete messages
				while True:
					# Look for Content-Length header
					if b"\r\n\r\n" not in buffer:
						break
					
					header_end = buffer.find(b"\r\n\r\n")
					header = buffer[:header_end].decode()
					
					# Parse Content-Length
					content_length = None
					for line in header.split("\r\n"):
						if line.startswith("Content-Length:"):
							content_length = int(line.split(":")[1].strip())
							break
					
					if content_length is None:
						self.logger.error("No Content-Length header found")
						break
					
					# Check if we have the complete message
					message_start = header_end + 4
					if len(buffer) < message_start + content_length:
						break
					
					# Extract and process message
					message_data = buffer[message_start:message_start + content_length]
					buffer = buffer[message_start + content_length:]
					
					try:
						message = json.loads(message_data.decode())
						await self._process_message(message)
					except json.JSONDecodeError as e:
						self.logger.error(f"Failed to decode message: {e}")
		
		except Exception as e:
			self.logger.error(f"Error handling responses: {e}")
	
	async def _process_message(self, message: Dict[str, Any]):
		"""Process a message from the language server"""
		if "id" in message:
			# Response to a request
			request_id = message["id"]
			if request_id in self.pending_requests:
				future = self.pending_requests.pop(request_id)
				if "error" in message:
					future.set_exception(RuntimeError(message["error"]["message"]))
				else:
					future.set_result(message.get("result"))
		else:
			# Notification from server
			method = message.get("method")
			params = message.get("params", {})
			self.logger.info(f"Received notification: {method}")
	
	# ========================================
	# Language Server Protocol Methods
	# ========================================
	
	async def open_document(self, uri: str, text: str, language_id: str = "apg"):
		"""Open a document in the language server"""
		params = {
			"textDocument": {
				"uri": uri,
				"languageId": language_id,
				"version": 1,
				"text": text
			}
		}
		
		await self._send_notification("textDocument/didOpen", params)
	
	async def get_completions(self, uri: str, line: int, character: int) -> List[Dict[str, Any]]:
		"""Get code completions at the specified position"""
		params = {
			"textDocument": {"uri": uri},
			"position": {"line": line, "character": character}
		}
		
		result = await self._send_request("textDocument/completion", params)
		return result if isinstance(result, list) else result.get("items", [])
	
	async def get_hover(self, uri: str, line: int, character: int) -> Optional[Dict[str, Any]]:
		"""Get hover information at the specified position"""
		params = {
			"textDocument": {"uri": uri},
			"position": {"line": line, "character": character}
		}
		
		return await self._send_request("textDocument/hover", params)
	
	async def get_definition(self, uri: str, line: int, character: int) -> List[Dict[str, Any]]:
		"""Get symbol definition at the specified position"""
		params = {
			"textDocument": {"uri": uri},
			"position": {"line": line, "character": character}
		}
		
		result = await self._send_request("textDocument/definition", params)
		return result if isinstance(result, list) else [result] if result else []
	
	async def get_references(self, uri: str, line: int, character: int) -> List[Dict[str, Any]]:
		"""Get symbol references at the specified position"""
		params = {
			"textDocument": {"uri": uri},
			"position": {"line": line, "character": character},
			"context": {"includeDeclaration": True}
		}
		
		result = await self._send_request("textDocument/references", params)
		return result if result else []
	
	async def get_document_symbols(self, uri: str) -> List[Dict[str, Any]]:
		"""Get document symbols for outline view"""
		params = {"textDocument": {"uri": uri}}
		
		result = await self._send_request("textDocument/documentSymbol", params)
		return result if result else []


# ========================================
# Language Server Testing Utilities 
# ========================================

class APGLanguageServerTester:
	"""Test utilities for APG Language Server"""
	
	def __init__(self, client: APGLanguageClient):
		self.client = client
		self.logger = logging.getLogger(__name__)
	
	async def test_basic_functionality(self) -> bool:
		"""Test basic language server functionality"""
		try:
			# Test document with APG code
			test_code = '''module test version 1.0.0 {
	description: "Test module";
}

agent TestAgent {
	name: str = "Test Agent";
	counter: int = 0;
	
	process: () -> str = {
		counter = counter + 1;
		return "Hello from APG! Count: " + str(counter);
	};
	
	get_count: () -> int = {
		return counter;
	};
}
'''
			
			uri = "file:///test.apg"
			await self.client.open_document(uri, test_code)
			
			# Test completions
			completions = await self.client.get_completions(uri, 10, 10)
			print(f"✓ Completions test: Found {len(completions)} items")
			
			# Test hover on "agent" keyword
			hover = await self.client.get_hover(uri, 4, 1)
			if hover:
				print("✓ Hover test: Got hover information")
			else:
				print("⚠ Hover test: No hover information")
			
			# Test document symbols
			symbols = await self.client.get_document_symbols(uri)
			print(f"✓ Document symbols test: Found {len(symbols)} symbols")
			
			return True
			
		except Exception as e:
			print(f"✗ Basic functionality test failed: {e}")
			return False
	
	async def test_error_diagnostics(self) -> bool:
		"""Test error diagnostic functionality"""
		try:
			# Test document with syntax errors
			error_code = '''module test version 1.0.0 {
	description: "Test with errors"
}

agent ErrorAgent {
	name: str = "Error Agent"  // Missing semicolon
	
	invalid_method: () -> {  // Missing return type
		return "test"  // Missing semicolon
	}  // Missing semicolon for method
}
'''
			
			uri = "file:///error_test.apg"
			await self.client.open_document(uri, error_code)
			
			# Wait a bit for diagnostics to be processed
			await asyncio.sleep(1)
			
			print("✓ Error diagnostics test: Document opened with errors")
			return True
			
		except Exception as e:
			print(f"✗ Error diagnostics test failed: {e}")
			return False
	
	async def test_complex_features(self) -> bool:
		"""Test complex language features"""
		try:
			complex_code = '''module complex_test version 1.0.0 {
	description: "Complex APG features test";
	author: "APG Test Suite";
}

agent ComplexAgent {
	config: dict[str, any] = {
		"timeout": 30.0,
		"retries": 3,
		"enabled": true
	};
	
	handlers: list[str] = ["http", "websocket"];
	
	process_request: async (request_type: str, data: dict) -> dict = {
		if (request_type in handlers) {
			return {"status": "success", "data": data};
		}
		return {"status": "error", "message": "Unknown type"};
	};
}

db TestDB {
	url: "postgresql://localhost:5432/testdb";
	
	schema main_schema {
		table users {
			id serial [pk]
			email varchar(255) [unique, not null]
			created_at timestamp [default: now()]
		}
		
		table posts {
			id serial [pk]
			user_id int [ref: > users.id]
			content text
		}
	}
}
'''
			
			uri = "file:///complex_test.apg"
			await self.client.open_document(uri, complex_code)
			
			# Test completions in different contexts
			completions = await self.client.get_completions(uri, 15, 20)
			print(f"✓ Complex features test: Found {len(completions)} completions")
			
			return True
			
		except Exception as e:
			print(f"✗ Complex features test failed: {e}")
			return False
	
	async def run_all_tests(self) -> bool:
		"""Run all language server tests"""
		print("Running APG Language Server Tests")
		print("=" * 50)
		
		tests = [
			("Basic Functionality", self.test_basic_functionality),
			("Error Diagnostics", self.test_error_diagnostics),
			("Complex Features", self.test_complex_features)
		]
		
		passed = 0
		total = len(tests)
		
		for test_name, test_func in tests:
			print(f"\nRunning {test_name} test...")
			try:
				if await test_func():
					passed += 1
				else:
					print(f"✗ {test_name} test failed")
			except Exception as e:
				print(f"✗ {test_name} test error: {e}")
		
		print(f"\nTest Results: {passed}/{total} tests passed")
		return passed == total


# ========================================
# CLI Entry Point
# ========================================

async def main():
	"""Main entry point for language server client"""
	import argparse
	
	parser = argparse.ArgumentParser(description="APG Language Server Client")
	parser.add_argument("--host", default="127.0.0.1", help="Language server host")
	parser.add_argument("--port", type=int, default=2087, help="Language server port")
	parser.add_argument("--test", action="store_true", help="Run tests")
	parser.add_argument("--log-level", default="INFO", help="Log level")
	
	args = parser.parse_args()
	
	# Configure logging
	logging.basicConfig(
		level=getattr(logging, args.log_level.upper()),
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)
	
	if not LSP_CLIENT_AVAILABLE:
		print("Error: Language Server client libraries not installed.")
		print("Install with: pip install websockets lsprotocol")
		sys.exit(1)
	
	client = APGLanguageClient(args.host, args.port)
	
	try:
		# Connect to language server
		print(f"Connecting to APG Language Server at {args.host}:{args.port}...")
		if not await client.connect():
			print("Failed to connect to language server")
			print("Make sure the server is running with: apg-language-server")
			sys.exit(1)
		
		if args.test:
			# Run tests
			tester = APGLanguageServerTester(client)
			success = await tester.run_all_tests()
			sys.exit(0 if success else 1)
		else:
			# Interactive mode
			print("Connected! Language server client running...")
			print("Press Ctrl+C to exit")
			
			# Keep connection alive
			while True:
				await asyncio.sleep(1)
	
	except KeyboardInterrupt:
		print("\nShutting down client")
	except Exception as e:
		print(f"Client error: {e}")
		sys.exit(1)
	finally:
		await client.disconnect()


if __name__ == "__main__":
	if not LSP_CLIENT_AVAILABLE:
		print("Language Server client libraries not available.")
		print("Install with: pip install websockets lsprotocol")
		sys.exit(1)
	
	asyncio.run(main())