import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER = REPO_ROOT / "language_server" / "server.py"


class LSPClient:
	def __init__(self):
		self.proc = subprocess.Popen(
			[sys.executable, str(SERVER)],
			cwd=REPO_ROOT,
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
		)
		self.next_id = 1

	def close(self):
		if self.proc.poll() is None:
			try:
				self.request("shutdown", {})
				self.notify("exit", {})
			except Exception:
				self.proc.terminate()
			try:
				self.proc.wait(timeout=5)
			except subprocess.TimeoutExpired:
				self.proc.kill()
				self.proc.wait(timeout=5)

	def send(self, payload):
		body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
		self.proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
		self.proc.stdin.flush()

	def request(self, method, params):
		request_id = self.next_id
		self.next_id += 1
		self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
		while True:
			message = self.read_message()
			if message.get("id") == request_id:
				return message

	def notify(self, method, params):
		self.send({"jsonrpc": "2.0", "method": method, "params": params})

	def read_message(self, timeout=10):
		deadline = time.monotonic() + timeout
		header_bytes = self._read_until(b"\r\n\r\n", deadline)
		headers = {}
		for line in header_bytes.decode("ascii").split("\r\n"):
			if not line:
				continue
			name, _, value = line.partition(":")
			headers[name.lower()] = value.strip()
		length = int(headers["content-length"])
		body = self._read_exact(length, deadline)
		return json.loads(body.decode("utf-8"))

	def _read_until(self, marker, deadline):
		data = getattr(self, "_pushback", b"")
		self._pushback = b""
		while marker not in data:
			data += self._read_available(deadline)
		before, _, after = data.partition(marker)
		if after:
			self._pushback = getattr(self, "_pushback", b"") + after
		return before

	def _read_exact(self, length, deadline):
		data = getattr(self, "_pushback", b"")
		self._pushback = b""
		while len(data) < length:
			data += self._read_available(deadline)
		if len(data) > length:
			self._pushback = data[length:]
			data = data[:length]
		return data

	def _read_available(self, deadline):
		remaining = deadline - time.monotonic()
		if remaining <= 0:
			stderr = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.poll() is not None else ""
			raise TimeoutError(f"Timed out waiting for LSP output. stderr={stderr}")
		ready, _, _ = select.select([self.proc.stdout.fileno()], [], [], remaining)
		if not ready:
			return b""
		return os.read(self.proc.stdout.fileno(), 4096)

	def read_notification(self, method, timeout=10):
		deadline = time.monotonic() + timeout
		while time.monotonic() < deadline:
			message = self.read_message(timeout=deadline - time.monotonic())
			if message.get("method") == method:
				return message
		raise TimeoutError(f"Timed out waiting for notification {method}")


def initialize(client):
	return client.request("initialize", {
		"processId": None,
		"rootUri": REPO_ROOT.as_uri(),
		"capabilities": {},
	})


def open_doc(client, uri, text):
	client.notify("textDocument/didOpen", {
		"textDocument": {
			"uri": uri,
			"languageId": "apg",
			"version": 1,
			"text": text,
		}
	})


def test_lsp_initialize():
	client = LSPClient()
	try:
		response = initialize(client)
		capabilities = response["result"]["capabilities"]

		assert capabilities["textDocumentSync"] == 2
		assert capabilities["completionProvider"]["triggerCharacters"] == [" ", ":"]
		assert capabilities["hoverProvider"] is True
		assert capabilities["diagnosticProvider"] is True
		assert capabilities["definitionProvider"] is True
	finally:
		client.close()


def test_lsp_completion_types(tmp_path):
	client = LSPClient()
	try:
		initialize(client)
		source = "module sample version 1.0.0 {\n}\n\nentity Customer {\n  name: \n}\n"
		uri = (tmp_path / "customer.apg").as_uri()
		open_doc(client, uri, source)

		line = source.splitlines().index("  name: ")
		position = {"line": line, "character": len("  name: ")}
		response = client.request("textDocument/completion", {
			"textDocument": {"uri": uri},
			"position": position,
		})
		labels = {item["label"] for item in response["result"]}

		assert {"str", "int", "float", "bool", "date", "datetime", "text", "uuid", "json", "file"}.issubset(labels)
	finally:
		client.close()


def test_lsp_diagnostics_on_error(tmp_path):
	client = LSPClient()
	try:
		initialize(client)
		uri = (tmp_path / "bad.apg").as_uri()
		open_doc(client, uri, "invalid_entity Broken {\n")

		notification = client.read_notification("textDocument/publishDiagnostics")
		diagnostics = notification["params"]["diagnostics"]

		assert notification["params"]["uri"] == uri
		assert diagnostics
		assert any("Unknown entity declaration" in item["message"] or "Unclosed brace" in item["message"] for item in diagnostics)
	finally:
		client.close()


def test_lsp_hover_type(tmp_path):
	client = LSPClient()
	try:
		initialize(client)
		source = "module sample version 1.0.0 {\n}\n\nentity Customer {\n  name: str;\n}\n"
		uri = (tmp_path / "customer.apg").as_uri()
		open_doc(client, uri, source)

		line = source.splitlines().index("  name: str;")
		position = {"line": line, "character": len("  name: s")}
		response = client.request("textDocument/hover", {
			"textDocument": {"uri": uri},
			"position": position,
		})
		value = response["result"]["contents"]["value"]

		assert response["result"]["contents"]["kind"] == "markdown"
		assert "**str**" in value
		assert "String value" in value
		assert "name: str;" in value
	finally:
		client.close()
