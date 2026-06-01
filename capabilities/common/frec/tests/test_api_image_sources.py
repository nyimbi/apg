"""FREC API image source handling tests."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from capabilities.common.frec import api
from capabilities.common.frec.views import EnrollmentRequest, IdentificationRequest, VerificationRequest


class _FakeImageResponse:
	def __init__(self, payload: bytes, content_type: str = "image/jpeg") -> None:
		self._payload = payload
		self.headers = {
			"Content-Type": content_type,
			"Content-Length": str(len(payload)),
		}

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		return False

	def read(self, size: int) -> bytes:
		return self._payload[:size]


def test_face_image_from_base64_request():
	payload = b"face-bytes"
	request = EnrollmentRequest(
		user_id="user-1",
		image_data=base64.b64encode(payload).decode("ascii"),
	)

	image = api._face_image_from_request(request)

	assert image.tobytes() == payload


def test_face_image_from_data_url_request():
	payload = b"face-url-bytes"
	request = VerificationRequest(
		user_id="user-1",
		image_url=f"data:image/jpeg;base64,{base64.b64encode(payload).decode('ascii')}",
	)

	image = api._face_image_from_request(request)

	assert image.tobytes() == payload


def test_face_image_from_https_url_with_public_host(monkeypatch):
	payload = b"remote-face-bytes"
	request = IdentificationRequest(image_url="https://images.example.test/face.jpg")

	monkeypatch.setattr(
		api.socket,
		"getaddrinfo",
		lambda hostname, port: [(api.socket.AF_INET, api.socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))],
	)
	monkeypatch.setattr(api, "urlopen", lambda req, timeout: _FakeImageResponse(payload))

	image = api._face_image_from_request(request)

	assert image.tobytes() == payload


def test_face_image_url_blocks_private_hosts(monkeypatch):
	request = SimpleNamespace(image_data=None, image_url="https://internal.example.test/face.jpg")
	monkeypatch.setattr(
		api.socket,
		"getaddrinfo",
		lambda hostname, port: [(api.socket.AF_INET, api.socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))],
	)

	with pytest.raises(ValueError, match="public address"):
		api._face_image_from_request(request)


def test_face_image_url_rejects_unsupported_scheme():
	request = SimpleNamespace(image_data=None, image_url="file:///etc/passwd")

	with pytest.raises(ValueError, match="Unsupported image URL scheme"):
		api._face_image_from_request(request)
