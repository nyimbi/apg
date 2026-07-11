"""Generated app file upload regressions."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from compiler.compiler import compile_apg_string


UPLOAD_SOURCE = """
module upload_probe version 1.0.0 {}

table Profile {
	name: str;
	avatar: file;
}
"""

PNG_BYTES = b"\x89PNG\r\n\x1a\napg-upload-probe"

_UPLOAD_ENV_KEYS = (
	"APG_API_KEY",
	"APG_AUTH_USERS",
	"APG_DATABASE_URL",
	"APG_DATA_FILE",
	"APG_DATA_PATH",
	"APG_DB_PATH",
	"APG_ENV",
	"APG_JWT_SECRET",
	"APG_PRODUCTION",
	"APG_SESSION_SECRET",
	"APG_SQLITE_PATH",
	"APG_UPLOAD_ALLOWED_TYPES",
	"APG_UPLOAD_DIR",
	"APG_UPLOAD_MAX_BYTES",
	"DATABASE_URL",
)


@pytest.fixture()
def generated_upload_app(monkeypatch, tmp_path):
	for key in _UPLOAD_ENV_KEYS:
		monkeypatch.delenv(key, raising=False)
	upload_dir = tmp_path / "uploads"
	monkeypatch.setenv("APG_UPLOAD_DIR", str(upload_dir))
	result = compile_apg_string(UPLOAD_SOURCE)
	assert result.success, result.errors
	namespace: dict[str, object] = {"__file__": str(tmp_path / "generated_upload_app.py")}
	exec(compile(result.generated_files["app.py"], "generated_upload_app.py", "exec"), namespace)
	namespace["_flask_app"].config["TESTING"] = True
	namespace["_upload_dir"] = upload_dir
	return namespace


@pytest.fixture()
def client(generated_upload_app):
	return generated_upload_app["_flask_app"].test_client()


def _upload(client, payload: bytes = PNG_BYTES, *, mime: str = "image/png", filename: str = "avatar.png"):
	return client.post(
		"/records/Profile",
		data={
			"name": "Asha",
			"avatar": (io.BytesIO(payload), filename, mime),
		},
		content_type="multipart/form-data",
	)


def test_file_upload_stores_file(client):
	response = _upload(client)

	assert response.status_code == 201, response.get_json()
	record = response.get_json()["record"]
	stored_path = Path(record["avatar_path"])
	assert stored_path.exists()
	assert stored_path.read_bytes() == PNG_BYTES
	assert record["avatar_mime"] == "image/png"
	assert record["avatar_size"] == len(PNG_BYTES)
	assert record["avatar_url"].startswith("/uploads/Profile/")


def test_upload_rejects_disallowed_mime(client, generated_upload_app):
	response = _upload(client, b"<html>no</html>", mime="text/html", filename="avatar.html")

	assert response.status_code == 415
	assert response.get_json()["error"] == "unsupported_media_type"
	assert not list((generated_upload_app["_upload_dir"] / "Profile").glob("*"))


def test_upload_rejects_oversized(client, monkeypatch):
	monkeypatch.setenv("APG_UPLOAD_MAX_BYTES", "4")

	response = _upload(client, b"12345", mime="image/png")

	assert response.status_code == 413
	assert response.get_json()["error"] == "payload_too_large"


def test_upload_serves_file(client):
	created = _upload(client).get_json()["record"]

	response = client.get(created["avatar_url"])

	assert response.status_code == 200
	assert response.data == PNG_BYTES
	assert response.headers["Content-Type"].startswith("image/png")
	assert response.headers["ETag"]
