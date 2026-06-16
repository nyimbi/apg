"""OpenAPI connector stub generator for APG.

Reads an OpenAPI 3.x spec and generates a Python APGConnector subclass with:
- Circuit-breaker aware HTTP client
- Per-operation stub methods
- Connector manifest for marketplace registration
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


def _to_py_name(s: str) -> str:
	return re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_") or "op"


def _class_name_from_title(title: str) -> str:
	parts = re.split(r"[\s\-_]+", title)
	return "".join(p.capitalize() for p in parts if p) + "Connector"


def generate_connector_stub(
	spec_path: str,
	output_path: str | None = None,
	class_name: str | None = None,
) -> str:
	"""Generate a Python connector stub from an OpenAPI 3.x spec.

	Args:
		spec_path: Path to OpenAPI YAML or JSON file.
		output_path: Where to write the generated .py file. Defaults to
		             same directory as spec with _connector.py suffix.
		class_name: Class name override. Defaults to title-derived name.

	Returns:
		The generated Python source code.
	"""
	spec_file = Path(spec_path)
	if not spec_file.exists():
		raise FileNotFoundError(f"OpenAPI spec not found: {spec_path}")

	raw = spec_file.read_text(encoding="utf-8")
	if spec_path.endswith((".yaml", ".yml")):
		try:
			import yaml
			spec: dict[str, Any] = yaml.safe_load(raw)
		except ImportError as exc:
			raise ImportError("PyYAML required: pip install PyYAML") from exc
	else:
		spec = json.loads(raw)

	info = spec.get("info", {})
	title: str = info.get("title", spec_file.stem)
	version: str = info.get("version", "1.0.0")
	description: str = info.get("description", "")

	servers = spec.get("servers", [])
	base_url = servers[0].get("url", "") if servers else ""

	if class_name is None:
		class_name = _class_name_from_title(title)

	# Extract operations from paths
	operations: list[dict[str, Any]] = []
	for path_str, path_item in spec.get("paths", {}).items():
		if not isinstance(path_item, dict):
			continue
		for method, op in path_item.items():
			if method.startswith("x-") or not isinstance(op, dict):
				continue
			op_id = op.get("operationId") or f"{method}_{_to_py_name(path_str)}"
			summary = op.get("summary", op.get("description", ""))
			# Path + query parameters
			params = [p["name"] for p in op.get("parameters", []) if isinstance(p, dict) and "name" in p]
			# Request body presence
			has_body = "requestBody" in op
			operations.append({
				"op_id": op_id,
				"py_name": _to_py_name(op_id),
				"method": method.upper(),
				"path": path_str,
				"summary": summary,
				"params": params,
				"has_body": has_body,
			})

	# Build the Python stub — use regular string concatenation to avoid f-string brace confusion
	lines: list[str] = []

	lines += [
		'"""Auto-generated APG connector stub for ' + title + ' ' + version + '.',
		'',
		'Source spec: ' + spec_path,
		'Base URL: ' + base_url,
		'',
		'DO NOT EDIT — regenerate with: apg connector generate --spec ' + spec_path,
		'"""',
		'from __future__ import annotations',
		'',
		'import json as _json',
		'import urllib.request as _urllib_req',
		'import urllib.error as _urllib_err',
		'from typing import Any',
		'',
		'',
		'class ' + class_name + ':',
		'    """' + title + ' — APG connector.',
		'    ',
		'    ' + description,
		'    """',
		'',
		'    BASE_URL: str = ' + repr(base_url),
		'    TIMEOUT: int = 30',
		'    MAX_RETRIES: int = 3',
		'    CIRCUIT_THRESHOLD: int = 5',
		'',
		'    MANIFEST: dict[str, Any] = {',
		'        "name": ' + repr(class_name) + ',',
		'        "title": ' + repr(title) + ',',
		'        "version": ' + repr(version) + ',',
		'        "base_url": ' + repr(base_url) + ',',
		'        "operations": ' + repr([op["op_id"] for op in operations]) + ',',
		'    }',
		'',
		'    def __init__(',
		'        self,',
		'        base_url: str | None = None,',
		'        timeout: int | None = None,',
		'        auth_header: str | None = None,',
		'        api_key: str | None = None,',
		'    ) -> None:',
		'        self._base_url = (base_url or self.BASE_URL).rstrip("/")',
		'        self._timeout = timeout or self.TIMEOUT',
		'        self._auth_header = auth_header or (f"Bearer {api_key}" if api_key else None)',
		'        self._failures = 0',
		'',
		'    def _call(',
		'        self,',
		'        method: str,',
		'        path: str,',
		'        payload: dict[str, Any] | None = None,',
		'        params: dict[str, str] | None = None,',
		'    ) -> dict[str, Any]:',
		'        if self._failures >= self.CIRCUIT_THRESHOLD:',
		'            return {"error": "circuit_open", "failures": self._failures}',
		'        url = self._base_url + path',
		'        if params:',
		'            from urllib.parse import urlencode',
		'            url += "?" + urlencode(params)',
		'        data = _json.dumps(payload).encode() if payload else None',
		'        req = _urllib_req.Request(url, data=data, method=method)',
		'        req.add_header("Content-Type", "application/json")',
		'        req.add_header("Accept", "application/json")',
		'        if self._auth_header:',
		'            req.add_header("Authorization", self._auth_header)',
		'        try:',
		'            with _urllib_req.urlopen(req, timeout=self._timeout) as resp:',
		'                self._failures = 0',
		'                body = resp.read().decode()',
		'                return _json.loads(body) if body else {}',
		'        except _urllib_err.HTTPError as exc:',
		'            self._failures += 1',
		'            return {"error": exc.reason, "status": exc.code}',
		'        except Exception as exc:',
		'            self._failures += 1',
		'            return {"error": str(exc)}',
		'',
		'    def health_check(self) -> bool:',
		'        """Return False if circuit breaker has tripped."""',
		'        return self._failures < self.CIRCUIT_THRESHOLD',
		'',
	]

	for op in operations:
		py_params = ""
		if op["params"]:
			py_params = ", ".join(f'{_to_py_name(p)}: str | None = None' for p in op["params"]) + ", "
		if op["has_body"]:
			py_params += "payload: dict[str, Any] | None = None, "

		lines += [
			'    def ' + op["py_name"] + '(self, ' + py_params + ') -> dict[str, Any]:',
			'        """' + (op["summary"] or op["op_id"]) + '"""',
		]
		# Build path expression with parameter substitution
		path_expr = op["path"]
		for p in op["params"]:
			py_p = _to_py_name(p)
			path_expr = path_expr.replace("{" + p + "}", "{str(" + py_p + " or '')}")
		lines.append('        _path = f' + repr(path_expr).replace("'", '"'))
		if op["has_body"]:
			lines.append('        return self._call("' + op["method"] + '", _path, payload=payload)')
		else:
			lines.append('        return self._call("' + op["method"] + '", _path)')
		lines.append('')

	cn_lower = _to_py_name(class_name.lower())
	cn_upper = _to_py_name(class_name).upper()
	lines += [
		'',
		'# Default instance (configure via env vars)',
		'import os as _os',
		'_default_' + cn_lower + ' = ' + class_name + '(',
		'    base_url=_os.environ.get("APG_' + cn_upper + '_BASE_URL"),',
		'    auth_header=_os.environ.get("APG_' + cn_upper + '_AUTH"),',
		')',
	]

	code = "\n".join(lines)

	if output_path is None:
		output_path = str(spec_file.with_name(spec_file.stem + "_connector.py"))

	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(code, encoding="utf-8")
	return code


def scan_connectors(directory: str | None = None) -> list[dict[str, Any]]:
	"""Scan a directory for *_connector.py files and extract their MANIFESTs.

	Returns a list of connector manifest dicts.
	"""
	import importlib.util

	scan_dir = Path(directory or "connectors")
	if not scan_dir.exists():
		return []

	manifests: list[dict[str, Any]] = []
	for py_file in sorted(scan_dir.rglob("*_connector.py")):
		try:
			spec = importlib.util.spec_from_file_location("_tmp_conn", py_file)
			if spec and spec.loader:
				mod = importlib.util.module_from_spec(spec)
				spec.loader.exec_module(mod)  # type: ignore
				for attr in dir(mod):
					cls = getattr(mod, attr)
					if (isinstance(cls, type) and hasattr(cls, "MANIFEST")
							and isinstance(cls.MANIFEST, dict)):
						manifests.append({**cls.MANIFEST, "file": str(py_file)})
		except Exception:
			pass
	return manifests
