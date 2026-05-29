"""APG packaging profiles layered on generated Python applications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compiler import APGCompiler, CodeGenConfig, CompilationResult
from .release import build_release_report


SUPPORTED_PACKAGE_TARGETS = ("python", "web", "desktop", "mobile", "container")


def build_package_report(
	source_file: Path,
	target: str = "web",
	out_dir: Path | None = None,
	catalog: Path | None = None,
) -> dict[str, Any]:
	"""Compile, verify, and package an APG application profile."""
	profile = _normalize_package_target(target)
	output_root = Path(out_dir or "dist")
	report: dict[str, Any] = {
		"format": "apg.package-report.v1",
		"ok": False,
		"source": str(source_file),
		"target": profile,
		"catalog": str(catalog) if catalog is not None else None,
		"output_dir": "",
		"manifest_path": "",
		"release": {},
		"manifest": {},
		"files": [],
		"checks": {},
		"errors": [],
		"warnings": [],
	}

	if profile not in SUPPORTED_PACKAGE_TARGETS:
		report["errors"].append(
			f"Unsupported package target {target!r}. Supported targets: {', '.join(SUPPORTED_PACKAGE_TARGETS)}"
		)
		return report

	release_report = build_release_report(source_file, target="python", catalog=catalog)
	report["release"] = _release_summary(release_report)
	report["warnings"].extend(str(warning) for warning in release_report.get("warnings", []))
	if not release_report.get("ok"):
		report["errors"].extend(str(error) for error in release_report.get("errors", []))
		return report

	compiler = APGCompiler(CodeGenConfig(target_language="python"))
	result = compiler.compile_file(source_file, target_language="python")
	report["warnings"].extend(str(warning) for warning in result.warnings)
	if not result.success:
		report["errors"].extend(str(error) for error in result.errors)
		return report

	application_name = result.module.name if result.module else source_file.stem
	package_dir = output_root / f"{_safe_name(application_name)}-{profile}"
	report["output_dir"] = str(package_dir)
	report["manifest_path"] = str(package_dir / "package_manifest.json")
	manifest = _package_manifest(source_file, profile, result, release_report)
	profile_files = _profile_files(profile, manifest)
	_write_package(package_dir, result.generated_files, profile_files, manifest, release_report)

	report["manifest"] = manifest
	report["files"] = sorted(
		str(path.relative_to(package_dir))
		for path in package_dir.rglob("*")
		if path.is_file() and "__pycache__" not in path.relative_to(package_dir).parts
	)
	report["checks"] = _package_checks(profile, package_dir, manifest, release_report)
	for error in report["checks"].get("errors", []):
		report["errors"].append(str(error))
	for warning in report["checks"].get("warnings", []):
		report["warnings"].append(str(warning))
	report["ok"] = not report["errors"]
	return report


def _normalize_package_target(target: str) -> str:
	return (target or "web").lower().replace("_", "-")


def _safe_name(name: str) -> str:
	cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in name)
	return cleaned.strip("-_") or "apg-app"


def _release_summary(release_report: dict[str, Any]) -> dict[str, Any]:
	evidence = release_report.get("evidence", {})
	return {
		"format": release_report.get("format"),
		"ok": release_report.get("ok"),
		"target": release_report.get("target"),
		"preflight": release_report.get("preflight", {}),
		"generated_file_count": release_report.get("generated", {}).get("file_count", 0),
		"self_test": evidence.get("self_test", {}),
		"semantic_model": evidence.get("semantic_model", {}),
		"contracts": evidence.get("contracts", {}),
	}


def _package_manifest(
	source_file: Path,
	profile: str,
	result: CompilationResult,
	release_report: dict[str, Any],
) -> dict[str, Any]:
	application_name = result.module.name if result.module else source_file.stem
	generated_files = sorted(result.generated_files)
	return {
		"format": "apg.package-manifest.v1",
		"name": application_name,
		"version": result.module.version if result.module else "1.0.0",
		"source": str(source_file),
		"profile": profile,
		"base_target": "python",
		"generated_artifacts": generated_files,
		"profile_artifacts": _profile_artifact_names(profile),
		"entrypoints": _profile_entrypoints(profile),
		"signing": _signing_posture(profile),
		"release_evidence": {
			"ok": bool(release_report.get("ok")),
			"self_test": release_report.get("evidence", {}).get("self_test", {}),
			"semantic_model": release_report.get("evidence", {}).get("semantic_model", {}),
			"contracts": release_report.get("evidence", {}).get("contracts", {}),
		},
	}


def _profile_artifact_names(profile: str) -> list[str]:
	if profile == "web":
		return ["run_web.py"]
	if profile == "desktop":
		return ["run_desktop.py"]
	if profile == "mobile":
		return ["mobile_profile.json"]
	if profile == "container":
		return ["container_profile.json"]
	return ["python_profile.json"]


def _profile_entrypoints(profile: str) -> dict[str, str]:
	if profile == "web":
		return {"run": "python run_web.py", "self_test": "python app.py --self-test"}
	if profile == "desktop":
		return {"run": "python run_desktop.py", "self_test": "python app.py --self-test"}
	if profile == "mobile":
		return {"profile": "mobile_profile.json", "self_test": "python app.py --self-test"}
	if profile == "container":
		return {"build": "docker build -t apg-packaged-app .", "self_test": "python app.py --self-test"}
	return {"run": "python app.py", "self_test": "python app.py --self-test"}


def _signing_posture(profile: str) -> dict[str, Any]:
	requires_signing = profile in {"desktop", "mobile"}
	return {
		"required_for_distribution": requires_signing,
		"status": "unsigned_development_profile" if requires_signing else "not_required_for_profile",
		"policy": "Declare signing before external distribution; development packages remain executable without signing.",
	}


def _profile_files(profile: str, manifest: dict[str, Any]) -> dict[str, str]:
	if profile == "web":
		return {
			"run_web.py": _launcher("0.0.0.0", 8080),
		}
	if profile == "desktop":
		return {
			"run_desktop.py": _launcher("127.0.0.1", 8080),
		}
	if profile == "mobile":
		return {
			"mobile_profile.json": json.dumps({
				"format": "apg.mobile-profile.v1",
				"name": manifest["name"],
				"runtime": "python",
				"offline": {"supported": True, "storage": "memory"},
				"permissions": [],
				"launch": manifest["entrypoints"]["self_test"],
				"signing": manifest["signing"],
			}, indent=2, sort_keys=True) + "\n",
		}
	if profile == "container":
		return {
			"container_profile.json": json.dumps({
				"format": "apg.container-profile.v1",
				"name": manifest["name"],
				"dockerfile": "Dockerfile",
				"healthcheck": "python app.py --self-test",
			}, indent=2, sort_keys=True) + "\n",
		}
	return {
		"python_profile.json": json.dumps({
			"format": "apg.python-profile.v1",
			"name": manifest["name"],
			"entrypoints": manifest["entrypoints"],
		}, indent=2, sort_keys=True) + "\n",
	}


def _launcher(host: str, port: int) -> str:
	return (
		"#!/usr/bin/env python3\n"
		"\"\"\"Generated APG package launcher.\"\"\"\n\n"
		"import app\n\n"
		"if __name__ == \"__main__\":\n"
		f"    app.main([\"app.py\", \"--host\", \"{host}\", \"--port\", \"{port}\"])\n"
	)


def _write_package(
	package_dir: Path,
	generated_files: dict[str, str],
	profile_files: dict[str, str],
	manifest: dict[str, Any],
	release_report: dict[str, Any],
) -> None:
	package_dir.mkdir(parents=True, exist_ok=True)
	files = {
		**generated_files,
		**profile_files,
		"package_manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
		"release_report.json": json.dumps(release_report, indent=2, sort_keys=True) + "\n",
	}
	for file_name, content in files.items():
		path = package_dir / file_name
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(content, encoding="utf-8")


def _package_checks(
	profile: str,
	package_dir: Path,
	manifest: dict[str, Any],
	release_report: dict[str, Any],
) -> dict[str, Any]:
	errors: list[str] = []
	warnings: list[str] = []
	for artifact in manifest["generated_artifacts"]:
		if not (package_dir / artifact).exists():
			errors.append(f"missing generated artifact {artifact}")
	for artifact in [*manifest["profile_artifacts"], "package_manifest.json", "release_report.json"]:
		if not (package_dir / artifact).exists():
			errors.append(f"missing package artifact {artifact}")
	if not release_report.get("ok"):
		errors.append("release evidence did not pass")
	if profile in {"desktop", "mobile"} and manifest["signing"]["status"] == "unsigned_development_profile":
		warnings.append(f"{profile} package uses unsigned development signing posture")
	return {
		"errors": errors,
		"warnings": warnings,
		"release_evidence_ok": bool(release_report.get("ok")),
		"manifest_written": (package_dir / "package_manifest.json").exists(),
		"profile_artifacts_present": all(
			(package_dir / artifact).exists()
			for artifact in manifest["profile_artifacts"]
		),
	}
