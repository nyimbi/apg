#!/usr/bin/env python3
"""Developer-experience commands for APG projects."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click

from compiler.compiler import APGCompiler, CodeGenConfig
from compiler.code_generator import CodeGenerator


@dataclass(frozen=True)
class CompileOutcome:
	success: bool
	elapsed_ms: float
	file_count: int = 0
	error: str | None = None


def _compile_apg_source(source_path: Path, output_dir: Path) -> CompileOutcome:
	"""Compile one APG source file to an output directory with quiet CLI output."""
	t0 = time.monotonic()
	try:
		source = source_path.read_text(encoding="utf-8")
		config = CodeGenConfig(
			target_language=CodeGenerator.normalize_target("python"),
			output_directory=str(output_dir),
			include_runtime=True,
			verbose=False,
		)
		result = APGCompiler(config).compile_string(source, source_path.stem)
	except Exception as exc:  # pragma: no cover - defensive boundary
		elapsed_ms = (time.monotonic() - t0) * 1000
		return CompileOutcome(False, elapsed_ms, error=str(exc))

	elapsed_ms = (time.monotonic() - t0) * 1000
	if not result.success:
		message = "; ".join(str(error) for error in result.errors) or "compilation failed"
		return CompileOutcome(False, elapsed_ms, error=message)

	output_dir.mkdir(parents=True, exist_ok=True)
	for filename, content in result.generated_files.items():
		file_path = output_dir / filename
		file_path.parent.mkdir(parents=True, exist_ok=True)
		file_path.write_text(content, encoding="utf-8")

	return CompileOutcome(True, elapsed_ms, file_count=len(result.generated_files))


def _require_source(source_file: str) -> Path:
	source_path = Path(source_file)
	if not source_path.exists():
		raise click.ClickException(f"APG source file not found: {source_path}")
	if not source_path.is_file():
		raise click.ClickException(f"APG source path is not a file: {source_path}")
	return source_path


def _timestamp() -> str:
	return datetime.now().strftime("%H:%M:%S")


def _print_compile_result(outcome: CompileOutcome) -> None:
	if outcome.success:
		click.echo(f"{_timestamp()} Recompiled in {outcome.elapsed_ms:.0f}ms")
	else:
		click.echo(
			f"{_timestamp()} ERROR: {outcome.error or 'compilation failed'}",
			err=True,
		)


@click.command(name="watch")
@click.argument("source_file")
@click.option("--output", "-o", default="generated", help="Output directory for generated app")
def watch(source_file: str, output: str) -> None:
	"""Watch an APG file and recompile on save."""
	source_path = _require_source(source_file).resolve()
	output_dir = Path(output)
	clear_screen = os.getenv("APG_WATCH_CLEAR") == "1"

	def recompile() -> None:
		if clear_screen:
			click.clear()
		_print_compile_result(_compile_apg_source(source_path, output_dir))

	click.echo(f"Watching {source_path}")
	recompile()

	try:
		from watchdog.events import FileSystemEventHandler
		from watchdog.observers import Observer
	except ImportError:
		_poll_watch(source_path, recompile)
		return

	class APGFileHandler(FileSystemEventHandler):
		def __init__(self) -> None:
			self._last_mtime = source_path.stat().st_mtime

		def on_modified(self, event) -> None:  # type: ignore[no-untyped-def]
			self._maybe_recompile(event.src_path)

		def on_created(self, event) -> None:  # type: ignore[no-untyped-def]
			self._maybe_recompile(event.src_path)

		def on_moved(self, event) -> None:  # type: ignore[no-untyped-def]
			self._maybe_recompile(getattr(event, "dest_path", event.src_path))

		def _maybe_recompile(self, changed_path: str) -> None:
			if Path(changed_path).resolve() != source_path:
				return
			try:
				mtime = source_path.stat().st_mtime
			except OSError:
				return
			if mtime == self._last_mtime:
				return
			self._last_mtime = mtime
			recompile()

	observer = Observer()
	observer.schedule(APGFileHandler(), str(source_path.parent), recursive=False)
	observer.start()
	try:
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		click.echo("Stopped watching.")
	finally:
		observer.stop()
		observer.join()


def _poll_watch(source_path: Path, recompile) -> None:  # type: ignore[no-untyped-def]
	try:
		last_mtime = source_path.stat().st_mtime
	except OSError:
		last_mtime = 0.0
	try:
		while True:
			time.sleep(1)
			try:
				mtime = source_path.stat().st_mtime
			except OSError:
				continue
			if mtime != last_mtime:
				last_mtime = mtime
				recompile()
	except KeyboardInterrupt:
		click.echo("Stopped watching.")


@click.command(name="serve")
@click.argument("source_file")
def serve(source_file: str) -> None:
	"""Compile an APG file to a temp directory and run the generated app."""
	source_path = _require_source(source_file)
	temp_dir = Path(tempfile.mkdtemp(prefix="apg-serve-"))
	outcome = _compile_apg_source(source_path, temp_dir)
	if not outcome.success:
		raise click.ClickException(outcome.error or "compilation failed")

	env = os.environ.copy()
	os.chdir(temp_dir)
	click.echo(f"Serving generated app from {temp_dir}")
	os.execvpe(sys.executable, [sys.executable, "app.py"], env)


@click.command(name="export")
@click.argument("source_file")
@click.option(
	"--format",
	"export_format",
	default="docker",
	type=click.Choice(["docker"]),
	help="Export target format",
)
@click.option("--output", "-o", default=".", help="Output directory")
def export_cmd(source_file: str, export_format: str, output: str) -> None:
	"""Export APG deployment assets."""
	source_path = _require_source(source_file)
	output_dir = Path(output)

	if export_format != "docker":  # pragma: no cover - click constrains choices
		raise click.ClickException(f"Unsupported export format: {export_format}")

	outcome = _compile_apg_source(source_path, output_dir)
	if not outcome.success:
		raise click.ClickException(outcome.error or "compilation failed")

	_write_docker_export(output_dir)
	click.echo(f"Exported docker assets to {output_dir}")


def _write_docker_export(output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "Dockerfile").write_text(
		"""\
FROM python:3.12-slim
WORKDIR /app
COPY app.py .
RUN pip install --no-cache-dir flask
EXPOSE 5000
CMD ["python", "app.py"]
""",
		encoding="utf-8",
	)
	(output_dir / "docker-compose.yml").write_text(
		"""\
services:
  apg-app:
    build: .
    env_file:
      - .env
    ports:
      - "5000:5000"
    restart: unless-stopped
""",
		encoding="utf-8",
	)
