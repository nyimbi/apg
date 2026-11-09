"""Generated app CSS regressions."""

from __future__ import annotations

import re
from pathlib import Path

from compiler.compiler import APGCompiler


SIMPLE_SOURCE = """
module generated_app_css version 1.0.0 {}

table Customer {
    name: str;
}
"""


def _generated_app_source() -> str:
	result = APGCompiler().compile_string(SIMPLE_SOURCE, "generated_app_css")
	assert result.success, result.errors
	return result.generated_files["app.py"]


def test_dark_mode_css_present():
	app_source = _generated_app_source()

	assert "@media (prefers-color-scheme: dark)" in app_source


def test_print_css_present():
	app_source = _generated_app_source()

	assert "@media print" in app_source


def test_mobile_breakpoint_present():
	app_source = _generated_app_source()

	assert "max-width: 768px" in app_source


def test_fluid_typography_uses_clamp():
	app_source = _generated_app_source()

	assert "clamp(" in app_source


def test_container_queries_present():
	app_source = _generated_app_source()

	assert "container-type" in app_source


def _css_text() -> str:
	path = Path(__file__).resolve().parents[1] / "compiler" / "assets" / "apg.css"
	return path.read_text(encoding="utf-8")


def _token_layers() -> tuple[dict[str, str], dict[str, str]]:
	"""Return (light_tokens, dark_tokens) from apg.css.

	Declaration order is the cascade: the first :root block seeds the light
	theme, every later --apg-*: value overrides it for the dark theme.
	"""
	pairs = re.findall(r"(--apg-[a-z0-9-]+)\s*:\s*([^;]+);", _css_text())
	light: dict[str, str] = {}
	dark: dict[str, str] = {}
	for name, value in pairs:
		value = value.strip()
		if name not in light:
			light[name] = value
		dark[name] = value
	for tokens in (light, dark):
		for name, value in list(tokens.items()):
			inner = re.fullmatch(r"var\((--apg-[a-z0-9-]+)\)", value)
			if inner and inner.group(1) in tokens:
				tokens[name] = tokens[inner.group(1)]
	return light, dark


def _relative_luminance(hex_color: str) -> float:
	def _channel(raw: int) -> float:
		c = raw / 255
		return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

	digits = hex_color.removeprefix("#")
	red, green, blue = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
	return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def _contrast_ratio(fg: str, bg: str) -> float:
	first, second = sorted((_relative_luminance(fg), _relative_luminance(bg)), reverse=True)
	return (first + 0.05) / (second + 0.05)


_TEXT_PAIRS = (
	("--apg-text", "--apg-bg-canvas"),
	("--apg-text", "--apg-bg-card"),
	("--apg-muted", "--apg-bg-canvas"),
	("--apg-muted", "--apg-bg-card"),
	("--apg-primary", "--apg-bg-canvas"),
	("--apg-primary", "--apg-bg-card"),
	("--apg-danger", "--apg-bg-canvas"),
	("--apg-danger", "--apg-bg-card"),
	("--apg-info", "--apg-bg-canvas"),
	("--apg-info", "--apg-bg-card"),
	("--apg-success", "--apg-bg-canvas"),
	("--apg-success", "--apg-bg-card"),
	("--apg-warning", "--apg-bg-canvas"),
	("--apg-warning", "--apg-bg-card"),
)


def test_light_theme_text_contrast_meets_wcag_aa():
	light, _dark = _token_layers()

	for fg_name, bg_name in _TEXT_PAIRS:
		ratio = _contrast_ratio(light[fg_name], light[bg_name])
		assert ratio >= 4.5, f"light {fg_name} on {bg_name}: {ratio:.2f}:1"


def test_dark_theme_text_contrast_meets_wcag_aa():
	_light, dark = _token_layers()

	for fg_name, bg_name in _TEXT_PAIRS:
		ratio = _contrast_ratio(dark[fg_name], dark[bg_name])
		assert ratio >= 4.5, f"dark {fg_name} on {bg_name}: {ratio:.2f}:1"


def test_generated_theme_overrides_semantic_colors_for_dark():
	app_source = _generated_app_source()

	# primary is declared in light + both dark blocks; danger/info/success/warning
	# only need the two dark blocks (light values come from apg.css).
	requirements = {"--apg-primary": 3, "--apg-danger": 2, "--apg-info": 2, "--apg-success": 2, "--apg-warning": 2}
	for name, minimum in requirements.items():
		overrides = re.findall(re.escape(name) + r"\s*:", app_source)
		assert len(overrides) >= minimum, f"{name} expected >= {minimum} declarations, got {len(overrides)}"


def test_accent_token_meets_non_text_contrast():
	light, dark = _token_layers()

	for theme, tokens in (("light", light), ("dark", dark)):
		ratio = _contrast_ratio(tokens["--apg-accent"], tokens["--apg-bg-card"])
		assert ratio >= 3.0, f"{theme} --apg-accent on --apg-bg-card: {ratio:.2f}:1"
