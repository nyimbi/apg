"""APG shared UI component library for Flask blueprint capabilities.

Provides:
- apg_base.html: base template (Tailwind CDN + htmx + Alpine.js)
- _tokens.html: macro for APG design tokens → CSS :root {}
- _nav.html: macro for capability route list → accessible nav strip
- macros/dashboard.html: kpi_card, activity_feed, health_strip
- macros/table.html: data_table with htmx sort/filter/paginate
- macros/form.html: field, form_section with ARIA
- macros/workbench.html: split_pane, queue_item, detail_panel
- macros/settings.html: settings_group
- macros/shell.html: full APG shell with MANIFEST-driven sidebar

Usage in a Flask blueprint::

    from capabilities.common.ui import register_templates

    blueprint = Blueprint("my_cap", __name__)
    register_templates(blueprint)

Then in templates extend apg_base.html::

    {% extends "apg_base.html" %}
    {% from "macros/table.html" import data_table %}
    {% block content %}
      {{ data_table(rows, cols, hx_url="/api/my_cap/records") }}
    {% endblock %}

IMPORTANT: Do NOT extend Flask-AppBuilder base.html.
Bootstrap 3 / Tailwind CSS specificity conflicts are unresolvable.
"""
from __future__ import annotations

import os
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def register_templates(app_or_blueprint: object) -> None:
	"""Register the APG shared template folder on a Flask app or blueprint.

	Call this once during blueprint setup so Jinja2 finds apg_base.html
	and the macro files before the blueprint's own templates.
	"""
	template_folder = str(_TEMPLATES_DIR)
	if hasattr(app_or_blueprint, "template_folder"):
		# Blueprint: set/extend template search path
		existing = getattr(app_or_blueprint, "template_folder", None)
		if existing and existing != template_folder:
			# Flask doesn't support multiple template folders per blueprint natively;
			# add to app's Jinja2 loader instead when possible
			_add_to_app_loader(app_or_blueprint, template_folder)
		else:
			object.__setattr__(app_or_blueprint, "template_folder", template_folder)
	elif hasattr(app_or_blueprint, "jinja_loader"):
		_add_to_app_loader(app_or_blueprint, template_folder)


def _add_to_app_loader(obj: object, path: str) -> None:
	try:
		from jinja2 import FileSystemLoader, ChoiceLoader
		loader = getattr(obj, "jinja_loader", None)
		if loader is None:
			return
		new_loader = FileSystemLoader(path)
		if isinstance(loader, ChoiceLoader):
			loader.loaders.insert(0, new_loader)
		else:
			object.__setattr__(obj, "_jinja_loader", ChoiceLoader([new_loader, loader]))
	except Exception:
		pass


def templates_dir() -> Path:
	"""Return path to the APG shared templates directory."""
	return _TEMPLATES_DIR
