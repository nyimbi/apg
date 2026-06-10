"""APG Studio — standalone Flask application.

Run:  python -m capabilities.composition.studio.app
      OR: uv run python capabilities/composition/studio/app.py

Serves:
  /studio/           — landing page
  /studio/compositor — web IDE
  /studio/api/*      — JSON APIs
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, redirect

_log = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__, instance_relative_config=False)
    app.secret_key = os.environ.get("SECRET_KEY", "apg-studio-dev-key-change-in-prod")
    app.config.update(
        DEBUG=os.environ.get("DEBUG", "true").lower() == "true",
        TEMPLATES_AUTO_RELOAD=True,
    )

    try:
        from .api import studio_api
    except ImportError:
        # Absolute import when run as a script
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from capabilities.composition.studio.api import studio_api  # type: ignore[no-redef]

    app.register_blueprint(studio_api)

    @app.route("/")
    def root():
        return redirect("/studio/")

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "not found"}, 404

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
    port = int(os.environ.get("STUDIO_PORT", "5100"))
    app = create_app()
    _log.info("APG Studio starting on http://localhost:%d/studio/", port)
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
