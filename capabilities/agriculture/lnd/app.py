"""Flask application entry point for agr_lnd."""
from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    from .api import blueprint
    app.register_blueprint(blueprint)
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
