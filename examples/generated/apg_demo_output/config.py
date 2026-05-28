"""
Flask-AppBuilder Configuration
=============================

Configuration file for the APG Flask-AppBuilder application.
"""

import os
from flask_appbuilder.security.manager import AUTH_OID, AUTH_REMOTE_USER, AUTH_DB, AUTH_LDAP, AUTH_OAUTH

basedir = os.path.abspath(os.path.dirname(__file__))

# Your App secret key
SECRET_KEY = '\2\1thisismyscretkey\1\2\e\y\y\h'

# Database configuration - SQLite for demo
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')

# Flask-WTF flag for CSRF
CSRF_ENABLED = True

# ------------------------------
# GLOBALS FOR APP Builder 
# ------------------------------
APP_NAME = "APG Task Manager"

# ----------------------------------------------------
# AUTHENTICATION CONFIG
# ----------------------------------------------------
AUTH_TYPE = AUTH_DB

# ----------------------------------------------------
# APG SPECIFIC CONFIG
# ----------------------------------------------------
APG_AGENT_POLL_INTERVAL = 5  # seconds
APG_WORKFLOW_TIMEOUT = 300   # seconds

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
}
