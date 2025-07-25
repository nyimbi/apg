#!/usr/bin/env python3
"""
APG Composable Template System
==============================

Composable capability-based templating system for generating world-class applications
by combining base templates with focused capability modules.

Architecture:
- Base Templates: Core application architectures (Flask webapp, microservice, etc.)
- Capabilities: Focused feature modules (AI, payments, auth, etc.)
- Integration Patterns: Pre-defined capability combinations for common use cases
- Composition Engine: Intelligently combines capabilities based on APG AST analysis
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

__version__ = "1.0.0"
__author__ = "APG Development Team"

# System paths
COMPOSABLE_ROOT = Path(__file__).parent
BASE_TEMPLATES_DIR = COMPOSABLE_ROOT / "bases"
CAPABILITIES_DIR = COMPOSABLE_ROOT / "capabilities"
INTEGRATIONS_DIR = COMPOSABLE_ROOT / "integrations"

# Core system components
SYSTEM_COMPONENTS = {
    'base_templates': BASE_TEMPLATES_DIR,
    'capabilities': CAPABILITIES_DIR,
    'integrations': INTEGRATIONS_DIR,
}

# Capability categories for organization
CAPABILITY_CATEGORIES = {
    'auth': 'Authentication & Authorization',
    'ai': 'Artificial Intelligence & ML',
    'data': 'Data Storage & Processing',
    'payments': 'Payment Processing',
    'iot': 'Internet of Things',
    'business': 'Business Logic',
    'communication': 'Communication & Messaging',
    'analytics': 'Analytics & Reporting',
    'security': 'Security & Compliance',
    'infrastructure': 'Infrastructure & DevOps'
}

def get_system_info() -> Dict[str, Any]:
    """Get system information and status"""
    return {
        'version': __version__,
        'components': SYSTEM_COMPONENTS,
        'categories': CAPABILITY_CATEGORIES,
        'paths': {
            'root': COMPOSABLE_ROOT,
            'bases': BASE_TEMPLATES_DIR,
            'capabilities': CAPABILITIES_DIR,
            'integrations': INTEGRATIONS_DIR
        }
    }