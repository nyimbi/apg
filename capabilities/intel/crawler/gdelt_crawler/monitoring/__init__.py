"""
GDELT Monitoring and Alerting Components
========================================

Real-time monitoring, alerting, and metrics collection for GDELT crawling
operations with conflict detection and system health monitoring.

Components:
- **GDELTAlertSystem**: Alert system for critical events and conflicts
- **GDELTMetrics**: Performance and content metrics collection
- **Health Monitoring**: System health checks and diagnostics

Features:
- Real-time conflict detection and alerting
- Performance metrics and reporting
- System health monitoring
- Customizable alert thresholds

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

try:
    from .alerts import GDELTAlertSystem
    from .metrics import GDELTMetrics
    
    __all__ = [
        'GDELTAlertSystem',
        'GDELTMetrics'
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    __all__ = []
    import logging
    logging.getLogger(__name__).warning(f"GDELT monitoring components not available: {e}")

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__company__ = "Datacraft"
__website__ = "www.datacraft.co.ke"
__license__ = "MIT"