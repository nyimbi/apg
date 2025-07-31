"""
GDELT Utility Functions and Helper Components
=============================================

Utility functions and helper components for GDELT data processing,
geographic enhancement, and ML integration support.

Components:
- **Parsers**: GDELT format parsers and data transformation utilities
- **Geocoding**: Geographic data enhancement and location processing
- **ML Integration**: ML scoring integration and helper functions

Features:
- GDELT format parsing and validation
- Geographic coordinate processing
- ML model integration utilities
- Data transformation helpers

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

try:
    from .parsers import (
        GDELTFormatParser,
        EventCodeMapper,
        parse_gdelt_date,
        parse_gdelt_coordinates
    )
    
    from .geocoding import (
        GDELTGeocoder,
        LocationEnhancer,
        coordinate_validator
    )
    
    from .ml_integration import (
        MLScorerIntegration,
        prepare_content_for_ml,
        extract_event_features
    )
    
    __all__ = [
        # Parsers
        'GDELTFormatParser',
        'EventCodeMapper',
        'parse_gdelt_date',
        'parse_gdelt_coordinates',
        
        # Geocoding
        'GDELTGeocoder',
        'LocationEnhancer', 
        'coordinate_validator',
        
        # ML Integration
        'MLScorerIntegration',
        'prepare_content_for_ml',
        'extract_event_features'
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    __all__ = []
    import logging
    logging.getLogger(__name__).warning(f"GDELT utils components not available: {e}")

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__company__ = "Datacraft"
__website__ = "www.datacraft.co.ke"
__license__ = "MIT"