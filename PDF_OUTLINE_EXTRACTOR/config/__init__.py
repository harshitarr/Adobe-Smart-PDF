# Import all configuration modules
from .settings import *
from .patterns import *
from .language_patterns import *

# Export all configuration constants and settings
__all__ = [
    # Core settings
    'LOGGING_CONFIG', 
    'INPUT_DIR', 
    'OUTPUT_DIR', 
    'LOGS_DIR',
    
    # Document processing patterns
    'DOCUMENT_PATTERNS', 
    'HEADING_PATTERNS', 
    'CONTENT_FILTERS',
    
    # Language detection and processing
    'LANGUAGE_PATTERNS',
    'LANGUAGE_CONFIG', 
    'LANGUAGE_PROCESSING_CONFIG',
    'SUPPORTED_LANGUAGES',
    'LANGUAGE_DISPLAY_NAMES',
    'LANGUAGE_CODES',
    
    # Language settings from settings.py
    'LANGUAGE_DETECTION_ENABLED',
    'LANGUAGE_CONFIDENCE_THRESHOLD',
    'DEFAULT_LANGUAGE',
    'LANGUAGE_DISPLAY_ENABLED',
    'LANGUAGE_AWARE_PROCESSING',
    'LANGUAGE_SPECIFIC_PATTERNS',
    
    # Processing configuration
    'EXTRACTION_CONFIG',
    'CLASSIFICATION_CONFIG',
    'OUTPUT_CONFIG'
]

# Version info for the config module
__version__ = '1.3.0'

# Configuration categories for organized access
CONFIG_CATEGORIES = {
    'core': [
        'LOGGING_CONFIG', 'INPUT_DIR', 'OUTPUT_DIR', 'LOGS_DIR'
    ],
    'document_processing': [
        'DOCUMENT_PATTERNS', 'HEADING_PATTERNS', 'CONTENT_FILTERS',
        'EXTRACTION_CONFIG', 'CLASSIFICATION_CONFIG'
    ],
    'language_support': [
        'LANGUAGE_PATTERNS', 'LANGUAGE_CONFIG', 'LANGUAGE_PROCESSING_CONFIG',
        'SUPPORTED_LANGUAGES', 'LANGUAGE_DISPLAY_NAMES', 'LANGUAGE_CODES'
    ],
    'language_settings': [
        'LANGUAGE_DETECTION_ENABLED', 'LANGUAGE_CONFIDENCE_THRESHOLD',
        'DEFAULT_LANGUAGE', 'LANGUAGE_DISPLAY_ENABLED',
        'LANGUAGE_AWARE_PROCESSING', 'LANGUAGE_SPECIFIC_PATTERNS'
    ],
    'output': [
        'OUTPUT_CONFIG'
    ]
}

def get_config_by_category(category: str):
    """
    Get configuration items by category
    
    Args:
        category (str): Category name ('core', 'document_processing', 'language_support', etc.)
        
    Returns:
        dict: Dictionary of configuration items for the category
    """
    if category not in CONFIG_CATEGORIES:
        return {}
    
    config_items = CONFIG_CATEGORIES[category]
    result = {}
    
    # Get current module's globals
    current_globals = globals()
    
    for item in config_items:
        if item in current_globals:
            result[item] = current_globals[item]
    
    return result

def list_config_categories():
    """
    List all available configuration categories
    
    Returns:
        list: List of available configuration category names
    """
    return list(CONFIG_CATEGORIES.keys())

def is_language_support_enabled():
    """
    Check if language support features are enabled
    
    Returns:
        bool: True if language detection and processing are enabled
    """
    try:
        return (
            globals().get('LANGUAGE_DETECTION_ENABLED', False) and
            globals().get('LANGUAGE_AWARE_PROCESSING', False) and
            len(globals().get('SUPPORTED_LANGUAGES', [])) > 1
        )
    except:
        return False

def get_supported_languages():
    """
    Get list of supported languages with display names
    
    Returns:
        dict: Dictionary mapping language codes to display names
    """
    try:
        languages = globals().get('SUPPORTED_LANGUAGES', ['english'])
        display_names = globals().get('LANGUAGE_DISPLAY_NAMES', {})
        return {lang: display_names.get(lang, lang.title()) for lang in languages}
    except:
        return {'english': 'English'}

def validate_configuration():
    """
    Validate that all required configuration items are present
    
    Returns:
        tuple: (is_valid, missing_items, validation_errors)
    """
    missing_items = []
    validation_errors = []
    current_globals = globals()
    
    # Required core configuration items
    required_core = [
        'LOGGING_CONFIG', 'INPUT_DIR', 'OUTPUT_DIR', 'LOGS_DIR'
    ]
    
    # Required pattern configuration items
    required_patterns = [
        'DOCUMENT_PATTERNS', 'HEADING_PATTERNS', 'CONTENT_FILTERS'
    ]
    
    # Check core requirements
    for item in required_core:
        if item not in current_globals:
            missing_items.append(item)
    
    # Check pattern requirements
    for item in required_patterns:
        if item not in current_globals:
            missing_items.append(item)
    
    # Validate language configuration if enabled
    if current_globals.get('LANGUAGE_DETECTION_ENABLED', False):
        required_language = [
            'LANGUAGE_PATTERNS', 'LANGUAGE_CONFIG', 'SUPPORTED_LANGUAGES'
        ]
        
        for item in required_language:
            if item not in current_globals:
                missing_items.append(item)
        
        # Validate language patterns structure
        language_patterns = current_globals.get('LANGUAGE_PATTERNS', {})
        if not isinstance(language_patterns, dict):
            validation_errors.append("LANGUAGE_PATTERNS must be a dictionary")
        
        # Validate supported languages
        supported_languages = current_globals.get('SUPPORTED_LANGUAGES', [])
        if not isinstance(supported_languages, list) or len(supported_languages) == 0:
            validation_errors.append("SUPPORTED_LANGUAGES must be a non-empty list")
    
    # Validate directory paths
    for dir_item in ['INPUT_DIR', 'OUTPUT_DIR', 'LOGS_DIR']:
        if dir_item in current_globals:
            dir_value = current_globals[dir_item]
            if not hasattr(dir_value, '__str__'):
                validation_errors.append(f"{dir_item} must be a valid path")
    
    is_valid = len(missing_items) == 0 and len(validation_errors) == 0
    
    return is_valid, missing_items, validation_errors

def print_configuration_summary():
    """
    Print a summary of the current configuration
    """
    print("=" * 60)
    print("PDF Outline Extractor - Configuration Summary")
    print("=" * 60)
    
    # Core settings
    print("\n📁 Core Settings:")
    core_config = get_config_by_category('core')
    for key, value in core_config.items():
        if key.endswith('_DIR'):
            print(f"  {key}: {value}")
        elif key == 'LOGGING_CONFIG':
            print(f"  {key}: {'Configured' if value else 'Not configured'}")
    
    # Language support
    print("\n🌐 Language Support:")
    if is_language_support_enabled():
        supported = get_supported_languages()
        print(f"  Status: ✅ Enabled")
        print(f"  Languages: {', '.join(supported.values())}")
        print(f"  Default: {globals().get('DEFAULT_LANGUAGE', 'english').title()}")
        print(f"  Confidence threshold: {globals().get('LANGUAGE_CONFIDENCE_THRESHOLD', 0.4)}")
    else:
        print(f"  Status: ❌ Disabled")
    
    # Document processing
    print("\n📄 Document Processing:")
    doc_config = get_config_by_category('document_processing')
    print(f"  Pattern sets: {len([k for k in doc_config.keys() if 'PATTERN' in k])}")
    print(f"  Content filters: {'✅ Configured' if 'CONTENT_FILTERS' in doc_config else '❌ Missing'}")
    
    # Validation
    print("\n🔍 Configuration Validation:")
    is_valid, missing, errors = validate_configuration()
    if is_valid:
        print("  Status: ✅ Valid")
    else:
        print("  Status: ❌ Issues found")
        if missing:
            print(f"  Missing items: {', '.join(missing)}")
        if errors:
            print(f"  Errors: {'; '.join(errors)}")
    
    print("=" * 60)

# Optional: Auto-validate on import (can be disabled)
AUTO_VALIDATE_ON_IMPORT = True

if AUTO_VALIDATE_ON_IMPORT:
    try:
        is_valid, missing_items, validation_errors = validate_configuration()
        if not is_valid:
            import warnings
            warning_msg = f"Configuration validation failed. Missing: {missing_items}, Errors: {validation_errors}"
            warnings.warn(warning_msg, UserWarning)
    except Exception as e:
        import warnings
        warnings.warn(f"Configuration validation error: {e}", UserWarning)
