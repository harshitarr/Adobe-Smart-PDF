import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Performance constraints
MAX_EXECUTION_TIME = int(os.getenv('MAX_EXECUTION_TIME', '10'))  # seconds
MAX_MODEL_SIZE = int(os.getenv('MAX_MODEL_SIZE', '200'))  # MB
MAX_PAGES = int(os.getenv('MAX_PAGES', '50'))

# Memory and processing limits (NEW)
MAX_MEMORY_USAGE_MB = int(os.getenv('MAX_MEMORY_USAGE_MB', '512'))
MAX_TEXT_BLOCKS_PER_ANALYSIS = int(os.getenv('MAX_TEXT_BLOCKS_PER_ANALYSIS', '10000'))
ANALYSIS_TIMEOUT_SECONDS = int(os.getenv('ANALYSIS_TIMEOUT_SECONDS', '30'))

# Enhanced logging configuration with multilingual support
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        },
        'simple': {
            'format': '[%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'extraction': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'extraction.log',
            'formatter': 'standard',
            'encoding': 'utf-8',  # Support for multilingual logging
        },
        'classification': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'classification.log',
            'formatter': 'standard',
            'encoding': 'utf-8',
        },
        'language_detection': {  # NEW: Language detection logging
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'language_detection.log',
            'formatter': 'detailed',
            'encoding': 'utf-8',
        },
        'errors': {
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'errors.log',
            'formatter': 'detailed',
            'encoding': 'utf-8',
        },
        'console': {  # NEW: Console output with UTF-8 support
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'debug': {  # NEW: Debug logging
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'debug.log',
            'formatter': 'detailed',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        'extraction': {
            'handlers': ['extraction', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        'classification': {
            'handlers': ['classification', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        'language_detection': {  # NEW: Language detection logger
            'handlers': ['language_detection', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        'errors': {
            'handlers': ['errors', 'console'],
            'level': 'ERROR',
            'propagate': False,
        },
        'debug': {  # NEW: Debug logger
            'handlers': ['debug'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'root': {  # NEW: Root logger configuration
            'handlers': ['console'],
            'level': 'WARNING',
        },
    },
}

# Language Detection Settings (ENHANCED)
LANGUAGE_DETECTION_ENABLED = os.getenv('LANGUAGE_DETECTION_ENABLED', 'true').lower() == 'true'
LANGUAGE_CONFIDENCE_THRESHOLD = float(os.getenv('LANGUAGE_CONFIDENCE_THRESHOLD', '0.4'))
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'english')
LANGUAGE_DISPLAY_ENABLED = os.getenv('LANGUAGE_DISPLAY_ENABLED', 'true').lower() == 'true'

# Advanced language detection settings (NEW)
LANGUAGE_DETECTION_METHOD = os.getenv('LANGUAGE_DETECTION_METHOD', 'pattern_based')  # pattern_based, statistical, hybrid
LANGUAGE_DETECTION_CACHE_ENABLED = os.getenv('LANGUAGE_DETECTION_CACHE_ENABLED', 'true').lower() == 'true'
LANGUAGE_DETECTION_CACHE_SIZE = int(os.getenv('LANGUAGE_DETECTION_CACHE_SIZE', '1000'))
LANGUAGE_DETECTION_TIMEOUT = float(os.getenv('LANGUAGE_DETECTION_TIMEOUT', '5.0'))

# Fallback behavior settings (NEW)
LANGUAGE_FALLBACK_ENABLED = os.getenv('LANGUAGE_FALLBACK_ENABLED', 'true').lower() == 'true'
LANGUAGE_AUTO_CORRECTION = os.getenv('LANGUAGE_AUTO_CORRECTION', 'true').lower() == 'true'
LANGUAGE_MIXED_DOCUMENT_HANDLING = os.getenv('LANGUAGE_MIXED_DOCUMENT_HANDLING', 'primary').lower()  # primary, mixed, separate

# Import language configuration with comprehensive error handling
try:
    from config.language_patterns import (
        LANGUAGE_PATTERNS, 
        LANGUAGE_CONFIG, 
        LANGUAGE_PROCESSING_CONFIG,
        DOCUMENT_TYPE_LANGUAGE_CONFIG,
        PERFORMANCE_CONFIG,
        VALIDATION_CONFIG,
        get_supported_languages,
        validate_language_config
    )
    
    # Extract configuration values safely
    SUPPORTED_LANGUAGES = LANGUAGE_CONFIG.get('analysis_settings', {}).get('supported_languages', ['english'])
    LANGUAGE_DISPLAY_NAMES = LANGUAGE_CONFIG.get('display_names', {'english': 'English'})
    LANGUAGE_CODES = LANGUAGE_CONFIG.get('language_codes', {'english': 'en'})
    LANGUAGE_NATIVE_NAMES = LANGUAGE_CONFIG.get('native_names', {'english': 'English'})
    LANGUAGE_SCRIPT_FAMILIES = LANGUAGE_CONFIG.get('script_families', {'english': ['latin']})
    LANGUAGE_DIFFICULTY_LEVELS = LANGUAGE_CONFIG.get('difficulty_levels', {'english': 'low'})
    
    # Validate configuration on import
    validation_result = validate_language_config()
    if not validation_result['valid']:
        print(f"⚠️  Language configuration validation warnings: {len(validation_result['errors'])} errors")
    
    LANGUAGE_CONFIG_LOADED = True
    
except ImportError as e:
    print(f"⚠️  Language configuration not available: {e}")
    print("   Falling back to English-only processing")
    
    # Fallback configuration
    SUPPORTED_LANGUAGES = ['english']
    LANGUAGE_DISPLAY_NAMES = {'english': 'English'}
    LANGUAGE_CODES = {'english': 'en'}
    LANGUAGE_NATIVE_NAMES = {'english': 'English'}
    LANGUAGE_SCRIPT_FAMILIES = {'english': ['latin']}
    LANGUAGE_DIFFICULTY_LEVELS = {'english': 'low'}
    LANGUAGE_PATTERNS = {}
    LANGUAGE_CONFIG = {}
    LANGUAGE_PROCESSING_CONFIG = {}
    DOCUMENT_TYPE_LANGUAGE_CONFIG = {}
    PERFORMANCE_CONFIG = {}
    VALIDATION_CONFIG = {}
    LANGUAGE_CONFIG_LOADED = False

except Exception as e:
    print(f"❌ Error loading language configuration: {e}")
    
    # Safe fallback
    SUPPORTED_LANGUAGES = ['english']
    LANGUAGE_DISPLAY_NAMES = {'english': 'English'}
    LANGUAGE_CODES = {'english': 'en'}
    LANGUAGE_NATIVE_NAMES = {'english': 'English'}
    LANGUAGE_SCRIPT_FAMILIES = {'english': ['latin']}
    LANGUAGE_DIFFICULTY_LEVELS = {'english': 'low'}
    LANGUAGE_PATTERNS = {}
    LANGUAGE_CONFIG = {}
    LANGUAGE_PROCESSING_CONFIG = {}
    DOCUMENT_TYPE_LANGUAGE_CONFIG = {}
    PERFORMANCE_CONFIG = {}
    VALIDATION_CONFIG = {}
    LANGUAGE_CONFIG_LOADED = False

# Language processing settings (ENHANCED)
LANGUAGE_AWARE_PROCESSING = os.getenv('LANGUAGE_AWARE_PROCESSING', 'true').lower() == 'true'
LANGUAGE_SPECIFIC_PATTERNS = os.getenv('LANGUAGE_SPECIFIC_PATTERNS', 'true').lower() == 'true'
LANGUAGE_CONFIDENCE_WEIGHTING = os.getenv('LANGUAGE_CONFIDENCE_WEIGHTING', 'true').lower() == 'true'
LANGUAGE_ADAPTIVE_THRESHOLDS = os.getenv('LANGUAGE_ADAPTIVE_THRESHOLDS', 'true').lower() == 'true'

# Cross-language processing settings (NEW)
ENABLE_MULTILINGUAL_DOCUMENTS = os.getenv('ENABLE_MULTILINGUAL_DOCUMENTS', 'true').lower() == 'true'
MULTILINGUAL_CONFIDENCE_THRESHOLD = float(os.getenv('MULTILINGUAL_CONFIDENCE_THRESHOLD', '0.3'))
LANGUAGE_MIXING_TOLERANCE = float(os.getenv('LANGUAGE_MIXING_TOLERANCE', '0.2'))

# Document type specific language settings (NEW)
DOCUMENT_TYPE_LANGUAGE_DETECTION = {
    'technical_manual': {
        'enabled': os.getenv('MANUAL_LANGUAGE_DETECTION', 'true').lower() == 'true',
        'confidence_threshold': float(os.getenv('MANUAL_LANGUAGE_THRESHOLD', '0.5')),
        'preferred_languages': os.getenv('MANUAL_PREFERRED_LANGUAGES', 'english,german,japanese').split(',')
    },
    'business_document': {
        'enabled': os.getenv('BUSINESS_LANGUAGE_DETECTION', 'true').lower() == 'true',
        'confidence_threshold': float(os.getenv('BUSINESS_LANGUAGE_THRESHOLD', '0.4')),
        'preferred_languages': os.getenv('BUSINESS_PREFERRED_LANGUAGES', 'english,german').split(',')
    },
    'form_document': {
        'enabled': os.getenv('FORM_LANGUAGE_DETECTION', 'true').lower() == 'true',
        'confidence_threshold': float(os.getenv('FORM_LANGUAGE_THRESHOLD', '0.6')),
        'preferred_languages': os.getenv('FORM_PREFERRED_LANGUAGES', 'english,tamil').split(',')
    },
    'program_document': {
        'enabled': os.getenv('PROGRAM_LANGUAGE_DETECTION', 'true').lower() == 'true',
        'confidence_threshold': float(os.getenv('PROGRAM_LANGUAGE_THRESHOLD', '0.4')),
        'preferred_languages': os.getenv('PROGRAM_PREFERRED_LANGUAGES', 'english').split(',')
    },
    'invitation_document': {
        'enabled': os.getenv('INVITATION_LANGUAGE_DETECTION', 'true').lower() == 'true',
        'confidence_threshold': float(os.getenv('INVITATION_LANGUAGE_THRESHOLD', '0.3')),
        'preferred_languages': os.getenv('INVITATION_PREFERRED_LANGUAGES', 'english,japanese,tamil').split(',')
    }
}

# Text processing settings (NEW)
TEXT_NORMALIZATION_ENABLED = os.getenv('TEXT_NORMALIZATION_ENABLED', 'true').lower() == 'true'
UNICODE_NORMALIZATION_FORM = os.getenv('UNICODE_NORMALIZATION_FORM', 'NFC')  # NFC, NFD, NFKC, NFKD
TEXT_ENCODING_DETECTION = os.getenv('TEXT_ENCODING_DETECTION', 'true').lower() == 'true'
TEXT_CLEANUP_ENABLED = os.getenv('TEXT_CLEANUP_ENABLED', 'true').lower() == 'true'

# Caching settings (NEW)
ENABLE_RESULT_CACHING = os.getenv('ENABLE_RESULT_CACHING', 'true').lower() == 'true'
CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '1000'))
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', '3600'))  # 1 hour
CACHE_LANGUAGE_RESULTS = os.getenv('CACHE_LANGUAGE_RESULTS', 'true').lower() == 'true'

# Output formatting settings (NEW)
OUTPUT_INCLUDE_LANGUAGE_INFO = os.getenv('OUTPUT_INCLUDE_LANGUAGE_INFO', 'true').lower() == 'true'
OUTPUT_INCLUDE_CONFIDENCE_SCORES = os.getenv('OUTPUT_INCLUDE_CONFIDENCE_SCORES', 'true').lower() == 'true'
OUTPUT_INCLUDE_DETECTION_DETAILS = os.getenv('OUTPUT_INCLUDE_DETECTION_DETAILS', 'false').lower() == 'true'
OUTPUT_FORMAT_TIMESTAMPS = os.getenv('OUTPUT_FORMAT_TIMESTAMPS', 'true').lower() == 'true'

# Error handling and recovery settings (NEW)
ENABLE_GRACEFUL_DEGRADATION = os.getenv('ENABLE_GRACEFUL_DEGRADATION', 'true').lower() == 'true'
RETRY_FAILED_LANGUAGE_DETECTION = os.getenv('RETRY_FAILED_LANGUAGE_DETECTION', 'true').lower() == 'true'
MAX_RETRY_ATTEMPTS = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))
FALLBACK_TO_ENGLISH_ON_ERROR = os.getenv('FALLBACK_TO_ENGLISH_ON_ERROR', 'true').lower() == 'true'

# Debug and development settings (NEW)
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
VERBOSE_LANGUAGE_DETECTION = os.getenv('VERBOSE_LANGUAGE_DETECTION', 'false').lower() == 'true'
SAVE_INTERMEDIATE_RESULTS = os.getenv('SAVE_INTERMEDIATE_RESULTS', 'false').lower() == 'true'
ENABLE_PERFORMANCE_PROFILING = os.getenv('ENABLE_PERFORMANCE_PROFILING', 'false').lower() == 'true'

# Security and validation settings (NEW)
VALIDATE_INPUT_ENCODING = os.getenv('VALIDATE_INPUT_ENCODING', 'true').lower() == 'true'
SANITIZE_OUTPUT = os.getenv('SANITIZE_OUTPUT', 'true').lower() == 'true'
MAX_INPUT_FILE_SIZE_MB = int(os.getenv('MAX_INPUT_FILE_SIZE_MB', '100'))
ALLOWED_FILE_EXTENSIONS = os.getenv('ALLOWED_FILE_EXTENSIONS', '.pdf').split(',')

# Feature flags (NEW)
FEATURE_FLAGS = {
    'advanced_language_detection': os.getenv('ENABLE_ADVANCED_LANGUAGE_DETECTION', 'true').lower() == 'true',
    'statistical_analysis': os.getenv('ENABLE_STATISTICAL_ANALYSIS', 'true').lower() == 'true',
    'cross_validation': os.getenv('ENABLE_CROSS_VALIDATION', 'false').lower() == 'true',
    'parallel_processing': os.getenv('ENABLE_PARALLEL_PROCESSING', 'false').lower() == 'true',
    'gpu_acceleration': os.getenv('ENABLE_GPU_ACCELERATION', 'false').lower() == 'true',
    'experimental_features': os.getenv('ENABLE_EXPERIMENTAL_FEATURES', 'false').lower() == 'true'
}

# Environment-specific overrides (NEW)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'production').lower()

if ENVIRONMENT == 'development':
    DEBUG_MODE = True
    VERBOSE_LANGUAGE_DETECTION = True
    SAVE_INTERMEDIATE_RESULTS = True
    ENABLE_PERFORMANCE_PROFILING = True
elif ENVIRONMENT == 'testing':
    LANGUAGE_DETECTION_CACHE_ENABLED = False
    ENABLE_RESULT_CACHING = False
    DEBUG_MODE = True
elif ENVIRONMENT == 'production':
    DEBUG_MODE = False
    VERBOSE_LANGUAGE_DETECTION = False
    SAVE_INTERMEDIATE_RESULTS = False

# Configuration validation and summary (NEW)
def get_configuration_summary():
    """Get a summary of current configuration"""
    return {
        'language_detection': {
            'enabled': LANGUAGE_DETECTION_ENABLED,
            'supported_languages': SUPPORTED_LANGUAGES,
            'confidence_threshold': LANGUAGE_CONFIDENCE_THRESHOLD,
            'default_language': DEFAULT_LANGUAGE,
            'method': LANGUAGE_DETECTION_METHOD
        },
        'processing': {
            'language_aware': LANGUAGE_AWARE_PROCESSING,
            'specific_patterns': LANGUAGE_SPECIFIC_PATTERNS,
            'multilingual_documents': ENABLE_MULTILINGUAL_DOCUMENTS,
            'adaptive_thresholds': LANGUAGE_ADAPTIVE_THRESHOLDS
        },
        'performance': {
            'max_execution_time': MAX_EXECUTION_TIME,
            'max_pages': MAX_PAGES,
            'caching_enabled': ENABLE_RESULT_CACHING,
            'timeout_seconds': ANALYSIS_TIMEOUT_SECONDS
        },
        'environment': {
            'mode': ENVIRONMENT,
            'debug': DEBUG_MODE,
            'config_loaded': LANGUAGE_CONFIG_LOADED
        },
        'feature_flags': FEATURE_FLAGS
    }

def validate_configuration():
    """Validate current configuration settings"""
    errors = []
    warnings = []
    
    # Validate language settings
    if LANGUAGE_CONFIDENCE_THRESHOLD < 0.0 or LANGUAGE_CONFIDENCE_THRESHOLD > 1.0:
        errors.append("LANGUAGE_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
    
    if DEFAULT_LANGUAGE not in SUPPORTED_LANGUAGES:
        warnings.append(f"DEFAULT_LANGUAGE '{DEFAULT_LANGUAGE}' not in SUPPORTED_LANGUAGES")
    
    # Validate performance settings
    if MAX_EXECUTION_TIME <= 0:
        errors.append("MAX_EXECUTION_TIME must be positive")
    
    if MAX_PAGES <= 0:
        errors.append("MAX_PAGES must be positive")
    
    # Validate directories
    for dir_path in [INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {e}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

# Utility functions for external access (NEW)
def get_language_display_name(language_code: str) -> str:
    """Get display name for a language code"""
    return LANGUAGE_DISPLAY_NAMES.get(language_code, language_code.title())

def get_supported_language_codes() -> list:
    """Get list of supported language codes"""
    return [LANGUAGE_CODES.get(lang, lang) for lang in SUPPORTED_LANGUAGES]

def is_language_supported(language: str) -> bool:
    """Check if a language is supported"""
    return language in SUPPORTED_LANGUAGES

def get_language_confidence_threshold(document_type: str = None) -> float:
    """Get confidence threshold for specific document type or default"""
    if document_type and document_type in DOCUMENT_TYPE_LANGUAGE_DETECTION:
        return DOCUMENT_TYPE_LANGUAGE_DETECTION[document_type]['confidence_threshold']
    return LANGUAGE_CONFIDENCE_THRESHOLD

# Initialize directories on import
try:
    for directory in [INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"⚠️  Warning: Could not create directories: {e}")

# Configuration validation on import
if __name__ != '__main__':
    validation = validate_configuration()
    if not validation['valid']:
        print(f"❌ Configuration validation failed: {validation['errors']}")
    elif validation['warnings']:
        print(f"⚠️  Configuration warnings: {validation['warnings']}")

# Export key configuration for easy access
__all__ = [
    'BASE_DIR', 'INPUT_DIR', 'OUTPUT_DIR', 'LOGS_DIR',
    'MAX_EXECUTION_TIME', 'MAX_MODEL_SIZE', 'MAX_PAGES',
    'LOGGING_CONFIG',
    'LANGUAGE_DETECTION_ENABLED', 'LANGUAGE_CONFIDENCE_THRESHOLD', 'DEFAULT_LANGUAGE',
    'SUPPORTED_LANGUAGES', 'LANGUAGE_DISPLAY_NAMES', 'LANGUAGE_CODES',
    'LANGUAGE_AWARE_PROCESSING', 'LANGUAGE_SPECIFIC_PATTERNS',
    'get_configuration_summary', 'validate_configuration',
    'get_language_display_name', 'is_language_supported'
]

if __name__ == '__main__':
    # Print configuration summary when run directly
    print("📋 PDF Outline Extractor Configuration Summary")
    print("=" * 50)
    
    summary = get_configuration_summary()
    for section, settings in summary.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    validation = validate_configuration()
    if validation['valid']:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
