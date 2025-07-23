import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Performance constraints
MAX_EXECUTION_TIME = 10  # seconds
MAX_MODEL_SIZE = 200  # MB
MAX_PAGES = 50

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'extraction': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'extraction.log',
            'formatter': 'standard',
        },
        'classification': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'classification.log',
            'formatter': 'standard',
        },
        'errors': {
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'errors.log',
            'formatter': 'standard',
        },
    },
    'loggers': {
        'extraction': {
            'handlers': ['extraction'],
            'level': 'INFO',
            'propagate': False,
        },
        'classification': {
            'handlers': ['classification'],
            'level': 'INFO',
            'propagate': False,
        },
        'errors': {
            'handlers': ['errors'],
            'level': 'ERROR',
            'propagate': False,
        },
    },
}
