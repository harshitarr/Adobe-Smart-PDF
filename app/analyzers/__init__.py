import sys
import os

# Fix import paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all analyzer classes
from analyzers.font_analyzer import FontAnalyzer
from analyzers.spatial_analyzer import SpatialAnalyzer
from analyzers.text_analyzer import TextAnalyzer
from analyzers.structure_analyzer import StructureAnalyzer
from analyzers.language_analyzer import LanguageAnalyzer

# Export all analyzer classes
__all__ = [
    'FontAnalyzer', 
    'SpatialAnalyzer', 
    'TextAnalyzer', 
    'StructureAnalyzer',
    'LanguageAnalyzer'
]

# Version info for the analyzers module
__version__ = '1.2.0'

# Analyzer registry for dynamic loading
ANALYZER_REGISTRY = {
    'font': FontAnalyzer,
    'spatial': SpatialAnalyzer,
    'text': TextAnalyzer,
    'structure': StructureAnalyzer,
    'language': LanguageAnalyzer
}

# Analyzer categories for organized access
ANALYZER_CATEGORIES = {
    'document_analysis': ['font', 'spatial', 'text', 'structure'],
    'content_analysis': ['text', 'structure', 'language'],
    'layout_analysis': ['font', 'spatial'],
    'multilingual': ['language', 'text']
}

def get_analyzer_class(analyzer_type: str):
    """
    Get analyzer class by type name
    
    Args:
        analyzer_type (str): Type of analyzer ('font', 'spatial', 'text', 'structure', 'language')
        
    Returns:
        Class: Analyzer class or None if not found
    """
    return ANALYZER_REGISTRY.get(analyzer_type.lower())

def list_available_analyzers():
    """
    List all available analyzer types
    
    Returns:
        list: List of available analyzer type names
    """
    return list(ANALYZER_REGISTRY.keys())

def get_analyzers_by_category(category: str):
    """
    Get analyzer classes by category
    
    Args:
        category (str): Category name ('document_analysis', 'content_analysis', etc.)
        
    Returns:
        dict: Dictionary of analyzer name to class mappings for the category
    """
    if category not in ANALYZER_CATEGORIES:
        return {}
    
    analyzer_names = ANALYZER_CATEGORIES[category]
    return {name: ANALYZER_REGISTRY[name] for name in analyzer_names if name in ANALYZER_REGISTRY}

def create_analyzer_suite(category: str = 'document_analysis'):
    """
    Create a suite of analyzers for a specific category
    
    Args:
        category (str): Category of analyzers to create
        
    Returns:
        dict: Dictionary of initialized analyzer instances
    """
    analyzers = get_analyzers_by_category(category)
    return {name: analyzer_class() for name, analyzer_class in analyzers.items()}

def is_multilingual_supported():
    """
    Check if multilingual analysis is supported
    
    Returns:
        bool: True if LanguageAnalyzer is available
    """
    return 'language' in ANALYZER_REGISTRY

# Initialize default analyzer configuration
DEFAULT_ANALYZER_CONFIG = {
    'font': {
        'clustering_threshold': 0.1,
        'min_font_size': 8,
        'max_font_size': 72
    },
    'spatial': {
        'column_detection_threshold': 0.3,
        'line_spacing_factor': 1.2
    },
    'text': {
        'min_text_length': 3,
        'pattern_confidence_threshold': 0.6
    },
    'structure': {
        'hierarchy_depth': 6,
        'structure_confidence_threshold': 0.5
    },
    'language': {
        'confidence_threshold': 0.4,
        'sample_size': 75,
        'supported_languages': ['japanese', 'german', 'tamil', 'english']
    }
}

def get_default_config(analyzer_type: str = None):
    """
    Get default configuration for analyzer(s)
    
    Args:
        analyzer_type (str, optional): Specific analyzer type, or None for all
        
    Returns:
        dict: Configuration dictionary
    """
    if analyzer_type:
        return DEFAULT_ANALYZER_CONFIG.get(analyzer_type.lower(), {})
    return DEFAULT_ANALYZER_CONFIG.copy()
