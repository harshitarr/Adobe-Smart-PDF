import sys
import os

# Fix import paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import base processor class
from processors.base_processor import BaseProcessor

# Import all specialized processors
from processors.form_processor import FormProcessor
from processors.manual_processor import ManualProcessor
from processors.business_processor import BusinessProcessor
from processors.program_processor import ProgramProcessor
from processors.invitation_processor import InvitationProcessor

# Export all processor classes
__all__ = [
    'BaseProcessor',
    'FormProcessor', 
    'ManualProcessor', 
    'BusinessProcessor', 
    'ProgramProcessor', 
    'InvitationProcessor'
]

# Version info for the processors module
__version__ = '1.1.0'

# Processor registry for dynamic loading
PROCESSOR_REGISTRY = {
    'form': FormProcessor,
    'manual': ManualProcessor,
    'business': BusinessProcessor,
    'program': ProgramProcessor,
    'invitation': InvitationProcessor
}

def get_processor_class(processor_type: str):
    """
    Get processor class by type name
    
    Args:
        processor_type (str): Type of processor ('form', 'manual', etc.)
        
    Returns:
        Class: Processor class or None if not found
    """
    return PROCESSOR_REGISTRY.get(processor_type.lower())

def list_available_processors():
    """
    List all available processor types
    
    Returns:
        list: List of available processor type names
    """
    return list(PROCESSOR_REGISTRY.keys())
