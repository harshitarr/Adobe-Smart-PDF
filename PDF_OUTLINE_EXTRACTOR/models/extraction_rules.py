from typing import Dict, List, Callable
from .document_types import DocumentType, TextBlock, HeadingLevel

class ExtractionRules:
    def __init__(self):
        self.rules = {
            DocumentType.FORM_DOCUMENT: {
                'extract_outline': False,
                'title_required': True,
                'max_outline_items': 0,
                'content_filters': ['form_fields', 'signatures', 'declarations']
            },
            
            DocumentType.TECHNICAL_MANUAL: {
                'extract_outline': True,
                'title_required': True,
                'max_outline_items': 50,
                'content_filters': ['toc_entries', 'section_headers']
            },
            
            DocumentType.BUSINESS_DOCUMENT: {
                'extract_outline': True,
                'title_required': True,
                'max_outline_items': 100,
                'content_filters': ['business_sections', 'appendices']
            },
            
            DocumentType.PROGRAM_DOCUMENT: {
                'extract_outline': True,
                'title_required': True,
                'max_outline_items': 5,
                'content_filters': ['program_sections']
            },
            
            DocumentType.INVITATION_DOCUMENT: {
                'extract_outline': True,
                'title_required': False,
                'max_outline_items': 2,
                'content_filters': ['invitation_elements']
            }
        }
    
    def should_extract_outline(self, doc_type: DocumentType) -> bool:
        return self.rules[doc_type]['extract_outline']
    
    def is_title_required(self, doc_type: DocumentType) -> bool:
        return self.rules[doc_type]['title_required']
    
    def get_max_outline_items(self, doc_type: DocumentType) -> int:
        return self.rules[doc_type]['max_outline_items']
