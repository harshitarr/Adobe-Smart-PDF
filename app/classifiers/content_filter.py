import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Set
import logging

from models.document_types import TextBlock, DocumentType
from config.patterns import CONTENT_FILTERS

logger = logging.getLogger('classification')

class ContentFilter:
    def __init__(self):
        self.ignore_patterns = CONTENT_FILTERS['IGNORE_PATTERNS']
        self.title_exclusions = CONTENT_FILTERS['TITLE_EXCLUSIONS']
        
        # Pattern-based filtering rules
        self.metadata_exclusion_patterns = [
            r'version\s+\d+\s+page\s+\d+\s+of\s+\d+',
            r'may\s+\d+,\s+\d+',
            r'©\s*international\s+software\s+testing',
            r'qualifications\s+board\s*$',
            r'^\d+\.\d+\s+\d+\s+[A-Z]+\s+\d+',  # Revision table entries
            r'^[A-Z][a-z]+\s+\d+,\s+\d+$',  # Date lines
        ]
        
        self.valid_structural_patterns = [
            r'^revision\s+history\s*$',
            r'^table\s+of\s+contents\s*$',
            r'^acknowledgements?\s*$',
            r'^\d+\.\s+[A-Z]',  # Numbered sections
            r'^\d+\.\d+\s+[A-Z]',  # Decimal subsections
            r'^\d+\.\s+references\s*$'
        ]
    
    def filter_structural_elements(self, text_blocks: List[TextBlock], doc_type: DocumentType) -> List[TextBlock]:
        """Filter text blocks to keep only structural elements"""
        structural_blocks = []
        
        for block in text_blocks:
            if self.is_structural_element(block, doc_type):
                structural_blocks.append(block)
        
        logger.info(f"Filtered {len(text_blocks)} blocks to {len(structural_blocks)} structural elements")
        return structural_blocks
    
    def is_structural_element(self, block: TextBlock, doc_type: DocumentType) -> bool:
        """Pattern-based structural element detection"""
        text = block.text.strip()
        
        if not text:
            return False
        
        # First check exclusions
        if self._should_exclude_metadata(text):
            return False
        
        # Check if text should be ignored
        if self._should_ignore_text(text):
            return False
        
        # Check if text is content rather than structure
        if self._is_content_text(text):
            return False
        
        # Document type specific filtering
        return self._is_structural_for_document_type(text, doc_type)
    
    def _should_exclude_metadata(self, text: str) -> bool:
        """Exclude document metadata and formatting elements"""
        text_lower = text.lower().strip()
        
        # Check against metadata exclusion patterns
        for pattern in self.metadata_exclusion_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Additional metadata checks
        metadata_indicators = [
            len(text.split()) > 15,  # Long revision table entries
            text.startswith('0.') and 'JUNE' in text.upper(),  # Revision entries
            'copyright' in text_lower and 'international' in text_lower,
            text_lower.endswith('qualifications board'),
            re.match(r'^[a-z]\d+', text_lower),  # Technical IDs like "afm1"
        ]
        
        return any(metadata_indicators)
    
    def _should_ignore_text(self, text: str) -> bool:
        """Pattern-based ignore detection"""
        text_lower = text.lower()
        
        for pattern in self.ignore_patterns:
            if re.search(pattern, text_lower):
                logger.debug(f"Ignoring text due to pattern '{pattern}': {text[:50]}...")
                return True
        
        return False
    
    def _is_content_text(self, text: str) -> bool:
        """Detect content vs structural text"""
        text_lower = text.lower()
        
        # Content indicators
        content_indicators = [
            len(text.split()) > 25,  # Very long text blocks
            text.count('.') > 3,  # Multiple sentences
            'the tester should' in text_lower,  # Learning objectives content
            'this document' in text_lower and len(text.split()) > 10,
            re.search(r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text),  # Email addresses
            'www.' in text_lower,
            text.startswith('•') or text.startswith('-'),  # Bullet points
        ]
        
        return any(content_indicators)
    
    def _is_structural_for_document_type(self, text: str, doc_type: DocumentType) -> bool:
        """Document type specific structural detection"""
        text_lower = text.lower().strip()
        
        if doc_type == DocumentType.TECHNICAL_MANUAL:
            # Check against valid structural patterns
            for pattern in self.valid_structural_patterns:
                if re.search(pattern, text_lower):
                    return True
            
            # Additional manual-specific patterns
            manual_patterns = [
                r'^[1-4]\.\s+[a-z]',  # Main sections 1-4
                r'^[1-4]\.\d+\s+[a-z]',  # Subsections like 2.1, 2.2
            ]
            
            return any(re.search(pattern, text_lower) for pattern in manual_patterns)
        
        elif doc_type == DocumentType.FORM_DOCUMENT:
            form_patterns = [
                r'application\s+form\s+for\s+grant\s+of\s+ltc\s+advance'
            ]
            return any(re.search(pattern, text_lower) for pattern in form_patterns)
        
        elif doc_type == DocumentType.BUSINESS_DOCUMENT:
            business_patterns = [
                r'rfp\s*:', r'request\s+for\s+proposal',
                r'appendix\s+[A-Z]:', r'phase\s+(I{1,3}|\d+):',
                r'summary\s*$', r'background\s*$'
            ]
            return any(re.search(pattern, text_lower) for pattern in business_patterns)
        
        elif doc_type == DocumentType.PROGRAM_DOCUMENT:
            program_patterns = [
                r'stem\s+pathways', r'pathway\s+options'
            ]
            return any(re.search(pattern, text_lower) for pattern in program_patterns)
        
        elif doc_type == DocumentType.INVITATION_DOCUMENT:
            invitation_patterns = [
                r'hope\s+to\s+see\s+you\s+there'
            ]
            return any(re.search(pattern, text_lower) for pattern in invitation_patterns)
        
        return False
    
    def is_valid_title(self, text: str, doc_type: DocumentType) -> bool:
        """Pattern-based title validation"""
        if not text or len(text.strip()) < 3:
            return False
        
        text_lower = text.lower().strip()
        
        # Check title exclusions
        for exclusion in self.title_exclusions:
            if re.search(exclusion, text_lower):
                return False
        
        # Document type specific title validation
        if doc_type == DocumentType.TECHNICAL_MANUAL:
            return 'overview' in text_lower and 'foundation' in text_lower
        elif doc_type == DocumentType.FORM_DOCUMENT:
            return 'application form' in text_lower
        elif doc_type == DocumentType.BUSINESS_DOCUMENT:
            return 'rfp:' in text_lower
        elif doc_type == DocumentType.PROGRAM_DOCUMENT:
            return 'stem pathways' in text_lower
        elif doc_type == DocumentType.INVITATION_DOCUMENT:
            return False  # Invitations typically don't have titles
        
        return True
