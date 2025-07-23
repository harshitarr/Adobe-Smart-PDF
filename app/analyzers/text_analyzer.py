import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Optional
import logging

from models.document_types import TextBlock, DocumentType
from config.patterns import HEADING_PATTERNS, CONTENT_FILTERS

logger = logging.getLogger('extraction')

class TextAnalyzer:
    def __init__(self):
        self.heading_patterns = HEADING_PATTERNS
        self.content_filters = CONTENT_FILTERS
    
    def analyze_text_patterns(self, text_blocks: List[TextBlock], doc_type: DocumentType) -> Dict[str, Any]:
        """Analyze text patterns to identify structure"""
        if not text_blocks:
            return {}
        
        # Analyze heading patterns
        heading_analysis = self._analyze_heading_patterns(text_blocks)
        
        # Analyze numbering patterns
        numbering_analysis = self._analyze_numbering_patterns(text_blocks)
        
        # Analyze content vs structure
        content_analysis = self._analyze_content_structure(text_blocks, doc_type)
        
        analysis = {
            'headings': heading_analysis,
            'numbering': numbering_analysis,
            'content': content_analysis,
            'total_blocks': len(text_blocks)
        }
        
        logger.info(f"Text pattern analysis completed for {len(text_blocks)} blocks")
        return analysis
    
    def _analyze_heading_patterns(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze heading patterns in text"""
        heading_matches = {'H1': [], 'H2': [], 'H3': [], 'H4': []}
        
        for block in text_blocks:
            text = block.text.strip()
            if not text:
                continue
            
            # Check against heading patterns
            for level, patterns in self.heading_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, text, re.IGNORECASE | re.MULTILINE):
                        heading_matches[level].append({
                            'text': text,
                            'page': block.page,
                            'pattern': pattern,
                            'block': block
                        })
                        break
        
        return heading_matches
    
    def _analyze_numbering_patterns(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze numbering patterns"""
        numbering_patterns = {
            'simple_numbered': r'^\d+\.\s+',  # 1. 2. 3.
            'decimal_numbered': r'^\d+\.\d+\s+',  # 1.1 2.1 3.2
            'roman_numbered': r'^[IVX]+\.\s+',  # I. II. III.
            'letter_numbered': r'^[A-Z]\.\s+',  # A. B. C.
            'phase_numbered': r'^Phase\s+(I{1,3}|\d+):\s+'  # Phase I: Phase II:
        }
        
        pattern_matches = {}
        
        for pattern_name, pattern in numbering_patterns.items():
            matches = []
            for block in text_blocks:
                if re.match(pattern, block.text.strip(), re.IGNORECASE):
                    matches.append({
                        'text': block.text.strip(),
                        'page': block.page,
                        'block': block
                    })
            pattern_matches[pattern_name] = matches
        
        return pattern_matches
    
    def _analyze_content_structure(self, text_blocks: List[TextBlock], doc_type: DocumentType) -> Dict[str, Any]:
        """Analyze content vs structural elements"""
        structural_blocks = []
        content_blocks = []
        ignored_blocks = []
        
        for block in text_blocks:
            text = block.text.strip()
            if not text:
                continue
            
            # Check if should be ignored
            if self._should_ignore_text(text):
                ignored_blocks.append(block)
                continue
            
            # Check if it's structural
            if self._is_structural_text(text, doc_type):
                structural_blocks.append(block)
            else:
                content_blocks.append(block)
        
        return {
            'structural': len(structural_blocks),
            'content': len(content_blocks),
            'ignored': len(ignored_blocks),
            'structural_blocks': structural_blocks,
            'content_blocks': content_blocks,
            'ignored_blocks': ignored_blocks
        }
    
    def _should_ignore_text(self, text: str) -> bool:
        """Check if text should be ignored based on filter patterns"""
        text_lower = text.lower()
        
        for pattern in self.content_filters['IGNORE_PATTERNS']:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _is_structural_text(self, text: str, doc_type: DocumentType) -> bool:
        """Determine if text is structural vs content"""
        # Check against heading patterns
        for level, patterns in self.heading_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return True
        
        # Check for structural indicators based on document type
        if doc_type == DocumentType.FORM_DOCUMENT:
            # Forms have minimal structure
            return False
        elif doc_type == DocumentType.TECHNICAL_MANUAL:
            # Look for section headers, TOC entries
            structural_indicators = [
                r'^\d+\.\s+',
                r'^\d+\.\d+\s+',
                r'^table\s+of\s+contents',
                r'^revision\s+history',
                r'^acknowledgements'
            ]
        elif doc_type == DocumentType.BUSINESS_DOCUMENT:
            # Look for business section headers
            structural_indicators = [
                r'^summary\s*$',
                r'^background\s*$',
                r'^appendix\s+[A-Z]:',
                r'^phase\s+(I{1,3}|\d+):',
                r'.*:\s*$'  # Colon-terminated
            ]
        elif doc_type == DocumentType.PROGRAM_DOCUMENT:
            # Look for program section headers
            structural_indicators = [
                r'^pathway\s+options',
                r'^regular\s+pathway',
                r'^distinction\s+pathway'
            ]
        elif doc_type == DocumentType.INVITATION_DOCUMENT:
            # Look for invitation structure
            structural_indicators = [
                r'^hope\s+to\s+see\s+you\s+there'
            ]
        else:
            return False
        
        # Check against structural indicators
        for pattern in structural_indicators:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def detect_heading_level(self, text: str, doc_type: DocumentType) -> Optional[str]:
        """Detect heading level for given text"""
        for level in ['H1', 'H2', 'H3', 'H4']:
            if level in self.heading_patterns:
                for pattern in self.heading_patterns[level]:
                    if re.match(pattern, text.strip(), re.IGNORECASE):
                        return level
        return None
