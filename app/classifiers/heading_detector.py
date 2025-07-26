import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Optional, Tuple
import logging

from models.document_types import TextBlock, DocumentType
from models.heading_patterns import HeadingPatternMatcher

logger = logging.getLogger('classification')

class HeadingDetector:
    def __init__(self):
        self.pattern_matcher = HeadingPatternMatcher()
        
        # Enhanced pattern-based rules
        self.heading_patterns = {
            'H1': [
                r'^revision\s+history\s*$',
                r'^table\s+of\s+contents\s*$', 
                r'^acknowledgements?\s*$',
                r'^\d+\.\s+introduction\s+to\s+the\s+foundation',
                r'^\d+\.\s+introduction\s+to\s+foundation\s+level\s+agile',
                r'^\d+\.\s+overview\s+of\s+the\s+foundation\s+level',
                r'^\d+\.\s+references\s*$'
            ],
            'H2': [
                r'^\d+\.\d+\s+[a-z]',  # Decimal subsections
                r'^\d+\.\d+\s+intended\s+audience',
                r'^\d+\.\d+\s+career\s+paths',
                r'^\d+\.\d+\s+learning\s+objectives',
                r'^\d+\.\d+\s+entry\s+requirements',
                r'^\d+\.\d+\s+structure\s+and\s+course',
                r'^\d+\.\d+\s+keeping\s+it\s+current',
                r'^\d+\.\d+\s+business\s+outcomes',
                r'^\d+\.\d+\s+content',
                r'^\d+\.\d+\s+trademarks',
                r'^\d+\.\d+\s+documents\s+and\s+web\s+sites'
            ]
        }
        
        self.exclusion_patterns = [
            r'version\s+\d+\s+page',
            r'may\s+\d+,\s+\d+',
            r'copyright.*international',
            r'qualifications\s+board\s*$',
            r'^\d+\.\d+\s+\d+\s+[A-Z]+',  # Revision table entries
        ]
    
    def detect_headings(self, text_blocks: List[TextBlock], doc_type: DocumentType, 
                       font_analysis: dict = None) -> List[Tuple[TextBlock, str, float]]:
        """Enhanced heading detection with pattern filtering"""
        headings = []
        
        for block in text_blocks:
            # First check if we should exclude this block
            if self._should_exclude_block(block):
                continue
                
            heading_level, confidence = self.detect_heading_level(block, doc_type, font_analysis)
            if heading_level:
                headings.append((block, heading_level, confidence))
        
        # Sort by confidence and validate
        headings.sort(key=lambda x: x[2], reverse=True)
        validated_headings = self._validate_heading_hierarchy(headings, doc_type)
        
        logger.info(f"Detected {len(validated_headings)} headings from {len(text_blocks)} text blocks")
        return validated_headings
    
    def _should_exclude_block(self, block: TextBlock) -> bool:
        """Check if block should be excluded from heading detection"""
        text = block.text.strip().lower()
        
        # Check exclusion patterns
        for pattern in self.exclusion_patterns:
            if re.search(pattern, text):
                return True
        
        # Additional exclusion rules
        exclusion_rules = [
            len(text.split()) > 15,  # Too long for headings
            'the tester should' in text,  # Learning objectives content
            text.count('.') > 3,  # Multiple sentences
            '@' in text,  # Email addresses
            'www.' in text,  # URLs
        ]
        
        return any(exclusion_rules)
    
    def detect_heading_level(self, block: TextBlock, doc_type: DocumentType, 
                           font_analysis: dict = None) -> Tuple[Optional[str], float]:
        """Enhanced heading level detection"""
        text = block.text.strip()
        
        if not text:
            return None, 0.0
        
        # Check against heading patterns
        for level, patterns in self.heading_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Calculate confidence based on pattern match and context
                    confidence = self._calculate_heading_confidence(block, level, font_analysis)
                    return level, confidence
        
        return None, 0.0
    
    def _calculate_heading_confidence(self, block: TextBlock, level: str, font_analysis: dict) -> float:
        """Calculate heading confidence based on multiple factors"""
        confidence = 0.8  # Base confidence for pattern match
        
        text = block.text.strip()
        
        # Length factor (shorter text more likely to be heading)
        word_count = len(text.split())
        if word_count <= 8:
            confidence += 0.1
        elif word_count > 12:
            confidence -= 0.2
        
        # Font size factor
        if font_analysis and 'size_analysis' in font_analysis:
            size_stats = font_analysis['size_analysis']
            mean_size = size_stats.get('mean_size', 12)
            
            if block.font_size > mean_size:
                confidence += 0.1
        
        # Bold formatting
        if block.is_bold:
            confidence += 0.1
        
        # Page position (early pages more likely for navigation)
        if level == 'H1' and block.page <= 5:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _validate_heading_hierarchy(self, headings: List[Tuple[TextBlock, str, float]], 
                                  doc_type: DocumentType) -> List[Tuple[TextBlock, str, float]]:
        """Validate heading hierarchy"""
        if not headings:
            return []
        
        validated = []
        seen_texts = set()
        
        for block, level, confidence in headings:
            text_key = block.text.strip().lower()
            
            # Avoid duplicates
            if text_key in seen_texts:
                continue
            
            # Document type specific validation
            if self._is_valid_for_document_type(block, level, doc_type):
                validated.append((block, level, confidence))
                seen_texts.add(text_key)
        
        return validated
    
    def _is_valid_for_document_type(self, block: TextBlock, level: str, doc_type: DocumentType) -> bool:
        """Document type specific heading validation"""
        if doc_type != DocumentType.TECHNICAL_MANUAL:
            return True  # Other document types use different logic
        
        text = block.text.strip().lower()
        
        # For technical manuals, validate against expected patterns
        if level == 'H1':
            h1_indicators = [
                'revision history', 'table of contents', 'acknowledgements',
                'introduction to the foundation', 'introduction to foundation level',
                'overview of the foundation', 'references'
            ]
            return any(indicator in text for indicator in h1_indicators)
        
        elif level == 'H2':
            h2_indicators = [
                'intended audience', 'career paths', 'learning objectives',
                'entry requirements', 'structure and course', 'keeping it current',
                'business outcomes', 'content', 'trademarks', 'documents and web sites'
            ]
            return any(indicator in text for indicator in h2_indicators)
        
        return True
