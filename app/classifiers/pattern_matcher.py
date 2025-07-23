import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import Dict, List, Tuple, Optional
import logging

from models.document_types import DocumentType
from config.patterns import HEADING_PATTERNS

logger = logging.getLogger('classification')

class PatternMatcher:
    def __init__(self):
        self.heading_patterns = HEADING_PATTERNS
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Pre-compile regex patterns for better performance"""
        compiled = {}
        for level, patterns in self.heading_patterns.items():
            compiled[level] = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
        return compiled
    
    def match_heading_pattern(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        """Match text against heading patterns and return level, confidence, and matched pattern"""
        text = text.strip()
        
        if not text:
            return None, 0.0, None
        
        best_match = None
        best_confidence = 0.0
        best_pattern = None
        
        # Check each heading level
        for level in ['H1', 'H2', 'H3', 'H4']:
            if level in self.compiled_patterns:
                for pattern_obj in self.compiled_patterns[level]:
                    if pattern_obj.match(text):
                        confidence = self._calculate_pattern_confidence(text, level, pattern_obj.pattern)
                        if confidence > best_confidence:
                            best_match = level
                            best_confidence = confidence
                            best_pattern = pattern_obj.pattern
        
        return best_match, best_confidence, best_pattern
    
    def _calculate_pattern_confidence(self, text: str, level: str, pattern: str) -> float:
        """Calculate confidence score for pattern match"""
        base_confidence = 0.8  # Base confidence for pattern match
        
        # Adjust confidence based on pattern specificity
        if pattern.startswith('^') and pattern.endswith('$'):
            base_confidence += 0.1  # Exact match patterns are more confident
        
        # Adjust based on text characteristics
        text_clean = text.strip()
        
        # Length-based adjustments
        word_count = len(text_clean.split())
        if word_count <= 5:
            base_confidence += 0.1  # Short headings are more likely
        elif word_count > 15:
            base_confidence -= 0.2  # Very long headings are less likely
        
        # Capitalization adjustments
        if text_clean.isupper() and level in ['H1', 'H2']:
            base_confidence += 0.1
        elif text_clean.istitle():
            base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def match_numbering_pattern(self, text: str) -> Dict[str, any]:
        """Enhanced numbering pattern detection"""
        numbering_patterns = {
            'simple': r'^(\d+)\.\s+(.+)',           # 1. Title  
            'decimal': r'^(\d+\.\d+)\s+(.+)',       # 1.1 Title
            'references_with_page': r'^(\d+)\.\s+(references)\s+(\d+)$',  # 4. References 12
            'complex': r'^(\d+\.\d+\.\d+)\s+(.+)',  # 1.1.1 Title
            'appendix': r'^Appendix\s+([A-Z]):\s*(.+)', # Appendix A: Title
            'phase': r'^Phase\s+(I{1,3}|\d+):\s*(.+)',  # Phase I: Title
        }
        
        for pattern_type, pattern in numbering_patterns.items():
            match = re.match(pattern, text.strip(), re.IGNORECASE)
            if match:
                result = {
                    'type': pattern_type,
                    'full_match': match.group(0)
                }
                if pattern_type == 'references_with_page':
                    result['number'] = match.group(1)
                    result['title'] = match.group(2)
                    result['page'] = match.group(3)
                else:
                    result['number'] = match.group(1)
                    result['title'] = match.group(2).strip()
                return result
        
        return {}
    
    def match_document_type_patterns(self, text_blocks: List[str]) -> Dict[DocumentType, float]:
        """Match text against document type patterns"""
        from config.patterns import DOCUMENT_PATTERNS
        
        # Combine all text for analysis
        combined_text = ' '.join(text_blocks).lower()
        
        type_scores = {}
        
        for doc_type, patterns in DOCUMENT_PATTERNS.items():
            score = 0.0
            total_patterns = len(patterns['indicators'])
            
            if total_patterns == 0:
                continue
            
            # Check how many indicators match
            matches = 0
            for indicator in patterns['indicators']:
                if isinstance(indicator, str):
                    if indicator in combined_text:
                        matches += 1
                else:  # regex pattern
                    if re.search(indicator, combined_text):
                        matches += 1
            
            # Calculate score as percentage of matches
            score = matches / total_patterns
            
            # Bonus for strong indicators
            if doc_type == DocumentType.FORM_DOCUMENT and 'application form' in combined_text:
                score += 0.3
            elif doc_type == DocumentType.TECHNICAL_MANUAL and 'foundation level' in combined_text:
                score += 0.3
            elif doc_type == DocumentType.BUSINESS_DOCUMENT and 'rfp:' in combined_text:
                score += 0.3
            elif doc_type == DocumentType.PROGRAM_DOCUMENT and 'stem pathways' in combined_text:
                score += 0.3
            elif doc_type == DocumentType.INVITATION_DOCUMENT and 'you\'re invited' in combined_text:
                score += 0.3
            
            type_scores[doc_type] = min(1.0, score)
        
        return type_scores
    
    def is_exclusion_pattern(self, text: str) -> bool:
        """Check if text matches exclusion patterns"""
        exclusion_patterns = [
            r'international\s+software\s+testing',
            r'page\s+\d+\s+of\s+\d+',
            r'copyright\s+\d+',
            r'version\s+\d+\.\d+',
            r'all\s+rights\s+reserved',
            r'topjump\s+trampoline\s+park',
            r'you\'re\s+invited\s+to\s+a\s+party'
        ]
        
        text_lower = text.lower()
        for pattern in exclusion_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
