import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
import unicodedata
from typing import List, Optional

class TextProcessor:
    def __init__(self):
        # Generalized stop words for title detection
        self.stop_words = {
            'international', 'software', 'testing', 'qualifications', 'board',
            'page', 'of', 'version', 'copyright', 'all', 'rights', 'reserved',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'
        }
        
        # Generalized patterns
        self.patterns = {
            'title_indicators': [
                r'overview', r'introduction', r'manual', r'guide',
                r'handbook', r'reference', r'foundation'
            ],
            'heading_patterns': [
                r'^\d+\.\s+', r'^\d+\.\d+\s+', r'^[A-Z][A-Z\s]+$',
                r'.*:\s*$', r'^Appendix\s+[A-Z]:', r'^Phase\s+(I{1,3}|\d+):'
            ],
            'content_patterns': [
                r'bullet\s+point', r'for\s+office\s+use\s+only',
                r'signature\s*:', r'date\s*:', r'email\s*:',
                r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', r'www\.',
                r'^\$[\d,]+', r'%\s*$'
            ]
        }
    
    def clean_text(self, text: str) -> str:
        """Generalized text cleaning for Adobe compliance"""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove all types of whitespace from ends
        text = text.strip(' \t\n\r\f\v')
        
        # Normalize internal spaces (multiple spaces to single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Final cleanup - one more strip
        text = text.strip()
        
        return text
    
    def is_likely_title(self, text: str) -> bool:
        """Generalized title detection"""
        if not text or len(text) < 3:
            return False
        
        clean_text = self.clean_text(text.lower())
        words = clean_text.split()
        
        if not words:
            return False
        
        # Check for title indicators
        indicator_count = sum(1 for pattern in self.patterns['title_indicators'] 
                            if re.search(pattern, clean_text))
        
        # Title characteristics
        has_indicators = indicator_count >= 1
        right_length = 2 <= len(words) <= 10
        not_too_generic = self._calculate_stop_word_ratio(words) <= 0.6
        no_numbering = not re.match(r'^\d+\.', clean_text)
        
        return has_indicators and right_length and not_too_generic and no_numbering
    
    def is_likely_heading(self, text: str) -> bool:
        """Generalized heading detection"""
        if not text:
            return False
        
        clean_text = self.clean_text(text)
        
        # Check against heading patterns
        for pattern in self.patterns['heading_patterns']:
            if re.match(pattern, clean_text, re.IGNORECASE):
                return True
        
        # Additional heading characteristics
        word_count = len(clean_text.split())
        is_short = word_count <= 8
        no_sentence_ending = not clean_text.endswith('.')
        
        return is_short and no_sentence_ending
    
    def is_content_text(self, text: str) -> bool:
        """Generalized content detection"""
        if not text:
            return False
        
        # Check against content patterns
        for pattern in self.patterns['content_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Length-based content detection
        word_count = len(text.split())
        is_long_text = word_count > 20
        is_sentence = text.endswith('.') and word_count > 5
        
        return is_long_text or is_sentence
    
    def _calculate_stop_word_ratio(self, words: List[str]) -> float:
        """Calculate ratio of stop words in text"""
        if not words:
            return 0.0
        
        stop_word_count = sum(1 for word in words if word.lower() in self.stop_words)
        return stop_word_count / len(words)
    
    def extract_numbering(self, text: str) -> Optional[str]:
        """Generalized numbering extraction"""
        patterns = [
            r'^(\d+)\.\s+',           # 1. 
            r'^(\d+\.\d+)\s+',        # 1.1 
            r'^(\d+\.\d+\.\d+)\s+'    # 1.1.1 
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def detect_toc_entry(self, text: str) -> bool:
        """Generalized TOC entry detection"""
        text_lower = text.lower().strip()
        
        # Generalized TOC patterns
        toc_patterns = [
            r'.*\d+\s*$',                    # Ends with page numbers
            r'.*\.{3,}.*\d+',               # Dots leading to page numbers
            r'\w+.*\d+.*\w+.*\d+',          # Multiple items with numbers
            r'\d+\.\d+\s+\w+.*\d+'          # Numbered items with page refs
        ]
        
        return any(re.search(pattern, text_lower) for pattern in toc_patterns)
    
    def smart_text_similarity(self, text1: str, text2: str) -> float:
        """Generalized text similarity calculation"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        norm1 = self.clean_text(text1.lower())
        norm2 = self.clean_text(text2.lower())
        
        # Word-based similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def format_section_text(self, text: str) -> str:
        """Generalized section text formatting"""
        clean_text = self.clean_text(text)
        
        # Ensure proper capitalization
        if clean_text and not clean_text[0].isupper():
            clean_text = clean_text[0].upper() + clean_text[1:]
        
        return clean_text
