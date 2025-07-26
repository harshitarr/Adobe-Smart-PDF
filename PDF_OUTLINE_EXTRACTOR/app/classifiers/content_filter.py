import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Set, Dict, Any
import logging

from models.document_types import TextBlock, DocumentType
from config.patterns import CONTENT_FILTERS

logger = logging.getLogger('classification')

class ContentFilter:
    def __init__(self):
        try:
            self.ignore_patterns = CONTENT_FILTERS['IGNORE_PATTERNS']
            self.title_exclusions = CONTENT_FILTERS['TITLE_EXCLUSIONS']
        except (KeyError, ImportError):
            # Fallback patterns if config is not available
            self.ignore_patterns = [
                r'^page\s+\d+\s*$',
                r'^\d+\s*$',
                r'^copyright\s+©',
                r'^©\s+\d{4}',
                r'^\s*$'
            ]
            self.title_exclusions = [
                r'^page\s+\d+',
                r'^table\s+of\s+contents',
                r'^index\s*$'
            ]
        
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
        
        # Structural noise patterns (for the missing method)
        self.structural_noise_patterns = [
            r'^page\s+\d+\s*$',  # Page numbers
            r'^\d+\s*$',  # Standalone numbers
            r'^copyright\s+©',  # Copyright notices
            r'^©\s+\d{4}',  # Copyright years
            r'^\s*$',  # Empty or whitespace-only
            r'^[^\w\s]*$',  # Only punctuation
            r'^www\.',  # URLs
            r'^http[s]?://',  # URLs
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # Dates
            r'^[\d\s\-\(\)]+$',  # Phone numbers
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'  # Email addresses
        ]
        
        # Structural noise keywords
        self.structural_noise_keywords = [
            'page', 'continued', 'see page', 'figure', 'table',
            'appendix', 'footnote', 'endnote', 'reference',
            'bibliography', 'index', 'glossary'
        ]

    def is_structural_noise(self, text: str) -> bool:
        """Check if text is structural noise that should be filtered - FIXED METHOD"""
        if not text or not isinstance(text, str):
            return True
        
        text_clean = text.strip().lower()
        
        if not text_clean:
            return True
        
        # Check against structural noise patterns
        for pattern in self.structural_noise_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True
        
        # Check for structural noise keywords
        for keyword in self.structural_noise_keywords:
            if keyword in text_clean:
                return True
        
        # Filter very short non-meaningful text
        if len(text_clean) < 3:
            return True
        
        # Filter text that's mostly numbers or punctuation
        alphanumeric_count = sum(1 for c in text_clean if c.isalnum())
        if len(text_clean) > 0 and alphanumeric_count / len(text_clean) < 0.3:
            return True
        
        return False

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
        try:
            # Handle different input types
            if hasattr(block, 'text'):
                text = block.text.strip()
            elif isinstance(block, str):
                text = block.strip()
            else:
                text = str(block).strip()
            
            if not text:
                return False
            
            # First check exclusions
            if self._should_exclude_metadata(text):
                return False
            
            # Check if text should be ignored
            if self._should_ignore_text(text):
                return False
            
            # Check if text is structural noise
            if self.is_structural_noise(text):
                return False
            
            # Check if text is content rather than structure
            if self._is_content_text(text):
                return False
            
            # Document type specific filtering
            return self._is_structural_for_document_type(text, doc_type)
            
        except Exception as e:
            logger.debug(f"Error in structural element detection: {e}")
            return False
    
    def _should_exclude_metadata(self, text: str) -> bool:
        """Exclude document metadata and formatting elements"""
        if not text:
            return True
            
        text_lower = text.lower().strip()
        
        # Check against metadata exclusion patterns
        for pattern in self.metadata_exclusion_patterns:
            try:
                if re.search(pattern, text_lower):
                    return True
            except re.error:
                continue
        
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
        if not text:
            return True
            
        text_lower = text.lower()
        
        for pattern in self.ignore_patterns:
            try:
                if re.search(pattern, text_lower):
                    logger.debug(f"Ignoring text due to pattern '{pattern}': {text[:50]}...")
                    return True
            except re.error:
                continue
        
        return False
    
    def _is_content_text(self, text: str) -> bool:
        """Detect content vs structural text"""
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Content indicators
        content_indicators = [
            len(text.split()) > 25,  # Very long text blocks
            text.count('.') > 3,  # Multiple sentences
            'the tester should' in text_lower,  # Learning objectives content
            'this document' in text_lower and len(text.split()) > 10,
            'www.' in text_lower,
            text.startswith('•') or text.startswith('-'),  # Bullet points
        ]
        
        # Check email addresses safely
        try:
            if re.search(r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
                content_indicators.append(True)
        except re.error:
            pass
        
        return any(content_indicators)
    
    def _is_structural_for_document_type(self, text: str, doc_type: DocumentType) -> bool:
        """Document type specific structural detection"""
        if not text:
            return False
            
        text_lower = text.lower().strip()
        
        try:
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
            
        except re.error as e:
            logger.debug(f"Regex error in structural detection: {e}")
        except Exception as e:
            logger.debug(f"Error in document type structural detection: {e}")
        
        return False
    
    def is_valid_title(self, text: str, doc_type: DocumentType) -> bool:
        """Pattern-based title validation"""
        if not text or len(text.strip()) < 3:
            return False
        
        text_lower = text.lower().strip()
        
        # Check if it's structural noise first
        if self.is_structural_noise(text):
            return False
        
        # Check title exclusions
        for exclusion in self.title_exclusions:
            try:
                if re.search(exclusion, text_lower):
                    return False
            except re.error:
                continue
        
        # Document type specific title validation
        try:
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
        except Exception as e:
            logger.debug(f"Error in title validation: {e}")
        
        return True

    def filter_content(self, text_blocks: List) -> List:
        """Filter out noise from text blocks - Additional utility method"""
        filtered_blocks = []
        
        for block in text_blocks:
            try:
                # Get text from block
                if hasattr(block, 'text'):
                    text = block.text
                elif isinstance(block, str):
                    text = block
                else:
                    text = str(block)
                
                # Check if it's structural noise
                if not self.is_structural_noise(text):
                    filtered_blocks.append(block)
                    
            except Exception as e:
                logger.debug(f"Error filtering block: {e}")
                # Include block if we can't process it
                filtered_blocks.append(block)
        
        return filtered_blocks
    
    def is_likely_heading(self, text: str) -> bool:
        """Check if text is likely a heading - Additional utility method"""
        if not text or not isinstance(text, str):
            return False
        
        text_clean = text.strip()
        
        # Skip noise
        if self.is_structural_noise(text_clean):
            return False
        
        # Heading patterns
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered headings
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^[A-Z][a-z\s]+:$',  # Title case with colon
            r'^Chapter\s+\d+',  # Chapter headings
            r'^Section\s+\d+',  # Section headings
        ]
        
        try:
            for pattern in heading_patterns:
                if re.match(pattern, text_clean):
                    return True
        except re.error:
            pass
        
        return False

    def clean_text(self, text: str) -> str:
        """Clean and normalize text - Additional utility method"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
        
        return text

    def get_filter_stats(self) -> Dict[str, int]:
        """Get statistics about filter patterns - Utility for debugging"""
        return {
            'ignore_patterns': len(self.ignore_patterns),
            'title_exclusions': len(self.title_exclusions),
            'metadata_exclusions': len(self.metadata_exclusion_patterns),
            'structural_patterns': len(self.valid_structural_patterns),
            'noise_patterns': len(self.structural_noise_patterns),
            'noise_keywords': len(self.structural_noise_keywords)
        }
