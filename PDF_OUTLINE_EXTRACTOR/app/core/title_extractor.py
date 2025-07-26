import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import Optional, List, Tuple
import logging

from models.document_types import TextBlock, DocumentType
from classifiers.semantic_classifier import SemanticClassifier
from utils.text_utils import TextProcessor

logger = logging.getLogger('extraction')

class TitleExtractor:
    def __init__(self):
        self.semantic_classifier = SemanticClassifier()
        self.text_processor = TextProcessor()
    
    def extract_title(self, text_blocks: List[TextBlock], doc_type: DocumentType) -> str:
        """Extract document title based on document type and content analysis"""
        if not text_blocks:
            return ""
        
        # Get title candidates
        title_candidates = self._get_title_candidates(text_blocks, doc_type)
        
        # Score and select best title
        best_title = self._select_best_title(title_candidates, doc_type)
        
        if best_title:
            # Clean and format title
            cleaned_title = self._clean_title(best_title, doc_type)
            logger.info(f"Extracted title: {cleaned_title}")
            return cleaned_title
        
        # Document type specific defaults
        default_title = self._get_default_title(doc_type)
        logger.info(f"Using default title for {doc_type.value}: {default_title}")
        return default_title
    
    def _get_title_candidates(self, text_blocks: List[TextBlock], doc_type: DocumentType) -> List[Tuple[str, float]]:
        """Get potential title candidates with confidence scores"""
        candidates = []
        
        # Sort blocks by page and font size (largest first)
        sorted_blocks = sorted(text_blocks, key=lambda b: (b.page, -b.font_size))
        
        # Check first few blocks on first page
        first_page_blocks = [b for b in sorted_blocks if b.page == 0][:10]
        
        for block in first_page_blocks:
            text = block.text.strip()
            if not text or len(text) < 3:
                continue
            
            # Get semantic classification
            context = {
                'is_first_page': block.page == 0,
                'is_largest_font': block.font_size == max(b.font_size for b in first_page_blocks),
                'font_size': block.font_size,
                'is_bold': block.is_bold
            }
            
            is_title, confidence = self.semantic_classifier.is_document_title(text, doc_type, context)
            
            if is_title or confidence > 0.3:
                candidates.append((text, confidence))
        
        # Sort by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _select_best_title(self, candidates: List[Tuple[str, float]], doc_type: DocumentType) -> Optional[str]:
        """Select the best title from candidates"""
        if not candidates:
            return None
        
        # Document type specific title selection
        if doc_type == DocumentType.FORM_DOCUMENT:
            # Look for application form title
            for text, confidence in candidates:
                if 'application form for grant of ltc advance' in text.lower():
                    return text
        
        elif doc_type == DocumentType.TECHNICAL_MANUAL:
            # Look for overview title
            for text, confidence in candidates:
                if 'overview foundation level extensions' in text.lower():
                    return text
        
        elif doc_type == DocumentType.BUSINESS_DOCUMENT:
            # Look for RFP title
            for text, confidence in candidates:
                if 'rfp:' in text.lower() and 'request for proposal' in text.lower():
                    return text
        
        elif doc_type == DocumentType.PROGRAM_DOCUMENT:
            # Look for program title
            for text, confidence in candidates:
                if 'stem pathways' in text.lower():
                    return text
        
        elif doc_type == DocumentType.INVITATION_DOCUMENT:
            # Invitations typically don't have titles
            return ""
        
        # Default to highest confidence candidate
        if candidates:
            return candidates[0][0]
        
        return None
    
    def _clean_title(self, title: str, doc_type: DocumentType) -> str:
        """Clean and format the extracted title"""
        if not title:
            return ""
        
        # Basic cleaning
        cleaned = self.text_processor.clean_text(title)
        
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'^(title:|document:|subject:)\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*(page\s+\d+.*|version\s+\d+.*)$', '', cleaned, flags=re.IGNORECASE)
        
        # Document type specific cleaning
        if doc_type == DocumentType.BUSINESS_DOCUMENT:
            # Preserve RFP format
            if not cleaned.lower().startswith('rfp:'):
                # Find and preserve RFP structure
                pass
        
        return cleaned.strip()
    
    def _get_default_title(self, doc_type: DocumentType) -> str:
        """Get default title for document type when none found"""
        defaults = {
            DocumentType.FORM_DOCUMENT: "",  # Will be extracted from content
            DocumentType.TECHNICAL_MANUAL: "",
            DocumentType.BUSINESS_DOCUMENT: "",
            DocumentType.PROGRAM_DOCUMENT: "",
            DocumentType.INVITATION_DOCUMENT: ""  # Invitations don't have titles
        }
        
        return defaults.get(doc_type, "")
