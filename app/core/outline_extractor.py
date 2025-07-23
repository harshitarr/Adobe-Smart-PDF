import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List, Dict, Any
import logging

from models.document_types import TextBlock, DocumentType, HeadingLevel, DocumentStructure
from analyzers.structure_analyzer import StructureAnalyzer
from classifiers.content_filter import ContentFilter
from classifiers.heading_detector import HeadingDetector

logger = logging.getLogger('extraction')

class OutlineExtractor:
    def __init__(self):
        self.structure_analyzer = StructureAnalyzer()
        self.content_filter = ContentFilter()
        self.heading_detector = HeadingDetector()
    
    def extract_outline(self, text_blocks: List[TextBlock], doc_type: DocumentType) -> List[HeadingLevel]:
        """Extract hierarchical outline from document"""
        if not text_blocks:
            return []
        
        logger.info(f"Extracting outline for {doc_type.value} document with {len(text_blocks)} text blocks")
        
        # Filter structural elements
        structural_blocks = self.content_filter.filter_structural_elements(text_blocks, doc_type)
        
        if not structural_blocks:
            logger.warning("No structural elements found")
            return []
        
        # Analyze document structure
        font_analysis = self.structure_analyzer.font_analyzer.analyze_fonts(structural_blocks)
        structure_analysis = self.structure_analyzer.analyze_document_structure(structural_blocks, doc_type)
        
        # Detect headings
        headings = self.heading_detector.detect_headings(structural_blocks, doc_type, font_analysis)
        
        # Convert to HeadingLevel objects
        outline = []
        for block, level, confidence in headings:
            heading = HeadingLevel(
                level=level,
                text=block.text.strip(),
                page=block.page,
                confidence=confidence,
                font_size=block.font_size,
                font_name=block.font_name
            )
            outline.append(heading)
        
        # Sort by page and position
        outline.sort(key=lambda h: (h.page, -h.confidence))
        
        # Apply document type specific filtering
        filtered_outline = self._filter_outline_by_document_type(outline, doc_type)
        
        logger.info(f"Extracted {len(filtered_outline)} outline items")
        return filtered_outline
    
    def _filter_outline_by_document_type(self, outline: List[HeadingLevel], doc_type: DocumentType) -> List[HeadingLevel]:
        """Filter outline based on document type rules"""
        if doc_type == DocumentType.FORM_DOCUMENT:
            # Forms have no outline
            return []
        
        elif doc_type == DocumentType.TECHNICAL_MANUAL:
            # Keep all valid headings for technical manuals
            return [h for h in outline if h.confidence > 0.6]
        
        elif doc_type == DocumentType.BUSINESS_DOCUMENT:
            # Keep all levels for business documents
            return [h for h in outline if h.confidence > 0.5]
        
        elif doc_type == DocumentType.PROGRAM_DOCUMENT:
            # Keep minimal structure for program documents
            filtered = [h for h in outline if h.confidence > 0.7]
            return filtered[:5]  # Max 5 items
        
        elif doc_type == DocumentType.INVITATION_DOCUMENT:
            # Keep only high-confidence items for invitations
            filtered = [h for h in outline if h.confidence > 0.8]
            return filtered[:2]  # Max 2 items
        
        return outline
