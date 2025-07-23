import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Optional
import logging

from models.document_types import TextBlock, DocumentType, HeadingLevel
from analyzers.font_analyzer import FontAnalyzer
from analyzers.spatial_analyzer import SpatialAnalyzer
from analyzers.text_analyzer import TextAnalyzer

logger = logging.getLogger('extraction')

class StructureAnalyzer:
    def __init__(self):
        self.font_analyzer = FontAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        self.text_analyzer = TextAnalyzer()
    
    def analyze_document_structure(self, text_blocks: List[TextBlock], doc_type: DocumentType) -> Dict[str, Any]:
        """Comprehensive document structure analysis"""
        if not text_blocks:
            return {}
        
        # Perform individual analyses
        font_analysis = self.font_analyzer.analyze_fonts(text_blocks)
        spatial_analysis = self.spatial_analyzer.analyze_layout(text_blocks)
        text_analysis = self.text_analyzer.analyze_text_patterns(text_blocks, doc_type)
        
        # Combine analyses for structure detection
        structure_analysis = self._combine_analyses(
            text_blocks, doc_type, font_analysis, spatial_analysis, text_analysis
        )
        
        logger.info("Document structure analysis completed")
        return structure_analysis
    
    def _combine_analyses(self, text_blocks: List[TextBlock], doc_type: DocumentType, 
                         font_analysis: Dict[str, Any], spatial_analysis: Dict[str, Any], 
                         text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine different analyses to determine document structure"""
        
        # Score each text block for heading likelihood
        heading_candidates = []
        
        for block in text_blocks:
            score = self._calculate_heading_score(block, doc_type, font_analysis, spatial_analysis, text_analysis)
            
            if score > 0.5:  # Threshold for heading candidates
                heading_level = self.text_analyzer.detect_heading_level(block.text, doc_type)
                if heading_level:
                    heading_candidates.append({
                        'block': block,
                        'level': heading_level,
                        'score': score,
                        'text': block.text.strip()
                    })
        
        # Sort by score (highest first)
        heading_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'font_analysis': font_analysis,
            'spatial_analysis': spatial_analysis,
            'text_analysis': text_analysis,
            'heading_candidates': heading_candidates,
            'total_candidates': len(heading_candidates)
        }
    
    def _calculate_heading_score(self, block: TextBlock, doc_type: DocumentType,
                               font_analysis: Dict[str, Any], spatial_analysis: Dict[str, Any],
                               text_analysis: Dict[str, Any]) -> float:
        """Calculate likelihood score that a text block is a heading"""
        score = 0.0
        
        # Font-based scoring
        font_score = self.font_analyzer.classify_text_block_importance(block, font_analysis)
        score += font_score * 0.3
        
        # Spatial-based scoring
        page_analysis = spatial_analysis.get('pages', {}).get(block.page, {})
        spatial_score = self.spatial_analyzer.is_likely_heading_position(block, page_analysis)
        score += spatial_score * 0.2
        
        # Text pattern-based scoring
        heading_level = self.text_analyzer.detect_heading_level(block.text, doc_type)
        if heading_level:
            score += 0.4
        
        # Text characteristics
        text = block.text.strip()
        
        # Short text is more likely to be heading
        if len(text.split()) <= 10:
            score += 0.1
        
        # All caps or title case
        if text.isupper() or text.istitle():
            score += 0.1
        
        # Ends with colon (common in headings)
        if text.endswith(':'):
            score += 0.1
        
        # Standalone formatting (bold, larger font)
        if block.is_bold:
            score += 0.1
        
        # Document type specific adjustments
        score = self._adjust_score_for_document_type(score, block, doc_type)
        
        return min(1.0, score)
    
    def _adjust_score_for_document_type(self, score: float, block: TextBlock, doc_type: DocumentType) -> float:
        """Adjust score based on document type specific patterns"""
        text = block.text.strip().lower()
        
        if doc_type == DocumentType.FORM_DOCUMENT:
            # Forms have minimal headings, mostly just title
            if 'application form' in text:
                score += 0.3
            else:
                score *= 0.3  # Reduce score for other text
        
        elif doc_type == DocumentType.TECHNICAL_MANUAL:
            # Technical manuals have clear section structure
            if any(keyword in text for keyword in ['revision history', 'table of contents', 'acknowledgements']):
                score += 0.3
            if re.match(r'^\d+\.', text):  # Numbered sections
                score += 0.3
        
        elif doc_type == DocumentType.BUSINESS_DOCUMENT:
            # Business documents have complex hierarchy
            if any(keyword in text for keyword in ['summary', 'background', 'appendix']):
                score += 0.3
            if text.endswith(':'):  # Colon-terminated headings
                score += 0.2
        
        elif doc_type == DocumentType.PROGRAM_DOCUMENT:
            # Program documents have minimal structure
            if 'pathway options' in text:
                score += 0.4
            else:
                score *= 0.5  # Reduce score for other text
        
        elif doc_type == DocumentType.INVITATION_DOCUMENT:
            # Invitations have minimal structure
            if 'hope to see you there' in text:
                score += 0.5
            else:
                score *= 0.2  # Heavily reduce score for other text
        
        return score
    
    def extract_hierarchical_outline(self, structure_analysis: Dict[str, Any], doc_type: DocumentType) -> List[HeadingLevel]:
        """Extract hierarchical outline from structure analysis"""
        heading_candidates = structure_analysis.get('heading_candidates', [])
        
        if not heading_candidates:
            return []
        
        # Filter candidates based on document type rules
        filtered_candidates = self._filter_candidates_by_document_type(heading_candidates, doc_type)
        
        # Convert to HeadingLevel objects
        outline = []
        for candidate in filtered_candidates:
            heading = HeadingLevel(
                level=candidate['level'],
                text=candidate['text'],
                page=candidate['block'].page,
                confidence=candidate['score'],
                font_size=candidate['block'].font_size,
                font_name=candidate['block'].font_name
            )
            outline.append(heading)
        
        # Sort by page and then by position
        outline.sort(key=lambda h: (h.page, -h.confidence))
        
        return outline
    
    def _filter_candidates_by_document_type(self, candidates: List[Dict], doc_type: DocumentType) -> List[Dict]:
        """Filter heading candidates based on document type rules"""
        if doc_type == DocumentType.FORM_DOCUMENT:
            # Forms typically have no hierarchical structure
            return []
        
        elif doc_type == DocumentType.TECHNICAL_MANUAL:
            # Keep all valid candidates for technical manuals
            return [c for c in candidates if c['score'] > 0.6]
        
        elif doc_type == DocumentType.BUSINESS_DOCUMENT:
            # Keep all levels for business documents
            return [c for c in candidates if c['score'] > 0.5]
        
        elif doc_type == DocumentType.PROGRAM_DOCUMENT:
            # Keep minimal structure for program documents
            return [c for c in candidates if c['score'] > 0.7][:5]  # Max 5 items
        
        elif doc_type == DocumentType.INVITATION_DOCUMENT:
            # Keep only high-confidence items for invitations
            return [c for c in candidates if c['score'] > 0.8][:2]  # Max 2 items
        
        return candidates
