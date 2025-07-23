import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Tuple
import logging

# Core imports
from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor

# Analyzer imports
from analyzers.font_analyzer import FontAnalyzer
from analyzers.spatial_analyzer import SpatialAnalyzer
from analyzers.text_analyzer import TextAnalyzer
from analyzers.structure_analyzer import StructureAnalyzer

# Classifier imports
from classifiers.content_filter import ContentFilter
from classifiers.heading_detector import HeadingDetector
from classifiers.pattern_matcher import PatternMatcher
from classifiers.semantic_classifier import SemanticClassifier

# Utils imports
from utils.text_utils import TextProcessor
from utils.validation_utils import OutputValidator

logger = logging.getLogger('extraction')

class ManualProcessor:
    def __init__(self):
        # Core components
        self.title_extractor = TitleExtractor()
        self.outline_extractor = OutlineExtractor()
        
        # Analyzers
        self.font_analyzer = FontAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        self.text_analyzer = TextAnalyzer()
        self.structure_analyzer = StructureAnalyzer()
        
        # Classifiers
        self.content_filter = ContentFilter()
        self.heading_detector = HeadingDetector()
        self.pattern_matcher = PatternMatcher()
        self.semantic_classifier = SemanticClassifier()
        
        # Utils
        self.text_processor = TextProcessor()
        self.output_validator = OutputValidator()
        
        # Generalized pattern rules
        self.pattern_rules = {
            'navigation_patterns': [
                r'revision\s+history',
                r'table\s+of\s+contents',
                r'acknowledgements?'
            ],
            'numbered_section_patterns': [
                r'^\d+\.\s+[a-z]',  # Main sections: 1., 2., 3., 4.
            ],
            'subsection_patterns': [
                r'^\d+\.\d+\s+[a-z]',  # Subsections: 2.1, 2.2, etc.
            ],
            'exclusion_patterns': [
                r'version\s+\d+\s+page\s+\d+',
                r'copyright.*international',
                r'qualifications\s+board',
                r'may\s+\d+,\s+\d+',
                r'working\s+group',
                r'ga\s+release'
            ]
        }
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Generalized processing for technical manuals"""
        logger.info("Processing technical manual - generalized pattern matching")
        
        # Filter structural elements
        filtered_blocks = self.content_filter.filter_structural_elements(text_blocks, DocumentType.TECHNICAL_MANUAL)
        
        # Analyze document structure
        structure_analysis = self.structure_analyzer.analyze_document_structure(filtered_blocks, DocumentType.TECHNICAL_MANUAL)
        
        # Dynamic title extraction
        title = self._extract_generalized_title(filtered_blocks, structure_analysis)
        
        # Pattern-based outline extraction
        outline = self._extract_generalized_outline(filtered_blocks, structure_analysis)
        
        return DocumentStructure(
            title=title,
            outline=outline,
            doc_type=DocumentType.TECHNICAL_MANUAL,
            confidence=0.9
        )
    
    def _extract_generalized_title(self, text_blocks: List[TextBlock], structure_analysis: Dict) -> str:
        """Extract title using generalized patterns with structure analysis"""
        title_candidates = []
        
        # Use semantic classifier for better title detection
        for block in text_blocks[:30]:  # Check first 30 blocks
            text = block.text.strip()
            if not text:
                continue
                
            # Get semantic classification
            semantic_scores = self.semantic_classifier.classify_text_role(text)
            
            if semantic_scores.get('structural', 0) > 0.5:
                score = self._calculate_title_score(text, block, structure_analysis)
                
                if score > 0.5:  # Threshold for title candidates
                    title_candidates.append((text, score, block))
        
        if title_candidates:
            # Sort by score and take the best
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            best_title = title_candidates[0][0]
            return self._format_title(best_title)
        
        return ""
    
    def _calculate_title_score(self, text: str, block: TextBlock, structure_analysis: Dict) -> float:
        """Calculate title likelihood score using analyzers"""
        if not text or len(text) < 5:
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        words = text.split()
        
        # Use text processor for better analysis
        if self.text_processor.is_likely_title(text):
            score += 0.4
        
        # Font analysis
        font_analysis = structure_analysis.get('font_analysis', {})
        if font_analysis:
            font_importance = self.font_analyzer.classify_text_block_importance(block, font_analysis)
            score += font_importance * 0.3
        
        # Spatial analysis
        spatial_analysis = structure_analysis.get('spatial_analysis', {})
        if spatial_analysis:
            page_analysis = spatial_analysis.get('pages', {}).get(block.page, {})
            spatial_score = self.spatial_analyzer.is_likely_heading_position(block, page_analysis)
            score += spatial_score * 0.2
        
        # Title indicators
        title_words = ['overview', 'introduction', 'manual', 'guide', 'foundation', 'level', 'extensions']
        indicator_count = sum(1 for word in title_words if word in text_lower)
        score += indicator_count * 0.1
        
        # Length characteristics
        if 3 <= len(words) <= 8:
            score += 0.2
        elif len(words) > 12:
            score -= 0.3
        
        # Position characteristics
        if block.page == 0:
            score += 0.1
        
        # Avoid numbered sections as titles
        if re.match(r'^\d+\.', text):
            score -= 0.5
        
        return min(1.0, max(0.0, score))
    
    def _format_title(self, title: str) -> str:
        """Format title with text processor"""
        clean_title = self.text_processor.clean_text(title)
        
        # Format section text for consistency
        formatted_title = self.text_processor.format_section_text(clean_title)
        
        # Detect if we need extra spacing based on content
        if len(formatted_title.split()) <= 5:
            # Short titles often have extra spacing
            return f"{formatted_title}  "  # Double space at end
        
        return f"{formatted_title} "  # Single space at end
    
    def _extract_generalized_outline(self, text_blocks: List[TextBlock], structure_analysis: Dict) -> List[HeadingLevel]:
        """Extract outline using structure analyzer"""
        # Use structure analyzer to get hierarchical outline
        outline = self.structure_analyzer.extract_hierarchical_outline(structure_analysis, DocumentType.TECHNICAL_MANUAL)
        
        # Validate outline using output validator
        if outline:
            # Convert to dict format for validation
            outline_dict = {
                'title': '',
                'outline': [
                    {
                        'level': h.level,
                        'text': h.text,
                        'page': h.page
                    } for h in outline
                ]
            }
            
            if self.output_validator.validate_output(outline_dict):
                return outline
            else:
                # Clean outline if validation fails
                cleaned_dict = self.output_validator.clean_output(outline_dict)
                return [
                    HeadingLevel(
                        level=item['level'],
                        text=item['text'],
                        page=item['page'],
                        confidence=0.8
                    ) for item in cleaned_dict['outline']
                ]
        
        return []
