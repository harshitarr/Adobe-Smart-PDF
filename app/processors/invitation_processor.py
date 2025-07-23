import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Optional
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

class InvitationProcessor:
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
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process invitation document with dynamic pattern recognition"""
        logger.info("Processing invitation document with dynamic analysis")
        
        # Filter blocks using content filter
        filtered_blocks = self.content_filter.filter_structural_elements(text_blocks, DocumentType.INVITATION_DOCUMENT)
        
        # Comprehensive document analysis using all analyzers
        doc_analysis = self._analyze_document_structure(filtered_blocks)
        
        # Dynamic title extraction
        title = self._extract_title_dynamically(filtered_blocks, doc_analysis)
        
        # Dynamic outline extraction using all classification methods
        outline = self._extract_outline_dynamically(filtered_blocks, doc_analysis)
        
        # Create and validate result
        result = DocumentStructure(
            title=title,
            outline=outline,
            doc_type=DocumentType.INVITATION_DOCUMENT,
            confidence=0.9
        )
        
        return self._validate_final_result(result)
    
    def _analyze_document_structure(self, text_blocks: List[TextBlock]) -> Dict:
        """Comprehensive document analysis using all available analyzers"""
        # Use structure analyzer for complete analysis
        structure_analysis = self.structure_analyzer.analyze_document_structure(
            text_blocks, DocumentType.INVITATION_DOCUMENT
        )
        
        # Get font and spatial analysis
        font_analysis = self.font_analyzer.analyze_fonts(text_blocks)
        spatial_analysis = self.spatial_analyzer.analyze_layout(text_blocks)
        
        analysis = {
            'structure_analysis': structure_analysis,
            'font_analysis': font_analysis,
            'spatial_analysis': spatial_analysis,
            'block_classifications': []
        }
        
        # Classify each block using semantic classifier and text analyzer
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            if not text:
                continue
            
            # Get semantic classification scores
            semantic_scores = self.semantic_classifier.classify_text_role(text)
            
            # Use text analyzer methods
            text_characteristics = {
                'is_likely_heading': self.text_processor.is_likely_heading(text),
                'is_likely_title': self.text_processor.is_likely_title(text),
                'is_content_text': self.text_processor.is_content_text(text)
            }
            
            # Calculate visual prominence using font analyzer
            visual_prominence = 0.0
            if font_analysis:
                visual_prominence = self.font_analyzer.classify_text_block_importance(block, font_analysis)
            
            # Get spatial context using spatial analyzer  
            spatial_context = {}
            if spatial_analysis and hasattr(block, 'page'):
                page_analysis = spatial_analysis.get('pages', {}).get(block.page, {})
                spatial_context = {
                    'is_prominent_position': self.spatial_analyzer.is_likely_heading_position(block, page_analysis),
                    'page_context': page_analysis
                }
            
            block_classification = {
                'index': i,
                'text': text,
                'block': block,
                'semantic_scores': semantic_scores,
                'text_characteristics': text_characteristics,
                'visual_prominence': visual_prominence,
                'spatial_context': spatial_context,
                'position_in_doc': i / len(text_blocks) if text_blocks else 0
            }
            
            analysis['block_classifications'].append(block_classification)
        
        return analysis
    
    def _extract_title_dynamically(self, text_blocks: List[TextBlock], doc_analysis: Dict) -> str:
        """Dynamic title extraction using semantic classifier"""
        title_candidates = []
        
        # Check blocks with high title semantic scores
        for classification in doc_analysis['block_classifications'][:5]:  # First 5 blocks
            semantic_scores = classification['semantic_scores']
            
            if semantic_scores.get('title', 0) > 0.4:
                # Validate using semantic classifier's title detection
                is_title, confidence = self.semantic_classifier.is_document_title(
                    classification['text'],
                    DocumentType.INVITATION_DOCUMENT,
                    {
                        'visual_prominence': classification['visual_prominence'],
                        'position_in_doc': classification['position_in_doc'],
                        'is_first_page': getattr(classification['block'], 'page', 0) == 0
                    }
                )
                
                if is_title and confidence > 0.6:
                    title_candidates.append({
                        'text': classification['text'],
                        'confidence': confidence
                    })
        
        # Use title extractor as fallback
        extracted_title = self.title_extractor.extract_title(text_blocks, DocumentType.INVITATION_DOCUMENT)
        if extracted_title:
            title_candidates.append({
                'text': extracted_title,
                'confidence': 0.5
            })
        
        # Select best candidate or return empty for invitations
        if title_candidates:
            title_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best_candidate = title_candidates[0]
            
            # Only return title if confidence is very high for invitations
            if best_candidate['confidence'] > 0.8:
                return self.text_processor.clean_text(best_candidate['text'])
        
        return ""  # Default empty for invitations
    
    def _extract_outline_dynamically(self, text_blocks: List[TextBlock], doc_analysis: Dict) -> List[HeadingLevel]:
        """Dynamic outline extraction using all available methods"""
        outline_candidates = []
        
        # Method 1: Use heading detector
        detected_headings = self.heading_detector.detect_headings(
            text_blocks,
            DocumentType.INVITATION_DOCUMENT,
            doc_analysis.get('font_analysis', {})
        )
        
        for block, level, confidence in detected_headings:
            if confidence > 0.3:
                outline_candidates.append({
                    'text': block.text,
                    'level': level,
                    'page': getattr(block, 'page', 0),
                    'confidence': confidence,
                    'method': 'heading_detector',
                    'block': block
                })
        
        # Method 2: Analyze semantic classifications for structural elements
        for classification in doc_analysis['block_classifications']:
            semantic_scores = classification['semantic_scores']
            
            # Check if semantically classified as structural
            if semantic_scores.get('structural', 0) > 0.5:
                # Validate it's not excluded content
                if semantic_scores.get('excluded', 0) < 0.3:
                    # Determine level using pattern matcher
                    level = self._determine_heading_level_dynamically(classification)
                    
                    if level:
                        outline_candidates.append({
                            'text': classification['text'],
                            'level': level,
                            'page': getattr(classification['block'], 'page', 0),
                            'confidence': semantic_scores['structural'],
                            'method': 'semantic_analysis',
                            'block': classification['block']
                        })
        
        # Method 3: Use structure analyzer's outline extraction
        if doc_analysis.get('structure_analysis'):
            structure_outline = self.structure_analyzer.extract_hierarchical_outline(
                doc_analysis['structure_analysis'], 
                DocumentType.INVITATION_DOCUMENT
            )
            
            for heading in structure_outline:
                outline_candidates.append({
                    'text': heading.text,
                    'level': heading.level,
                    'page': heading.page,
                    'confidence': heading.confidence,
                    'method': 'structure_analyzer',
                    'block': None
                })
        
        # Method 4: Use outline extractor
        extracted_outline = self.outline_extractor.extract_outline(text_blocks, DocumentType.INVITATION_DOCUMENT)
        if extracted_outline:
            for heading in extracted_outline:
                outline_candidates.append({
                    'text': heading.text,
                    'level': heading.level,
                    'page': heading.page,
                    'confidence': getattr(heading, 'confidence', 0.6),
                    'method': 'outline_extractor',
                    'block': None
                })
        
        # Build final outline
        return self._build_final_outline(outline_candidates)
    
    def _determine_heading_level_dynamically(self, classification: Dict) -> Optional[str]:
        """Determine heading level using pattern matcher and analyzers"""
        text = classification['text']
        
        # Use pattern matcher for level detection
        matched_level, pattern_confidence, pattern_info = self.pattern_matcher.match_heading_pattern(text)
        
        if matched_level and pattern_confidence > 0.5:
            return matched_level
        
        # Use visual prominence for level assignment
        visual_prominence = classification['visual_prominence']
        
        if visual_prominence > 0.8:
            return 'H1'
        elif visual_prominence > 0.6:
            return 'H2'
        elif visual_prominence > 0.4:
            return 'H3'
        
        # Use text characteristics
        text_chars = classification['text_characteristics']
        if text_chars['is_likely_heading']:
            # Check spatial context
            spatial_context = classification['spatial_context']
            if spatial_context.get('is_prominent_position', False):
                return 'H1'
            else:
                return 'H2'
        
        return None
    
    def _build_final_outline(self, candidates: List[Dict]) -> List[HeadingLevel]:
        """Build final outline with validation and deduplication"""
        if not candidates:
            return []
        
        # Remove duplicates and validate each candidate
        validated_candidates = []
        seen_texts = set()
        
        for candidate in candidates:
            text = candidate['text']
            clean_text = self.text_processor.clean_text(text)
            text_key = clean_text.lower().strip()
            
            # Skip if empty or duplicate
            if not clean_text or text_key in seen_texts:
                continue
            
            # Validate using pattern matcher
            if self.pattern_matcher.is_exclusion_pattern(clean_text):
                continue
            
            # Additional validation using semantic classifier if block available
            is_valid = True
            if candidate.get('block'):
                # Re-validate with semantic classifier
                semantic_scores = self.semantic_classifier.classify_text_role(clean_text)
                if semantic_scores.get('excluded', 0) > 0.7:
                    is_valid = False
            
            if is_valid:
                validated_candidates.append({
                    'text': clean_text,
                    'level': candidate['level'],
                    'page': candidate['page'],
                    'confidence': candidate['confidence'],
                    'method': candidate['method']
                })
                seen_texts.add(text_key)
        
        # Sort by confidence and method priority
        method_priority = {
            'heading_detector': 1,
            'semantic_analysis': 2,
            'structure_analyzer': 3,
            'outline_extractor': 4
        }
        
        validated_candidates.sort(
            key=lambda x: (method_priority.get(x['method'], 5), -x['confidence'])
        )
        
        # Build final outline
        final_outline = []
        for candidate in validated_candidates:
            final_outline.append(HeadingLevel(
                level=candidate['level'],
                text=candidate['text'],
                page=candidate['page'],
                confidence=candidate['confidence'],
                font_size=None,
                font_name=None
            ))
            
            logger.info(f"Added to outline: '{candidate['text']}' as {candidate['level']} "
                       f"(method: {candidate['method']}, confidence: {candidate['confidence']:.2f})")
        
        return final_outline
    
    def _validate_final_result(self, result: DocumentStructure) -> DocumentStructure:
        """Validate final result using output validator"""
        # Convert to dict for validation
        result_dict = {
            "title": result.title,
            "outline": [
                {
                    "level": h.level,
                    "text": h.text,
                    "page": h.page
                } for h in result.outline
            ]
        }
        
        # Use output validator
        if self.output_validator.validate_output(result_dict):
            logger.info("Output validation passed")
            return result
        else:
            logger.info("Output validation failed, cleaning result")
            # Clean using output validator
            cleaned_dict = self.output_validator.clean_output(result_dict)
            
            return DocumentStructure(
                title=cleaned_dict["title"],
                outline=[
                    HeadingLevel(
                        level=item["level"],
                        text=item["text"],
                        page=item["page"],
                        confidence=0.8
                    ) for item in cleaned_dict["outline"]
                ],
                doc_type=DocumentType.INVITATION_DOCUMENT,
                confidence=0.8
            )
