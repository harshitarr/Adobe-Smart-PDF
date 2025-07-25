import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

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

@dataclass
class InvitationProcessorConfig:
    """Configuration class for InvitationProcessor - removes all hardcoding"""
    
    # Document settings
    document_type: DocumentType = DocumentType.INVITATION_DOCUMENT
    document_confidence: float = 0.9
    fallback_confidence: float = 0.8
    
    # Title extraction settings
    title_search_blocks: int = 5
    title_semantic_threshold: float = 0.4
    title_confidence_threshold: float = 0.6
    title_high_confidence_threshold: float = 0.8
    fallback_title_confidence: float = 0.5
    enable_title_extraction: bool = True
    empty_title_for_invitations: bool = True
    
    # Outline extraction thresholds
    outline_confidence_threshold: float = 0.3
    semantic_structural_threshold: float = 0.5
    semantic_exclusion_threshold: float = 0.3
    pattern_confidence_threshold: float = 0.5
    semantic_exclusion_validation: float = 0.7
    
    # Visual prominence level thresholds
    h1_visual_threshold: float = 0.8
    h2_visual_threshold: float = 0.6
    h3_visual_threshold: float = 0.4
    default_heading_level: str = 'H2'
    
    # Method priorities for outline building
    method_priorities: Dict[str, int] = None
    
    # Processing toggles
    enable_content_filtering: bool = True
    enable_font_analysis: bool = True
    enable_spatial_analysis: bool = True
    enable_semantic_classification: bool = True
    enable_heading_detection: bool = True
    enable_pattern_matching: bool = True
    enable_structure_analysis: bool = True
    enable_output_validation: bool = True
    
    # Logging settings
    enable_detailed_logging: bool = True
    log_outline_additions: bool = True
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.method_priorities is None:
            self.method_priorities = {
                'heading_detector': 1,
                'semantic_analysis': 2,
                'structure_analyzer': 3,
                'outline_extractor': 4
            }

@dataclass
class AnalysisConfig:
    """Configuration for document analysis components"""
    
    # Font analysis settings
    font_analysis_enabled: bool = True
    font_importance_threshold: float = 0.5
    
    # Spatial analysis settings
    spatial_analysis_enabled: bool = True
    prominence_position_weight: float = 0.3
    
    # Semantic analysis settings
    semantic_analysis_enabled: bool = True
    semantic_confidence_weight: float = 0.7
    
    # Text analysis settings
    text_analysis_enabled: bool = True
    heading_likelihood_threshold: float = 0.6
    title_likelihood_threshold: float = 0.7

class InvitationProcessor:
    """Fully configurable invitation processor with no hardcoded values"""
    
    def __init__(self, 
                 config: Optional[InvitationProcessorConfig] = None,
                 analysis_config: Optional[AnalysisConfig] = None):
        
        self.config = config or InvitationProcessorConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
        
        # Core components
        self.title_extractor = TitleExtractor()
        self.outline_extractor = OutlineExtractor()
        
        # Analyzers - conditional initialization based on config
        self.font_analyzer = FontAnalyzer() if self.config.enable_font_analysis else None
        self.spatial_analyzer = SpatialAnalyzer() if self.config.enable_spatial_analysis else None
        self.text_analyzer = TextAnalyzer() if self.analysis_config.text_analysis_enabled else None
        self.structure_analyzer = StructureAnalyzer() if self.config.enable_structure_analysis else None
        
        # Classifiers - conditional initialization
        self.content_filter = ContentFilter() if self.config.enable_content_filtering else None
        self.heading_detector = HeadingDetector() if self.config.enable_heading_detection else None
        self.pattern_matcher = PatternMatcher() if self.config.enable_pattern_matching else None
        self.semantic_classifier = SemanticClassifier() if self.config.enable_semantic_classification else None
        
        # Utils
        self.text_processor = TextProcessor()
        self.output_validator = OutputValidator() if self.config.enable_output_validation else None
        
        # Set up logging
        if self.config.enable_detailed_logging:
            logger.setLevel(logging.DEBUG)
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process invitation document with configurable analysis pipeline"""
        if self.config.enable_detailed_logging:
            logger.info(f"Processing invitation document with {len(text_blocks)} blocks")
        
        # Step 1: Content filtering (configurable)
        if self.config.enable_content_filtering and self.content_filter:
            filtered_blocks = self.content_filter.filter_structural_elements(
                text_blocks, self.config.document_type
            )
            if self.config.enable_detailed_logging:
                logger.debug(f"Filtered to {len(filtered_blocks)} blocks")
        else:
            filtered_blocks = text_blocks
        
        # Step 2: Document analysis
        doc_analysis = self._analyze_document_structure(filtered_blocks)
        
        # Step 3: Title extraction (configurable)
        title = ""
        if self.config.enable_title_extraction:
            title = self._extract_title_dynamically(filtered_blocks, doc_analysis)
        
        # Step 4: Outline extraction
        outline = self._extract_outline_dynamically(filtered_blocks, doc_analysis)
        
        # Step 5: Create result
        result = DocumentStructure(
            title=title,
            outline=outline,
            doc_type=self.config.document_type,
            confidence=self.config.document_confidence
        )
        
        # Step 6: Validation (configurable)
        if self.config.enable_output_validation and self.output_validator:
            return self._validate_final_result(result)
        else:
            return result
    
    def _analyze_document_structure(self, text_blocks: List[TextBlock]) -> Dict:
        """Comprehensive document analysis using configured analyzers"""
        analysis = {
            'structure_analysis': None,
            'font_analysis': None,
            'spatial_analysis': None,
            'block_classifications': []
        }
        
        # Structure analysis (configurable)
        if self.config.enable_structure_analysis and self.structure_analyzer:
            analysis['structure_analysis'] = self.structure_analyzer.analyze_document_structure(
                text_blocks, self.config.document_type
            )
        
        # Font analysis (configurable)
        if self.config.enable_font_analysis and self.font_analyzer:
            analysis['font_analysis'] = self.font_analyzer.analyze_fonts(text_blocks)
        
        # Spatial analysis (configurable)
        if self.config.enable_spatial_analysis and self.spatial_analyzer:
            analysis['spatial_analysis'] = self.spatial_analyzer.analyze_layout(text_blocks)
        
        # Block-level classification
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            if not text:
                continue
            
            block_classification = self._classify_block(block, i, len(text_blocks), analysis)
            analysis['block_classifications'].append(block_classification)
        
        return analysis
    
    def _classify_block(self, block: TextBlock, index: int, total_blocks: int, analysis: Dict) -> Dict:
        """Classify individual block using configured components"""
        text = block.text.strip()
        
        # Base classification structure
        classification = {
            'index': index,
            'text': text,
            'block': block,
            'position_in_doc': index / total_blocks if total_blocks > 0 else 0,
            'semantic_scores': {},
            'text_characteristics': {},
            'visual_prominence': 0.0,
            'spatial_context': {}
        }
        
        # Semantic classification (configurable)
        if self.config.enable_semantic_classification and self.semantic_classifier:
            classification['semantic_scores'] = self.semantic_classifier.classify_text_role(text)
        
        # Text characteristics analysis (configurable)
        if self.analysis_config.text_analysis_enabled and self.text_analyzer:
            classification['text_characteristics'] = {
                'is_likely_heading': self.text_processor.is_likely_heading(text),
                'is_likely_title': self.text_processor.is_likely_title(text),
                'is_content_text': self.text_processor.is_content_text(text)
            }
        
        # Visual prominence (configurable)
        if (self.config.enable_font_analysis and 
            self.font_analyzer and 
            analysis.get('font_analysis')):
            classification['visual_prominence'] = self.font_analyzer.classify_text_block_importance(
                block, analysis['font_analysis']
            )
        
        # Spatial context (configurable)
        if (self.config.enable_spatial_analysis and 
            self.spatial_analyzer and 
            analysis.get('spatial_analysis') and 
            hasattr(block, 'page')):
            page_analysis = analysis['spatial_analysis'].get('pages', {}).get(block.page, {})
            classification['spatial_context'] = {
                'is_prominent_position': self.spatial_analyzer.is_likely_heading_position(block, page_analysis),
                'page_context': page_analysis
            }
        
        return classification
    
    def _extract_title_dynamically(self, text_blocks: List[TextBlock], doc_analysis: Dict) -> str:
        """Dynamic title extraction using configured thresholds and methods"""
        title_candidates = []
        
        # Method 1: Semantic classification based title detection
        if self.config.enable_semantic_classification and self.semantic_classifier:
            search_limit = min(self.config.title_search_blocks, len(doc_analysis['block_classifications']))
            
            for classification in doc_analysis['block_classifications'][:search_limit]:
                semantic_scores = classification.get('semantic_scores', {})
                
                if semantic_scores.get('title', 0) > self.config.title_semantic_threshold:
                    # Validate using semantic classifier's title detection
                    context = {
                        'visual_prominence': classification.get('visual_prominence', 0),
                        'position_in_doc': classification.get('position_in_doc', 0),
                        'is_first_page': getattr(classification['block'], 'page', 0) == 0
                    }
                    
                    is_title, confidence = self.semantic_classifier.is_document_title(
                        classification['text'],
                        self.config.document_type,
                        context
                    )
                    
                    if is_title and confidence > self.config.title_confidence_threshold:
                        title_candidates.append({
                            'text': classification['text'],
                            'confidence': confidence,
                            'method': 'semantic_classification'
                        })
        
        # Method 2: Use title extractor as fallback
        extracted_title = self.title_extractor.extract_title(text_blocks, self.config.document_type)
        if extracted_title:
            title_candidates.append({
                'text': extracted_title,
                'confidence': self.config.fallback_title_confidence,
                'method': 'title_extractor'
            })
        
        # Select best candidate
        if title_candidates:
            title_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best_candidate = title_candidates[0]
            
            # Apply invitation-specific logic (configurable)
            if self.config.empty_title_for_invitations:
                # Only return title if confidence is very high for invitations
                if best_candidate['confidence'] > self.config.title_high_confidence_threshold:
                    if self.config.enable_detailed_logging:
                        logger.debug(f"Selected title: '{best_candidate['text']}' "
                                   f"(method: {best_candidate['method']}, "
                                   f"confidence: {best_candidate['confidence']:.2f})")
                    return self.text_processor.clean_text(best_candidate['text'])
            else:
                # Return any title above basic threshold
                if best_candidate['confidence'] > self.config.title_confidence_threshold:
                    return self.text_processor.clean_text(best_candidate['text'])
        
        return ""  # Default based on configuration
    
    def _extract_outline_dynamically(self, text_blocks: List[TextBlock], doc_analysis: Dict) -> List[HeadingLevel]:
        """Dynamic outline extraction using all configured methods"""
        outline_candidates = []
        
        # Method 1: Heading detector (configurable)
        if self.config.enable_heading_detection and self.heading_detector:
            detected_headings = self.heading_detector.detect_headings(
                text_blocks,
                self.config.document_type,
                doc_analysis.get('font_analysis', {})
            )
            
            for block, level, confidence in detected_headings:
                if confidence > self.config.outline_confidence_threshold:
                    outline_candidates.append({
                        'text': block.text,
                        'level': level,
                        'page': getattr(block, 'page', 0),
                        'confidence': confidence,
                        'method': 'heading_detector',
                        'block': block
                    })
        
        # Method 2: Semantic classification analysis (configurable)
        if self.config.enable_semantic_classification and self.semantic_classifier:
            for classification in doc_analysis['block_classifications']:
                semantic_scores = classification.get('semantic_scores', {})
                
                # Check if semantically classified as structural
                if semantic_scores.get('structural', 0) > self.config.semantic_structural_threshold:
                    # Validate it's not excluded content
                    if semantic_scores.get('excluded', 0) < self.config.semantic_exclusion_threshold:
                        # Determine level dynamically
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
        
        # Method 3: Structure analyzer (configurable)
        if (self.config.enable_structure_analysis and 
            self.structure_analyzer and 
            doc_analysis.get('structure_analysis')):
            
            structure_outline = self.structure_analyzer.extract_hierarchical_outline(
                doc_analysis['structure_analysis'], 
                self.config.document_type
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
        
        # Method 4: Outline extractor fallback
        extracted_outline = self.outline_extractor.extract_outline(text_blocks, self.config.document_type)
        if extracted_outline:
            for heading in extracted_outline:
                outline_candidates.append({
                    'text': heading.text,
                    'level': heading.level,
                    'page': heading.page,
                    'confidence': getattr(heading, 'confidence', self.config.outline_confidence_threshold),
                    'method': 'outline_extractor',
                    'block': None
                })
        
        # Build final outline
        return self._build_final_outline(outline_candidates)
    
    def _determine_heading_level_dynamically(self, classification: Dict) -> Optional[str]:
        """Determine heading level using configured thresholds and methods"""
        text = classification['text']
        
        # Method 1: Pattern matcher (configurable)
        if self.config.enable_pattern_matching and self.pattern_matcher:
            matched_level, pattern_confidence, pattern_info = self.pattern_matcher.match_heading_pattern(text)
            
            if matched_level and pattern_confidence > self.config.pattern_confidence_threshold:
                return matched_level
        
        # Method 2: Visual prominence based level assignment (configurable)
        visual_prominence = classification.get('visual_prominence', 0)
        
        if visual_prominence > self.config.h1_visual_threshold:
            return 'H1'
        elif visual_prominence > self.config.h2_visual_threshold:
            return 'H2'
        elif visual_prominence > self.config.h3_visual_threshold:
            return 'H3'
        
        # Method 3: Text characteristics and spatial context
        text_chars = classification.get('text_characteristics', {})
        if text_chars.get('is_likely_heading', False):
            spatial_context = classification.get('spatial_context', {})
            if spatial_context.get('is_prominent_position', False):
                return 'H1'
            else:
                return self.config.default_heading_level
        
        return None
    
    def _build_final_outline(self, candidates: List[Dict]) -> List[HeadingLevel]:
        """Build final outline with configurable validation and deduplication"""
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
            
            # Validate using pattern matcher (configurable)
            if (self.config.enable_pattern_matching and 
                self.pattern_matcher and 
                self.pattern_matcher.is_exclusion_pattern(clean_text)):
                continue
            
            # Additional semantic validation (configurable)
            is_valid = True
            if (self.config.enable_semantic_classification and 
                self.semantic_classifier and 
                candidate.get('block')):
                semantic_scores = self.semantic_classifier.classify_text_role(clean_text)
                if semantic_scores.get('excluded', 0) > self.config.semantic_exclusion_validation:
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
        
        # Sort by configured method priority and confidence
        validated_candidates.sort(
            key=lambda x: (self.config.method_priorities.get(x['method'], 999), -x['confidence'])
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
            
            if self.config.log_outline_additions:
                logger.info(f"Added to outline: '{candidate['text']}' as {candidate['level']} "
                           f"(method: {candidate['method']}, confidence: {candidate['confidence']:.2f})")
        
        return final_outline
    
    def _validate_final_result(self, result: DocumentStructure) -> DocumentStructure:
        """Validate final result using configured output validator"""
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
            if self.config.enable_detailed_logging:
                logger.info("Output validation passed")
            return result
        else:
            if self.config.enable_detailed_logging:
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
                        confidence=self.config.fallback_confidence
                    ) for item in cleaned_dict["outline"]
                ],
                doc_type=self.config.document_type,
                confidence=self.config.fallback_confidence
            )

# Factory functions for easy processor creation
def create_invitation_processor(processor_type: str = 'standard') -> InvitationProcessor:
    """Factory function to create preconfigured invitation processors"""
    
    if processor_type == 'minimal':
        config = InvitationProcessorConfig(
            enable_font_analysis=False,
            enable_spatial_analysis=False,
            enable_pattern_matching=False,
            enable_detailed_logging=False,
            document_confidence=0.8
        )
        return InvitationProcessor(config=config)
    
    elif processor_type == 'strict':
        config = InvitationProcessorConfig(
            title_confidence_threshold=0.8,
            title_high_confidence_threshold=0.9,
            outline_confidence_threshold=0.5,
            semantic_structural_threshold=0.7,
            document_confidence=0.95
        )
        return InvitationProcessor(config=config)
    
    elif processor_type == 'lenient':
        config = InvitationProcessorConfig(
            title_confidence_threshold=0.3,
            title_high_confidence_threshold=0.5,
            outline_confidence_threshold=0.2,
            semantic_structural_threshold=0.3,
            empty_title_for_invitations=False,
            document_confidence=0.8
        )
        return InvitationProcessor(config=config)
    
    elif processor_type == 'debug':
        config = InvitationProcessorConfig(
            enable_detailed_logging=True,
            log_outline_additions=True,
            enable_output_validation=True
        )
        analysis_config = AnalysisConfig(
            semantic_confidence_weight=0.8,
            heading_likelihood_threshold=0.5
        )
        return InvitationProcessor(config=config, analysis_config=analysis_config)
    
    else:  # 'standard'
        return InvitationProcessor()

# Usage examples
if __name__ == "__main__":
    # Standard processor
    processor = InvitationProcessor()
    
    # Custom configuration
    custom_config = InvitationProcessorConfig(
        document_confidence=0.95,
        title_search_blocks=10,
        enable_title_extraction=False,  # Skip title extraction entirely
        method_priorities={
            'semantic_analysis': 1,
            'heading_detector': 2,
            'structure_analyzer': 3,
            'outline_extractor': 4
        }
    )
    custom_processor = InvitationProcessor(config=custom_config)
    
    # Using factory
    debug_processor = create_invitation_processor('debug')
    strict_processor = create_invitation_processor('strict')
