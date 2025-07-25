import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

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
class ProgramProcessorConfig:
    """Configuration class to remove all hardcoding from ProgramProcessor"""
    
    # Document settings
    document_type: DocumentType = DocumentType.PROGRAM_DOCUMENT
    document_confidence: float = 0.9
    fallback_confidence: float = 0.8
    empty_confidence: float = 0.0
    
    # Title extraction settings
    title_search_range: int = 5
    title_confidence_threshold: float = 0.7
    
    # Hierarchical importance thresholds
    primary_sections_threshold: float = 0.8
    secondary_sections_threshold: float = 0.6
    pattern_confidence_minimum: float = 0.0
    
    # Semantic validation thresholds
    exclusion_penalty_threshold: float = 0.5
    semantic_exclusion_threshold: float = 0.7
    structural_significance_threshold: float = 0.5
    final_structural_threshold: float = 0.6
    final_exclusion_threshold: float = 0.3
    
    # Confidence thresholds for outline extraction
    high_confidence_detection_threshold: float = 0.7
    structure_analyzer_default_confidence: float = 0.8
    
    # Scoring weights for hierarchical importance
    semantic_structural_weight: float = 0.5
    pattern_recognition_weight: float = 0.3
    exclusion_penalty_multiplier: float = 0.5
    max_importance_score: float = 1.0
    
    # Level-based importance weights
    h1_weight: float = 0.2
    h2_weight: float = 0.15
    h3_weight: float = 0.1
    h4_weight: float = 0.05
    default_level_weight: float = 0.0
    
    # Source priorities (lower number = higher priority)
    hierarchical_primary_priority: int = 1
    heading_detector_priority: int = 2
    structure_analyzer_priority: int = 3
    default_source_priority: int = 4
    
    # Logging settings
    enable_debug_logging: bool = True

class ProgramProcessor:
    """Fully configurable processor for program documents with no hardcoded values"""
    
    def __init__(self, config: Optional[ProgramProcessorConfig] = None):
        self.config = config or ProgramProcessorConfig()
        
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
        
        # Set up logging
        if self.config.enable_debug_logging:
            logger.setLevel(logging.DEBUG)
        
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process program document using pure component intelligence"""
        logger.info("Processing program document - pure component-driven analysis")
        
        if not text_blocks:
            return self._create_empty_result()
        
        # Pure component analysis
        analysis = self._perform_pure_component_analysis(text_blocks)
        
        # Component-driven title extraction
        title = self._extract_title_pure_components(text_blocks, analysis)
        
        # Intelligent outline filtering using component hierarchy
        outline = self._extract_intelligent_filtered_outline(text_blocks, analysis, title)
        
        # Create result
        result = DocumentStructure(
            title=title,
            outline=outline,
            doc_type=self.config.document_type,
            confidence=self.config.document_confidence
        )
        
        return self._validate_final_output(result)
    
    def _perform_pure_component_analysis(self, text_blocks: List[TextBlock]) -> Dict:
        """Pure component-driven analysis without any hardcoded logic"""
        analysis = {
            'structure_result': {},
            'heading_detections': [],
            'semantic_classifications': [],
            'filtered_content': [],
            'hierarchical_analysis': {}
        }
        
        # Structure analyzer - complete document understanding
        try:
            analysis['structure_result'] = self.structure_analyzer.analyze_document_structure(
                text_blocks, self.config.document_type
            )
        except Exception as e:
            logger.warning(f"Structure analyzer error: {e}")
            analysis['structure_result'] = {}
        
        # Heading detector - identify all potential headings
        try:
            font_analysis = self.font_analyzer.analyze_fonts(text_blocks)
            analysis['heading_detections'] = self.heading_detector.detect_headings(
                text_blocks, self.config.document_type, font_analysis
            )
        except Exception as e:
            logger.warning(f"Heading detection error: {e}")
            analysis['heading_detections'] = []
        
        # Semantic classifier - classify all content
        for i, block in enumerate(text_blocks):
            text = getattr(block, 'text', '').strip()
            if not text:
                continue
                
            try:
                semantic_scores = self.semantic_classifier.classify_text_role(text)
                
                classification = {
                    'index': i,
                    'text': text,
                    'block': block,
                    'page': getattr(block, 'page', 0),
                    'semantic_scores': semantic_scores
                }
                analysis['semantic_classifications'].append(classification)
                
                # Filter using content filter
                if not self.content_filter.is_structural_noise(text):
                    analysis['filtered_content'].append(classification)
                    
            except Exception as e:
                logger.warning(f"Semantic classification error: {e}")
        
        # Hierarchical analysis using pattern matcher
        analysis['hierarchical_analysis'] = self._analyze_content_hierarchy(analysis)
        
        return analysis
    
    def _analyze_content_hierarchy(self, analysis: Dict) -> Dict:
        """Analyze content hierarchy using pattern matcher intelligence"""
        hierarchy = {
            'primary_sections': [],
            'secondary_sections': [],
            'content_elements': []
        }
        
        # Use pattern matcher to understand hierarchical relationships
        for classification in analysis['filtered_content']:
            text = classification['text']
            semantic_scores = classification['semantic_scores']
            
            try:
                # Pattern matcher determines hierarchical importance
                matched_level, pattern_confidence, pattern_info = self.pattern_matcher.match_heading_pattern(text)
                
                if matched_level and pattern_confidence > self.config.pattern_confidence_minimum:
                    classification['pattern_level'] = matched_level
                    classification['pattern_confidence'] = pattern_confidence
                    classification['pattern_info'] = pattern_info
                    
                    # Combine semantic and pattern analysis for hierarchy determination
                    combined_importance = self._calculate_hierarchical_importance(
                        semantic_scores, pattern_confidence, matched_level
                    )
                    
                    classification['hierarchical_importance'] = combined_importance
                    
                    # Categorize based on configurable component intelligence
                    if combined_importance > self.config.primary_sections_threshold:
                        hierarchy['primary_sections'].append(classification)
                    elif combined_importance > self.config.secondary_sections_threshold:
                        hierarchy['secondary_sections'].append(classification)
                    else:
                        hierarchy['content_elements'].append(classification)
                        
            except Exception as e:
                logger.warning(f"Pattern analysis error: {e}")
        
        return hierarchy
    
    def _calculate_hierarchical_importance(self, semantic_scores: Dict, pattern_confidence: float, matched_level: str) -> float:
        """Calculate hierarchical importance using configurable component scores"""
        importance = 0.0
        
        # Semantic structural importance (configurable weight)
        importance += semantic_scores.get('structural', 0) * self.config.semantic_structural_weight
        
        # Pattern recognition confidence (configurable weight)
        importance += pattern_confidence * self.config.pattern_recognition_weight
        
        # Level-based importance using configurable weights
        level_weights = {
            'H1': self.config.h1_weight,
            'H2': self.config.h2_weight,
            'H3': self.config.h3_weight,
            'H4': self.config.h4_weight
        }
        importance += level_weights.get(matched_level, self.config.default_level_weight)
        
        # Exclusion penalty (configurable threshold and multiplier)
        if semantic_scores.get('excluded', 0) > self.config.exclusion_penalty_threshold:
            importance *= self.config.exclusion_penalty_multiplier
        
        return min(self.config.max_importance_score, importance)
    
    def _extract_title_pure_components(self, text_blocks: List[TextBlock], analysis: Dict) -> str:
        """Extract title using pure component intelligence"""
        # Method 1: Title extractor (primary method)
        try:
            extracted_title = self.title_extractor.extract_title(text_blocks, self.config.document_type)
            if extracted_title:
                return self.text_processor.clean_text(extracted_title)
        except Exception as e:
            logger.warning(f"Title extractor error: {e}")
        
        # Method 2: Semantic classifier on early content (configurable search range)
        search_limit = min(self.config.title_search_range, len(analysis['semantic_classifications']))
        
        for classification in analysis['semantic_classifications'][:search_limit]:
            try:
                is_title, confidence = self.semantic_classifier.is_document_title(
                    classification['text'], self.config.document_type, {}
                )
                
                if is_title and confidence > self.config.title_confidence_threshold:
                    return self.text_processor.clean_text(classification['text'])
            except Exception as e:
                logger.warning(f"Semantic title detection error: {e}")
        
        # Method 3: Structure analyzer title extraction
        structure_result = analysis.get('structure_result', {})
        if structure_result.get('extracted_title'):
            return self.text_processor.clean_text(structure_result['extracted_title'])
        
        return ""
    
    def _extract_intelligent_filtered_outline(self, text_blocks: List[TextBlock], analysis: Dict, title: str) -> List[HeadingLevel]:
        """Extract outline using intelligent component-based filtering"""
        candidates = []
        
        # Method 1: Use hierarchical analysis results
        hierarchy = analysis.get('hierarchical_analysis', {})
        
        # Process primary sections (highest importance)
        for primary in hierarchy.get('primary_sections', []):
            text = primary['text']
            
            # Skip title duplicates
            if title and text.lower().strip() == title.lower().strip():
                continue
            
            # Component validation
            if self._validate_outline_candidate_with_components(primary, analysis):
                try:
                    clean_text = self.text_processor.clean_text(text)
                    
                    candidates.append({
                        'text': clean_text,
                        'level': primary.get('pattern_level', 'H1'),
                        'page': primary['page'],
                        'confidence': primary['hierarchical_importance'],
                        'source': 'hierarchical_primary'
                    })
                except Exception as e:
                    logger.warning(f"Text processing error: {e}")
        
        # Method 2: Heading detector validation (configurable threshold)
        for block, level, confidence in analysis['heading_detections']:
            text = block.text.strip()
            
            # Skip title duplicates
            if title and text.lower().strip() == title.lower().strip():
                continue
            
            # High confidence detections only (configurable threshold)
            if confidence > self.config.high_confidence_detection_threshold:
                try:
                    # Validate with pattern matcher
                    if not self.pattern_matcher.is_exclusion_pattern(text):
                        clean_text = self.text_processor.clean_text(text)
                        
                        # Avoid duplicates
                        if not any(c['text'].lower() == clean_text.lower() for c in candidates):
                            candidates.append({
                                'text': clean_text,
                                'level': level,
                                'page': getattr(block, 'page', 0),
                                'confidence': confidence,
                                'source': 'heading_detector'
                            })
                except Exception as e:
                    logger.warning(f"Heading validation error: {e}")
        
        # Method 3: Structure analyzer outline
        try:
            structure_outline = self.structure_analyzer.extract_hierarchical_outline(
                analysis['structure_result'], self.config.document_type
            )
            
            for heading in structure_outline:
                text = heading.text.strip()
                
                # Skip title duplicates
                if title and text.lower().strip() == title.lower().strip():
                    continue
                
                # Validate with pattern matcher
                if not self.pattern_matcher.is_exclusion_pattern(text):
                    # Avoid duplicates
                    if not any(c['text'].lower() == text.lower() for c in candidates):
                        candidates.append({
                            'text': text,
                            'level': heading.level,
                            'page': heading.page,
                            'confidence': getattr(heading, 'confidence', self.config.structure_analyzer_default_confidence),
                            'source': 'structure_analyzer'
                        })
        except Exception as e:
            logger.warning(f"Structure outline error: {e}")
        
        return self._build_component_filtered_outline(candidates)
    
    def _validate_outline_candidate_with_components(self, candidate: Dict, analysis: Dict) -> bool:
        """Validate outline candidate using configurable components"""
        text = candidate['text']
        
        # Pattern matcher exclusion check
        try:
            if self.pattern_matcher.is_exclusion_pattern(text):
                return False
        except Exception as e:
            logger.warning(f"Pattern exclusion check error: {e}")
        
        # Semantic exclusion check (configurable threshold)
        semantic_scores = candidate.get('semantic_scores', {})
        if semantic_scores.get('excluded', 0) > self.config.semantic_exclusion_threshold:
            return False
        
        # Must have structural significance (configurable threshold)
        if semantic_scores.get('structural', 0) < self.config.structural_significance_threshold:
            return False
        
        return True
    
    def _build_component_filtered_outline(self, candidates: List[Dict]) -> List[HeadingLevel]:
        """Build final outline using configurable component intelligence"""
        if not candidates:
            return []
        
        # Component-based prioritization using configurable priorities
        source_priority = {
            'hierarchical_primary': self.config.hierarchical_primary_priority,
            'heading_detector': self.config.heading_detector_priority,
            'structure_analyzer': self.config.structure_analyzer_priority
        }
        
        candidates.sort(key=lambda x: (
            source_priority.get(x['source'], self.config.default_source_priority),
            -x['confidence']
        ))
        
        # Remove duplicates (keep highest priority)
        unique_candidates = {}
        for candidate in candidates:
            text_key = candidate['text'].lower().strip()
            if text_key not in unique_candidates:
                unique_candidates[text_key] = candidate
        
        final_candidates = list(unique_candidates.values())
        
        # Final component validation using configurable thresholds
        validated_candidates = []
        for candidate in final_candidates:
            try:
                # Final semantic validation
                semantic_scores = self.semantic_classifier.classify_text_role(candidate['text'])
                
                # Only include if components agree it's structural (configurable thresholds)
                if (semantic_scores.get('structural', 0) > self.config.final_structural_threshold and 
                    semantic_scores.get('excluded', 0) < self.config.final_exclusion_threshold):
                    validated_candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Final validation error: {e}")
                # Include if validation fails (fail-safe)
                validated_candidates.append(candidate)
        
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
            
            logger.info(f"Component-validated outline item: '{candidate['text']}' as {candidate['level']} (source: {candidate['source']})")
        
        return final_outline
    
    def _validate_final_output(self, result: DocumentStructure) -> DocumentStructure:
        """Final validation using output validator with configurable confidence"""
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
        
        try:
            if self.output_validator.validate_output(result_dict):
                logger.info(f"Final validation successful - Title: '{result.title}', Outline: {len(result.outline)} items")
                return result
            else:
                logger.warning("Final validation failed, cleaning")
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
        except Exception as e:
            logger.error(f"Final validation error: {e}")
            return result
    
    def _create_empty_result(self) -> DocumentStructure:
        """Create empty result with configurable confidence"""
        return DocumentStructure(
            title="",
            outline=[],
            doc_type=self.config.document_type,
            confidence=self.config.empty_confidence
        )

# Factory function for easy processor creation
def create_program_processor(processor_type: str = 'standard') -> ProgramProcessor:
    """Factory function to create preconfigured program processors"""
    
    if processor_type == 'strict':
        config = ProgramProcessorConfig(
            document_confidence=0.95,
            title_confidence_threshold=0.8,
            primary_sections_threshold=0.9,
            secondary_sections_threshold=0.7,
            high_confidence_detection_threshold=0.8
        )
        return ProgramProcessor(config=config)
    
    elif processor_type == 'lenient':
        config = ProgramProcessorConfig(
            document_confidence=0.8,
            title_confidence_threshold=0.5,
            primary_sections_threshold=0.6,
            secondary_sections_threshold=0.4,
            high_confidence_detection_threshold=0.5
        )
        return ProgramProcessor(config=config)
    
    elif processor_type == 'debug':
        config = ProgramProcessorConfig(
            enable_debug_logging=True,
            document_confidence=0.9
        )
        return ProgramProcessor(config=config)
    
    else:  # 'standard'
        return ProgramProcessor()

# Usage examples
if __name__ == "__main__":
    # Standard processor
    processor = ProgramProcessor()
    
    # Custom configuration
    custom_config = ProgramProcessorConfig(
        document_type=DocumentType.PROGRAM_DOCUMENT,
        document_confidence=0.95,
        title_search_range=10,
        primary_sections_threshold=0.85,
        semantic_structural_weight=0.6,
        pattern_recognition_weight=0.4
    )
    custom_processor = ProgramProcessor(config=custom_config)
    
    # Using factory
    strict_processor = create_program_processor('strict')
    lenient_processor = create_program_processor('lenient')
