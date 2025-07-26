import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

# Core imports
from processors.base_processor import BaseProcessor
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
    
    # Language-aware settings (new)
    enable_language_aware_processing: bool = True
    apply_language_confidence_adjustments: bool = True
    use_language_specific_validation: bool = True
    language_component_weighting: bool = True

class ProgramProcessor(BaseProcessor):
    """Fully configurable processor for program documents with no hardcoded values and multilingual support"""
    
    def __init__(self, config: Optional[ProgramProcessorConfig] = None):
        # Initialize BaseProcessor for language support
        super().__init__()
        
        self.config = config or ProgramProcessorConfig()
        
        # Language-specific adjustment cache
        self._language_adjusted_thresholds = {}
        self._language_component_weights = {}
        
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
        
        logger.info("ProgramProcessor initialized with multilingual support")
    
    def set_language_config(self, language: str, config: Dict[str, Any]):
        """Override from BaseProcessor - apply language-specific component adjustments"""
        super().set_language_config(language, config)
        
        if self.config.enable_language_aware_processing:
            self._apply_language_component_adjustments(language, config)
            logger.info(f"Applied {language} component adjustments to ProgramProcessor")
    
    def _apply_language_component_adjustments(self, language: str, lang_config: Dict[str, Any]):
        """Apply language-specific adjustments to component analysis"""
        
        # Store original thresholds if not cached
        if not self._language_adjusted_thresholds:
            self._language_adjusted_thresholds = {
                'original_title_confidence': self.config.title_confidence_threshold,
                'original_primary_threshold': self.config.primary_sections_threshold,
                'original_structural_threshold': self.config.final_structural_threshold,
                'original_semantic_weight': self.config.semantic_structural_weight
            }
        
        # Apply language-specific threshold adjustments
        if self.config.apply_language_confidence_adjustments:
            adjustments = self._get_language_component_adjustments(language)
            
            # Adjust component thresholds
            self.config.title_confidence_threshold *= adjustments.get('title_confidence_multiplier', 1.0)
            self.config.primary_sections_threshold *= adjustments.get('primary_threshold_multiplier', 1.0)
            self.config.final_structural_threshold *= adjustments.get('structural_threshold_multiplier', 1.0)
            
            # Adjust component weights if enabled
            if self.config.language_component_weighting:
                self.config.semantic_structural_weight *= adjustments.get('semantic_weight_multiplier', 1.0)
                self.config.pattern_recognition_weight *= adjustments.get('pattern_weight_multiplier', 1.0)
        
        # Cache language-specific component weights
        self._language_component_weights[language] = lang_config
    
    def _get_language_component_adjustments(self, language: str) -> Dict[str, float]:
        """Get language-specific component adjustment factors"""
        
        adjustments = {
            'japanese': {
                'title_confidence_multiplier': 0.9,        # Japanese titles may be more subtle
                'primary_threshold_multiplier': 1.05,      # Structure is often clearer
                'structural_threshold_multiplier': 0.95,   # Be more lenient
                'semantic_weight_multiplier': 1.1,         # Boost semantic analysis
                'pattern_weight_multiplier': 0.9           # Patterns may be different
            },
            'german': {
                'title_confidence_multiplier': 1.0,
                'primary_threshold_multiplier': 1.0,
                'structural_threshold_multiplier': 1.0,
                'semantic_weight_multiplier': 1.0,
                'pattern_weight_multiplier': 1.05          # German patterns are often explicit
            },
            'tamil': {
                'title_confidence_multiplier': 0.85,       # Be more lenient
                'primary_threshold_multiplier': 0.95,
                'structural_threshold_multiplier': 0.9,
                'semantic_weight_multiplier': 1.15,        # Boost semantic analysis
                'pattern_weight_multiplier': 0.85          # Patterns may be less clear
            },
            'english': {
                'title_confidence_multiplier': 1.0,
                'primary_threshold_multiplier': 1.0,
                'structural_threshold_multiplier': 1.0,
                'semantic_weight_multiplier': 1.0,
                'pattern_weight_multiplier': 1.0
            }
        }
        
        return adjustments.get(language, adjustments['english'])
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process program document using pure component intelligence with language awareness"""
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
        
        # Add language metadata if available
        if hasattr(result, 'metadata') and hasattr(self, 'language'):
            result.metadata.update({
                'detected_language': self.language,
                'language_component_adjustments': bool(self._language_component_weights),
                'component_analysis_language': self.language if self._language_component_weights else 'english'
            })
        
        return self._validate_final_output(result)
    
    def _perform_pure_component_analysis(self, text_blocks: List[TextBlock]) -> Dict:
        """Pure component-driven analysis without any hardcoded logic with language awareness"""
        analysis = {
            'structure_result': {},
            'heading_detections': [],
            'semantic_classifications': [],
            'filtered_content': [],
            'hierarchical_analysis': {},
            'language_context': self._language_component_weights.get(self.language, {}) if hasattr(self, 'language') else {}
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
        
        # Semantic classifier - classify all content with language awareness
        for i, block in enumerate(text_blocks):
            text = getattr(block, 'text', '').strip()
            if not text:
                continue
                
            try:
                semantic_scores = self.semantic_classifier.classify_text_role(text)
                
                # Apply language-specific semantic score adjustments
                if (hasattr(self, 'language') and 
                    self.config.use_language_specific_validation):
                    semantic_scores = self._adjust_semantic_scores_for_language(semantic_scores, text)
                
                classification = {
                    'index': i,
                    'text': text,
                    'block': block,
                    'page': getattr(block, 'page', 0),
                    'semantic_scores': semantic_scores,
                    'language_adjusted': hasattr(self, 'language') and self.config.use_language_specific_validation
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
    
    def _adjust_semantic_scores_for_language(self, semantic_scores: Dict[str, float], text: str) -> Dict[str, float]:
        """Adjust semantic scores based on language characteristics"""
        if not hasattr(self, 'language'):
            return semantic_scores
        
        language = self.language
        adjusted_scores = semantic_scores.copy()
        
        # Language-specific semantic adjustments
        if language == 'japanese':
            # Japanese may need different semantic weighting
            if 'structural' in adjusted_scores:
                adjusted_scores['structural'] *= 1.1
            if 'title' in adjusted_scores:
                adjusted_scores['title'] *= 1.05
        elif language == 'german':
            # German structure is often explicit
            if 'structural' in adjusted_scores:
                adjusted_scores['structural'] *= 1.05
        elif language == 'tamil':
            # Tamil may need more generous scoring
            if 'structural' in adjusted_scores:
                adjusted_scores['structural'] *= 1.15
            if 'title' in adjusted_scores:
                adjusted_scores['title'] *= 1.1
        
        return adjusted_scores
    
    def _analyze_content_hierarchy(self, analysis: Dict) -> Dict:
        """Analyze content hierarchy using pattern matcher intelligence with language awareness"""
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
                
                # Apply language-specific pattern confidence adjustment
                if (hasattr(self, 'language') and 
                    self.config.language_component_weighting):
                    pattern_confidence = self._adjust_pattern_confidence_for_language(pattern_confidence, text)
                
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
    
    def _adjust_pattern_confidence_for_language(self, pattern_confidence: float, text: str) -> float:
        """Adjust pattern confidence based on language characteristics"""
        if not hasattr(self, 'language'):
            return pattern_confidence
        
        language = self.language
        
        # Language-specific pattern confidence adjustments
        adjustments = {
            'japanese': 0.95,   # Japanese patterns may be less explicit
            'german': 1.05,     # German patterns are often very clear
            'tamil': 0.9,       # Tamil patterns may need more lenience
            'english': 1.0
        }
        
        multiplier = adjustments.get(language, 1.0)
        return min(1.0, pattern_confidence * multiplier)
    
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
        
        # Apply language-specific importance adjustments
        if (hasattr(self, 'language') and 
            self.config.language_component_weighting):
            importance = self._adjust_hierarchical_importance_for_language(importance, semantic_scores)
        
        return min(self.config.max_importance_score, importance)
    
    def _adjust_hierarchical_importance_for_language(self, importance: float, semantic_scores: Dict) -> float:
        """Adjust hierarchical importance based on language characteristics"""
        if not hasattr(self, 'language'):
            return importance
        
        language = self.language
        
        # Language-specific importance adjustments
        if language == 'japanese':
            # Japanese structure may need slight boost
            return importance * 1.02
        elif language == 'german':
            # German structure is reliable
            return importance
        elif language == 'tamil':
            # Tamil may need more generous importance scoring
            return importance * 1.05
        
        return importance
    
    def _extract_title_pure_components(self, text_blocks: List[TextBlock], analysis: Dict) -> str:
        """Extract title using pure component intelligence with language awareness"""
        # Method 1: Title extractor (primary method)
        try:
            extracted_title = self.title_extractor.extract_title(text_blocks, self.config.document_type)
            if extracted_title:
                cleaned_title = self.text_processor.clean_text(extracted_title)
                
                # Apply language-specific title processing
                if (hasattr(self, 'language') and 
                    self.config.enable_language_aware_processing):
                    cleaned_title = self._apply_language_title_processing(cleaned_title)
                
                return cleaned_title
        except Exception as e:
            logger.warning(f"Title extractor error: {e}")
        
        # Method 2: Semantic classifier on early content (configurable search range)
        search_limit = min(self.config.title_search_range, len(analysis['semantic_classifications']))
        
        for classification in analysis['semantic_classifications'][:search_limit]:
            try:
                # Add language context if available
                context = {'language_context': analysis.get('language_context', {})}
                
                is_title, confidence = self.semantic_classifier.is_document_title(
                    classification['text'], self.config.document_type, context
                )
                
                # Apply language-specific confidence adjustment
                if (hasattr(self, 'language') and 
                    self.config.apply_language_confidence_adjustments):
                    confidence = self._adjust_title_confidence_for_language(confidence, classification['text'])
                
                if is_title and confidence > self.config.title_confidence_threshold:
                    cleaned_title = self.text_processor.clean_text(classification['text'])
                    
                    # Apply language-specific title processing
                    if (hasattr(self, 'language') and 
                        self.config.enable_language_aware_processing):
                        cleaned_title = self._apply_language_title_processing(cleaned_title)
                    
                    return cleaned_title
            except Exception as e:
                logger.warning(f"Semantic title detection error: {e}")
        
        # Method 3: Structure analyzer title extraction
        structure_result = analysis.get('structure_result', {})
        if structure_result.get('extracted_title'):
            cleaned_title = self.text_processor.clean_text(structure_result['extracted_title'])
            
            # Apply language-specific title processing
            if (hasattr(self, 'language') and 
                self.config.enable_language_aware_processing):
                cleaned_title = self._apply_language_title_processing(cleaned_title)
            
            return cleaned_title
        
        return ""
    
    def _adjust_title_confidence_for_language(self, confidence: float, text: str) -> float:
        """Adjust title confidence based on language characteristics"""
        if not hasattr(self, 'language'):
            return confidence
        
        language = self.language
        
        # Language-specific title confidence adjustments
        adjustments = {
            'japanese': 1.05,   # Boost Japanese title confidence
            'german': 1.0,      # German titles are usually clear
            'tamil': 1.1,       # Be more generous with Tamil titles
            'english': 1.0
        }
        
        multiplier = adjustments.get(language, 1.0)
        return min(1.0, confidence * multiplier)
    
    def _apply_language_title_processing(self, title: str) -> str:
        """Apply language-specific title processing"""
        if not hasattr(self, 'language'):
            return title
        
        language = self.language
        
        # Language-specific title processing
        if language == 'japanese':
            # Japanese title processing
            return title.strip()
        elif language == 'german':
            # German title processing - preserve compound words
            return title.strip()
        elif language == 'tamil':
            # Tamil title processing
            return title.strip()
        
        return title
    
    def _extract_intelligent_filtered_outline(self, text_blocks: List[TextBlock], analysis: Dict, title: str) -> List[HeadingLevel]:
        """Extract outline using intelligent component-based filtering with language awareness"""
        candidates = []
        
        # Method 1: Use hierarchical analysis results
        hierarchy = analysis.get('hierarchical_analysis', {})
        
        # Process primary sections (highest importance)
        for primary in hierarchy.get('primary_sections', []):
            text = primary['text']
            
            # Skip title duplicates
            if title and text.lower().strip() == title.lower().strip():
                continue
            
            # Component validation with language awareness
            if self._validate_outline_candidate_with_components(primary, analysis):
                try:
                    clean_text = self.text_processor.clean_text(text)
                    
                    # Apply language-specific text processing
                    if (hasattr(self, 'language') and 
                        self.config.enable_language_aware_processing):
                        clean_text = self._apply_language_outline_processing(clean_text)
                    
                    candidates.append({
                        'text': clean_text,
                        'level': primary.get('pattern_level', 'H1'),
                        'page': primary['page'],
                        'confidence': primary['hierarchical_importance'],
                        'source': 'hierarchical_primary',
                        'language_processed': hasattr(self, 'language')
                    })
                except Exception as e:
                    logger.warning(f"Text processing error: {e}")
        
        # Method 2: Heading detector validation (configurable threshold)
        for block, level, confidence in analysis['heading_detections']:
            text = block.text.strip()
            
            # Skip title duplicates
            if title and text.lower().strip() == title.lower().strip():
                continue
            
            # Apply language-specific confidence adjustment
            if (hasattr(self, 'language') and 
                self.config.apply_language_confidence_adjustments):
                confidence = self._adjust_heading_confidence_for_language(confidence, text)
            
            # High confidence detections only (configurable threshold)
            if confidence > self.config.high_confidence_detection_threshold:
                try:
                    # Validate with pattern matcher
                    if not self.pattern_matcher.is_exclusion_pattern(text):
                        clean_text = self.text_processor.clean_text(text)
                        
                        # Apply language-specific text processing
                        if (hasattr(self, 'language') and 
                            self.config.enable_language_aware_processing):
                            clean_text = self._apply_language_outline_processing(clean_text)
                        
                        # Avoid duplicates
                        if not any(c['text'].lower() == clean_text.lower() for c in candidates):
                            candidates.append({
                                'text': clean_text,
                                'level': level,
                                'page': getattr(block, 'page', 0),
                                'confidence': confidence,
                                'source': 'heading_detector',
                                'language_processed': hasattr(self, 'language')
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
                    processed_text = text
                    
                    # Apply language-specific text processing
                    if (hasattr(self, 'language') and 
                        self.config.enable_language_aware_processing):
                        processed_text = self._apply_language_outline_processing(text)
                    
                    # Avoid duplicates
                    if not any(c['text'].lower() == processed_text.lower() for c in candidates):
                        candidates.append({
                            'text': processed_text,
                            'level': heading.level,
                            'page': heading.page,
                            'confidence': getattr(heading, 'confidence', self.config.structure_analyzer_default_confidence),
                            'source': 'structure_analyzer',
                            'language_processed': hasattr(self, 'language')
                        })
        except Exception as e:
            logger.warning(f"Structure outline error: {e}")
        
        return self._build_component_filtered_outline(candidates)
    
    def _adjust_heading_confidence_for_language(self, confidence: float, text: str) -> float:
        """Adjust heading confidence based on language characteristics"""
        if not hasattr(self, 'language'):
            return confidence
        
        language = self.language
        
        # Language-specific heading confidence adjustments
        adjustments = {
            'japanese': 1.02,   # Slight boost for Japanese headings
            'german': 1.0,      # German headings are usually reliable
            'tamil': 1.05,      # Be more generous with Tamil headings
            'english': 1.0
        }
        
        multiplier = adjustments.get(language, 1.0)
        return min(1.0, confidence * multiplier)
    
    def _apply_language_outline_processing(self, text: str) -> str:
        """Apply language-specific outline text processing"""
        if not hasattr(self, 'language'):
            return text
        
        language = self.language
        
        # Language-specific outline processing
        if language == 'japanese':
            # Japanese outline processing
            return text.strip()
        elif language == 'german':
            # German outline processing
            return text.strip()
        elif language == 'tamil':
            # Tamil outline processing
            return text.strip()
        
        return text
    
    def _validate_outline_candidate_with_components(self, candidate: Dict, analysis: Dict) -> bool:
        """Validate outline candidate using configurable components with language awareness"""
        text = candidate['text']
        
        # Pattern matcher exclusion check
        try:
            if self.pattern_matcher.is_exclusion_pattern(text):
                return False
        except Exception as e:
            logger.warning(f"Pattern exclusion check error: {e}")
        
        # Semantic exclusion check (configurable threshold)
        semantic_scores = candidate.get('semantic_scores', {})
        exclusion_threshold = self.config.semantic_exclusion_threshold
        
        # Apply language-specific validation adjustments
        if (hasattr(self, 'language') and 
            self.config.use_language_specific_validation):
            exclusion_threshold = self._adjust_exclusion_threshold_for_language(exclusion_threshold)
        
        if semantic_scores.get('excluded', 0) > exclusion_threshold:
            return False
        
        # Must have structural significance (configurable threshold)
        structural_threshold = self.config.structural_significance_threshold
        
        # Apply language-specific validation adjustments
        if (hasattr(self, 'language') and 
            self.config.use_language_specific_validation):
            structural_threshold = self._adjust_structural_threshold_for_language(structural_threshold)
        
        if semantic_scores.get('structural', 0) < structural_threshold:
            return False
        
        return True
    
    def _adjust_exclusion_threshold_for_language(self, threshold: float) -> float:
        """Adjust exclusion threshold based on language characteristics"""
        if not hasattr(self, 'language'):
            return threshold
        
        language = self.language
        
        # Language-specific exclusion threshold adjustments
        adjustments = {
            'japanese': 0.95,   # Be more lenient with Japanese exclusions
            'german': 1.0,      # German exclusions are usually reliable
            'tamil': 0.9,       # Be more lenient with Tamil exclusions
            'english': 1.0
        }
        
        multiplier = adjustments.get(language, 1.0)
        return threshold * multiplier
    
    def _adjust_structural_threshold_for_language(self, threshold: float) -> float:
        """Adjust structural threshold based on language characteristics"""
        if not hasattr(self, 'language'):
            return threshold
        
        language = self.language
        
        # Language-specific structural threshold adjustments
        adjustments = {
            'japanese': 0.9,    # Be more lenient with Japanese structure detection
            'german': 1.0,      # German structure is usually clear
            'tamil': 0.85,      # Be more lenient with Tamil structure detection
            'english': 1.0
        }
        
        multiplier = adjustments.get(language, 1.0)
        return threshold * multiplier
    
    def _build_component_filtered_outline(self, candidates: List[Dict]) -> List[HeadingLevel]:
        """Build final outline using configurable component intelligence with language awareness"""
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
        
        # Final component validation using configurable thresholds with language awareness  
        validated_candidates = []
        for candidate in final_candidates:
            try:
                # Final semantic validation
                semantic_scores = self.semantic_classifier.classify_text_role(candidate['text'])
                
                # Apply language-specific semantic adjustments
                if (hasattr(self, 'language') and 
                    self.config.use_language_specific_validation):
                    semantic_scores = self._adjust_semantic_scores_for_language(semantic_scores, candidate['text'])
                
                # Get language-adjusted thresholds
                structural_threshold = self.config.final_structural_threshold
                exclusion_threshold = self.config.final_exclusion_threshold
                
                if (hasattr(self, 'language') and 
                    self.config.use_language_specific_validation):
                    structural_threshold = self._adjust_structural_threshold_for_language(structural_threshold)
                    exclusion_threshold = self._adjust_exclusion_threshold_for_language(exclusion_threshold)
                
                # Only include if components agree it's structural (configurable thresholds)
                if (semantic_scores.get('structural', 0) > structural_threshold and 
                    semantic_scores.get('excluded', 0) < exclusion_threshold):
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
            
            lang_info = f" (lang: {self.language})" if hasattr(self, 'language') else ""
            logger.info(f"Component-validated outline item: '{candidate['text']}' as {candidate['level']} "
                       f"(source: {candidate['source']}){lang_info}")
        
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
        
        # Add language context if available
        if hasattr(self, 'language'):
            result_dict['language_context'] = {
                'language': self.language,
                'component_adjustments': bool(self._language_component_weights)
            }
        
        try:
            if self.output_validator.validate_output(result_dict):
                lang_info = f" (lang: {self.language})" if hasattr(self, 'language') else ""
                logger.info(f"Final validation successful - Title: '{result.title}', "
                           f"Outline: {len(result.outline)} items{lang_info}")
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
    
    elif processor_type == 'multilingual':
        config = ProgramProcessorConfig(
            enable_language_aware_processing=True,
            apply_language_confidence_adjustments=True,
            use_language_specific_validation=True,
            language_component_weighting=True,
            document_confidence=0.9
        )
        return ProgramProcessor(config=config)
    
    elif processor_type == 'debug':
        config = ProgramProcessorConfig(
            enable_debug_logging=True,
            enable_language_aware_processing=True,
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
        pattern_recognition_weight=0.4,
        enable_language_aware_processing=True
    )
    custom_processor = ProgramProcessor(config=custom_config)
    
    # Using factory
    multilingual_processor = create_program_processor('multilingual')
    strict_processor = create_program_processor('strict')
    lenient_processor = create_program_processor('lenient')
