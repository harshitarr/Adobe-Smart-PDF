import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

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

class ProgramProcessor:
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
            doc_type=DocumentType.PROGRAM_DOCUMENT,
            confidence=0.9
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
                text_blocks, DocumentType.PROGRAM_DOCUMENT
            )
        except Exception as e:
            logger.warning(f"Structure analyzer error: {e}")
            analysis['structure_result'] = {}
        
        # Heading detector - identify all potential headings
        try:
            font_analysis = self.font_analyzer.analyze_fonts(text_blocks)
            analysis['heading_detections'] = self.heading_detector.detect_headings(
                text_blocks, DocumentType.PROGRAM_DOCUMENT, font_analysis
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
                
                if matched_level and pattern_confidence > 0.0:
                    classification['pattern_level'] = matched_level
                    classification['pattern_confidence'] = pattern_confidence
                    classification['pattern_info'] = pattern_info
                    
                    # Combine semantic and pattern analysis for hierarchy determination
                    combined_importance = self._calculate_hierarchical_importance(
                        semantic_scores, pattern_confidence, matched_level
                    )
                    
                    classification['hierarchical_importance'] = combined_importance
                    
                    # Categorize based on component intelligence
                    if combined_importance > 0.8:
                        hierarchy['primary_sections'].append(classification)
                    elif combined_importance > 0.6:
                        hierarchy['secondary_sections'].append(classification)
                    else:
                        hierarchy['content_elements'].append(classification)
                        
            except Exception as e:
                logger.warning(f"Pattern analysis error: {e}")
        
        return hierarchy
    
    def _calculate_hierarchical_importance(self, semantic_scores: Dict, pattern_confidence: float, matched_level: str) -> float:
        """Calculate hierarchical importance using component scores"""
        importance = 0.0
        
        # Semantic structural importance
        importance += semantic_scores.get('structural', 0) * 0.5
        
        # Pattern recognition confidence
        importance += pattern_confidence * 0.3
        
        # Level-based importance (H1 > H2 > H3)
        level_weights = {'H1': 0.2, 'H2': 0.15, 'H3': 0.1, 'H4': 0.05}
        importance += level_weights.get(matched_level, 0)
        
        # Exclusion penalty
        if semantic_scores.get('excluded', 0) > 0.5:
            importance *= 0.5
        
        return min(1.0, importance)
    
    def _extract_title_pure_components(self, text_blocks: List[TextBlock], analysis: Dict) -> str:
        """Extract title using pure component intelligence"""
        # Method 1: Title extractor (primary method)
        try:
            extracted_title = self.title_extractor.extract_title(text_blocks, DocumentType.PROGRAM_DOCUMENT)
            if extracted_title:
                return self.text_processor.clean_text(extracted_title)
        except Exception as e:
            logger.warning(f"Title extractor error: {e}")
        
        # Method 2: Semantic classifier on early content
        for classification in analysis['semantic_classifications'][:5]:
            try:
                is_title, confidence = self.semantic_classifier.is_document_title(
                    classification['text'], DocumentType.PROGRAM_DOCUMENT, {}
                )
                
                if is_title and confidence > 0.7:
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
        
        # Method 2: Heading detector validation
        for block, level, confidence in analysis['heading_detections']:
            text = block.text.strip()
            
            # Skip title duplicates
            if title and text.lower().strip() == title.lower().strip():
                continue
            
            # High confidence detections only
            if confidence > 0.7:
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
                analysis['structure_result'], DocumentType.PROGRAM_DOCUMENT
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
                            'confidence': getattr(heading, 'confidence', 0.8),
                            'source': 'structure_analyzer'
                        })
        except Exception as e:
            logger.warning(f"Structure outline error: {e}")
        
        return self._build_component_filtered_outline(candidates)
    
    def _validate_outline_candidate_with_components(self, candidate: Dict, analysis: Dict) -> bool:
        """Validate outline candidate using components"""
        text = candidate['text']
        
        # Pattern matcher exclusion check
        try:
            if self.pattern_matcher.is_exclusion_pattern(text):
                return False
        except Exception as e:
            logger.warning(f"Pattern exclusion check error: {e}")
        
        # Semantic exclusion check
        semantic_scores = candidate.get('semantic_scores', {})
        if semantic_scores.get('excluded', 0) > 0.7:
            return False
        
        # Must have structural significance
        if semantic_scores.get('structural', 0) < 0.5:
            return False
        
        return True
    
    def _build_component_filtered_outline(self, candidates: List[Dict]) -> List[HeadingLevel]:
        """Build final outline using component intelligence"""
        if not candidates:
            return []
        
        # Component-based prioritization
        # Sort by confidence and source priority
        source_priority = {
            'hierarchical_primary': 1,
            'heading_detector': 2, 
            'structure_analyzer': 3
        }
        
        candidates.sort(key=lambda x: (
            source_priority.get(x['source'], 4),
            -x['confidence']
        ))
        
        # Remove duplicates (keep highest priority)
        unique_candidates = {}
        for candidate in candidates:
            text_key = candidate['text'].lower().strip()
            if text_key not in unique_candidates:
                unique_candidates[text_key] = candidate
        
        final_candidates = list(unique_candidates.values())
        
        # Final component validation
        validated_candidates = []
        for candidate in final_candidates:
            try:
                # Final semantic validation
                semantic_scores = self.semantic_classifier.classify_text_role(candidate['text'])
                
                # Only include if components agree it's structural
                if (semantic_scores.get('structural', 0) > 0.6 and 
                    semantic_scores.get('excluded', 0) < 0.3):
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
        """Final validation using output validator"""
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
                            confidence=0.8
                        ) for item in cleaned_dict["outline"]
                    ],
                    doc_type=DocumentType.PROGRAM_DOCUMENT,
                    confidence=0.8
                )
        except Exception as e:
            logger.error(f"Final validation error: {e}")
            return result
    
    def _create_empty_result(self) -> DocumentStructure:
        """Create empty result"""
        return DocumentStructure(
            title="",
            outline=[],
            doc_type=DocumentType.PROGRAM_DOCUMENT,
            confidence=0.0
        )
