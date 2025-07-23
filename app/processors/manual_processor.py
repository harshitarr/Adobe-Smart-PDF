import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List, Dict
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
        try:
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
            
            logger.info("ManualProcessor initialized successfully")
        except Exception as e:
            logger.error(f"ManualProcessor initialization error: {e}")
            raise
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Component-driven processing for manual documents with enhanced safety"""
        logger.info("Processing manual document - component-driven analysis")
        
        try:
            if not text_blocks:
                logger.warning("No text blocks provided to ManualProcessor")
                return self._create_safe_result()
            
            logger.info(f"Processing {len(text_blocks)} text blocks")
            
            # Use TECHNICAL_MANUAL to match main.py mapping
            doc_type = DocumentType.TECHNICAL_MANUAL  # ← Fixed: changed from MANUAL_DOCUMENT
            
            # Safe filtering with fallback
            filtered_blocks = self._filter_blocks_safe(text_blocks, doc_type)
            
            # Safe structure analysis with fallback
            structure_analysis = self._analyze_structure_safe(filtered_blocks, doc_type)
            
            # Component-driven title extraction
            title = self._extract_title_safe(filtered_blocks, structure_analysis, doc_type)
            logger.info(f"Extracted title: '{title}'")
            
            # Component-driven outline extraction
            outline = self._extract_outline_safe(filtered_blocks, structure_analysis, doc_type)
            logger.info(f"Extracted outline: {len(outline)} items")
            
            result = DocumentStructure(
                title=title,
                outline=outline,
                doc_type=doc_type,  # ← Uses TECHNICAL_MANUAL
                confidence=0.9
            )
            
            logger.info("ManualProcessor completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"ManualProcessor critical error: {e}", exc_info=True)
            return self._create_safe_result()
    
    def _filter_blocks_safe(self, text_blocks: List[TextBlock], doc_type: DocumentType) -> List[TextBlock]:
        """Safe filtering with fallback"""
        try:
            filtered = self.content_filter.filter_structural_elements(text_blocks, doc_type)
            return filtered if filtered else text_blocks
        except Exception as e:
            logger.warning(f"Content filter error, using original blocks: {e}")
            return text_blocks
    
    def _analyze_structure_safe(self, filtered_blocks: List[TextBlock], doc_type: DocumentType) -> Dict:
        """Safe structure analysis with fallback"""
        try:
            return self.structure_analyzer.analyze_document_structure(filtered_blocks, doc_type)
        except Exception as e:
            logger.warning(f"Structure analyzer error, using empty analysis: {e}")
            return {}
    
    def _extract_title_safe(self, text_blocks: List[TextBlock], structure_analysis: Dict, doc_type: DocumentType) -> str:
        """Safe title extraction with multiple fallbacks"""
        title_candidates = []
        
        # Method 1: Title extractor
        try:
            extracted_title = self.title_extractor.extract_title(text_blocks, doc_type)
            if extracted_title and extracted_title.strip():
                # Use semantic classifier to avoid TOC contamination
                try:
                    semantic_scores = self.semantic_classifier.classify_text_role(extracted_title)
                    if semantic_scores.get('navigation', 0) < 0.5:  # Not navigation content
                        title_candidates.append({
                            'text': extracted_title,
                            'confidence': 0.9,
                            'method': 'title_extractor'
                        })
                except Exception:
                    # If semantic classification fails, still consider the title
                    title_candidates.append({
                        'text': extracted_title,
                        'confidence': 0.7,
                        'method': 'title_extractor_fallback'
                    })
        except Exception as e:
            logger.warning(f"Title extractor error: {e}")
        
        # Method 2: Direct analysis of early blocks
        for i, block in enumerate(text_blocks[:5]):
            try:
                text = getattr(block, 'text', '').strip()
                if not text or len(text) < 5:
                    continue
                
                # Use text processor for basic validation
                try:
                    if self.text_processor.is_likely_title(text):
                        title_candidates.append({
                            'text': text,
                            'confidence': 0.8 - (i * 0.1),
                            'method': f'text_processor_block_{i}'
                        })
                except Exception:
                    pass
                
                # Semantic analysis if available
                try:
                    semantic_scores = self.semantic_classifier.classify_text_role(text)
                    if (semantic_scores.get('structural', 0) > 0.5 and 
                        semantic_scores.get('navigation', 0) < 0.5):
                        
                        title_candidates.append({
                            'text': text,
                            'confidence': 0.7 - (i * 0.1),
                            'method': f'semantic_block_{i}'
                        })
                except Exception:
                    pass
                
            except Exception as e:
                logger.warning(f"Error analyzing block {i}: {e}")
        
        # Select best candidate
        if title_candidates:
            title_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best_title = title_candidates[0]['text']
            
            # Format title safely
            formatted_title = self._format_title_safe(best_title)
            logger.info(f"Selected title: '{formatted_title}' using {title_candidates[0]['method']}")
            return formatted_title
        
        logger.warning("No title candidates found")
        return ""
    
    def _format_title_safe(self, title: str) -> str:
        """Safe title formatting"""
        if not title:
            return ""
        
        try:
            clean_title = self.text_processor.clean_text(title)
            if clean_title:
                try:
                    formatted_title = self.text_processor.format_section_text(clean_title)
                    # Add appropriate spacing
                    if len(formatted_title.split()) <= 5:
                        return f"{formatted_title}  "  # Double space for short titles
                    return f"{formatted_title} "  # Single space for longer titles
                except Exception:
                    # Fallback formatting
                    return clean_title + "  "
            else:
                return title.strip() + "  "
        except Exception as e:
            logger.warning(f"Title formatting error: {e}")
            return title.strip() + "  "
    
    def _extract_outline_safe(self, text_blocks: List[TextBlock], structure_analysis: Dict, doc_type: DocumentType) -> List[HeadingLevel]:
        """Safe outline extraction with multiple methods"""
        
        # Method 1: Outline extractor
        try:
            outline = self.outline_extractor.extract_outline(text_blocks, doc_type)
            if outline and len(outline) > 0:
                validated = self._validate_outline_safe(outline)
                if validated:
                    logger.info(f"Outline extractor success: {len(validated)} items")
                    return validated
        except Exception as e:
            logger.warning(f"Outline extractor error: {e}")
        
        # Method 2: Structure analyzer
        try:
            if structure_analysis:
                outline = self.structure_analyzer.extract_hierarchical_outline(structure_analysis, doc_type)
                if outline and len(outline) > 0:
                    validated = self._validate_outline_safe(outline)
                    if validated:
                        logger.info(f"Structure analyzer success: {len(validated)} items")
                        return validated
        except Exception as e:
            logger.warning(f"Structure analyzer outline error: {e}")
        
        logger.warning("No outline extracted")
        return []
    
    def _validate_outline_safe(self, outline: List[HeadingLevel]) -> List[HeadingLevel]:
        """Safe outline validation"""
        if not outline:
            return []
        
        try:
            # Convert to dict format for validation
            outline_dict = {
                'title': '',
                'outline': [
                    {
                        'level': getattr(h, 'level', 'H1'),
                        'text': getattr(h, 'text', ''),
                        'page': getattr(h, 'page', 0)
                    } for h in outline if hasattr(h, 'text') and h.text
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
        except Exception as e:
            logger.warning(f"Outline validation error: {e}")
            return outline  # Return original if validation fails
    
    def _create_safe_result(self) -> DocumentStructure:
        """Create guaranteed safe result"""
        return DocumentStructure(
            title="",
            outline=[],
            doc_type=DocumentType.TECHNICAL_MANUAL,  # ← Fixed: changed from MANUAL_DOCUMENT
            confidence=0.0
        )
