import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List
import logging

from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor
from analyzers.font_analyzer import FontAnalyzer
from classifiers.heading_detector import HeadingDetector
from classifiers.pattern_matcher import PatternMatcher
from classifiers.semantic_classifier import SemanticClassifier
from utils.text_utils import TextProcessor
from utils.validation_utils import OutputValidator

logger = logging.getLogger('extraction')

class BusinessProcessor:
    def __init__(self):
        self.title_extractor = TitleExtractor()
        self.outline_extractor = OutlineExtractor()
        self.font_analyzer = FontAnalyzer()
        self.heading_detector = HeadingDetector()
        self.pattern_matcher = PatternMatcher()
        self.semantic_classifier = SemanticClassifier()
        self.text_processor = TextProcessor()
        self.output_validator = OutputValidator()
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process business document with enhanced error handling"""
        logger.info("Processing business document")
        
        try:
            if not text_blocks:
                return self._create_empty_result()
            
            # Component-driven title extraction with corruption repair
            title = self._extract_title_with_corruption_repair(text_blocks)
            
            # Component-driven outline extraction
            outline = self._extract_outline_comprehensive(text_blocks)
            
            result = DocumentStructure(
                title=title,
                outline=outline,
                doc_type=DocumentType.BUSINESS_DOCUMENT,
                confidence=0.9
            )
            
            return self._validate_result(result)
            
        except Exception as e:
            logger.error(f"Business processing error: {e}")
            return self._create_empty_result()
    
    def _extract_title_with_corruption_repair(self, text_blocks: List[TextBlock]) -> str:
        """Extract title with enhanced corruption repair"""
        title_candidates = []
        
        # Method 1: Title extractor (handles corruption repair)
        try:
            extracted = self.title_extractor.extract_title(text_blocks, DocumentType.BUSINESS_DOCUMENT)
            if extracted and extracted.strip():
                cleaned = self.text_processor.clean_text(extracted)
                if cleaned and len(cleaned) > 5:
                    title_candidates.append({
                        'text': cleaned,
                        'confidence': 0.9,
                        'method': 'title_extractor'
                    })
        except Exception as e:
            logger.warning(f"Title extractor error: {e}")
        
        # Method 2: Semantic analysis on early blocks
        for i, block in enumerate(text_blocks[:5]):
            text = getattr(block, 'text', '').strip()
            if not text or len(text) < 5:
                continue
            
            try:
                is_title, confidence = self.semantic_classifier.is_document_title(
                    text, DocumentType.BUSINESS_DOCUMENT, {'position': i}
                )
                if is_title and confidence > 0.5:
                    cleaned = self.text_processor.clean_text(text)
                    if cleaned and len(cleaned) > 5:
                        title_candidates.append({
                            'text': cleaned,
                            'confidence': confidence,
                            'method': f'semantic_block_{i}'
                        })
            except Exception as e:
                logger.warning(f"Semantic analysis error for block {i}: {e}")
        
        # Select best candidate
        return self._select_best_title(title_candidates)
    
    def _select_best_title(self, candidates: List[dict]) -> str:
        """Select best title candidate"""
        if not candidates:
            return ""
        
        # Sort by confidence and select best
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        best = candidates[0]
        
        title = best['text']
        # Format with trailing spaces
        if not title.endswith('  '):
            title += '  '
        
        logger.info(f"Selected title: '{title}' using {best['method']} (confidence: {best['confidence']:.2f})")
        return title
    
    def _extract_outline_comprehensive(self, text_blocks: List[TextBlock]) -> List[HeadingLevel]:
        """Comprehensive outline extraction for business documents"""
        
        # Method 1: Outline extractor (primary for business docs)
        try:
            extracted_outline = self.outline_extractor.extract_outline(text_blocks, DocumentType.BUSINESS_DOCUMENT)
            if extracted_outline and len(extracted_outline) > 5:  # Expect substantial outline
                logger.info(f"Outline extractor found {len(extracted_outline)} items")
                return extracted_outline
        except Exception as e:
            logger.warning(f"Outline extractor error: {e}")
        
        # Method 2: Heading detector with comprehensive analysis
        try:
            font_analysis = self.font_analyzer.analyze_fonts(text_blocks)
            detected_headings = self.heading_detector.detect_headings(
                text_blocks, DocumentType.BUSINESS_DOCUMENT, font_analysis
            )
            
            outline_candidates = []
            for block, level, confidence in detected_headings:
                if confidence > 0.2:  # Lower threshold for comprehensive extraction
                    text = getattr(block, 'text', '').strip()
                    if text:
                        clean_text = self.text_processor.clean_text(text)
                        
                        # Validate with pattern matcher
                        try:
                            if not self.pattern_matcher.is_exclusion_pattern(clean_text):
                                outline_candidates.append(HeadingLevel(
                                    level=level,
                                    text=clean_text,
                                    page=getattr(block, 'page', 0),
                                    confidence=confidence
                                ))
                        except Exception as e:
                            # Include anyway if pattern matcher fails
                            outline_candidates.append(HeadingLevel(
                                level=level,
                                text=clean_text,
                                page=getattr(block, 'page', 0),
                                confidence=confidence
                            ))
            
            if outline_candidates:
                logger.info(f"Heading detector found {len(outline_candidates)} items")
                return self._sort_outline_items(outline_candidates)
        except Exception as e:
            logger.warning(f"Heading detector error: {e}")
        
        return []
    
    def _sort_outline_items(self, outline_items: List[HeadingLevel]) -> List[HeadingLevel]:
        """Sort outline items by page and hierarchical level"""
        level_order = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4}
        outline_items.sort(key=lambda x: (
            x.page,
            level_order.get(x.level, 5),
            -x.confidence
        ))
        return outline_items
    
    def _validate_result(self, result: DocumentStructure) -> DocumentStructure:
        """Validate result using output validator"""
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
                logger.info(f"Business validation successful - Title: '{result.title}', Outline: {len(result.outline)} items")
                return result
            else:
                logger.warning("Business validation failed, cleaning")
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
                    doc_type=DocumentType.BUSINESS_DOCUMENT,
                    confidence=0.8
                )
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return result
    
    def _create_empty_result(self) -> DocumentStructure:
        """Create empty result structure"""
        return DocumentStructure(
            title="",
            outline=[],
            doc_type=DocumentType.BUSINESS_DOCUMENT,
            confidence=0.0
        )
