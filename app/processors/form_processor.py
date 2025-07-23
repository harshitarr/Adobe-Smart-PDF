import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List
import logging

from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from analyzers.font_analyzer import FontAnalyzer
from classifiers.semantic_classifier import SemanticClassifier
from utils.text_utils import TextProcessor
from utils.validation_utils import OutputValidator

logger = logging.getLogger('extraction')

class FormProcessor:
    def __init__(self):
        self.title_extractor = TitleExtractor()
        self.font_analyzer = FontAnalyzer()
        self.semantic_classifier = SemanticClassifier()
        self.text_processor = TextProcessor()
        self.output_validator = OutputValidator()
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process form document with enhanced error handling"""
        logger.info("Processing form document")
        
        try:
            if not text_blocks:
                return self._create_empty_result()
            
            # Extract title with multiple methods
            title = self._extract_title_comprehensive(text_blocks)
            
            # Forms have no outline
            outline = []
            
            result = DocumentStructure(
                title=title,
                outline=outline,
                doc_type=DocumentType.FORM_DOCUMENT,
                confidence=0.9
            )
            
            return self._validate_result(result)
            
        except Exception as e:
            logger.error(f"Form processing error: {e}")
            return self._create_empty_result()
    
    def _extract_title_comprehensive(self, text_blocks: List[TextBlock]) -> str:
        """Comprehensive title extraction for forms"""
        title_candidates = []
        
        # Method 1: Title extractor
        try:
            extracted = self.title_extractor.extract_title(text_blocks, DocumentType.FORM_DOCUMENT)
            if extracted and extracted.strip():
                cleaned = self.text_processor.clean_text(extracted)
                title_candidates.append({
                    'text': cleaned,
                    'confidence': 0.9,
                    'method': 'title_extractor'
                })
        except Exception as e:
            logger.warning(f"Title extractor error: {e}")
        
        # Method 2: Semantic analysis on early blocks
        for i, block in enumerate(text_blocks[:3]):
            text = getattr(block, 'text', '').strip()
            if not text or len(text) < 5:
                continue
            
            try:
                is_title, confidence = self.semantic_classifier.is_document_title(
                    text, DocumentType.FORM_DOCUMENT, {'position': i}
                )
                if is_title and confidence > 0.4:
                    cleaned = self.text_processor.clean_text(text)
                    title_candidates.append({
                        'text': cleaned,
                        'confidence': confidence,
                        'method': f'semantic_block_{i}'
                    })
            except Exception as e:
                logger.warning(f"Semantic analysis error for block {i}: {e}")
        
        # Method 3: Text processor validation on early blocks
        for i, block in enumerate(text_blocks[:3]):
            text = getattr(block, 'text', '').strip()
            if not text or len(text) < 5:
                continue
            
            try:
                if self.text_processor.is_likely_title(text):
                    # Additional validation: not a field label
                    if not text.endswith(':') and not text.isdigit():
                        cleaned = self.text_processor.clean_text(text)
                        title_candidates.append({
                            'text': cleaned,
                            'confidence': 0.7,
                            'method': f'text_processor_block_{i}'
                        })
            except Exception as e:
                logger.warning(f"Text processor error for block {i}: {e}")
        
        # Select best candidate
        return self._select_best_title(title_candidates)
    
    def _select_best_title(self, candidates: List[dict]) -> str:
        """Select best title candidate"""
        if not candidates:
            logger.warning("No title candidates found")
            return ""
        
        # Remove duplicates and select highest confidence
        unique_candidates = {}
        for candidate in candidates:
            text_key = candidate['text'].lower().strip()
            if text_key not in unique_candidates or candidate['confidence'] > unique_candidates[text_key]['confidence']:
                unique_candidates[text_key] = candidate
        
        # Sort by confidence
        final_candidates = list(unique_candidates.values())
        final_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        if final_candidates:
            best = final_candidates[0]
            title = best['text']
            
            # Format with trailing spaces
            if title and not title.endswith('  '):
                title += '  '
            
            logger.info(f"Selected title: '{title}' using {best['method']} (confidence: {best['confidence']:.2f})")
            return title
        
        return ""
    
    def _validate_result(self, result: DocumentStructure) -> DocumentStructure:
        """Validate result using output validator"""
        result_dict = {
            "title": result.title,
            "outline": []
        }
        
        try:
            if self.output_validator.validate_output(result_dict):
                logger.info(f"Form validation successful - Title: '{result.title}'")
                return result
            else:
                logger.warning("Form validation failed, cleaning")
                cleaned_dict = self.output_validator.clean_output(result_dict)
                return DocumentStructure(
                    title=cleaned_dict["title"],
                    outline=[],
                    doc_type=DocumentType.FORM_DOCUMENT,
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
            doc_type=DocumentType.FORM_DOCUMENT,
            confidence=0.0
        )
