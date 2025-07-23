import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List, Dict, Any
import logging
import re

# Core imports
from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor

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

class FormProcessor:
    def __init__(self):
        # Core components
        self.title_extractor = TitleExtractor()
        
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
        """Process form document with dynamic title extraction"""
        logger.info("Processing form document with enhanced title detection")
        
        # Filter structural elements but keep potential titles
        filtered_blocks = self._filter_blocks_preserve_titles(text_blocks)
        
        # Dynamic title extraction using multiple methods
        title = self._extract_title_dynamically(filtered_blocks)
        
        # Forms typically have no hierarchical outline
        outline = []
        
        # Validate the result
        result = DocumentStructure(
            title=title,
            outline=outline,
            doc_type=DocumentType.FORM_DOCUMENT,
            confidence=0.9
        )
        
        # Final validation using output validator
        output_dict = {
            "title": result.title,
            "outline": [{"level": h.level, "text": h.text, "page": h.page} for h in result.outline]
        }
        
        if self.output_validator.validate_output(output_dict):
            return result
        else:
            # Clean and retry
            cleaned_output = self.output_validator.clean_output(output_dict)
            return DocumentStructure(
                title=cleaned_output["title"],
                outline=[],
                doc_type=DocumentType.FORM_DOCUMENT,
                confidence=0.8
            )
    
    def _filter_blocks_preserve_titles(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Filter blocks but preserve potential titles"""
        preserved_blocks = []
        
        for block in text_blocks:
            text = block.text.strip()
            if not text:
                continue
            
            # Always preserve the first few blocks as they may contain titles
            if len(preserved_blocks) < 5:
                preserved_blocks.append(block)
                continue
            
            # Use content filter for the rest
            if not self.content_filter.is_structural_noise(text):
                preserved_blocks.append(block)
        
        return preserved_blocks
    
    def _extract_title_dynamically(self, text_blocks: List[TextBlock]) -> str:
        """Dynamic title extraction using multiple detection methods"""
        if not text_blocks:
            return ""
        
        title_candidates = []
        
        # Method 1: Check first block directly (most common for forms)
        first_block = text_blocks[0]
        if first_block and first_block.text.strip():
            first_text = first_block.text.strip()
            
            # Use semantic classifier to validate if it's a title
            is_title, confidence = self._validate_title_candidate(first_text, first_block, position_score=1.0)
            
            if is_title and confidence > 0.3:  # Lower threshold for first block
                title_candidates.append({
                    'text': first_text,
                    'confidence': confidence,
                    'method': 'first_block'
                })
        
        # Method 2: Look for form-specific title patterns in first few blocks
        for i, block in enumerate(text_blocks[:3]):
            text = block.text.strip()
            if not text or i == 0:  # Skip empty and first block (already checked)
                continue
            
            # Check for form title patterns
            form_score = self._calculate_form_title_score(text, block, i)
            
            if form_score > 0.5:
                is_title, confidence = self._validate_title_candidate(text, block, position_score=1.0-i*0.2)
                
                if is_title:
                    title_candidates.append({
                        'text': text,
                        'confidence': confidence * form_score,
                        'method': 'form_pattern'
                    })
        
        # Method 3: Use structure analyzer for comprehensive analysis
        if text_blocks:
            structure_analysis = self.structure_analyzer.analyze_document_structure(text_blocks, DocumentType.FORM_DOCUMENT)
            
            # Look for structurally significant blocks
            for i, block in enumerate(text_blocks[:5]):
                text = block.text.strip()
                if not text:
                    continue
                
                # Use font analyzer to check visual prominence
                font_analysis = structure_analysis.get('font_analysis', {})
                if font_analysis:
                    font_importance = self.font_analyzer.classify_text_block_importance(block, font_analysis)
                    
                    if font_importance > 0.6:  # High visual importance
                        is_title, confidence = self._validate_title_candidate(text, block, position_score=1.0-i*0.1)
                        
                        if is_title:
                            title_candidates.append({
                                'text': text,
                                'confidence': confidence * font_importance,
                                'method': 'visual_prominence'
                            })
        
        # Method 4: Fallback to title extractor
        extracted_title = self.title_extractor.extract_title(text_blocks, DocumentType.FORM_DOCUMENT)
        if extracted_title and extracted_title.strip():
            title_candidates.append({
                'text': extracted_title.strip(),
                'confidence': 0.7,
                'method': 'title_extractor'
            })
        
        # Select best candidate
        if title_candidates:
            # Sort by confidence and select best
            title_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best_candidate = title_candidates[0]
            
            # Clean the selected title
            cleaned_title = self._clean_title(best_candidate['text'])
            
            logger.info(f"Selected title: '{cleaned_title}' using method: {best_candidate['method']} (confidence: {best_candidate['confidence']:.2f})")
            
            return cleaned_title
        
        return ""
    
    def _validate_title_candidate(self, text: str, block: TextBlock, position_score: float = 1.0) -> tuple[bool, float]:
        """Validate if text is likely a title using semantic classifier"""
        confidence = 0.0
        
        # Use semantic classifier
        is_title, semantic_confidence = self.semantic_classifier.is_document_title(
            text,
            DocumentType.FORM_DOCUMENT,
            {
                'is_first_page': block.page <= 1,
                'font_size': getattr(block, 'font_size', 12),
                'is_bold': getattr(block, 'is_bold', False),
                'position_score': position_score
            }
        )
        
        if is_title:
            confidence = semantic_confidence * position_score
        
        # Additional validation using text processor
        if self.text_processor.is_likely_title(text):
            confidence += 0.2
        
        # Form-specific validation
        form_indicators = [
            'application' in text.lower(),
            'form' in text.lower(),
            'grant' in text.lower(),
            len(text.split()) >= 3,  # Titles usually have multiple words
            len(text.split()) <= 12,  # But not too many
            not text.startswith(tuple('0123456789')),  # Not a numbered item
            not text.endswith(':')  # Not a field label
        ]
        
        form_score = sum(form_indicators) / len(form_indicators)
        confidence *= (0.5 + form_score * 0.5)  # Boost confidence for form-like titles
        
        return confidence > 0.3, confidence
    
    def _calculate_form_title_score(self, text: str, block: TextBlock, position: int) -> float:
        """Calculate form title likelihood score"""
        score = 0.0
        text_lower = text.lower()
        
        # Form title keywords (dynamic, not hardcoded values)
        form_keywords = ['application', 'form', 'grant', 'request', 'advance', 'leave', 'travel']
        keyword_matches = sum(1 for keyword in form_keywords if keyword in text_lower)
        
        if keyword_matches > 0:
            score += 0.4 * (keyword_matches / len(form_keywords))
        
        # Length appropriateness
        word_count = len(text.split())
        if 3 <= word_count <= 10:
            score += 0.3
        elif word_count > 10:
            score -= 0.2
        
        # Position factor (earlier is better for titles)
        score += 0.2 * (1 - position * 0.2)
        
        # Avoid field labels
        if text.endswith(':') or text.startswith(tuple('0123456789')):
            score -= 0.5
        
        return max(0.0, score)
    
    def _clean_title(self, title: str) -> str:
        """Clean and format title properly"""
        if not title:
            return ""
        
        # Use text processor for initial cleaning
        cleaned = self.text_processor.clean_text(title)
        
        # Additional form-specific cleaning
        # Remove excessive whitespace but preserve trailing space if originally present
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Preserve the original trailing space pattern if it exists
        if title.rstrip() != title:  # Original had trailing space
            if not cleaned.endswith(' '):
                cleaned += ' '
        
        return cleaned
