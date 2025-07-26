import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import Dict, List, Tuple
import logging
from collections import Counter

from models.document_types import TextBlock, DocumentType
from classifiers.pattern_matcher import PatternMatcher

logger = logging.getLogger('classification')

class DocumentClassifier:
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.classification_threshold = 0.3
    
    def classify_document(self, text_blocks: List[TextBlock]) -> Tuple[DocumentType, float]:
        """Classify document type based on text content"""
        if not text_blocks:
            logger.warning("No text blocks provided for classification")
            return DocumentType.FORM_DOCUMENT, 0.0
        
        # Extract text for analysis
        text_content = [block.text for block in text_blocks if block.text.strip()]
        
        if not text_content:
            logger.warning("No valid text content found")
            return DocumentType.FORM_DOCUMENT, 0.0
        
        # Get pattern-based scores
        type_scores = self.pattern_matcher.match_document_type_patterns(text_content)
        
        # Enhance with structural analysis
        enhanced_scores = self._enhance_with_structural_analysis(text_blocks, type_scores)
        
        # Find best classification
        if enhanced_scores:
            best_type = max(enhanced_scores, key=enhanced_scores.get)
            confidence = enhanced_scores[best_type]
            
            if confidence >= self.classification_threshold:
                logger.info(f"Document classified as {best_type.value} with confidence {confidence:.2f}")
                return best_type, confidence
        
        # Default classification
        logger.warning("Unable to classify document confidently, defaulting to FORM_DOCUMENT")
        return DocumentType.FORM_DOCUMENT, 0.1
    
    def _enhance_with_structural_analysis(self, text_blocks: List[TextBlock], 
                                        initial_scores: Dict[DocumentType, float]) -> Dict[DocumentType, float]:
        """Enhance classification with structural analysis"""
        enhanced_scores = initial_scores.copy()
        
        # Analyze document structure characteristics
        structure_features = self._analyze_structure_features(text_blocks)
        
        # Apply structural bonuses/penalties
        for doc_type in DocumentType:
            structural_bonus = self._calculate_structural_bonus(doc_type, structure_features)
            enhanced_scores[doc_type] = enhanced_scores.get(doc_type, 0.0) + structural_bonus
        
        # Normalize scores
        max_score = max(enhanced_scores.values()) if enhanced_scores else 1.0
        if max_score > 0:
            enhanced_scores = {k: v / max_score for k, v in enhanced_scores.items()}
        
        return enhanced_scores
    
    def _analyze_structure_features(self, text_blocks: List[TextBlock]) -> Dict[str, any]:
        """Analyze structural features of the document"""
        features = {
            'has_numbered_sections': False,
            'has_decimal_sections': False,
            'has_appendices': False,
            'has_form_fields': False,
            'has_table_of_contents': False,
            'has_phases': False,
            'has_pathways': False,
            'has_invitation_elements': False,
            'total_pages': 0,
            'avg_text_length': 0,
            'font_variety': 0
        }
        
        combined_text = ' '.join([block.text for block in text_blocks]).lower()
        
        # Check for structural patterns
        features['has_numbered_sections'] = bool(re.search(r'\d+\.\s+[A-Z]', combined_text, re.IGNORECASE))
        features['has_decimal_sections'] = bool(re.search(r'\d+\.\d+\s+[A-Z]', combined_text, re.IGNORECASE))
        features['has_appendices'] = 'appendix' in combined_text
        features['has_form_fields'] = bool(re.search(r'(signature|date|place)\s*:', combined_text))
        features['has_table_of_contents'] = 'table of contents' in combined_text
        features['has_phases'] = 'phase' in combined_text and ':' in combined_text
        features['has_pathways'] = 'pathway' in combined_text
        features['has_invitation_elements'] = any(phrase in combined_text for phrase in [
            'you\'re invited', 'rsvp', 'hope to see you', 'party invitation'
        ])
        
        # Document characteristics
        features['total_pages'] = len(set(block.page for block in text_blocks))
        features['avg_text_length'] = sum(len(block.text) for block in text_blocks) / len(text_blocks)
        features['font_variety'] = len(set(block.font_name for block in text_blocks if block.font_name))
        
        return features
    
    def _calculate_structural_bonus(self, doc_type: DocumentType, features: Dict[str, any]) -> float:
        """Calculate structural bonus for document type"""
        bonus = 0.0
        
        if doc_type == DocumentType.FORM_DOCUMENT:
            if features['has_form_fields']:
                bonus += 0.4
            if features['total_pages'] <= 2:
                bonus += 0.2
        
        elif doc_type == DocumentType.TECHNICAL_MANUAL:
            if features['has_numbered_sections']:
                bonus += 0.3
            if features['has_decimal_sections']:
                bonus += 0.3
            if features['has_table_of_contents']:
                bonus += 0.2
        
        elif doc_type == DocumentType.BUSINESS_DOCUMENT:
            if features['has_appendices']:
                bonus += 0.3
            if features['has_phases']:
                bonus += 0.2
            if features['total_pages'] > 5:
                bonus += 0.2
        
        elif doc_type == DocumentType.PROGRAM_DOCUMENT:
            if features['has_pathways']:
                bonus += 0.4
            if features['total_pages'] <= 3:
                bonus += 0.2
        
        elif doc_type == DocumentType.INVITATION_DOCUMENT:
            if features['has_invitation_elements']:
                bonus += 0.5
            if features['total_pages'] == 1:
                bonus += 0.3
        
        return bonus
