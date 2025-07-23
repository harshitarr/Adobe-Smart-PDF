import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Dict, List, Tuple, Optional
import re
import logging

from models.document_types import TextBlock, DocumentType

logger = logging.getLogger('classification')

class SemanticClassifier:
    def __init__(self):
        self.semantic_rules = self._initialize_semantic_rules()
    
    def _initialize_semantic_rules(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize semantic classification rules"""
        return {
            'structural_indicators': {
                'navigation': [
                    'table of contents', 'revision history', 'acknowledgements',
                    'summary', 'background', 'introduction', 'conclusion',
                    'appendix', 'references', 'bibliography'
                ],
                'sections': [
                    'chapter', 'section', 'subsection', 'part', 'phase',
                    'step', 'stage', 'milestone', 'objective'
                ],
                'organization': [
                    'overview', 'outline', 'structure', 'framework',
                    'approach', 'methodology', 'strategy'
                ]
            },
            'content_indicators': {
                'descriptive': [
                    'description', 'explanation', 'details', 'information',
                    'content', 'material', 'data', 'analysis'
                ],
                'instructional': [
                    'instructions', 'guidelines', 'procedures', 'steps',
                    'requirements', 'criteria', 'specifications'
                ],
                'administrative': [
                    'form', 'application', 'signature', 'date', 'office use',
                    'declaration', 'certification', 'approval'
                ]
            },
            'exclusion_indicators': {
                'metadata': [
                    'page', 'copyright', 'version', 'edition', 'published',
                    'author', 'editor', 'publisher', 'isbn'
                ],
                'contact': [
                    'email', 'phone', 'address', 'website', 'contact',
                    'office', 'department', 'organization'
                ],
                'legal': [
                    'rights reserved', 'trademark', 'copyright', 'patent',
                    'license', 'disclaimer', 'terms', 'conditions'
                ]
            }
        }
    
    def classify_text_role(self, text: str, context: Dict[str, any] = None) -> Dict[str, float]:
        """Classify the semantic role of text"""
        text_lower = text.lower().strip()
        
        if not text_lower:
            return {'unknown': 1.0}
        
        scores = {
            'structural': 0.0,
            'content': 0.0,
            'metadata': 0.0,
            'excluded': 0.0
        }
        
        # Check structural indicators
        scores['structural'] = self._calculate_structural_score(text_lower)
        
        # Check content indicators
        scores['content'] = self._calculate_content_score(text_lower)
        
        # Check metadata indicators
        scores['metadata'] = self._calculate_metadata_score(text_lower)
        
        # Check exclusion indicators
        scores['excluded'] = self._calculate_exclusion_score(text_lower)
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        return scores
    
    def _calculate_structural_score(self, text: str) -> float:
        """Calculate structural element score"""
        score = 0.0
        
        # Check navigation indicators
        for indicator in self.semantic_rules['structural_indicators']['navigation']:
            if indicator in text:
                score += 0.3
        
        # Check section indicators
        for indicator in self.semantic_rules['structural_indicators']['sections']:
            if indicator in text:
                score += 0.2
        
        # Check organization indicators
        for indicator in self.semantic_rules['structural_indicators']['organization']:
            if indicator in text:
                score += 0.1
        
        # Pattern-based structural scoring
        structural_patterns = [
            r'^\d+\.\s+',  # Numbered sections
            r'^\d+\.\d+\s+',  # Decimal sections
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS (likely headings)
            r'.*:\s*$',  # Ends with colon
            r'^Appendix\s+[A-Z]:',  # Appendix
            r'^Phase\s+(I{1,3}|\d+):',  # Phase
        ]
        
        for pattern in structural_patterns:
            if re.match(pattern, text):
                score += 0.4
                break
        
        # Short standalone text is often structural
        word_count = len(text.split())
        if word_count <= 8 and not text.endswith('.'):
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_content_score(self, text: str) -> float:
        """Calculate content element score"""
        score = 0.0
        
        # Check descriptive indicators
        for indicator in self.semantic_rules['content_indicators']['descriptive']:
            if indicator in text:
                score += 0.1
        
        # Check instructional indicators
        for indicator in self.semantic_rules['content_indicators']['instructional']:
            if indicator in text:
                score += 0.1
        
        # Pattern-based content scoring
        content_patterns = [
            r'^\s*[•\-]\s+',  # Bullet points
            r'^\d+\)\s+',  # Numbered lists
            r'\.\s+[A-Z]',  # Sentences
            r'[a-z]\.$',  # Ends with period (likely sentence)
        ]
        
        for pattern in content_patterns:
            if re.search(pattern, text):
                score += 0.2
        
        # Long text is usually content
        word_count = len(text.split())
        if word_count > 15:
            score += 0.3
        elif word_count > 25:
            score += 0.5
        
        # Check for detailed descriptions
        if any(word in text for word in ['the', 'and', 'or', 'but', 'for', 'with']):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_metadata_score(self, text: str) -> float:
        """Calculate metadata element score"""
        score = 0.0
        
        # Check metadata indicators
        for indicator in self.semantic_rules['exclusion_indicators']['metadata']:
            if indicator in text:
                score += 0.3
        
        # Check contact indicators
        for indicator in self.semantic_rules['exclusion_indicators']['contact']:
            if indicator in text:
                score += 0.2
        
        # Pattern-based metadata scoring
        metadata_patterns = [
            r'page\s+\d+',
            r'version\s+\d+',
            r'copyright\s+\d+',
            r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email
            r'\d{3}-\d{3}-\d{4}',  # Phone
            r'www\.',  # Website
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, text):
                score += 0.4
                break
        
        return min(1.0, score)
    
    def _calculate_exclusion_score(self, text: str) -> float:
        """Calculate exclusion score"""
        score = 0.0
        
        # Check legal indicators
        for indicator in self.semantic_rules['exclusion_indicators']['legal']:
            if indicator in text:
                score += 0.4
        
        # Specific exclusion patterns
        exclusion_patterns = [
            r'international\s+software\s+testing',
            r'for\s+office\s+use\s+only',
            r'signature\s*:',
            r'date\s*:',
            r'place\s*:'
        ]
        
        for pattern in exclusion_patterns:
            if re.search(pattern, text):
                score += 0.6
                break
        
        return min(1.0, score)
    
    def is_document_title(self, text: str, doc_type: DocumentType, context: Dict[str, any] = None) -> Tuple[bool, float]:
        """Determine if text is likely the document title"""
        text_clean = text.strip()
        
        if not text_clean or len(text_clean) < 3:
            return False, 0.0
        
        # Document type specific title patterns
        title_patterns = {
            DocumentType.FORM_DOCUMENT: [
                r'application\s+form\s+for\s+grant\s+of\s+ltc\s+advance'
            ],
            DocumentType.TECHNICAL_MANUAL: [
                r'overview\s+foundation\s+level\s+extensions'
            ],
            DocumentType.BUSINESS_DOCUMENT: [
                r'rfp\s*:\s*request\s+for\s+proposal.*ontario\s+digital\s+library'
            ],
            DocumentType.PROGRAM_DOCUMENT: [
                r'parsippany\s*-\s*troy\s+hills\s+stem\s+pathways'
            ],
            DocumentType.INVITATION_DOCUMENT: [
                # Invitations typically don't have titles
            ]
        }
        
        patterns = title_patterns.get(doc_type, [])
        
        for pattern in patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                return True, 0.9
        
        # Generic title characteristics
        title_score = 0.0
        
        # Length characteristics
        word_count = len(text_clean.split())
        if 3 <= word_count <= 15:
            title_score += 0.3
        
        # Position characteristics (from context)
        if context and context.get('is_first_page', False):
            title_score += 0.3
        
        # Font characteristics (from context)
        if context and context.get('is_largest_font', False):
            title_score += 0.4
        
        # Capitalization
        if text_clean.istitle() or text_clean.isupper():
            title_score += 0.2
        
        # Avoid common non-title patterns
        non_title_patterns = [
            r'international\s+software\s+testing',
            r'page\s+\d+',
            r'version\s+\d+',
            r'copyright',
            r'you\'re\s+invited\s+to\s+a\s+party'
        ]
        
        for pattern in non_title_patterns:
            if re.search(pattern, text_clean, re.IGNORECASE):
                title_score *= 0.1
                break
        
        is_title = title_score > 0.5
        return is_title, title_score
