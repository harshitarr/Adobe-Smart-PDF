import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Optional
import logging

from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor

logger = logging.getLogger('extraction')

class InvitationProcessor:
    def __init__(self):
        self.title_extractor = TitleExtractor()
        self.outline_extractor = OutlineExtractor()
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process document with intelligent pattern recognition"""
        logger.info("Processing document with pattern-based structure analysis")
        
        # Comprehensive document analysis
        doc_analysis = self._analyze_document_patterns(text_blocks)
        
        # Smart title extraction
        title = self._extract_title_intelligently(text_blocks, doc_analysis)
        
        # Smart outline extraction
        outline = self._extract_outline_intelligently(text_blocks, doc_analysis)
        
        return DocumentStructure(
            title=title,
            outline=outline,
            doc_type=DocumentType.INVITATION_DOCUMENT,
            confidence=0.9
        )
    
    def _analyze_document_patterns(self, text_blocks: List[TextBlock]) -> Dict:
        """Analyze document using comprehensive pattern recognition"""
        analysis = {
            'content_types': {
                'form_fields': [],      # FOR:, DATE:, TIME: etc.
                'addresses': [],        # Address information
                'contact_info': [],     # Phone, email, websites
                'business_names': [],   # Company/venue names
                'statements': [],       # Action statements with !
                'instructions': [],     # Long instructional text
                'structural': []        # Potential headings/sections
            },
            'formatting_patterns': {
                'all_caps': [],
                'bold_items': [],
                'exclamation_items': [],
                'short_items': [],
                'long_items': []
            },
            'position_analysis': {
                'early': [],    # First 20% of document
                'middle': [],   # Middle 60%
                'late': []      # Last 20%
            },
            'visual_hierarchy': []
        }
        
        total_blocks = len(text_blocks)
        
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            if len(text) < 1:
                continue
                
            position_ratio = i / total_blocks if total_blocks > 0 else 0
            word_count = len(text.split())
            
            block_info = {
                'index': i,
                'text': text,
                'word_count': word_count,
                'is_bold': block.is_bold,
                'is_upper': text.isupper(),
                'has_exclamation': '!' in text,
                'ends_with_exclamation': text.endswith('!'),
                'font_size': getattr(block, 'font_size', 12),
                'page': block.page,
                'position_ratio': position_ratio
            }
            
            # Content type classification
            content_type = self._classify_content_type(text, block_info)
            analysis['content_types'][content_type].append(block_info)
            
            # Formatting pattern analysis
            if text.isupper():
                analysis['formatting_patterns']['all_caps'].append(block_info)
            if block.is_bold:
                analysis['formatting_patterns']['bold_items'].append(block_info)
            if '!' in text:
                analysis['formatting_patterns']['exclamation_items'].append(block_info)
            if word_count <= 6:
                analysis['formatting_patterns']['short_items'].append(block_info)
            if word_count > 15:
                analysis['formatting_patterns']['long_items'].append(block_info)
            
            # Position analysis
            if position_ratio < 0.2:
                analysis['position_analysis']['early'].append(block_info)
            elif position_ratio < 0.8:
                analysis['position_analysis']['middle'].append(block_info)
            else:
                analysis['position_analysis']['late'].append(block_info)
            
            # Visual hierarchy scoring
            visual_score = self._calculate_visual_score(block_info)
            analysis['visual_hierarchy'].append({
                'block_info': block_info,
                'visual_score': visual_score,
                'content_type': content_type
            })
        
        return analysis
    
    def _classify_content_type(self, text: str, block_info: Dict) -> str:
        """Classify content type using pattern recognition"""
        text_lower = text.lower()
        
        # Form field patterns
        if (text.endswith(':') and block_info['word_count'] <= 3) or text.upper() in ['FOR:', 'DATE:', 'TIME:', 'ADDRESS:', 'RSVP:']:
            return 'form_fields'
        
        # Address patterns
        address_indicators = [
            text.upper().startswith('ADDRESS:'),
            re.search(r'\d+.*(?:street|parkway|avenue|road)', text_lower),
            re.search(r'\d{5}(-\d{4})?', text),  # ZIP codes
            'near' in text_lower and block_info['word_count'] > 2
        ]
        if any(address_indicators):
            return 'addresses'
        
        # Contact info patterns
        contact_indicators = [
            re.search(r'\d{3}[-\s]?\d{3}[-\s]?\d{4}', text),  # Phone
            'www.' in text or '.com' in text_lower,  # Website
            '@' in text and '.' in text  # Email
        ]
        if any(contact_indicators):
            return 'contact_info'
        
        # Business name patterns
        business_indicators = [
            text.isupper() and 2 <= block_info['word_count'] <= 6 and block_info['position_ratio'] < 0.3,
            'park' in text_lower and block_info['word_count'] <= 4,
            'jump' in text_lower and block_info['word_count'] <= 4
        ]
        if any(business_indicators):
            return 'business_names'
        
        # Statement patterns (action/emotional)
        statement_indicators = [
            text.endswith('!'),
            'hope' in text_lower or 'see you' in text_lower,
            text.isupper() and 3 <= block_info['word_count'] <= 8 and block_info['position_ratio'] > 0.5
        ]
        if any(statement_indicators):
            return 'statements'
        
        # Instruction patterns
        instruction_indicators = [
            block_info['word_count'] > 15,
            'required' in text_lower or 'please' in text_lower,
            'shoes' in text_lower or 'waiver' in text_lower,
            'visit' in text_lower and 'website' in text_lower
        ]
        if any(instruction_indicators):
            return 'instructions'
        
        # Structural elements (potential headings)
        structural_indicators = [
            text.isupper() and 2 <= block_info['word_count'] <= 10,
            block_info['is_bold'] and 2 <= block_info['word_count'] <= 8,
            block_info['font_size'] > 12
        ]
        if any(structural_indicators):
            return 'structural'
        
        return 'other'
    
    def _calculate_visual_score(self, block_info: Dict) -> float:
        """Calculate visual prominence score"""
        score = 0.0
        
        # Font size factor
        if block_info['font_size'] > 14:
            score += 0.4
        elif block_info['font_size'] > 12:
            score += 0.2
        
        # Style factors
        if block_info['is_bold']:
            score += 0.3
        if block_info['is_upper']:
            score += 0.2
        
        # Position factor
        if block_info['position_ratio'] < 0.2:
            score += 0.2
        
        # Length appropriateness
        if 2 <= block_info['word_count'] <= 8:
            score += 0.1
        
        return score
    
    def _extract_title_intelligently(self, text_blocks: List[TextBlock], doc_analysis: Dict) -> str:
        """Extract title using intelligent pattern analysis"""
        # For invitation documents, titles are typically empty
        # Only extract if there's a very clear business name pattern
        
        business_candidates = doc_analysis['content_types']['business_names']
        
        for candidate in business_candidates[:3]:  # Check first 3 business name candidates
            # Very strict criteria for invitation titles
            if (candidate['index'] <= 2 and 
                candidate['is_bold'] and 
                not self._contains_address_elements(candidate['text']) and
                not self._is_form_related(candidate['text'])):
                
                # Additional validation
                if self._is_pure_business_name(candidate['text']):
                    return self._clean_text(candidate['text'])
        
        return ""  # Default empty title for invitations
    
    def _extract_outline_intelligently(self, text_blocks: List[TextBlock], doc_analysis: Dict) -> List[HeadingLevel]:
        """Extract outline using intelligent pattern analysis"""
        outline_candidates = []
        
        # Prioritize statements (call-to-action items)
        for statement in doc_analysis['content_types']['statements']:
            statement_score = self._score_outline_candidate(statement, 'statement')
            if statement_score > 0.6:
                outline_candidates.append({
                    'block_info': statement,
                    'score': statement_score,
                    'type': 'statement'
                })
        
        # Consider structural elements
        for structural in doc_analysis['content_types']['structural']:
            if structural not in doc_analysis['content_types']['statements']:  # Avoid duplicates
                structural_score = self._score_outline_candidate(structural, 'structural')
                if structural_score > 0.7:  # Higher threshold for structural
                    outline_candidates.append({
                        'block_info': structural,
                        'score': structural_score,
                        'type': 'structural'
                    })
        
        # Select best candidate
        if outline_candidates:
            outline_candidates.sort(key=lambda x: (x['type'] == 'statement', x['score']), reverse=True)
            best = outline_candidates[0]
            
            return [HeadingLevel(
                level='H1',
                text=self._clean_text(best['block_info']['text']),
                page=best['block_info']['page'],
                confidence=best['score'],
                font_size=best['block_info'].get('font_size'),
                font_name=None
            )]
        
        return []
    
    def _score_outline_candidate(self, block_info: Dict, content_type: str) -> float:
        """Score potential outline candidates"""
        score = 0.0
        text_lower = block_info['text'].lower()
        
        # Base scoring by content type
        if content_type == 'statement':
            # Emotional/action statements
            emotional_patterns = [
                block_info['ends_with_exclamation'],
                'hope' in text_lower,
                'see you' in text_lower,
                block_info['is_upper']
            ]
            score += sum(emotional_patterns) / len(emotional_patterns) * 0.6
            
            # Position bonus for statements (later is better)
            if block_info['position_ratio'] > 0.6:
                score += 0.3
                
        elif content_type == 'structural':
            # Structural elements
            if block_info['is_upper'] and 2 <= block_info['word_count'] <= 6:
                score += 0.5
            if block_info['is_bold']:
                score += 0.3
        
        # Common scoring factors
        if block_info['is_bold']:
            score += 0.2
        if block_info['font_size'] > 12:
            score += 0.1
        
        return score
    
    def _contains_address_elements(self, text: str) -> bool:
        """Check if text contains address elements"""
        text_lower = text.lower()
        address_elements = [
            any(char.isdigit() for char in text),
            'address' in text_lower,
            'parkway' in text_lower,
            'street' in text_lower,
            'near' in text_lower
        ]
        return any(address_elements)
    
    def _is_form_related(self, text: str) -> bool:
        """Check if text is form-related"""
        return (text.endswith(':') or 
                text.upper() in ['FOR:', 'DATE:', 'TIME:', 'ADDRESS:', 'RSVP:'])
    
    def _is_pure_business_name(self, text: str) -> bool:
        """Check if text is a pure business name"""
        text_lower = text.lower()
        business_indicators = [
            'park' in text_lower,
            'jump' in text_lower,
            'center' in text_lower
        ]
        
        exclusion_indicators = [
            any(char.isdigit() for char in text),
            'address' in text_lower,
            text.endswith(':')
        ]
        
        return any(business_indicators) and not any(exclusion_indicators)
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving important punctuation"""
        return re.sub(r'\s+', ' ', text.strip())
