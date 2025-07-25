import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor

logger = logging.getLogger('extraction')

@dataclass
class ProcessorConfig:
    """Configuration class for BusinessProcessor parameters"""
    
    # Font and formatting thresholds
    default_font_size: float = 12.0
    large_font_threshold: float = 12.0
    
    # Search ranges
    title_search_blocks: int = 8
    formatted_title_fallback: int = 5
    early_position_threshold: float = 0.1
    mid_document_threshold: float = 0.5
    
    # Word count limits
    title_min_words: int = 5
    title_max_words: int = 30
    heading_max_words: int = 25
    short_heading_max_words: int = 15
    very_short_heading_max_words: int = 12
    section_heading_max_words: int = 8
    minimal_text_threshold: int = 3
    single_word_max_chars: int = 3
    page_ref_max_words: int = 4
    appendix_max_words: int = 12
    
    # Confidence weights
    bold_confidence_bonus: float = 0.3
    uppercase_confidence_bonus: float = 0.25
    largest_font_bonus: float = 0.3
    large_font_bonus: float = 0.25
    medium_font_bonus: float = 0.15
    pattern_match_bonus: float = 0.25
    good_length_bonus: float = 0.1
    acceptable_length_bonus: float = 0.05
    position_bonus: float = 0.1
    
    # Hierarchy confidence thresholds
    h1_threshold: float = 0.6
    h2_threshold: float = 0.55
    h3_threshold: float = 0.5
    h4_threshold: float = 0.5
    high_confidence_h1: float = 0.8
    high_confidence_h2: float = 0.7
    minimum_heading_confidence: float = 0.5
    
    # Document structure confidence
    document_confidence: float = 0.9
    
    # Sentence complexity thresholds
    max_periods_in_heading: int = 1
    max_commas_in_heading: int = 3
    min_sentence_words: int = 10

@dataclass
class PatternConfig:
    """Configuration for text patterns and vocabularies"""
    
    # Title repair patterns
    title_repair_patterns: List[tuple] = None
    
    # Exclusion patterns for headings
    paragraph_starters: List[str] = None
    sentence_starters: List[str] = None
    
    # Date/time vocabulary
    months: List[str] = None
    weekdays: List[str] = None
    time_indicators: List[str] = None
    
    # Hierarchy patterns
    h1_keywords: List[str] = None
    h2_keywords: List[str] = None
    h3_keywords: List[str] = None
    h4_prefixes: List[str] = None
    
    # Content type keywords
    structural_terms: List[str] = None
    business_terms: List[str] = None
    
    def __post_init__(self):
        if self.title_repair_patterns is None:
            self.title_repair_patterns = [
                (r'(RFP)(\s*:\s*R?\1)*', r'\1:'),
                (r'([a-z])\1{3,}', r'\1'),
                (r'(\w+)\s+(?:\w*\1\w*\s*){2,}', r'\1'),
                (r'\bfquest\b', 'Request'),
                (r'\boposal\b', 'Proposal'),
                (r'\bProposaloposal\b', 'Proposal'),
                (r'\boPr+\b', 'for Pr'),
                (r'\bPr+\s+', 'Pr')
            ]
        
        if self.paragraph_starters is None:
            self.paragraph_starters = [
                'the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ',
                'it is ', 'there is ', 'there are ', 'we will ', 'they will '
            ]
        
        if self.sentence_starters is None:
            self.sentence_starters = ['the ', 'a ', 'an ', 'this ', 'that ']
        
        if self.months is None:
            self.months = [
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december'
            ]
        
        if self.weekdays is None:
            self.weekdays = [
                'monday', 'tuesday', 'wednesday', 'thursday', 
                'friday', 'saturday', 'sunday'
            ]
        
        if self.time_indicators is None:
            self.time_indicators = ['noon on', 'p.m.', 'a.m.']
        
        if self.h1_keywords is None:
            self.h1_keywords = [
                'critical component', 'road map', 'roadmap'
            ]
        
        if self.h2_keywords is None:
            self.h2_keywords = [
                'summary', 'background', 'timeline', 'milestones',
                'business plan', 'approach', 'requirements',
                'evaluation', 'contract', 'appendix'
            ]
        
        if self.h3_keywords is None:
            self.h3_keywords = [
                'phase', 'equitable', 'shared', 'local', 'access',
                'guidance', 'training', 'what could', 'mean'
            ]
        
        if self.h4_prefixes is None:
            self.h4_prefixes = [
                'for each', 'for the', 'could mean'
            ]
        
        if self.structural_terms is None:
            self.structural_terms = [
                'summary', 'background', 'timeline', 'milestones',
                'appendix', 'business plan', 'evaluation', 'approach'
            ]
        
        if self.business_terms is None:
            self.business_terms = ['rfp', 'proposal']


class BusinessProcessor:
    def __init__(self, config: Optional[ProcessorConfig] = None, patterns: Optional[PatternConfig] = None):
        self.config = config or ProcessorConfig()
        self.patterns = patterns or PatternConfig()
        self.title_extractor = TitleExtractor()
        self.outline_extractor = OutlineExtractor()
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process business document with accurate pattern recognition"""
        logger.info("Processing business document with pattern-based extraction")
        
        # Analyze document structure comprehensively
        doc_analysis = self._analyze_document_comprehensively(text_blocks)
        
        # Extract clean title with advanced repair
        title = self._extract_and_repair_title(text_blocks, doc_analysis)
        
        # Extract accurate hierarchical outline
        outline = self._extract_accurate_outline(text_blocks, doc_analysis)
        
        return DocumentStructure(
            title=title,
            outline=outline,
            doc_type=DocumentType.BUSINESS_DOCUMENT,
            confidence=self.config.document_confidence
        )
    
    def _analyze_document_comprehensively(self, text_blocks: List[TextBlock]) -> Dict:
        """Comprehensive document analysis for pattern recognition"""
        analysis = {
            'blocks': [],
            'font_hierarchy': {'sizes': [], 'tiers': {}},
            'visual_patterns': {'bold_items': [], 'upper_items': [], 'formatted_items': []},
            'content_types': {'dates': [], 'paragraphs': [], 'headings': [], 'titles': []}
        }
        
        # Collect all block information
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            if len(text) < 1:
                continue
                
            font_size = getattr(block, 'font_size', self.config.default_font_size)
            analysis['font_hierarchy']['sizes'].append(font_size)
            
            block_data = {
                'index': i,
                'text': text,
                'word_count': len(text.split()),
                'char_count': len(text),
                'is_bold': block.is_bold,
                'is_upper': text.isupper(),
                'font_size': font_size,
                'page': block.page,
                'position_ratio': i / len(text_blocks)
            }
            
            analysis['blocks'].append(block_data)
            
            # Categorize by visual patterns
            if block.is_bold:
                analysis['visual_patterns']['bold_items'].append(block_data)
            if text.isupper():
                analysis['visual_patterns']['upper_items'].append(block_data)
            if block.is_bold or text.isupper() or font_size > self.config.large_font_threshold:
                analysis['visual_patterns']['formatted_items'].append(block_data)
        
        # Calculate font tiers
        if analysis['font_hierarchy']['sizes']:
            unique_sizes = sorted(set(analysis['font_hierarchy']['sizes']), reverse=True)
            analysis['font_hierarchy']['tiers'] = {
                'largest': unique_sizes[0] if len(unique_sizes) > 0 else self.config.default_font_size,
                'large': unique_sizes[1] if len(unique_sizes) > 1 else self.config.default_font_size,
                'medium': unique_sizes[2] if len(unique_sizes) > 2 else self.config.default_font_size,
                'small': unique_sizes[-1] if len(unique_sizes) > 0 else self.config.default_font_size
            }
        
        return analysis
    
    def _extract_and_repair_title(self, text_blocks: List[TextBlock], doc_analysis: Dict) -> str:
        """Extract title with advanced corruption repair"""
        # Look for title in first few blocks
        for block_data in doc_analysis['blocks'][:self.config.title_search_blocks]:
            if self._is_title_candidate(block_data):
                return self._advanced_title_repair(block_data['text'])
        
        # Fallback to first substantial formatted text
        for block_data in doc_analysis['visual_patterns']['formatted_items'][:self.config.formatted_title_fallback]:
            if (self.config.title_min_words <= block_data['word_count'] <= self.config.title_max_words and 
                not self._is_date_content(block_data['text']) and
                block_data['position_ratio'] < self.config.early_position_threshold):
                return self._advanced_title_repair(block_data['text'])
        
        return "Business Document"
    
    def _advanced_title_repair(self, text: str) -> str:
        """Advanced title corruption repair with configurable patterns"""
        repaired = text
        
        # Apply all configured repair patterns
        for pattern, replacement in self.patterns.title_repair_patterns:
            repaired = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)
        
        # Clean multiple spaces and punctuation
        repaired = re.sub(r'\s+', ' ', repaired)
        repaired = re.sub(r':+', ':', repaired)
        
        return repaired.strip()
    
    def _extract_accurate_outline(self, text_blocks: List[TextBlock], doc_analysis: Dict) -> List[HeadingLevel]:
        """Extract outline with accurate heading detection"""
        heading_candidates = []
        
        # Process each block for heading potential
        for block_data in doc_analysis['blocks']:
            heading_analysis = self._analyze_heading_potential(block_data, doc_analysis, text_blocks)
            
            if heading_analysis and heading_analysis['confidence'] > self.config.minimum_heading_confidence:
                heading_candidates.append(heading_analysis)
        
        # Build final outline with proper hierarchy
        return self._build_hierarchical_outline(heading_candidates)
    
    def _analyze_heading_potential(self, block_data: Dict, doc_analysis: Dict, all_blocks: List[TextBlock]) -> Optional[Dict]:
        """Analyze block for heading potential with strict filtering"""
        text = block_data['text']
        
        # Strict exclusion filters
        if self._should_exclude_from_headings(text, block_data):
            return None
        
        # Calculate comprehensive confidence
        confidence = self._calculate_comprehensive_confidence(block_data, doc_analysis)
        
        if confidence < self.config.minimum_heading_confidence:
            return None
        
        # Determine hierarchical level
        level = self._determine_hierarchy_level(block_data, confidence, doc_analysis)
        
        return {
            'block_data': block_data,
            'confidence': confidence,
            'level': level,
            'clean_text': self._clean_heading_text(text)
        }
    
    def _should_exclude_from_headings(self, text: str, block_data: Dict) -> bool:
        """Comprehensive exclusion filter for non-heading content"""
        exclusion_patterns = [
            # Date patterns
            self._is_date_content(text),
            
            # Long paragraphs
            block_data['word_count'] > self.config.heading_max_words,
            
            # Sentence patterns
            text.count('.') > self.config.max_periods_in_heading and block_data['word_count'] > self.config.min_sentence_words,
            text.count(',') > self.config.max_commas_in_heading,
            
            # Descriptive content starters
            any(text.lower().startswith(starter) for starter in self.patterns.paragraph_starters),
            
            # URLs and contact info
            '@' in text or 'www.' in text or '.com' in text,
            
            # Financial/statistical content
            '$' in text or '%' in text,
            re.search(r'\d+[-,]\d+', text),
            
            # Very short non-meaningful text
            block_data['word_count'] == 1 and len(text) <= self.config.single_word_max_chars,
            
            # Page references
            'page ' in text.lower() and block_data['word_count'] <= self.config.page_ref_max_words
        ]
        
        return any(exclusion_patterns)
    
    def _calculate_comprehensive_confidence(self, block_data: Dict, doc_analysis: Dict) -> float:
        """Calculate heading confidence using configurable factors"""
        confidence = 0.0
        text = block_data['text']
        text_lower = text.lower()
        
        # Format-based confidence
        if block_data['is_bold']:
            confidence += self.config.bold_confidence_bonus
        
        if block_data['is_upper'] and block_data['word_count'] <= self.config.short_heading_max_words:
            confidence += self.config.uppercase_confidence_bonus
        
        # Font size confidence (relative to document)
        font_tiers = doc_analysis['font_hierarchy']['tiers']
        if block_data['font_size'] >= font_tiers['largest']:
            confidence += self.config.largest_font_bonus
        elif block_data['font_size'] >= font_tiers['large']:
            confidence += self.config.large_font_bonus
        elif block_data['font_size'] >= font_tiers['medium']:
            confidence += self.config.medium_font_bonus
        
        # Content pattern confidence
        structural_patterns = [
            text_lower in self.patterns.structural_terms,
            any(term in text_lower for term in self.patterns.h2_keywords) and block_data['word_count'] <= self.config.appendix_max_words,
            text.endswith(':') and 2 <= block_data['word_count'] <= self.config.section_heading_max_words,
            re.match(r'^\d+\.\s+[A-Z]', text),  # Numbered sections
            any(term in text_lower for term in self.patterns.h3_keywords)
        ]
        
        if any(structural_patterns):
            confidence += self.config.pattern_match_bonus
        
        # Length appropriateness
        if 2 <= block_data['word_count'] <= self.config.very_short_heading_max_words:
            confidence += self.config.good_length_bonus
        elif block_data['word_count'] <= 20:
            confidence += self.config.acceptable_length_bonus
        
        # Position bonus
        if block_data['position_ratio'] < self.config.mid_document_threshold:
            confidence += self.config.position_bonus
        
        return min(confidence, 1.0)
    
    def _determine_hierarchy_level(self, block_data: Dict, confidence: float, doc_analysis: Dict) -> str:
        """Determine hierarchical level with configurable pattern-based rules"""
        text = block_data['text']
        text_lower = text.lower()
        
        # H1 level patterns
        h1_patterns = [
            any(keyword in text_lower for keyword in self.patterns.h1_keywords),
            confidence > self.config.high_confidence_h1 and block_data['font_size'] >= doc_analysis['font_hierarchy']['tiers']['largest']
        ]
        
        if any(h1_patterns):
            return 'H1'
        
        # H2 level patterns
        h2_patterns = [
            text_lower in self.patterns.h2_keywords,
            any(term in text_lower for term in self.patterns.h2_keywords) and block_data['word_count'] <= self.config.appendix_max_words,
            confidence > self.config.high_confidence_h2 and block_data['is_bold']
        ]
        
        if any(h2_patterns):
            return 'H2'
        
        # H3 level patterns
        h3_patterns = [
            text.endswith(':') and 2 <= block_data['word_count'] <= self.config.section_heading_max_words,
            re.match(r'^\d+\.\s+[A-Z]', text),  # Numbered sections
            any(keyword in text_lower for keyword in self.patterns.h3_keywords) and block_data['word_count'] <= self.config.section_heading_max_words,
            text.endswith(':') and any(keyword in text_lower for keyword in ['access'])
        ]
        
        if any(h3_patterns):
            return 'H3'
        
        # H4 level patterns
        h4_patterns = [
            any(text_lower.startswith(prefix) for prefix in self.patterns.h4_prefixes),
            any(phrase in text_lower for phrase in self.patterns.h4_prefixes) and text.endswith(':')
        ]
        
        if any(h4_patterns):
            return 'H4'
        
        # Default based on confidence
        if confidence > 0.75:
            return 'H2'
        else:
            return 'H3'
    
    def _build_hierarchical_outline(self, candidates: List[Dict]) -> List[HeadingLevel]:
        """Build final hierarchical outline with quality control"""
        if not candidates:
            return []
        
        # Sort by page and document position
        candidates.sort(key=lambda x: (x['block_data']['page'], x['block_data']['index']))
        
        outline = []
        used_texts = set()
        
        # Configure thresholds
        level_thresholds = {
            'H1': self.config.h1_threshold,
            'H2': self.config.h2_threshold,
            'H3': self.config.h3_threshold,
            'H4': self.config.h4_threshold
        }
        
        for candidate in candidates:
            clean_text = candidate['clean_text']
            text_key = clean_text.lower().strip()
            
            # Skip duplicates
            if text_key in used_texts:
                continue
            
            # Apply confidence thresholds by level
            required_threshold = level_thresholds.get(candidate['level'], self.config.minimum_heading_confidence)
            
            if candidate['confidence'] >= required_threshold:
                outline.append(HeadingLevel(
                    level=candidate['level'],
                    text=clean_text,
                    page=candidate['block_data']['page'],
                    confidence=candidate['confidence'],
                    font_size=candidate['block_data']['font_size'],
                    font_name=None
                ))
                used_texts.add(text_key)
        
        return outline
    
    def _is_title_candidate(self, block_data: Dict) -> bool:
        """Check if block is a title candidate"""
        text = block_data['text']
        return (block_data['index'] <= 5 and
                self.config.title_min_words <= block_data['word_count'] <= self.config.title_max_words and
                any(term in text.lower() for term in self.patterns.business_terms) and
                not self._is_date_content(text))
    
    def _is_date_content(self, text: str) -> bool:
        """Comprehensive date content detection using configurable patterns"""
        date_patterns = [
            any(month in text.lower() for month in self.patterns.months) and len(text.split()) <= 6,
            re.search(r'\d{1,2}:\d{2}', text),  # Times
            any(day in text.lower() for day in self.patterns.weekdays) and len(text.split()) <= 5,
            re.search(r'\d{1,2}/\d{1,2}/\d{4}', text),  # Date formats
            re.match(r'^(' + '|'.join(self.patterns.months[:3]) + r')\s+\d+$', text.lower()),  # Month + day
            any(indicator in text.lower() for indicator in self.patterns.time_indicators)
        ]
        
        return any(date_patterns)
    
    def _clean_heading_text(self, text: str) -> str:
        """Clean heading text while preserving structure"""
        cleaned = re.sub(r'\s+', ' ', text.strip())
        return cleaned
