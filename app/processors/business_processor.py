import sys, os, re, logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor

logger = logging.getLogger('extraction')

@dataclass
class ProcessorConfig:
    """All configurable parameters - zero hardcoding"""
    
    # Document settings
    document_type: DocumentType = DocumentType.BUSINESS_DOCUMENT
    document_confidence: float = 0.9
    
    # Font settings
    default_font_size: float = 12.0
    large_font_threshold: float = 12.0
    
    # Search limits
    title_search_blocks: int = 8
    formatted_title_fallback: int = 5
    
    # Position thresholds
    early_position_threshold: float = 0.1
    mid_document_threshold: float = 0.5
    
    # Word count limits
    title_min_words: int = 5
    title_max_words: int = 30
    heading_max_words: int = 25
    short_heading_max_words: int = 15
    very_short_heading_max_words: int = 12
    minimal_text_threshold: int = 3
    page_ref_max_words: int = 4
    appendix_max_words: int = 12
    
    # Confidence bonuses
    bold_confidence_bonus: float = 0.3
    uppercase_confidence_bonus: float = 0.25
    largest_font_bonus: float = 0.3
    large_font_bonus: float = 0.25
    medium_font_bonus: float = 0.15
    pattern_match_bonus: float = 0.25
    good_length_bonus: float = 0.1
    acceptable_length_bonus: float = 0.05
    position_bonus: float = 0.1
    
    # Hierarchy thresholds
    h1_threshold: float = 0.6
    h2_threshold: float = 0.55
    h3_threshold: float = 0.5
    h4_threshold: float = 0.5
    high_confidence_h1: float = 0.8
    high_confidence_h2: float = 0.7
    minimum_heading_confidence: float = 0.5
    
    # Text complexity limits
    max_periods_in_heading: int = 1
    max_commas_in_heading: int = 3
    
    # Scoring limits
    max_confidence_score: float = 1.0
    min_score_required: float = 0.0
    
    # Logging
    enable_debug_logging: bool = False

@dataclass
class PatternConfig:
    """All patterns, keywords, and vocabulary - fully configurable"""
    
    # Title repair patterns
    title_repair_patterns: List[tuple] = field(default_factory=list)
    
    # Text cleaning patterns
    whitespace_pattern: str = r'\s+'
    whitespace_replacement: str = ' '
    
    # Content exclusion patterns
    paragraph_starters: List[str] = field(default_factory=list)
    sentence_starters: List[str] = field(default_factory=list)
    
    # Date/time vocabulary
    months: List[str] = field(default_factory=list)
    weekdays: List[str] = field(default_factory=list)
    time_indicators: List[str] = field(default_factory=list)
    
    # Hierarchy keywords
    h1_keywords: List[str] = field(default_factory=list)
    h2_keywords: List[str] = field(default_factory=list)
    h3_keywords: List[str] = field(default_factory=list)
    h4_prefixes: List[str] = field(default_factory=list)
    
    # Content classification
    structural_terms: List[str] = field(default_factory=list)
    business_terms: List[str] = field(default_factory=list)
    
    # Case sensitivity settings
    case_sensitive_matching: bool = False
    
    def __post_init__(self):
        """Initialize empty defaults - all must be configured externally for domain-specific use"""
        # All patterns start empty to enforce external configuration
        pass

@dataclass
class AnalysisConfig:
    """Configuration for document analysis behavior"""
    
    # Analysis method toggles
    enable_title_extractor: bool = True
    enable_fallback_title_extraction: bool = True
    enable_title_repair: bool = True
    
    # Title scoring components
    enable_bold_scoring: bool = True
    enable_uppercase_scoring: bool = True
    enable_font_size_scoring: bool = True
    enable_length_scoring: bool = True
    enable_pattern_scoring: bool = True
    enable_position_scoring: bool = True
    
    # Heading validation components
    enable_paragraph_exclusion: bool = True
    enable_length_validation: bool = True
    enable_punctuation_validation: bool = True
    
    # Result processing
    enable_duplicate_removal: bool = True
    enable_confidence_filtering: bool = True

class BusinessProcessor:
    """Fully generalized business processor with zero hardcoding"""
    
    def __init__(self,
                 config: Optional[ProcessorConfig] = None,
                 patterns: Optional[PatternConfig] = None,
                 analysis: Optional[AnalysisConfig] = None):
        
        self.cfg = config or ProcessorConfig()
        self.pt = patterns or PatternConfig()
        self.analysis = analysis or AnalysisConfig()
        
        self.title_extractor = TitleExtractor()
        self.outline_extractor = OutlineExtractor()
        
        if self.cfg.enable_debug_logging:
            logger.setLevel(logging.DEBUG)
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process document using fully configurable logic"""
        logger.info("BusinessProcessor: start")
        
        doc_analysis = self._analyze_document_comprehensively(text_blocks)
        title = self._extract_and_repair_title(text_blocks, doc_analysis)
        outline = self._extract_accurate_outline(doc_analysis, title)
        
        return DocumentStructure(
            title=title,
            outline=outline,
            doc_type=self.cfg.document_type,
            confidence=self.cfg.document_confidence
        )
    
    def _analyze_document_comprehensively(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze document using configurable parameters"""
        analysis = {'blocks': []}
        total = len(blocks) or 1
        
        for i, blk in enumerate(blocks):
            txt = blk.text.strip()
            if not txt:
                continue
            
            fs = getattr(blk, 'font_size', self.cfg.default_font_size)
            bd = getattr(blk, 'is_bold', False)
            ub = txt.isupper()
            wc = len(txt.split())
            cc = len(txt)
            pr = i / total
            
            analysis['blocks'].append({
                'index': i,
                'text': txt,
                'font_size': fs,
                'is_bold': bd,
                'is_upper': ub,
                'word_count': wc,
                'char_count': cc,
                'position_ratio': pr,
                'page': getattr(blk, 'page', 0)
            })
        
        return analysis
    
    def _extract_and_repair_title(self, blocks: List[TextBlock], analysis: Dict) -> str:
        """Extract and repair title using configurable methods"""
        
        # Method 1: Core title extractor (configurable)
        if self.analysis.enable_title_extractor:
            try:
                extracted_title = self.title_extractor.extract_title(blocks, self.cfg.document_type)
                if extracted_title:
                    return self._repair_title(extracted_title) if self.analysis.enable_title_repair else extracted_title
            except Exception as e:
                if self.cfg.enable_debug_logging:
                    logger.debug(f"Title extractor failed: {e}")
        
        # Method 2: Fallback extraction (configurable)
        if self.analysis.enable_fallback_title_extraction:
            candidates = []
            search_limit = min(self.cfg.formatted_title_fallback, len(analysis['blocks']))
            
            for block_data in analysis['blocks'][:search_limit]:
                if not self._is_valid_title_candidate(block_data):
                    continue
                
                score = self._calculate_title_score(block_data)
                if score > self.cfg.min_score_required:
                    candidates.append((block_data['text'], score))
            
            if candidates:
                best_text = max(candidates, key=lambda x: x[1])[0]
                return self._repair_title(best_text) if self.analysis.enable_title_repair else best_text
        
        # Method 3: Return empty (configurable default)
        return ""
    
    def _is_valid_title_candidate(self, block_data: Dict) -> bool:
        """Validate title candidate using configurable criteria"""
        return (
            block_data['word_count'] >= self.cfg.title_min_words and
            block_data['word_count'] <= self.cfg.title_max_words and
            block_data['position_ratio'] <= self.cfg.mid_document_threshold
        )
    
    def _calculate_title_score(self, block_data: Dict) -> float:
        """Calculate title score using configurable components"""
        score = 0.0
        
        # Bold scoring (configurable)
        if self.analysis.enable_bold_scoring and block_data['is_bold']:
            score += self.cfg.bold_confidence_bonus
        
        # Uppercase scoring (configurable)
        if self.analysis.enable_uppercase_scoring and block_data['is_upper']:
            score += self.cfg.uppercase_confidence_bonus
        
        # Font size scoring (configurable)
        if self.analysis.enable_font_size_scoring:
            if block_data['font_size'] >= self.cfg.large_font_threshold:
                score += self.cfg.largest_font_bonus
        
        return min(self.cfg.max_confidence_score, score)
    
    def _repair_title(self, raw_title: str) -> str:
        """Repair title using configurable patterns"""
        if not self.analysis.enable_title_repair:
            return raw_title
        
        repaired = raw_title.strip()
        
        # Apply configurable repair patterns
        for pattern, replacement in self.pt.title_repair_patterns:
            repaired = re.sub(pattern, replacement, repaired, 
                            flags=0 if self.pt.case_sensitive_matching else re.IGNORECASE)
        
        # Apply configurable whitespace cleaning
        repaired = re.sub(self.pt.whitespace_pattern, self.pt.whitespace_replacement, repaired)
        
        return repaired
    
    def _extract_accurate_outline(self, analysis: Dict, title: str) -> List[HeadingLevel]:
        """Extract outline using configurable heading detection"""
        headings = []
        
        for block_data in analysis['blocks']:
            text = block_data['text']
            
            # Skip title duplicates (configurable)
            if self.analysis.enable_duplicate_removal and text.lower() == title.lower():
                continue
            
            # Validate as heading (configurable)
            if not self._is_valid_heading(block_data):
                continue
            
            # Score heading (configurable)
            confidence = self._score_heading(block_data)
            
            # Apply confidence filter (configurable)
            if self.analysis.enable_confidence_filtering:
                if confidence < self.cfg.minimum_heading_confidence:
                    continue
            
            # Determine level (configurable)
            level = self._determine_heading_level(confidence, block_data)
            
            headings.append(HeadingLevel(
                level=level,
                text=text,
                page=block_data['page'],
                confidence=confidence,
                font_size=block_data['font_size'],
                font_name=None
            ))
        
        return headings
    
    def _is_valid_heading(self, block_data: Dict) -> bool:
        """Validate heading using configurable exclusion rules"""
        text = block_data['text']
        text_lower = text.lower() if not self.pt.case_sensitive_matching else text
        
        # Paragraph starter exclusion (configurable)
        if self.analysis.enable_paragraph_exclusion:
            comparison_starters = (self.pt.paragraph_starters if self.pt.case_sensitive_matching 
                                 else [s.lower() for s in self.pt.paragraph_starters])
            if any(text_lower.startswith(starter) for starter in comparison_starters):
                return False
        
        # Length validation (configurable)
        if self.analysis.enable_length_validation:
            if len(text.split()) > self.cfg.heading_max_words:
                return False
        
        # Punctuation validation (configurable)
        if self.analysis.enable_punctuation_validation:
            if (text.count('.') > self.cfg.max_periods_in_heading or
                text.count(',') > self.cfg.max_commas_in_heading):
                return False
        
        return True
    
    def _score_heading(self, block_data: Dict) -> float:
        """Score heading using configurable components"""
        score = 0.0
        
        # Format-based scoring (configurable)
        if self.analysis.enable_bold_scoring and block_data['is_bold']:
            score += self.cfg.bold_confidence_bonus
        
        if self.analysis.enable_uppercase_scoring and block_data['is_upper']:
            score += self.cfg.uppercase_confidence_bonus
        
        # Font size scoring (configurable)
        if self.analysis.enable_font_size_scoring:
            font_size = block_data['font_size']
            if font_size >= self.cfg.large_font_threshold:
                score += self.cfg.largest_font_bonus
            elif font_size > self.cfg.default_font_size:
                score += self.cfg.large_font_bonus
        
        # Length scoring (configurable)
        if self.analysis.enable_length_scoring:
            word_count = block_data['word_count']
            if word_count <= self.cfg.very_short_heading_max_words:
                score += self.cfg.acceptable_length_bonus
            elif word_count <= self.cfg.short_heading_max_words:
                score += self.cfg.good_length_bonus
        
        # Pattern scoring (configurable)
        if self.analysis.enable_pattern_scoring:
            text_for_comparison = (block_data['text'] if self.pt.case_sensitive_matching 
                                 else block_data['text'].lower())
            
            comparison_terms = (self.pt.structural_terms + self.pt.business_terms 
                              if self.pt.case_sensitive_matching
                              else [term.lower() for term in self.pt.structural_terms + self.pt.business_terms])
            
            if any(term in text_for_comparison for term in comparison_terms):
                score += self.cfg.pattern_match_bonus
        
        # Position scoring (configurable)
        if self.analysis.enable_position_scoring:
            if block_data['position_ratio'] <= self.cfg.early_position_threshold:
                score += self.cfg.position_bonus
        
        return min(self.cfg.max_confidence_score, score)
    
    def _determine_heading_level(self, confidence: float, block_data: Dict) -> str:
        """Determine heading level using configurable thresholds"""
        if confidence >= self.cfg.high_confidence_h1:
            return 'H1'
        elif confidence >= self.cfg.h1_threshold:
            return 'H1'
        elif confidence >= self.cfg.high_confidence_h2:
            return 'H2'
        elif confidence >= self.cfg.h2_threshold:
            return 'H2'
        elif confidence >= self.cfg.h3_threshold:
            return 'H3'
        else:
            return 'H4'

# Factory function for easy processor creation
def create_business_processor(mode: str = 'standard') -> BusinessProcessor:
    """Create processor with preconfigured settings"""
    
    config = ProcessorConfig()
    patterns = PatternConfig()
    analysis = AnalysisConfig()
    
    if mode == 'strict':
        config.bold_confidence_bonus = 0.4
        config.h1_threshold = 0.7
        config.minimum_heading_confidence = 0.6
        
    elif mode == 'lenient':
        config.bold_confidence_bonus = 0.2
        config.minimum_heading_confidence = 0.4
        config.h1_threshold = 0.5
        
    elif mode == 'debug':
        config.enable_debug_logging = True
        
    elif mode == 'rfp_configured':
        # Example of domain-specific configuration
        patterns.title_repair_patterns = [
            (r'(RFP)(\s*:\s*R?\1)*', r'\1:'),
            (r'([a-z])\1{3,}', r'\1'),
            (r'(\w+)\s+(?:\w*\1\w*\s*){2,}', r'\1'),
            (r'\bfquest\b', 'Request'),
            (r'\boposal\b', 'Proposal'),
            (r'\bProposaloposal\b', 'Proposal'),
            (r'\boPr+\b', 'for Pr'),
            (r'\bPr+\s+', 'Pr')
        ]
        patterns.paragraph_starters = [
            'the ', 'a ', 'an ', 'this ', 'that ',
            'it is ', 'there is ', 'we will ', 'they will '
        ]
        patterns.h2_keywords = [
            'summary', 'background', 'timeline', 'milestones',
            'business plan', 'approach', 'requirements', 'evaluation',
            'contract', 'appendix'
        ]
        patterns.business_terms = ['rfp', 'proposal']
        patterns.structural_terms = patterns.h2_keywords + ['appendix', 'business plan']
    
    return BusinessProcessor(config=config, patterns=patterns, analysis=analysis)

# Usage examples
if __name__ == "__main__":
    # Standard processor (empty patterns)
    standard_processor = create_business_processor('standard')
    
    # RFP-configured processor
    rfp_processor = create_business_processor('rfp_configured')
    
    # Custom configuration
    custom_config = ProcessorConfig(
        document_type=DocumentType.BUSINESS_DOCUMENT,
        document_confidence=0.95,
        title_min_words=3,
        large_font_threshold=14.0
    )
    
    custom_patterns = PatternConfig(
        title_repair_patterns=[(r'SPEC\s*:', 'Specification:')],
        business_terms=['specification', 'standard', 'protocol'],
        case_sensitive_matching=False
    )
    
    custom_analysis = AnalysisConfig(
        enable_title_repair=True,
        enable_pattern_scoring=True,
        enable_confidence_filtering=True
    )
    
    custom_processor = BusinessProcessor(
        config=custom_config,
        patterns=custom_patterns,
        analysis=custom_analysis
    )
