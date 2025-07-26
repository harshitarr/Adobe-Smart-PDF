import sys, os, re, logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor
from processors.base_processor import BaseProcessor
from config.language_patterns import LANGUAGE_PROCESSING_CONFIG

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
    
    # Language-aware settings (new)
    enable_language_aware_processing: bool = True
    apply_language_font_adjustments: bool = True
    use_language_specific_patterns: bool = True

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
    
    # Language-specific patterns (new)
    language_specific_h1_keywords: Dict[str, List[str]] = field(default_factory=dict)
    language_specific_h2_keywords: Dict[str, List[str]] = field(default_factory=dict)
    language_specific_business_terms: Dict[str, List[str]] = field(default_factory=dict)
    language_specific_structural_terms: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize empty defaults - all must be configured externally for domain-specific use"""
        # All patterns start empty to enforce external configuration
        
        # Initialize language-specific pattern dictionaries if empty
        if not self.language_specific_h1_keywords:
            self.language_specific_h1_keywords = {
                'japanese': [],
                'german': [],
                'tamil': [],
                'english': []
            }
        
        if not self.language_specific_h2_keywords:
            self.language_specific_h2_keywords = {
                'japanese': [],
                'german': [],
                'tamil': [],
                'english': []
            }
        
        if not self.language_specific_business_terms:
            self.language_specific_business_terms = {
                'japanese': [],
                'german': [],
                'tamil': [],
                'english': []
            }
        
        if not self.language_specific_structural_terms:
            self.language_specific_structural_terms = {
                'japanese': [],
                'german': [],
                'tamil': [],
                'english': []
            }

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
    
    # Language-aware analysis (new)
    enable_language_aware_scoring: bool = True
    enable_language_specific_validation: bool = True
    apply_language_font_thresholds: bool = True

class BusinessProcessor(BaseProcessor):
    """Fully generalized business processor with zero hardcoding and multilingual support"""
    
    def __init__(self,
                 config: Optional[ProcessorConfig] = None,
                 patterns: Optional[PatternConfig] = None,
                 analysis: Optional[AnalysisConfig] = None):
        
        # Initialize base processor for language support
        super().__init__()
        
        self.cfg = config or ProcessorConfig()
        self.pt = patterns or PatternConfig()
        self.analysis = analysis or AnalysisConfig()
        
        self.title_extractor = TitleExtractor()
        self.outline_extractor = OutlineExtractor()
        
        # Language-aware configuration cache
        self._language_adjusted_config = {}
        self._current_language_patterns = {}
        
        if self.cfg.enable_debug_logging:
            logger.setLevel(logging.DEBUG)
        
        logger.info("BusinessProcessor initialized with multilingual support")
    
    def set_language_config(self, language: str, config: Dict[str, Any]):
        """Override from BaseProcessor - set language-specific configuration"""
        super().set_language_config(language, config)
        
        if self.cfg.enable_language_aware_processing:
            self._apply_language_adjustments(language, config)
            logger.info(f"Applied {language} configuration to BusinessProcessor")
    
    def _apply_language_adjustments(self, language: str, lang_config: Dict[str, Any]):
        """Apply language-specific adjustments to processing configuration"""
        
        # Store original config if not already cached
        if not self._language_adjusted_config:
            self._language_adjusted_config = {
                'font_weight_threshold': self.cfg.large_font_threshold,
                'spacing_multiplier': 1.0,  # Default spacing
                'title_patterns': self.pt.title_repair_patterns.copy()
            }
        
        # Apply language-specific font adjustments
        if self.cfg.apply_language_font_adjustments:
            font_threshold = lang_config.get('font_weight_threshold', self.cfg.large_font_threshold)
            spacing_mult = lang_config.get('spacing_multiplier', 1.0)
            
            # Adjust font thresholds based on language characteristics
            self.cfg.large_font_threshold = font_threshold
            
            logger.debug(f"Font threshold adjusted to {font_threshold} for {language}")
        
        # Apply language-specific patterns
        if self.cfg.use_language_specific_patterns:
            self._update_language_patterns(language, lang_config)
    
    def _update_language_patterns(self, language: str, lang_config: Dict[str, Any]):
        """Update patterns based on language-specific configuration"""
        
        # Get language-specific title patterns
        title_patterns = lang_config.get('title_patterns', [])
        if title_patterns:
            # Merge with existing patterns rather than replace
            combined_patterns = self.pt.title_repair_patterns + [
                (pattern, pattern) for pattern in title_patterns
            ]
            self.pt.title_repair_patterns = combined_patterns
        
        # Update heading indicators
        heading_indicators = lang_config.get('heading_indicators', [])
        if heading_indicators:
            # Add language-specific indicators to existing business/structural terms
            self.pt.structural_terms.extend(heading_indicators)
            self.pt.h2_keywords.extend(heading_indicators)
        
        # Cache current language patterns
        self._current_language_patterns = {
            'language': language,
            'patterns': lang_config
        }
        
        logger.debug(f"Updated patterns for {language} processing")
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process document using fully configurable logic with language awareness"""
        logger.info("BusinessProcessor: start")
        
        doc_analysis = self._analyze_document_comprehensively(text_blocks)
        title = self._extract_and_repair_title(text_blocks, doc_analysis)
        outline = self._extract_accurate_outline(doc_analysis, title)
        
        # Create document structure with language metadata
        doc_structure = DocumentStructure(
            title=title,
            outline=outline,
            doc_type=self.cfg.document_type,
            confidence=self.cfg.document_confidence
        )
        
        # Add language information to metadata if available
        if hasattr(doc_structure, 'metadata') and hasattr(self, 'language'):
            doc_structure.metadata.update({
                'detected_language': self.language,
                'language_config_applied': bool(self._current_language_patterns),
                'processing_language': self._current_language_patterns.get('language', 'english')
            })
        
        return doc_structure
    
    def _analyze_document_comprehensively(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze document using configurable parameters with language awareness"""
        analysis = {'blocks': []}
        total = len(blocks) or 1
        
        # Get language-specific font adjustments
        font_multiplier = self.language_config.get('spacing_multiplier', 1.0) if hasattr(self, 'language_config') else 1.0
        
        for i, blk in enumerate(blocks):
            txt = blk.text.strip()
            if not txt:
                continue
            
            fs = getattr(blk, 'font_size', self.cfg.default_font_size)
            
            # Apply language-specific font size adjustments
            if self.analysis.apply_language_font_thresholds and font_multiplier != 1.0:
                fs = fs * font_multiplier
            
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
                'page': getattr(blk, 'page', 0),
                'language_adjusted_font': fs != getattr(blk, 'font_size', self.cfg.default_font_size)
            })
        
        return analysis
    
    def _extract_and_repair_title(self, blocks: List[TextBlock], analysis: Dict) -> str:
        """Extract and repair title using configurable methods with language awareness"""
        
        # Method 1: Core title extractor (configurable)
        if self.analysis.enable_title_extractor:
            try:
                extracted_title = self.title_extractor.extract_title(blocks, self.cfg.document_type)
                if extracted_title:
                    repaired_title = self._repair_title(extracted_title) if self.analysis.enable_title_repair else extracted_title
                    
                    # Apply language-specific title processing if available
                    if self.cfg.enable_language_aware_processing and hasattr(self, 'language'):
                        repaired_title = self._apply_language_title_processing(repaired_title)
                    
                    return repaired_title
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
                repaired_title = self._repair_title(best_text) if self.analysis.enable_title_repair else best_text
                
                # Apply language-specific title processing if available
                if self.cfg.enable_language_aware_processing and hasattr(self, 'language'):
                    repaired_title = self._apply_language_title_processing(repaired_title)
                
                return repaired_title
        
        # Method 3: Return empty (configurable default)
        return ""
    
    def _apply_language_title_processing(self, title: str) -> str:
        """Apply language-specific title processing"""
        if not hasattr(self, 'language') or not self._current_language_patterns:
            return title
        
        processed_title = title
        language = self.language
        
        # Apply language-specific title patterns if available
        lang_patterns = self._current_language_patterns.get('patterns', {})
        title_patterns = lang_patterns.get('title_patterns', [])
        
        for pattern in title_patterns:
            if isinstance(pattern, str):
                # Simple pattern - use as regex to clean common issues
                processed_title = re.sub(pattern, '', processed_title, flags=re.IGNORECASE)
        
        # Language-specific title formatting
        if language == 'german':
            # German titles often have compound words - preserve capitalization
            processed_title = processed_title.strip()
        elif language == 'japanese':
            # Japanese titles may have mixed scripts - preserve all characters
            processed_title = processed_title.strip()
        elif language == 'tamil':
            # Tamil titles - preserve script integrity
            processed_title = processed_title.strip()
        
        return processed_title
    
    def _is_valid_title_candidate(self, block_data: Dict) -> bool:
        """Validate title candidate using configurable criteria with language awareness"""
        base_valid = (
            block_data['word_count'] >= self.cfg.title_min_words and
            block_data['word_count'] <= self.cfg.title_max_words and
            block_data['position_ratio'] <= self.cfg.mid_document_threshold
        )
        
        if not base_valid:
            return False
        
        # Apply language-specific validation if enabled
        if self.analysis.enable_language_specific_validation and hasattr(self, 'language'):
            return self._validate_title_for_language(block_data)
        
        return True
    
    def _validate_title_for_language(self, block_data: Dict) -> bool:
        """Apply language-specific title validation"""
        if not hasattr(self, 'language'):
            return True
        
        language = self.language
        text = block_data['text']
        
        # Language-specific validation rules
        if language == 'japanese':
            # Japanese titles should have reasonable character balance
            # Allow more flexibility in word count due to different tokenization
            return len(text.strip()) >= 3
        elif language == 'german':
            # German can have very long compound words
            # Be more lenient with word count limits
            return block_data['word_count'] <= self.cfg.title_max_words * 1.5
        elif language == 'tamil':
            # Tamil script validation - ensure meaningful content
            return len(text.strip()) >= 3
        
        return True
    
    def _calculate_title_score(self, block_data: Dict) -> float:
        """Calculate title score using configurable components with language awareness"""
        score = 0.0
        
        # Bold scoring (configurable)
        if self.analysis.enable_bold_scoring and block_data['is_bold']:
            score += self._get_language_adjusted_bonus('bold', self.cfg.bold_confidence_bonus)
        
        # Uppercase scoring (configurable with language awareness)
        if self.analysis.enable_uppercase_scoring and block_data['is_upper']:
            # Some languages don't use uppercase the same way
            uppercase_bonus = self._get_language_adjusted_bonus('uppercase', self.cfg.uppercase_confidence_bonus)
            score += uppercase_bonus
        
        # Font size scoring (configurable)
        if self.analysis.enable_font_size_scoring:
            if block_data['font_size'] >= self.cfg.large_font_threshold:
                score += self._get_language_adjusted_bonus('font', self.cfg.largest_font_bonus)
        
        # Language-aware pattern scoring
        if self.analysis.enable_language_aware_scoring and hasattr(self, 'language'):
            language_score = self._calculate_language_specific_score(block_data)
            score += language_score
        
        return min(self.cfg.max_confidence_score, score)
    
    def _get_language_adjusted_bonus(self, bonus_type: str, default_bonus: float) -> float:
        """Get language-adjusted bonus values"""
        if not hasattr(self, 'language') or not self._current_language_patterns:
            return default_bonus
        
        language = self.language
        
        # Language-specific bonus adjustments
        adjustments = {
            'japanese': {
                'bold': 0.8,      # Bold is less common in Japanese
                'uppercase': 0.3,  # Uppercase doesn't apply to Japanese scripts
                'font': 1.0
            },
            'german': {
                'bold': 1.0,
                'uppercase': 0.7,  # German uses less all-caps
                'font': 1.0
            },
            'tamil': {
                'bold': 0.9,
                'uppercase': 0.2,  # Uppercase doesn't apply to Tamil script
                'font': 1.0
            },
            'english': {
                'bold': 1.0,
                'uppercase': 1.0,
                'font': 1.0
            }
        }
        
        multiplier = adjustments.get(language, {}).get(bonus_type, 1.0)
        return default_bonus * multiplier
    
    def _calculate_language_specific_score(self, block_data: Dict) -> float:
        """Calculate additional score based on language-specific patterns"""
        if not hasattr(self, 'language'):
            return 0.0
        
        language = self.language
        text = block_data['text'].lower()
        score = 0.0
        
        # Check language-specific business terms
        lang_business_terms = self.pt.language_specific_business_terms.get(language, [])
        if lang_business_terms:
            for term in lang_business_terms:
                if term.lower() in text:
                    score += self.cfg.pattern_match_bonus * 0.5
        
        # Check language-specific structural terms
        lang_structural_terms = self.pt.language_specific_structural_terms.get(language, [])
        if lang_structural_terms:
            for term in lang_structural_terms:
                if term.lower() in text:
                    score += self.cfg.pattern_match_bonus * 0.3
        
        return min(self.cfg.pattern_match_bonus, score)
    
    def _repair_title(self, raw_title: str) -> str:
        """Repair title using configurable patterns with language awareness"""
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
        """Extract outline using configurable heading detection with language awareness"""
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
        """Validate heading using configurable exclusion rules with language awareness"""
        text = block_data['text']
        text_lower = text.lower() if not self.pt.case_sensitive_matching else text
        
        # Paragraph starter exclusion (configurable)
        if self.analysis.enable_paragraph_exclusion:
            comparison_starters = (self.pt.paragraph_starters if self.pt.case_sensitive_matching 
                                 else [s.lower() for s in self.pt.paragraph_starters])
            if any(text_lower.startswith(starter) for starter in comparison_starters):
                return False
        
        # Length validation (configurable with language awareness)
        if self.analysis.enable_length_validation:
            max_words = self._get_language_adjusted_max_words(self.cfg.heading_max_words)
            if len(text.split()) > max_words:
                return False
        
        # Punctuation validation (configurable)
        if self.analysis.enable_punctuation_validation:
            if (text.count('.') > self.cfg.max_periods_in_heading or
                text.count(',') > self.cfg.max_commas_in_heading):
                return False
        
        return True
    
    def _get_language_adjusted_max_words(self, default_max: int) -> int:
        """Get language-adjusted maximum word count for headings"""
        if not hasattr(self, 'language'):
            return default_max
        
        # Language-specific adjustments for word count limits
        adjustments = {
            'japanese': 1.5,   # Japanese may have different tokenization
            'german': 1.3,     # German compound words may affect word count
            'tamil': 1.4,      # Tamil script considerations
            'english': 1.0
        }
        
        multiplier = adjustments.get(self.language, 1.0)
        return int(default_max * multiplier)
    
    def _score_heading(self, block_data: Dict) -> float:
        """Score heading using configurable components with language awareness"""
        score = 0.0
        
        # Format-based scoring (configurable)
        if self.analysis.enable_bold_scoring and block_data['is_bold']:
            score += self._get_language_adjusted_bonus('bold', self.cfg.bold_confidence_bonus)
        
        if self.analysis.enable_uppercase_scoring and block_data['is_upper']:
            score += self._get_language_adjusted_bonus('uppercase', self.cfg.uppercase_confidence_bonus)
        
        # Font size scoring (configurable)
        if self.analysis.enable_font_size_scoring:
            font_size = block_data['font_size']
            if font_size >= self.cfg.large_font_threshold:
                score += self._get_language_adjusted_bonus('font', self.cfg.largest_font_bonus)
            elif font_size > self.cfg.default_font_size:
                score += self._get_language_adjusted_bonus('font', self.cfg.large_font_bonus)
        
        # Length scoring (configurable)
        if self.analysis.enable_length_scoring:
            word_count = block_data['word_count']
            if word_count <= self.cfg.very_short_heading_max_words:
                score += self.cfg.acceptable_length_bonus
            elif word_count <= self.cfg.short_heading_max_words:
                score += self.cfg.good_length_bonus
        
        # Pattern scoring (configurable with language awareness)
        if self.analysis.enable_pattern_scoring:
            pattern_score = self._calculate_pattern_score(block_data)
            score += pattern_score
        
        # Position scoring (configurable)
        if self.analysis.enable_position_scoring:
            if block_data['position_ratio'] <= self.cfg.early_position_threshold:
                score += self.cfg.position_bonus
        
        # Language-specific heading scoring
        if self.analysis.enable_language_aware_scoring and hasattr(self, 'language'):
            language_heading_score = self._calculate_language_heading_score(block_data)
            score += language_heading_score
        
        return min(self.cfg.max_confidence_score, score)
    
    def _calculate_pattern_score(self, block_data: Dict) -> float:
        """Calculate pattern-based score with language awareness"""
        score = 0.0
        text_for_comparison = (block_data['text'] if self.pt.case_sensitive_matching 
                             else block_data['text'].lower())
        
        # Standard pattern matching
        comparison_terms = (self.pt.structural_terms + self.pt.business_terms 
                          if self.pt.case_sensitive_matching
                          else [term.lower() for term in self.pt.structural_terms + self.pt.business_terms])
        
        if any(term in text_for_comparison for term in comparison_terms):
            score += self.cfg.pattern_match_bonus
        
        # Language-specific pattern matching
        if hasattr(self, 'language') and self.language in self.pt.language_specific_h2_keywords:
            lang_terms = self.pt.language_specific_h2_keywords[self.language]
            lang_comparison_terms = (lang_terms if self.pt.case_sensitive_matching
                                   else [term.lower() for term in lang_terms])
            
            if any(term in text_for_comparison for term in lang_comparison_terms):
                score += self.cfg.pattern_match_bonus * 0.8
        
        return score
    
    def _calculate_language_heading_score(self, block_data: Dict) -> float:
        """Calculate language-specific heading scores"""
        if not hasattr(self, 'language'):
            return 0.0
        
        language = self.language
        text = block_data['text']
        score = 0.0
        
        # Get language-specific heading patterns from configuration
        lang_config = self.language_config if hasattr(self, 'language_config') else {}
        
        # Check against language-specific heading indicators
        heading_indicators = lang_config.get('heading_indicators', [])
        for indicator in heading_indicators:
            if indicator.lower() in text.lower():
                score += self.cfg.pattern_match_bonus * 0.6
        
        return score
    
    def _determine_heading_level(self, confidence: float, block_data: Dict) -> str:
        """Determine heading level using configurable thresholds with language awareness"""
        
        # Apply language-specific threshold adjustments if available
        if hasattr(self, 'language') and self._current_language_patterns:
            confidence = self._adjust_confidence_for_language(confidence, block_data)
        
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
    
    def _adjust_confidence_for_language(self, confidence: float, block_data: Dict) -> float:
        """Adjust confidence score based on language-specific factors"""
        if not hasattr(self, 'language'):
            return confidence
        
        language = self.language
        adjusted_confidence = confidence
        
        # Language-specific confidence adjustments
        if language == 'japanese':
            # Japanese may have different formatting conventions
            if block_data.get('language_adjusted_font', False):
                adjusted_confidence *= 1.1
        elif language == 'german':
            # German compound words might affect scoring
            if block_data['word_count'] > self.cfg.short_heading_max_words:
                adjusted_confidence *= 0.95  # Slight penalty for very long headings
        elif language == 'tamil':
            # Tamil script considerations
            if len(block_data['text']) > 50:  # Character-based check for Tamil
                adjusted_confidence *= 1.05
        
        return min(self.cfg.max_confidence_score, adjusted_confidence)

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
        
    elif mode == 'multilingual':
        # Enhanced multilingual mode
        config.enable_language_aware_processing = True
        config.apply_language_font_adjustments = True
        config.use_language_specific_patterns = True
        analysis.enable_language_aware_scoring = True
        analysis.enable_language_specific_validation = True
        analysis.apply_language_font_thresholds = True
        
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
        
        # Add language-specific patterns for RFP documents
        patterns.language_specific_business_terms = {
            'german': ['antrag', 'vorschlag', 'ausschreibung'],
            'japanese': ['提案書', '要求仕様書', '仕様書'],
            'tamil': ['முன்மொழிவு', 'கோரிக்கை'],
            'english': ['rfp', 'proposal', 'request']
        }
    
    return BusinessProcessor(config=config, patterns=patterns, analysis=analysis)

# Usage examples
if __name__ == "__main__":
    # Standard processor (empty patterns)
    standard_processor = create_business_processor('standard')
    
    # Multilingual processor
    multilingual_processor = create_business_processor('multilingual')
    
    # RFP-configured processor
    rfp_processor = create_business_processor('rfp_configured')
    
    # Custom configuration
    custom_config = ProcessorConfig(
        document_type=DocumentType.BUSINESS_DOCUMENT,
        document_confidence=0.95,
        title_min_words=3,
        large_font_threshold=14.0,
        enable_language_aware_processing=True
    )
    
    custom_patterns = PatternConfig(
        title_repair_patterns=[(r'SPEC\s*:', 'Specification:')],
        business_terms=['specification', 'standard', 'protocol'],
        case_sensitive_matching=False
    )
    
    custom_analysis = AnalysisConfig(
        enable_title_repair=True,
        enable_pattern_scoring=True,
        enable_confidence_filtering=True,
        enable_language_aware_scoring=True
    )
    
    custom_processor = BusinessProcessor(
        config=custom_config,
        patterns=custom_patterns,
        analysis=custom_analysis
    )
