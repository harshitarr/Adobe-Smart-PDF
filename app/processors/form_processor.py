import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor

logger = logging.getLogger('extraction')

@dataclass
class FormProcessorConfig:
    """Configuration class for FormProcessor parameters"""
    
    # Document processing settings
    document_confidence: float = 0.9
    document_type: DocumentType = DocumentType.FORM_DOCUMENT
    
    # Outline extraction settings
    enable_outline_extraction: bool = False
    outline_extraction_method: str = 'simple'  # 'simple', 'pattern_based', 'none'
    
    # Section detection parameters (if outline extraction is enabled)
    section_min_font_size: float = 12.0
    section_keywords: List[str] = None
    section_confidence_threshold: float = 0.6
    max_section_word_count: int = 15
    
    # Logging and debugging
    enable_debug_logging: bool = False
    log_extraction_details: bool = False
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.section_keywords is None:
            self.section_keywords = [
                'section', 'part', 'step', 'instructions', 
                'details', 'information', 'requirements'
            ]

@dataclass
class FormPatternConfig:
    """Configuration for form-specific patterns and vocabularies"""
    
    # Form section indicators
    section_prefixes: List[str] = None
    section_suffixes: List[str] = None
    numbered_section_pattern: str = r'^\d+\.\s+[A-Z]'
    lettered_section_pattern: str = r'^[A-Z]\.\s+[A-Z]'
    
    # Form field indicators (for future enhancement)
    field_indicators: List[str] = None
    required_field_markers: List[str] = None
    
    def __post_init__(self):
        """Initialize default pattern values"""
        if self.section_prefixes is None:
            self.section_prefixes = ['section', 'part', 'step']
        
        if self.section_suffixes is None:
            self.section_suffixes = [':', '.-']
        
        if self.field_indicators is None:
            self.field_indicators = ['name:', 'address:', 'phone:', 'email:', 'date:']
        
        if self.required_field_markers is None:
            self.required_field_markers = ['*', '(required)', '(mandatory)']

class FormProcessor:
    """Generalized form processor with configurable parameters"""
    
    def __init__(self, 
                 config: Optional[FormProcessorConfig] = None,
                 patterns: Optional[FormPatternConfig] = None):
        self.config = config or FormProcessorConfig()
        self.patterns = patterns or FormPatternConfig()
        self.title_extractor = TitleExtractor()
        
        # Set up logging based on configuration
        if self.config.enable_debug_logging:
            logger.setLevel(logging.DEBUG)
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process form document with configurable extraction options"""
        
        if self.config.log_extraction_details:
            logger.info(f"Processing form document (type: {self.config.document_type})")
            logger.debug(f"Processing {len(text_blocks)} text blocks")
        
        # Extract title using configured document type
        title = self.title_extractor.extract_title(text_blocks, self.config.document_type)
        
        # Extract outline based on configuration
        outline = self._extract_outline(text_blocks) if self.config.enable_outline_extraction else []
        
        if self.config.log_extraction_details:
            logger.debug(f"Extracted title: '{title}'")
            logger.debug(f"Extracted {len(outline)} outline items")
        
        return DocumentStructure(
            title=title,
            outline=outline,
            doc_type=self.config.document_type,
            confidence=self.config.document_confidence
        )
    
    def _extract_outline(self, text_blocks: List[TextBlock]) -> List[HeadingLevel]:
        """Extract outline based on configured method"""
        
        if self.config.outline_extraction_method == 'none':
            return []
        elif self.config.outline_extraction_method == 'simple':
            return self._extract_simple_outline(text_blocks)
        elif self.config.outline_extraction_method == 'pattern_based':
            return self._extract_pattern_based_outline(text_blocks)
        else:
            logger.warning(f"Unknown outline extraction method: {self.config.outline_extraction_method}")
            return []
    
    def _extract_simple_outline(self, text_blocks: List[TextBlock]) -> List[HeadingLevel]:
        """Simple outline extraction based on basic formatting"""
        outline = []
        
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            if not text:
                continue
            
            # Simple section detection based on font size and formatting
            font_size = getattr(block, 'font_size', self.config.section_min_font_size)
            word_count = len(text.split())
            
            if (font_size >= self.config.section_min_font_size and
                block.is_bold and
                word_count <= self.config.max_section_word_count and
                not self._is_form_field(text)):
                
                outline.append(HeadingLevel(
                    level='H3',  # Default level for form sections
                    text=text,
                    page=block.page,
                    confidence=self.config.section_confidence_threshold,
                    font_size=font_size,
                    font_name=getattr(block, 'font_name', None)
                ))
        
        return outline
    
    def _extract_pattern_based_outline(self, text_blocks: List[TextBlock]) -> List[HeadingLevel]:
        """Pattern-based outline extraction using configured patterns"""
        outline = []
        
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            if not text:
                continue
            
            confidence = self._calculate_section_confidence(text, block)
            
            if confidence >= self.config.section_confidence_threshold:
                level = self._determine_section_level(text, confidence)
                
                outline.append(HeadingLevel(
                    level=level,
                    text=text,
                    page=block.page,
                    confidence=confidence,
                    font_size=getattr(block, 'font_size', self.config.section_min_font_size),
                    font_name=getattr(block, 'font_name', None)
                ))
        
        return outline
    
    def _calculate_section_confidence(self, text: str, block: TextBlock) -> float:
        """Calculate confidence that a text block is a section heading"""
        confidence = 0.0
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Font and formatting indicators
        if block.is_bold:
            confidence += 0.3
        
        font_size = getattr(block, 'font_size', self.config.section_min_font_size)
        if font_size >= self.config.section_min_font_size:
            confidence += 0.2
        
        # Pattern matching
        if any(keyword in text_lower for keyword in self.config.section_keywords):
            confidence += 0.3
        
        # Structural patterns
        import re
        if re.match(self.patterns.numbered_section_pattern, text):
            confidence += 0.4
        elif re.match(self.patterns.lettered_section_pattern, text):
            confidence += 0.3
        
        # Prefix/suffix patterns
        if any(text_lower.startswith(prefix) for prefix in self.patterns.section_prefixes):
            confidence += 0.2
        
        if any(text.endswith(suffix) for suffix in self.patterns.section_suffixes):
            confidence += 0.2
        
        # Length appropriateness
        if 1 <= word_count <= self.config.max_section_word_count:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _determine_section_level(self, text: str, confidence: float) -> str:
        """Determine hierarchical level for form sections"""
        text_lower = text.lower()
        
        # H2 for major sections
        if any(keyword in text_lower for keyword in ['instructions', 'requirements', 'details']):
            return 'H2'
        
        # H3 for subsections
        if confidence > 0.8:
            return 'H2'
        else:
            return 'H3'
    
    def _is_form_field(self, text: str) -> bool:
        """Check if text appears to be a form field rather than a section"""
        text_lower = text.lower()
        
        # Check for field indicators
        field_patterns = [
            any(indicator in text_lower for indicator in self.patterns.field_indicators),
            any(marker in text for marker in self.patterns.required_field_markers),
            text.endswith('_' * 3),  # Common field placeholder pattern
            len(text.split()) == 1 and ':' in text  # Single word with colon
        ]
        
        return any(field_patterns)

# Factory function for easy processor creation
def create_form_processor(processor_type: str = 'standard') -> FormProcessor:
    """Factory function to create preconfigured form processors"""
    
    if processor_type == 'simple':
        config = FormProcessorConfig(
            enable_outline_extraction=False,
            document_confidence=0.8
        )
        return FormProcessor(config=config)
    
    elif processor_type == 'sectioned':
        config = FormProcessorConfig(
            enable_outline_extraction=True,
            outline_extraction_method='pattern_based',
            document_confidence=0.9,
            section_confidence_threshold=0.5
        )
        return FormProcessor(config=config)
    
    elif processor_type == 'debug':
        config = FormProcessorConfig(
            enable_debug_logging=True,
            log_extraction_details=True,
            enable_outline_extraction=True
        )
        return FormProcessor(config=config)
    
    else:  # 'standard'
        return FormProcessor()
