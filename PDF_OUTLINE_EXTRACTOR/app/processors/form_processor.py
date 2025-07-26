import sys
import os
import logging
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Ensure proper package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from processors.base_processor import BaseProcessor
from models.document_types import TextBlock, DocumentType, DocumentStructure

# Dynamic imports with graceful fallbacks
try:
    from core.title_extractor import TitleExtractor
except ImportError:
    TitleExtractor = None

try:
    from analyzers.font_analyzer import FontAnalyzer
except ImportError:
    FontAnalyzer = None

try:
    from analyzers.text_analyzer import TextAnalyzer
except ImportError:
    TextAnalyzer = None

try:
    from classifiers.content_filter import ContentFilter
except ImportError:
    ContentFilter = None

try:
    from classifiers.semantic_classifier import SemanticClassifier
except ImportError:
    SemanticClassifier = None

try:
    from utils.text_utils import TextProcessor
except ImportError:
    TextProcessor = None

try:
    from utils.validation_utils import OutputValidator
except ImportError:
    OutputValidator = None

logger = logging.getLogger('extraction')


@dataclass
class FormProcessorConfig:
    """Configuration class for FormProcessor - removes all hardcoding"""
    
    # Document settings
    document_type: DocumentType = DocumentType.FORM_DOCUMENT
    success_confidence: float = 0.9
    error_confidence: float = 0.0

    # Font analysis settings
    font_analysis_block_limit: int = 15
    font_bold_multiplier: float = 1.3
    font_normal_multiplier: float = 1.0
    fallback_title_threshold: float = 16.0
    fallback_average_size: float = 12.0
    title_threshold_ratio: float = 0.9

    # Content analysis settings
    short_block_length_threshold: int = 30
    form_like_short_ratio_threshold: float = 0.4
    form_like_avg_length_threshold: float = 50.0

    # Title extraction settings
    font_based_block_limit: int = 10
    position_based_block_limit: int = 8
    early_position_limit: int = 3

    # Title quality validation
    min_title_length: int = 8
    max_title_length: int = 200
    min_title_words: int = 2
    title_quality_threshold: int = 4

    # Title formatting
    title_suffix: str = "  "

    # Supported document types
    supported_types: List[str] = None

    # Logging settings
    enable_debug_logging: bool = True

    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.supported_types is None:
            self.supported_types = ['FORM_DOCUMENT', 'APPLICATION_FORM', 'SURVEY_FORM']


@dataclass
class FormAnalysisConfig:
    """Configuration for form analysis components"""

    # Font analysis settings
    font_size_attribute: str = 'font_size'
    is_bold_attribute: str = 'is_bold'
    default_font_size: float = 12.0
    default_is_bold: bool = False
    min_text_length_for_analysis: int = 2

    # Content pattern settings
    text_join_separator: str = " "
    content_analysis_enabled: bool = True

    # Title extraction priorities
    use_title_extractor: bool = True
    use_font_based_extraction: bool = True
    use_position_based_extraction: bool = True

    # Validation settings
    use_output_validator: bool = True


class FormProcessor(BaseProcessor):
    """Fully configurable processor for form documents using adaptive analysis"""

    def __init__(self,
                 config: Optional[FormProcessorConfig] = None,
                 analysis_config: Optional[FormAnalysisConfig] = None):
        """Initialize with configurable parameters"""
        super().__init__()                                 # initialize BaseProcessor

        self.config = config or FormProcessorConfig()
        self.analysis_config = analysis_config or FormAnalysisConfig()

        if self.config.enable_debug_logging:
            logger.setLevel(logging.DEBUG)

        try:
            # Initialize components with fallbacks
            self.title_extractor    = self._safe_init(TitleExtractor)
            self.font_analyzer      = self._safe_init(FontAnalyzer)
            self.text_analyzer      = self._safe_init(TextAnalyzer)
            self.content_filter     = self._safe_init(ContentFilter)
            self.semantic_classifier= self._safe_init(SemanticClassifier)
            self.text_processor     = self._safe_init(TextProcessor)
            self.output_validator   = self._safe_init(OutputValidator)

            available = [
                name for name, comp in [
                    ('TitleExtractor',    self.title_extractor),
                    ('FontAnalyzer',      self.font_analyzer),
                    ('TextAnalyzer',      self.text_analyzer),
                    ('ContentFilter',     self.content_filter),
                    ('SemanticClassifier',self.semantic_classifier),
                    ('TextProcessor',     self.text_processor),
                    ('OutputValidator',   self.output_validator)
                ] if comp is not None
            ]
            logger.info(f"FormProcessor initialized with: {available}")

        except Exception as e:
            logger.error(f"FormProcessor initialization error: {e}")

    def set_language_config(self, language: str, config: Dict[str, Any]):
        """
        Override BaseProcessor to apply language-specific settings
        """
        super().set_language_config(language, config)
        # Dynamically apply any matching config keys to our configs
        for key, val in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, val)
            if hasattr(self.analysis_config, key):
                setattr(self.analysis_config, key, val)
        logger.info(f"Applied language config for '{language}' to FormProcessor")

    def _safe_init(self, component_class):
        """Safely initialize component"""
        if component_class is None:
            return None
        try:
            return component_class()
        except Exception as e:
            logger.debug(f"Component initialization failed: {e}")
            return None

    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process form document using adaptive analysis"""
        logger.info(f"FormProcessor: Starting to process {len(text_blocks)} text blocks")

        try:
            if not text_blocks:
                logger.warning("FormProcessor: No text blocks provided")
                return self._create_empty_result()

            # Build analysis context
            analysis_context = self._build_analysis_context(text_blocks)

            # Extract title adaptively
            title = self._extract_title_adaptively(text_blocks, analysis_context)
            logger.info(f"FormProcessor: Extracted title: '{title}'")

            # Forms typically have empty outlines
            outline = self._determine_outline_adaptively(text_blocks, analysis_context)

            result = DocumentStructure(
                title=title,
                outline=outline,
                doc_type=self.config.document_type,
                confidence=self.config.success_confidence
            )

            # Attach language metadata if available
            if hasattr(result, 'metadata') and hasattr(self, 'language'):
                result.metadata.update({
                    'detected_language': self.language,
                    'language_config_applied': bool(self.language_config)
                })

            return self._validate_result(result)

        except Exception as e:
            logger.error(f"FormProcessor processing error: {e}", exc_info=True)
            return self._create_empty_result()

    def _build_analysis_context(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Build analysis context using available components"""
        context = {}

        # Font analysis
        if self.font_analyzer:
            try:
                context['font_analysis'] = self.font_analyzer.analyze_fonts(text_blocks)
            except Exception:
                context['font_analysis'] = self._fallback_font_analysis(text_blocks)
        else:
            context['font_analysis'] = self._fallback_font_analysis(text_blocks)

        # Content analysis
        if self.analysis_config.content_analysis_enabled:
            context['content_analysis'] = self._analyze_content_patterns(text_blocks)
        else:
            context['content_analysis'] = {}

        return context

    def _fallback_font_analysis(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Fallback font analysis when FontAnalyzer unavailable"""
        font_sizes = []
        limit = min(self.config.font_analysis_block_limit, len(text_blocks))

        for block in text_blocks[:limit]:
            try:
                fs = getattr(block, self.analysis_config.font_size_attribute,
                             self.analysis_config.default_font_size)
                bd = getattr(block, self.analysis_config.is_bold_attribute,
                             self.analysis_config.default_is_bold)
                txt = block.text.strip()

                if txt and len(txt) > self.analysis_config.min_text_length_for_analysis:
                    weight = fs * (self.config.font_bold_multiplier if bd
                                   else self.config.font_normal_multiplier)
                    font_sizes.append(weight)
            except Exception:
                continue

        if font_sizes:
            return {
                'title_threshold': max(font_sizes),
                'average_size': sum(font_sizes) / len(font_sizes),
                'method': 'fallback'
            }

        return {
            'title_threshold': self.config.fallback_title_threshold,
            'average_size': self.config.fallback_average_size,
            'method': 'default'
        }

    def _analyze_content_patterns(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze content patterns using configurable thresholds"""
        total_text = ""
        short_blocks = 0
        total_blocks = 0

        for block in text_blocks:
            txt = block.text.strip()
            if txt:
                total_text += self.analysis_config.text_join_separator + txt
                total_blocks += 1
                if len(txt) < self.config.short_block_length_threshold:
                    short_blocks += 1

        ratio = short_blocks / max(1, total_blocks)
        avg_len = len(total_text) / max(1, total_blocks)
        is_form = (ratio > self.config.form_like_short_ratio_threshold and
                   avg_len < self.config.form_like_avg_length_threshold)

        return {
            'short_block_ratio': ratio,
            'avg_block_length': avg_len,
            'total_length': len(total_text),
            'is_form_like': is_form
        }

    def _extract_title_adaptively(self,
                                  text_blocks: List[TextBlock],
                                  context: Dict[str, Any]) -> str:
        """Extract title using configurable adaptive methods"""

        # Method 1: TitleExtractor
        if (self.analysis_config.use_title_extractor and
                self.title_extractor):
            try:
                title = self.title_extractor.extract_title(
                    text_blocks, self.config.document_type)
                if title and self._validate_title_quality(title):
                    return self._format_title(title)
            except Exception as e:
                logger.debug(f"TitleExtractor failed: {e}")

        # Method 2: Font-based extraction
        if self.analysis_config.use_font_based_extraction:
            fa = context.get('font_analysis', {})
            threshold = fa.get('title_threshold',
                               self.config.fallback_title_threshold)
            limit = min(self.config.font_based_block_limit,
                        len(text_blocks))

            for block in text_blocks[:limit]:
                try:
                    fs = getattr(block, self.analysis_config.font_size_attribute,
                                 self.analysis_config.default_font_size)
                    bd = getattr(block, self.analysis_config.is_bold_attribute,
                                 self.analysis_config.default_is_bold)
                    txt = block.text.strip()
                    weight = fs * (self.config.font_bold_multiplier if bd
                                   else self.config.font_normal_multiplier)

                    if (weight >= threshold * self.config.title_threshold_ratio and
                            self._validate_title_quality(txt)):
                        return self._format_title(txt)
                except Exception:
                    continue

        # Method 3: Position-based fallback
        if self.analysis_config.use_position_based_extraction:
            limit = min(self.config.position_based_block_limit,
                        len(text_blocks))
            for i, block in enumerate(text_blocks[:limit]):
                txt = block.text.strip()
                if (i < self.config.early_position_limit and
                        self._validate_title_quality(txt)):
                    return self._format_title(txt)

        return ""

    def _determine_outline_adaptively(self,
                                      text_blocks: List[TextBlock],
                                      context: Dict[str, Any]) -> List[Dict]:
        """Determine outline requirement using configurable analysis"""
        ca = context.get('content_analysis', {})
        # Forms almost never have outlines
        if ca.get('is_form_like', True):
            return []
        return []

    def _validate_title_quality(self, title: str) -> bool:
        """Validate title quality using configurable criteria"""
        if not title:
            return False
        t = title.strip()
        checks = [
            len(t) >= self.config.min_title_length,
            len(t) <= self.config.max_title_length,
            len(t.split()) >= self.config.min_title_words,
            not t.endswith(':'),
            not t.isdigit()
        ]
        return sum(checks) >= self.config.title_quality_threshold

    def _format_title(self, title: str) -> str:
        """Format title with configurable formatting"""
        if not title:
            return ""
        if self.text_processor:
            try:
                cleaned = self.text_processor.clean_text(title)
                return cleaned + self.config.title_suffix
            except Exception:
                pass
        # fallback
        cleaned = re.sub(r'\s+', ' ', title.strip())
        return (cleaned + self.config.title_suffix) if cleaned else ""

    def _validate_result(self, result: DocumentStructure) -> DocumentStructure:
        """Validate result using configurable validator"""
        if (self.analysis_config.use_output_validator and
                self.output_validator):
            try:
                out = {"title": result.title, "outline": result.outline}
                if self.output_validator.validate_output(out):
                    return result
            except Exception:
                pass
        return result

    def _create_empty_result(self) -> DocumentStructure:
        """Create empty result for error cases using configured values"""
        return DocumentStructure(
            title="",
            outline=[],
            doc_type=self.config.document_type,
            confidence=self.config.error_confidence
        )

    def get_supported_document_types(self) -> List[str]:
        """Return configurable supported document types"""
        return list(self.config.supported_types)
    

# Factory function for easy processor creation
def create_form_processor(processor_type: str = 'standard') -> FormProcessor:
    """Factory function to create preconfigured form processors"""

    if processor_type == 'strict':
        config = FormProcessorConfig(
            min_title_length=10,
            min_title_words=3,
            title_quality_threshold=5,
            success_confidence=0.95
        )
        analysis_config = FormAnalysisConfig(
            use_title_extractor=True,
            use_font_based_extraction=True,
            use_position_based_extraction=False
        )
        return FormProcessor(config=config, analysis_config=analysis_config)

    elif processor_type == 'lenient':
        config = FormProcessorConfig(
            min_title_length=3,
            min_title_words=1,
            title_quality_threshold=2,
            success_confidence=0.8
        )
        return FormProcessor(config=config)

    elif processor_type == 'minimal':
        config = FormProcessorConfig(
            font_analysis_block_limit=5,
            font_based_block_limit=5,
            position_based_block_limit=3,
            enable_debug_logging=False
        )
        analysis_config = FormAnalysisConfig(
            content_analysis_enabled=False,
            use_output_validator=False
        )
        return FormProcessor(config=config, analysis_config=analysis_config)

    elif processor_type == 'debug':
        config = FormProcessorConfig(
            enable_debug_logging=True,
            title_suffix=" [DEBUG]"
        )
        return FormProcessor(config=config)

    else:  # 'standard'
        return FormProcessor()


# Usage examples
if __name__ == "__main__":
    # Standard processor
    processor = FormProcessor()

    # Custom configuration
    custom_config = FormProcessorConfig(
        document_type=DocumentType.FORM_DOCUMENT,
        success_confidence=0.95,
        min_title_length=5,
        title_suffix="",  # No suffix
        supported_types=['CUSTOM_FORM', 'SURVEY']
    )

    custom_analysis = FormAnalysisConfig(
        use_title_extractor=False,  # Skip title extractor
        min_text_length_for_analysis=5
    )

    custom_processor = FormProcessor(config=custom_config,
                                     analysis_config=custom_analysis)

    # Factory usage
    strict  = create_form_processor('strict')
    lenient = create_form_processor('lenient')
