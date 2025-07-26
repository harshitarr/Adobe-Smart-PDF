from typing import Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger('extraction')

class BaseProcessor(ABC):
    """Base processor with comprehensive language awareness and safe configuration handling"""
    
    def __init__(self):
        self.language = 'english'
        self.language_config = {}
        self.processing_config = {}
        self.original_config = {}  # Store original values for reset capability
        
        logger.debug(f"Initialized {self.__class__.__name__} with default language: {self.language}")
        
    def set_language_config(self, language: str, config: Dict[str, Any]):
        """Set language-specific processing configuration with validation"""
        try:
            if not language or not isinstance(language, str):
                logger.warning("Invalid language provided, using default")
                language = 'english'
                
            if not isinstance(config, dict):
                logger.warning("Invalid config provided, using empty config")
                config = {}
            
            # Store original config on first call
            if not self.original_config:
                self.original_config = {
                    'language': self.language,
                    'language_config': self.language_config.copy(),
                    'processing_config': self.processing_config.copy()
                }
            
            self.language = language
            self.language_config = config.copy()  # Create a copy to avoid reference issues
            
            # Safely update processing config
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool, list, dict)):
                    self.processing_config[key] = value
                else:
                    logger.debug(f"Skipping unsupported config type for {key}: {type(value)}")
            
            logger.info(f"Applied {language} configuration to {self.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Error setting language config: {e}")
            # Fallback to safe defaults
            self.language = 'english'
            self.language_config = {}
    
    def reset_to_default_config(self):
        """Reset to original configuration"""
        if self.original_config:
            self.language = self.original_config['language']
            self.language_config = self.original_config['language_config'].copy()
            self.processing_config = self.original_config['processing_config'].copy()
            logger.debug(f"Reset {self.__class__.__name__} to default configuration")
    
    def get_title_patterns(self) -> List[str]:
        """Get language-specific title patterns with fallback"""
        try:
            patterns = self.language_config.get('title_patterns', [])
            
            # If no language-specific patterns, return default patterns
            if not patterns:
                default_patterns = [
                    r'^\d+\.?\s+[A-Z]',           # Numbered sections: "1. Introduction"
                    r'^[IVXLCDM]+\.?\s+[A-Z]',    # Roman numerals: "I. Overview"
                    r'^[A-Z][a-z\s]+:$',          # Title with colon: "Introduction:"
                    r'^[A-Z][A-Z\s]+$',           # ALL CAPS: "INTRODUCTION"
                    r'^\d+\.\d+\s+[A-Z]',         # Subsections: "1.1 Overview"
                    r'^Chapter\s+\d+',            # Chapter format
                    r'^Section\s+\d+',            # Section format
                    r'^Part\s+[IVXLCDM]+',        # Part format
                    r'^Appendix\s+[A-Z]'          # Appendix format
                ]
                return default_patterns
            
            return patterns if isinstance(patterns, list) else []
            
        except Exception as e:
            logger.debug(f"Error getting title patterns: {e}")
            return []
    
    def get_heading_indicators(self) -> List[str]:
        """Get language-specific heading indicators with comprehensive fallbacks"""
        try:
            indicators = self.language_config.get('heading_indicators', [])
            
            # Language-specific defaults if none provided
            if not indicators:
                language_defaults = {
                    'english': [
                        'Chapter', 'Section', 'Part', 'Article', 'Introduction',
                        'Overview', 'Summary', 'Conclusion', 'Background',
                        'Methodology', 'Results', 'Discussion', 'Appendix',
                        'References', 'Bibliography', 'Index', 'Glossary'
                    ],
                    'japanese': [
                        '章', '節', '項', '編', '部', '巻', '序章', '概要',
                        '背景', '方法', '結果', '考察', '結論', '付録',
                        'はじめに', 'おわりに', '参考文献', '索引'
                    ],
                    'german': [
                        'Kapitel', 'Abschnitt', 'Teil', 'Artikel', 'Einführung',
                        'Überblick', 'Zusammenfassung', 'Schlussfolgerung',
                        'Hintergrund', 'Methodik', 'Ergebnisse', 'Diskussion',
                        'Anhang', 'Literatur', 'Bibliographie', 'Index'
                    ],
                    'tamil': [
                        'அத்தியாயம்', 'பிரிவு', 'பகுதி', 'கட்டுரை', 'அறிமுகம்',
                        'கண்ணோட்டம்', 'சுருக்கம்', 'முடிவு', 'பின்னணி',
                        'முறைமை', 'முடிவுகள்', 'விவாதம்', 'இணைப்பு',
                        'குறிப்புகள்', 'பட்டியல்', 'அட்டவணை'
                    ]
                }
                
                indicators = language_defaults.get(self.language, language_defaults['english'])
            
            return indicators if isinstance(indicators, list) else []
            
        except Exception as e:
            logger.debug(f"Error getting heading indicators: {e}")
            return ['Chapter', 'Section', 'Part', 'Article']  # Safe fallback
    
    def get_font_weight_threshold(self) -> float:
        """Get language-specific font weight threshold with validation"""
        try:
            threshold = self.language_config.get('font_weight_threshold', 0.6)
            
            # Validate threshold is a reasonable float
            if isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0:
                return float(threshold)
            else:
                logger.debug(f"Invalid font weight threshold: {threshold}, using default")
                return 0.6
                
        except Exception as e:
            logger.debug(f"Error getting font weight threshold: {e}")
            return 0.6
    
    def get_spacing_multiplier(self) -> float:
        """Get language-specific spacing multiplier with validation"""
        try:
            multiplier = self.language_config.get('spacing_multiplier', 1.0)
            
            # Validate multiplier is a reasonable positive float
            if isinstance(multiplier, (int, float)) and 0.1 <= multiplier <= 5.0:
                return float(multiplier)
            else:
                logger.debug(f"Invalid spacing multiplier: {multiplier}, using default")
                return 1.0
                
        except Exception as e:
            logger.debug(f"Error getting spacing multiplier: {e}")
            return 1.0
    
    def get_confidence_threshold(self) -> float:
        """Get language-specific confidence threshold"""
        try:
            threshold = self.language_config.get('confidence_threshold', 0.4)
            
            if isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0:
                return float(threshold)
            else:
                return 0.4
                
        except Exception as e:
            logger.debug(f"Error getting confidence threshold: {e}")
            return 0.4
    
    def get_language_display_name(self) -> str:
        """Get human-readable display name for current language"""
        try:
            display_names = {
                'english': 'English',
                'japanese': 'Japanese (日本語)',
                'german': 'German (Deutsch)',
                'tamil': 'Tamil (தமிழ்)'
            }
            
            return display_names.get(self.language, self.language.title())
            
        except Exception as e:
            logger.debug(f"Error getting language display name: {e}")
            return self.language.title()
    
    def is_language_aware(self) -> bool:
        """Check if processor has language-specific configuration applied"""
        return self.language != 'english' and bool(self.language_config)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of current processing configuration"""
        try:
            return {
                'language': self.language,
                'display_name': self.get_language_display_name(),
                'is_language_aware': self.is_language_aware(),
                'config_keys': list(self.language_config.keys()),
                'font_threshold': self.get_font_weight_threshold(),
                'spacing_multiplier': self.get_spacing_multiplier(),
                'confidence_threshold': self.get_confidence_threshold(),
                'title_patterns_count': len(self.get_title_patterns()),
                'heading_indicators_count': len(self.get_heading_indicators())
            }
        except Exception as e:
            logger.debug(f"Error getting processing summary: {e}")
            return {'error': str(e)}
    
    def validate_text_blocks(self, text_blocks: List) -> bool:
        """Validate input text blocks"""
        try:
            if not text_blocks:
                logger.warning("Empty text blocks provided")
                return False
            
            if not isinstance(text_blocks, list):
                logger.warning("Text blocks must be a list")
                return False
            
            # Check if blocks have required attributes
            valid_blocks = 0
            for block in text_blocks[:5]:  # Check first 5 blocks
                if hasattr(block, 'text') and hasattr(block, 'page'):
                    if block.text and isinstance(block.text, str):
                        valid_blocks += 1
            
            if valid_blocks == 0:
                logger.warning("No valid text blocks found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating text blocks: {e}")
            return False
    
    @abstractmethod
    def process(self, text_blocks: List) -> Any:
        """
        Process the document - to be implemented by subclasses
        
        Args:
            text_blocks (List): List of TextBlock objects to process
            
        Returns:
            DocumentStructure: Processed document structure
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    def __str__(self) -> str:
        """String representation of processor"""
        return f"{self.__class__.__name__}(language={self.language}, config_keys={len(self.language_config)})"
    
    def __repr__(self) -> str:
        """Detailed representation of processor"""
        return (f"{self.__class__.__name__}("
                f"language='{self.language}', "
                f"is_language_aware={self.is_language_aware()}, "
                f"config_keys={list(self.language_config.keys())})")
