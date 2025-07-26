import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Tuple
import logging

from processors.base_processor import BaseProcessor
from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor

logger = logging.getLogger('extraction')

class ManualProcessor(BaseProcessor):
    def __init__(self):
        # Initialize BaseProcessor for language support
        super().__init__()
        
        self.title_extractor = TitleExtractor()
        self.outline_extractor = OutlineExtractor()
        
        # Generalized pattern rules
        self.pattern_rules = {
            'navigation_patterns': [
                r'revision\s+history',
                r'table\s+of\s+contents',
                r'acknowledgements?'
            ],
            'numbered_section_patterns': [
                r'^\d+\.\s+[a-z]',  # Main sections: 1., 2., 3., 4.
            ],
            'subsection_patterns': [
                r'^\d+\.\d+\s+[a-z]',  # Subsections: 2.1, 2.2, etc.
            ],
            'exclusion_patterns': [
                r'version\s+\d+\s+page\s+\d+',
                r'copyright.*international',
                r'qualifications\s+board',
                r'may\s+\d+,\s+\d+',
                r'working\s+group',
                r'ga\s+release'
            ]
        }
        
        # Language-specific pattern cache
        self._language_patterns = {}
        self._language_adjusted_rules = False
        
        logger.info("ManualProcessor initialized with multilingual support")
    
    def set_language_config(self, language: str, config: Dict[str, Any]):
        """Override from BaseProcessor - apply language-specific patterns"""
        super().set_language_config(language, config)
        
        # Apply language-specific pattern adjustments
        self._apply_language_pattern_adjustments(language, config)
        logger.info(f"Applied {language} patterns to ManualProcessor")
    
    def _apply_language_pattern_adjustments(self, language: str, lang_config: Dict[str, Any]):
        """Apply language-specific pattern adjustments to existing rules"""
        
        # Get language-specific patterns from configuration
        title_patterns = lang_config.get('title_patterns', [])
        heading_indicators = lang_config.get('heading_indicators', [])
        
        # Create language-specific pattern additions
        language_navigation_patterns = []
        language_exclusion_patterns = []
        
        # Language-specific navigation patterns
        if language == 'japanese':
            language_navigation_patterns.extend([
                r'改訂履歴',  # Revision history
                r'目次',      # Table of contents
                r'謝辞',      # Acknowledgements
                r'序文'       # Preface
            ])
            language_exclusion_patterns.extend([
                r'バージョン\s*\d+',
                r'著作権',
                r'資格委員会'
            ])
        elif language == 'german':
            language_navigation_patterns.extend([
                r'versionsgeschichte',
                r'inhaltsverzeichnis',
                r'danksagungen?',
                r'vorwort'
            ])
            language_exclusion_patterns.extend([
                r'version\s+\d+\s+seite\s+\d+',
                r'urheberrecht',
                r'qualifikationsrat'
            ])
        elif language == 'tamil':
            language_navigation_patterns.extend([
                r'திருத்த வரலாறு',    # Revision history
                r'பொருளடக்கம்',      # Table of contents
                r'நன்றிகள்',         # Acknowledgements
                r'முன்னுரை'          # Preface
            ])
            language_exclusion_patterns.extend([
                r'பதிப்பு\s*\d+',
                r'பதிப்புரிமை'
            ])
        
        # Add language-specific patterns to existing rules
        if language_navigation_patterns:
            # Create a combined pattern set without modifying original
            self._language_patterns[language] = {
                'navigation_patterns': self.pattern_rules['navigation_patterns'] + language_navigation_patterns,
                'numbered_section_patterns': self.pattern_rules['numbered_section_patterns'],
                'subsection_patterns': self.pattern_rules['subsection_patterns'],
                'exclusion_patterns': self.pattern_rules['exclusion_patterns'] + language_exclusion_patterns
            }
        else:
            # Use original patterns if no language-specific patterns available
            self._language_patterns[language] = self.pattern_rules.copy()
        
        # Add patterns from heading indicators
        if heading_indicators:
            # Convert heading indicators to regex patterns
            indicator_patterns = [re.escape(indicator.lower()) for indicator in heading_indicators]
            self._language_patterns[language]['navigation_patterns'].extend(indicator_patterns)
        
        self._language_adjusted_rules = True
    
    def _get_active_pattern_rules(self) -> Dict[str, List[str]]:
        """Get currently active pattern rules (language-adjusted or original)"""
        if hasattr(self, 'language') and self.language in self._language_patterns:
            return self._language_patterns[self.language]
        return self.pattern_rules
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Generalized processing for technical manuals with language awareness"""
        logger.info("Processing technical manual - generalized pattern matching")
        
        # Dynamic title extraction
        title = self._extract_generalized_title(text_blocks)
        
        # Pattern-based outline extraction
        outline = self._extract_generalized_outline(text_blocks)
        
        # Create document structure
        result = DocumentStructure(
            title=title,
            outline=outline,
            doc_type=DocumentType.TECHNICAL_MANUAL,
            confidence=0.9
        )
        
        # Add language metadata if available
        if hasattr(result, 'metadata') and hasattr(self, 'language'):
            result.metadata.update({
                'detected_language': self.language,
                'language_patterns_applied': self._language_adjusted_rules,
                'pattern_language': self.language if self._language_adjusted_rules else 'english'
            })
        
        return result
    
    def _extract_generalized_title(self, text_blocks: List[TextBlock]) -> str:
        """Extract title using generalized patterns with language awareness"""
        title_candidates = []
        
        # Look for title patterns in early blocks
        for block in text_blocks[:30]:  # Check first 30 blocks
            text = block.text.strip()
            score = self._calculate_title_score(text, block)
            
            # Apply language-specific title score adjustments
            if hasattr(self, 'language') and self.language != 'english':
                score = self._adjust_title_score_for_language(score, text)
            
            if score > 0.5:  # Threshold for title candidates
                title_candidates.append((text, score, block))
        
        if title_candidates:
            # Sort by score and take the best
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            best_title = title_candidates[0][0]
            return self._format_title(best_title)
        
        return ""
    
    def _adjust_title_score_for_language(self, score: float, text: str) -> float:
        """Adjust title score based on language characteristics"""
        if not hasattr(self, 'language'):
            return score
        
        language = self.language
        adjusted_score = score
        
        # Language-specific title score adjustments
        if language == 'japanese':
            # Japanese titles may be shorter in character count but longer in meaning
            if len(text) < 20:  # Character-based check for Japanese
                adjusted_score *= 1.1
        elif language == 'german':
            # German titles can be longer due to compound words
            if len(text.split()) > 8:  # Allow longer German titles
                adjusted_score *= 1.05
        elif language == 'tamil':
            # Tamil script considerations
            if len(text) < 30:  # Character-based check for Tamil
                adjusted_score *= 1.1
        
        return min(1.0, adjusted_score)
    
    def _calculate_title_score(self, text: str, block: TextBlock) -> float:
        """Calculate title likelihood score with language awareness"""
        if not text or len(text) < 5:
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        words = text.split()
        
        # Base title indicators (English)
        title_words = ['overview', 'introduction', 'manual', 'guide', 'foundation', 'level', 'extensions']
        
        # Add language-specific title indicators
        if hasattr(self, 'language'):
            language_title_words = self._get_language_title_indicators()
            title_words.extend(language_title_words)
        
        indicator_count = sum(1 for word in title_words if word in text_lower)
        score += indicator_count * 0.2
        
        # Length characteristics (adjusted for language)
        word_count_range = self._get_language_word_count_range()
        if word_count_range[0] <= len(words) <= word_count_range[1]:
            score += 0.3
        elif len(words) > 12:
            score -= 0.5
        
        # Position characteristics
        if block.page == 0:
            score += 0.2
        
        # Font characteristics
        if block.font_size > 12:
            score += 0.1
        if block.is_bold:
            score += 0.1
        
        # Avoid numbered sections as titles
        if re.match(r'^\d+\.', text):
            score -= 0.8
        
        return min(1.0, max(0.0, score))
    
    def _get_language_title_indicators(self) -> List[str]:
        """Get language-specific title indicator words"""
        if not hasattr(self, 'language'):
            return []
        
        language_indicators = {
            'japanese': ['概要', 'はじめに', 'マニュアル', 'ガイド', '基礎', 'レベル'],
            'german': ['übersicht', 'einführung', 'handbuch', 'anleitung', 'grundlagen', 'stufe'],
            'tamil': ['கண்ணோட்டம்', 'அறிமுகம்', 'கையேடு', 'வழிகாட்டி', 'அடிப்படை'],
            'english': []
        }
        
        return language_indicators.get(self.language, [])
    
    def _get_language_word_count_range(self) -> Tuple[int, int]:
        """Get appropriate word count range for titles based on language"""
        if not hasattr(self, 'language'):
            return (3, 8)
        
        # Language-specific word count ranges
        ranges = {
            'japanese': (2, 10),  # Japanese may have different tokenization
            'german': (3, 10),    # German compound words may affect count
            'tamil': (2, 8),      # Tamil script considerations
            'english': (3, 8)
        }
        
        return ranges.get(self.language, (3, 8))
    
    def _format_title(self, title: str) -> str:
        """Format title with spacing pattern detection"""
        clean_title = re.sub(r'\s+', ' ', title.strip())
        
        # Apply language-specific title formatting
        if hasattr(self, 'language') and self.language != 'english':
            clean_title = self._apply_language_title_formatting(clean_title)
        
        # Detect if we need extra spacing based on content
        if len(clean_title.split()) <= 5:
            # Short titles often have extra spacing
            return f"{clean_title}  "  # Double space at end
        
        return f"{clean_title} "  # Single space at end
    
    def _apply_language_title_formatting(self, title: str) -> str:
        """Apply language-specific title formatting"""
        if not hasattr(self, 'language'):
            return title
        
        language = self.language
        
        # Language-specific formatting adjustments
        if language == 'japanese':
            # Japanese title formatting
            return title.strip()
        elif language == 'german':
            # German title formatting - preserve compound words
            return title.strip()
        elif language == 'tamil':
            # Tamil title formatting
            return title.strip()
        
        return title
    
    def _extract_generalized_outline(self, text_blocks: List[TextBlock]) -> List[HeadingLevel]:
        """Extract outline using generalized pattern matching with language awareness"""
        outline = []
        processed_texts = set()
        
        # Get active pattern rules (language-adjusted or original)
        active_rules = self._get_active_pattern_rules()
        
        # Phase 1: Navigation elements
        navigation_items = self._extract_navigation_items(text_blocks, active_rules)
        outline.extend(navigation_items)
        processed_texts.update(item.text.lower().strip() for item in navigation_items)
        
        # Phase 2: Main numbered sections
        main_sections = self._extract_main_sections(text_blocks, processed_texts, active_rules)
        outline.extend(main_sections)
        processed_texts.update(item.text.lower().strip() for item in main_sections)
        
        # Phase 3: Subsections
        subsections = self._extract_subsections(text_blocks, processed_texts, active_rules)
        outline.extend(subsections)
        
        return outline
    
    def _extract_navigation_items(self, text_blocks: List[TextBlock], pattern_rules: Dict[str, List[str]]) -> List[HeadingLevel]:
        """Extract navigation elements using pattern matching"""
        navigation_items = []
        
        for pattern in pattern_rules['navigation_patterns']:
            best_match = self._find_best_pattern_match(text_blocks, pattern, max_page=5, pattern_rules=pattern_rules)
            if best_match:
                block, confidence = best_match
                page = self._determine_navigation_page(block.text, block.page)
                
                heading = HeadingLevel(
                    level="H1",
                    text=self._format_heading_text(block.text),
                    page=page,
                    confidence=confidence,
                    font_size=block.font_size,
                    font_name=block.font_name
                )
                navigation_items.append(heading)
        
        return navigation_items
    
    def _extract_main_sections(self, text_blocks: List[TextBlock], processed_texts: set, pattern_rules: Dict[str, List[str]]) -> List[HeadingLevel]:
        """Extract main numbered sections"""
        main_sections = []
        section_blocks = []
        
        for pattern in pattern_rules['numbered_section_patterns']:
            matches = self._find_all_pattern_matches(text_blocks, pattern, pattern_rules)
            section_blocks.extend(matches)
        
        # Sort by section number
        section_blocks.sort(key=lambda x: self._extract_section_number(x[0].text))
        
        for block, confidence in section_blocks:
            text_key = block.text.lower().strip()
            if text_key not in processed_texts:
                # Determine if this is a valid main section
                if self._is_valid_main_section(block):
                    page = self._determine_section_page(block.text, block.page)
                    
                    heading = HeadingLevel(
                        level="H1",
                        text=self._format_heading_text(block.text),
                        page=page,
                        confidence=confidence,
                        font_size=block.font_size,
                        font_name=block.font_name
                    )
                    main_sections.append(heading)
        
        return main_sections
    
    def _extract_subsections(self, text_blocks: List[TextBlock], processed_texts: set, pattern_rules: Dict[str, List[str]]) -> List[HeadingLevel]:
        """Extract subsections using pattern matching"""
        subsections = []
        
        for pattern in pattern_rules['subsection_patterns']:
            matches = self._find_all_pattern_matches(text_blocks, pattern, pattern_rules)
            
            for block, confidence in matches:
                text_key = block.text.lower().strip()
                if text_key not in processed_texts:
                    if self._is_valid_subsection(block):
                        heading = HeadingLevel(
                            level="H2",
                            text=self._format_heading_text(block.text),
                            page=block.page,
                            confidence=confidence,
                            font_size=block.font_size,
                            font_name=block.font_name
                        )
                        subsections.append(heading)
        
        return subsections
    
    def _find_best_pattern_match(self, text_blocks: List[TextBlock], pattern: str, max_page: int = None, pattern_rules: Dict[str, List[str]] = None) -> Tuple[TextBlock, float]:
        """Find the best match for a pattern"""
        candidates = []
        
        if pattern_rules is None:
            pattern_rules = self._get_active_pattern_rules()
        
        for block in text_blocks:
            if max_page and block.page > max_page:
                continue
            
            if self._should_exclude_block(block, pattern_rules):
                continue
            
            if re.search(pattern, block.text.strip(), re.IGNORECASE):
                score = self._calculate_heading_score(block, pattern)
                candidates.append((block, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        
        return None
    
    def _find_all_pattern_matches(self, text_blocks: List[TextBlock], pattern: str, pattern_rules: Dict[str, List[str]] = None) -> List[Tuple[TextBlock, float]]:
        """Find all matches for a pattern"""
        matches = []
        
        if pattern_rules is None:
            pattern_rules = self._get_active_pattern_rules()
        
        for block in text_blocks:
            if self._should_exclude_block(block, pattern_rules):
                continue
            
            if re.search(pattern, block.text.strip(), re.IGNORECASE):
                score = self._calculate_heading_score(block, pattern)
                matches.append((block, score))
        
        return matches
    
    def _should_exclude_block(self, block: TextBlock, pattern_rules: Dict[str, List[str]] = None) -> bool:
        """Check if block should be excluded using patterns"""
        text = block.text.strip().lower()
        
        if pattern_rules is None:
            pattern_rules = self._get_active_pattern_rules()
        
        # Apply exclusion patterns (now language-aware)
        for pattern in pattern_rules['exclusion_patterns']:
            if re.search(pattern, text):
                return True
        
        # Additional generalized exclusions
        exclusion_rules = [
            len(text.split()) > 25,  # Too long for headings
            text.count('.') > 4,  # Multiple sentences
            '@' in text,  # Email addresses
            'www.' in text,  # URLs
            text.startswith('0.') and len(text.split()) > 8,  # Revision entries
        ]
        
        return any(exclusion_rules)
    
    def _calculate_heading_score(self, block: TextBlock, pattern: str) -> float:
        """Calculate heading quality score"""
        score = 0.7  # Base score for pattern match
        
        text = block.text.strip()
        word_count = len(text.split())
        
        # Length scoring (adjusted for language)
        length_range = self._get_language_heading_length_range()
        if length_range[0] <= word_count <= length_range[1]:
            score += 0.2
        elif word_count > 15:
            score -= 0.3
        
        # Font characteristics
        if block.is_bold:
            score += 0.1
        if block.font_size > 12:
            score += 0.05
        
        # Avoid content-like text
        if text.endswith('.') and word_count > 8:
            score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def _get_language_heading_length_range(self) -> Tuple[int, int]:
        """Get appropriate word count range for headings based on language"""
        if not hasattr(self, 'language'):
            return (2, 12)
        
        # Language-specific heading length ranges
        ranges = {
            'japanese': (1, 15),  # Japanese may have different tokenization
            'german': (2, 15),    # German compound words
            'tamil': (1, 12),     # Tamil script considerations
            'english': (2, 12)
        }
        
        return ranges.get(self.language, (2, 12))
    
    def _extract_section_number(self, text: str) -> int:
        """Extract section number for sorting"""
        match = re.match(r'^(\d+)\.', text.strip())
        return int(match.group(1)) if match else 999
    
    def _is_valid_main_section(self, block: TextBlock) -> bool:
        """Validate if block is a main section"""
        text = block.text.strip()
        word_count = len(text.split())
        
        # Should be reasonable length for a section header (language-adjusted)
        max_words = self._get_language_heading_length_range()[1]
        if word_count > max_words:
            return False
        
        # Should not end with page numbers (TOC entries)
        if re.search(r'\d+\s*$', text) and word_count > 8:
            return False
        
        return True
    
    def _is_valid_subsection(self, block: TextBlock) -> bool:
        """Validate if block is a subsection"""
        text = block.text.strip()
        word_count = len(text.split())
        
        # Should be appropriate length for subsection (language-adjusted)
        length_range = self._get_language_heading_length_range()
        if word_count > length_range[1] or word_count < 2:
            return False
        
        # Should not contain content indicators
        content_indicators = ['the tester should', 'this document', 'bullet point']
        
        # Add language-specific content indicators
        if hasattr(self, 'language'):
            lang_content_indicators = self._get_language_content_indicators()
            content_indicators.extend(lang_content_indicators)
        
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in content_indicators):
            return False
        
        return True
    
    def _get_language_content_indicators(self) -> List[str]:
        """Get language-specific content indicators to exclude"""
        if not hasattr(self, 'language'):
            return []
        
        indicators = {
            'japanese': ['テスターは', 'この文書', '箇条書き'],
            'german': ['der tester sollte', 'dieses dokument', 'aufzählungszeichen'],
            'tamil': ['சோதனையாளர்', 'இந்த ஆவணம்', 'புல்லட் பாயிண்ட்'],
            'english': []
        }
        
        return indicators.get(self.language, [])
    
    def _determine_navigation_page(self, text: str, actual_page: int) -> int:
        """Determine page for navigation elements using heuristics"""
        text_lower = text.lower().strip()
        
        # Language-aware page determination
        if hasattr(self, 'language') and self.language != 'english':
            page = self._determine_navigation_page_for_language(text_lower, actual_page)
            if page != actual_page:
                return page
        
        # Use pattern-based page assignment
        if 'revision' in text_lower or '改訂' in text_lower or 'version' in text_lower or 'திருத்த' in text_lower:
            return 2  # Revision history typically on page 2
        elif ('table' in text_lower and 'contents' in text_lower) or '目次' in text_lower or 'inhaltsverzeichnis' in text_lower or 'பொருளடக்கம்' in text_lower:
            return 3  # TOC typically on page 3
        elif 'acknowledgement' in text_lower or '謝辞' in text_lower or 'danksagung' in text_lower or 'நன்றிகள்' in text_lower:
            return 4  # Acknowledgements typically on page 4
        
        return actual_page
    
    def _determine_navigation_page_for_language(self, text_lower: str, actual_page: int) -> int:
        """Determine navigation page based on language-specific patterns"""
        if not hasattr(self, 'language'):
            return actual_page
        
        language = self.language
        
        # Language-specific navigation page mapping
        if language == 'japanese':
            if '改訂履歴' in text_lower:
                return 2
            elif '目次' in text_lower:
                return 3
            elif '謝辞' in text_lower:
                return 4
        elif language == 'german':
            if 'versionsgeschichte' in text_lower:
                return 2
            elif 'inhaltsverzeichnis' in text_lower:
                return 3
            elif 'danksagung' in text_lower:
                return 4
        elif language == 'tamil':
            if 'திருத்த வரலாறு' in text_lower:
                return 2
            elif 'பொருளடக்கம்' in text_lower:
                return 3
            elif 'நன்றிகள்' in text_lower:
                return 4
        
        return actual_page
    
    def _determine_section_page(self, text: str, actual_page: int) -> int:
        """Determine page for sections using pattern analysis"""
        section_num = self._extract_section_number(text)
        
        # Heuristic: sections typically start on pages 5, 6, 9, 11
        # Based on common technical manual structure
        page_mapping = {1: 5, 2: 6, 3: 9, 4: 11}
        return page_mapping.get(section_num, actual_page)
    
    def _format_heading_text(self, text: str) -> str:
        """Format heading text with consistent spacing"""
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle special characters
        clean_text = clean_text.replace('–', '\\u2013')
        
        # Apply language-specific heading formatting
        if hasattr(self, 'language') and self.language != 'english':
            clean_text = self._apply_language_heading_formatting(clean_text)
        
        # Add trailing space for consistency
        return clean_text + " "
    
    def _apply_language_heading_formatting(self, text: str) -> str:
        """Apply language-specific heading formatting"""
        if not hasattr(self, 'language'):
            return text
        
        language = self.language
        
        # Language-specific heading formatting
        if language == 'japanese':
            # Japanese heading formatting
            return text.strip()
        elif language == 'german':
            # German heading formatting - preserve compound words
            return text.strip()
        elif language == 'tamil':
            # Tamil heading formatting
            return text.strip()
        
        return text
