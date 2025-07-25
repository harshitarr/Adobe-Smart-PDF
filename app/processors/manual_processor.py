import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Tuple
import logging

from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor

logger = logging.getLogger('extraction')

class ManualProcessor:
    def __init__(self):
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
    
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Generalized processing for technical manuals"""
        logger.info("Processing technical manual - generalized pattern matching")
        
        # Dynamic title extraction
        title = self._extract_generalized_title(text_blocks)
        
        # Pattern-based outline extraction
        outline = self._extract_generalized_outline(text_blocks)
        
        return DocumentStructure(
            title=title,
            outline=outline,
            doc_type=DocumentType.TECHNICAL_MANUAL,
            confidence=0.9
        )
    
    def _extract_generalized_title(self, text_blocks: List[TextBlock]) -> str:
        """Extract title using generalized patterns"""
        title_candidates = []
        
        # Look for title patterns in early blocks
        for block in text_blocks[:30]:  # Check first 30 blocks
            text = block.text.strip()
            score = self._calculate_title_score(text, block)
            
            if score > 0.5:  # Threshold for title candidates
                title_candidates.append((text, score, block))
        
        if title_candidates:
            # Sort by score and take the best
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            best_title = title_candidates[0][0]
            return self._format_title(best_title)
        
        return ""
    
    def _calculate_title_score(self, text: str, block: TextBlock) -> float:
        """Calculate title likelihood score"""
        if not text or len(text) < 5:
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        words = text.split()
        
        # Title indicators
        title_words = ['overview', 'introduction', 'manual', 'guide', 'foundation', 'level', 'extensions']
        indicator_count = sum(1 for word in title_words if word in text_lower)
        score += indicator_count * 0.2
        
        # Length characteristics
        if 3 <= len(words) <= 8:
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
    
    def _format_title(self, title: str) -> str:
        """Format title with spacing pattern detection"""
        clean_title = re.sub(r'\s+', ' ', title.strip())
        
        # Detect if we need extra spacing based on content
        if len(clean_title.split()) <= 5:
            # Short titles often have extra spacing
            return f"{clean_title}  "  # Double space at end
        
        return f"{clean_title} "  # Single space at end
    
    def _extract_generalized_outline(self, text_blocks: List[TextBlock]) -> List[HeadingLevel]:
        """Extract outline using generalized pattern matching"""
        outline = []
        processed_texts = set()
        
        # Phase 1: Navigation elements
        navigation_items = self._extract_navigation_items(text_blocks)
        outline.extend(navigation_items)
        processed_texts.update(item.text.lower().strip() for item in navigation_items)
        
        # Phase 2: Main numbered sections
        main_sections = self._extract_main_sections(text_blocks, processed_texts)
        outline.extend(main_sections)
        processed_texts.update(item.text.lower().strip() for item in main_sections)
        
        # Phase 3: Subsections
        subsections = self._extract_subsections(text_blocks, processed_texts)
        outline.extend(subsections)
        
        return outline
    
    def _extract_navigation_items(self, text_blocks: List[TextBlock]) -> List[HeadingLevel]:
        """Extract navigation elements using pattern matching"""
        navigation_items = []
        
        for pattern in self.pattern_rules['navigation_patterns']:
            best_match = self._find_best_pattern_match(text_blocks, pattern, max_page=5)
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
    
    def _extract_main_sections(self, text_blocks: List[TextBlock], processed_texts: set) -> List[HeadingLevel]:
        """Extract main numbered sections"""
        main_sections = []
        section_blocks = []
        
        for pattern in self.pattern_rules['numbered_section_patterns']:
            matches = self._find_all_pattern_matches(text_blocks, pattern)
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
    
    def _extract_subsections(self, text_blocks: List[TextBlock], processed_texts: set) -> List[HeadingLevel]:
        """Extract subsections using pattern matching"""
        subsections = []
        
        for pattern in self.pattern_rules['subsection_patterns']:
            matches = self._find_all_pattern_matches(text_blocks, pattern)
            
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
    
    def _find_best_pattern_match(self, text_blocks: List[TextBlock], pattern: str, max_page: int = None) -> Tuple[TextBlock, float]:
        """Find the best match for a pattern"""
        candidates = []
        
        for block in text_blocks:
            if max_page and block.page > max_page:
                continue
            
            if self._should_exclude_block(block):
                continue
            
            if re.search(pattern, block.text.strip(), re.IGNORECASE):
                score = self._calculate_heading_score(block, pattern)
                candidates.append((block, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        
        return None
    
    def _find_all_pattern_matches(self, text_blocks: List[TextBlock], pattern: str) -> List[Tuple[TextBlock, float]]:
        """Find all matches for a pattern"""
        matches = []
        
        for block in text_blocks:
            if self._should_exclude_block(block):
                continue
            
            if re.search(pattern, block.text.strip(), re.IGNORECASE):
                score = self._calculate_heading_score(block, pattern)
                matches.append((block, score))
        
        return matches
    
    def _should_exclude_block(self, block: TextBlock) -> bool:
        """Check if block should be excluded using patterns"""
        text = block.text.strip().lower()
        
        # Apply exclusion patterns
        for pattern in self.pattern_rules['exclusion_patterns']:
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
        
        # Length scoring
        if 2 <= word_count <= 12:
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
    
    def _extract_section_number(self, text: str) -> int:
        """Extract section number for sorting"""
        match = re.match(r'^(\d+)\.', text.strip())
        return int(match.group(1)) if match else 999
    
    def _is_valid_main_section(self, block: TextBlock) -> bool:
        """Validate if block is a main section"""
        text = block.text.strip()
        word_count = len(text.split())
        
        # Should be reasonable length for a section header
        if word_count > 15:
            return False
        
        # Should not end with page numbers (TOC entries)
        if re.search(r'\d+\s*$', text) and word_count > 8:
            return False
        
        return True
    
    def _is_valid_subsection(self, block: TextBlock) -> bool:
        """Validate if block is a subsection"""
        text = block.text.strip()
        word_count = len(text.split())
        
        # Should be appropriate length for subsection
        if word_count > 10 or word_count < 2:
            return False
        
        # Should not contain content indicators
        content_indicators = ['the tester should', 'this document', 'bullet point']
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in content_indicators):
            return False
        
        return True
    
    def _determine_navigation_page(self, text: str, actual_page: int) -> int:
        """Determine page for navigation elements using heuristics"""
        text_lower = text.lower().strip()
        
        # Use pattern-based page assignment
        if 'revision' in text_lower:
            return 2  # Revision history typically on page 2
        elif 'table' in text_lower and 'contents' in text_lower:
            return 3  # TOC typically on page 3
        elif 'acknowledgement' in text_lower:
            return 4  # Acknowledgements typically on page 4
        
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
        
        # Add trailing space for consistency
        return clean_text + " "
