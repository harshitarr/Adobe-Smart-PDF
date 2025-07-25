import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
from typing import List, Dict, Any, Tuple, Optional
import logging

from models.document_types import TextBlock, DocumentType, DocumentStructure, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor

logger = logging.getLogger('extraction')

class ProgramProcessor:
    def __init__(self):
        self.title_extractor = TitleExtractor()
        self.outline_extractor = OutlineExtractor()
        
    def process(self, text_blocks: List[TextBlock]) -> DocumentStructure:
        """Process document using hierarchical context analysis"""
        logger.info("Processing document with context-aware hierarchy analysis")
        
        # Analyze document structural context
        context_analysis = self._analyze_document_context(text_blocks)
        
        # Extract title
        title = self._extract_title_contextual(text_blocks, context_analysis)
        
        # Extract outline with hierarchical context awareness
        outline = self._extract_hierarchical_outline(text_blocks, context_analysis)
        
        return DocumentStructure(
            title=title,
            outline=outline,
            doc_type=DocumentType.PROGRAM_DOCUMENT,
            confidence=0.9
        )
    
    def _analyze_document_context(self, text_blocks: List[TextBlock]) -> Dict:
        """Analyze document context for hierarchy detection"""
        context = {
            'content_sections': [],
            'potential_headers': [],
            'header_groups': {},
            'formatting_patterns': {'bold_blocks': [], 'upper_blocks': []},
            'content_flow': []
        }
        
        # Identify content sections and potential headers
        for i, block in enumerate(text_blocks):
            text = block.text.strip()
            if len(text) < 2:
                continue
                
            block_analysis = {
                'index': i,
                'text': text,
                'word_count': len(text.split()),
                'is_bold': block.is_bold,
                'is_upper': text.isupper(),
                'font_size': getattr(block, 'font_size', 12),
                'page': block.page
            }
            
            # Categorize content
            if self._is_descriptive_section(text):
                context['content_sections'].append(block_analysis)
            elif self._has_header_characteristics(text, block):
                context['potential_headers'].append(block_analysis)
                
                # Group headers by formatting similarity
                format_key = f"bold:{block.is_bold}_upper:{text.isupper()}"
                if format_key not in context['header_groups']:
                    context['header_groups'][format_key] = []
                context['header_groups'][format_key].append(block_analysis)
            
            context['content_flow'].append(block_analysis)
        
        return context
    
    def _extract_title_contextual(self, text_blocks: List[TextBlock], context_analysis: Dict) -> str:
        """Extract title using contextual position analysis"""
        # Look in document opening
        for block in text_blocks[:5]:
            text = block.text.strip()
            if len(text) < 3:
                continue
                
            if self._is_document_title_pattern(text, block, context_analysis):
                return self._clean_text(text)
        
        # Fallback to first substantial non-descriptive text
        for block in text_blocks[:8]:
            text = block.text.strip()
            if 8 <= len(text) <= 80 and not self._is_descriptive_section(text):
                return self._clean_text(text)
                
        return "Document Title"
    
    def _is_document_title_pattern(self, text: str, block: TextBlock, context: Dict) -> bool:
        """Identify document title using contextual patterns"""
        word_count = len(text.split())
        
        title_indicators = [
            3 <= word_count <= 12,  # Appropriate title length
            not self._is_descriptive_section(text),
            not text.startswith(('*', '-', '•')),
            block.page == 0,  # First page
            not self._is_list_item(text)
        ]
        
        return sum(title_indicators) >= 4
    
    def _extract_hierarchical_outline(self, text_blocks: List[TextBlock], context_analysis: Dict) -> List[HeadingLevel]:
        """Extract outline using hierarchical context analysis"""
        hierarchy_candidates = []
        
        # Analyze each potential header in context
        for header in context_analysis['potential_headers']:
            hierarchy_analysis = self._analyze_hierarchical_context(header, context_analysis, text_blocks)
            
            if hierarchy_analysis and hierarchy_analysis['hierarchical_significance'] > 0.6:
                hierarchy_candidates.append(hierarchy_analysis)
        
        # Select based on hierarchical importance
        return self._select_hierarchical_headers(hierarchy_candidates)
    
    def _analyze_hierarchical_context(self, header: Dict, context: Dict, all_blocks: List[TextBlock]) -> Optional[Dict]:
        """Analyze header's hierarchical significance in document context"""
        text = header['text']
        position = header['index']
        
        # Calculate hierarchical significance
        hierarchical_score = 0.0
        
        # 1. Context introducer analysis (key insight)
        introducer_score = self._calculate_introducer_significance(text, position, all_blocks)
        hierarchical_score += introducer_score * 0.5
        
        # 2. Format significance
        format_score = self._calculate_format_significance(header, context)
        hierarchical_score += format_score * 0.3
        
        # 3. Position significance
        position_score = self._calculate_hierarchical_position_score(position, len(all_blocks))
        hierarchical_score += position_score * 0.2
        
        if hierarchical_score < 0.6:
            return None
        
        # Determine hierarchical role
        hierarchical_role = self._determine_hierarchical_role(text, hierarchical_score, introducer_score)
        
        return {
            'header': header,
            'hierarchical_significance': hierarchical_score,
            'hierarchical_role': hierarchical_role,
            'introducer_score': introducer_score
        }
    
    def _calculate_introducer_significance(self, text: str, position: int, all_blocks: List[TextBlock]) -> float:
        """Calculate significance as content section introducer (CRITICAL for accuracy)"""
        if position >= len(all_blocks) - 1:
            return 0.0
        
        score = 0.0
        
        # Analyze what follows this text
        following_blocks = all_blocks[position + 1:position + 8]  # Look ahead 7 blocks
        
        similar_formatted_items = 0
        content_items_count = 0
        
        for block in following_blocks:
            following_text = block.text.strip()
            if len(following_text) < 2:
                continue
            
            # Count structured content items that could be under this header
            if self._is_structured_content_item(following_text):
                content_items_count += 1
                
                # Check if it has similar formatting to current text (indicates sub-items)
                if self._has_similar_formatting(text, following_text, block):
                    similar_formatted_items += 1
        
        # Strong introducer pattern: followed by multiple structured items
        if content_items_count >= 2:
            score += 0.8
            
            # Bonus if introducing similarly formatted items (like pathway types)
            if similar_formatted_items >= 2:
                score += 0.2
        elif content_items_count >= 1:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_format_significance(self, header: Dict, context: Dict) -> float:
        """Calculate format-based significance"""
        score = 0.0
        
        # ALL CAPS significance
        if header['is_upper']:
            if header['word_count'] <= 4:
                score += 0.8
            elif header['word_count'] <= 6:
                score += 0.6
            else:
                score += 0.3
        
        # Bold significance
        if header['is_bold']:
            score += 0.5
        
        # Font size significance
        if header['font_size'] > 12:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_hierarchical_position_score(self, position: int, total_blocks: int) -> float:
        """Calculate position-based hierarchical significance"""
        relative_pos = position / total_blocks
        
        # Middle sections often contain main structural divisions
        if 0.2 <= relative_pos <= 0.8:  # Middle 60%
            return 1.0
        elif 0.1 <= relative_pos <= 0.9:  # Broader middle
            return 0.7
        else:
            return 0.3
    
    def _determine_hierarchical_role(self, text: str, hierarchical_score: float, introducer_score: float) -> str:
        """Determine hierarchical role in document structure"""
        word_count = len(text.split())
        
        # Main section introducer (like "PATHWAY OPTIONS")
        if introducer_score > 0.6 and hierarchical_score > 0.8:
            return 'main_section_introducer'
        
        # Content item (like "REGULAR PATHWAY", "DISTINCTION PATHWAY")
        elif introducer_score < 0.3 and hierarchical_score > 0.6:
            return 'content_item'
        
        # Regular section header
        elif hierarchical_score > 0.7:
            return 'section_header'
        
        else:
            return 'minor_heading'
    
    def _select_hierarchical_headers(self, candidates: List[Dict]) -> List[HeadingLevel]:
        """Select headers based on hierarchical importance with improved logic"""
        if not candidates:
            return []
        
        # Sort by hierarchical role priority and significance
        role_priority = {
            'main_section_introducer': 1,
            'section_header': 2,
            'content_item': 3,
            'minor_heading': 4
        }
        
        candidates.sort(key=lambda x: (
            role_priority.get(x['hierarchical_role'], 5),
            -x['hierarchical_significance']
        ))
        
        outline = []
        used_texts = set()
        
        # Enhanced selection logic
        for candidate in candidates:
            text_key = candidate['header']['text'].lower().strip()
            
            # Avoid duplicates
            if text_key in used_texts:
                continue
            
            role = candidate['hierarchical_role']
            significance = candidate['hierarchical_significance']
            
            # Prioritize main section introducers
            if role == 'main_section_introducer' and significance > 0.75:
                outline.append(self._create_heading_level(candidate, 'H1'))
                used_texts.add(text_key)
                break  # Take the best main section introducer
        
        # If no main introducer found, consider best section header
        if not outline:
            for candidate in candidates:
                text_key = candidate['header']['text'].lower().strip()
                if text_key in used_texts:
                    continue
                    
                if candidate['hierarchical_role'] == 'section_header' and candidate['hierarchical_significance'] > 0.85:
                    outline.append(self._create_heading_level(candidate, 'H1'))
                    used_texts.add(text_key)
                    break
        
        # Last resort: highest scoring candidate if none found
        if not outline and candidates:
            best_candidate = candidates[0]
            if best_candidate['hierarchical_significance'] > 0.9:
                outline.append(self._create_heading_level(best_candidate, 'H1'))
        
        return outline
    
    def _create_heading_level(self, candidate: Dict, level: str) -> HeadingLevel:
        """Create HeadingLevel from candidate"""
        header = candidate['header']
        return HeadingLevel(
            level=level,
            text=self._clean_text(header['text']),
            page=header['page'],
            confidence=candidate['hierarchical_significance'],
            font_size=header.get('font_size'),
            font_name=None
        )
    
    def _has_header_characteristics(self, text: str, block: TextBlock) -> bool:
        """Check if text has header characteristics"""
        word_count = len(text.split())
        
        characteristics = [
            word_count <= 10,  # Reasonable header length
            not self._is_descriptive_section(text),
            not text.startswith(('*', '-', '•')),
            text.isupper() or text.istitle() or block.is_bold
        ]
        
        return sum(characteristics) >= 3
    
    def _is_structured_content_item(self, text: str) -> bool:
        """Identify structured content items (not section introducers)"""
        text_lower = text.lower()
        
        structured_patterns = [
            text.startswith('*') or text.startswith('-'),  # Bullet points
            'credits of' in text_lower,
            'maintain' in text_lower and ('gpa' in text_lower or 'overall' in text_lower),
            'participate' in text_lower,
            'join and' in text_lower,
            re.search(r'at least.*course', text_lower),
            'either participate' in text_lower,
            text_lower.endswith('pathway') and len(text.split()) <= 6,  # Specific pathway types
            # Pattern-based detection for pathway items
            self._is_pathway_type_item(text),
            # Requirement items
            self._is_requirement_item(text_lower)
        ]
        
        return any(structured_patterns)
    
    def _is_pathway_type_item(self, text: str) -> bool:
        """Detect pathway-type items without hardcoding"""
        text_upper = text.upper()
        word_count = len(text.split())
        
        # Pattern-based detection for items like "REGULAR PATHWAY", "DISTINCTION PATHWAY"
        pathway_patterns = [
            text_upper.endswith('PATHWAY') and word_count <= 4,
            text_upper.endswith('TRACK') and word_count <= 4,
            text_upper.endswith('OPTION') and word_count <= 4,
            text_upper.endswith('PROGRAM') and word_count <= 4,
            # Pattern: [ADJECTIVE] + [TYPE] (2 words, all caps)
            word_count == 2 and text.isupper() and not self._is_section_introducer_pattern(text)
        ]
        
        return any(pathway_patterns)
    
    def _is_section_introducer_pattern(self, text: str) -> bool:
        """Detect section introducer patterns"""
        text_upper = text.upper()
        
        # Section introducer patterns (like "PATHWAY OPTIONS")
        introducer_patterns = [
            'OPTIONS' in text_upper,
            'REQUIREMENTS' in text_upper,
            'OVERVIEW' in text_upper,
            'SUMMARY' in text_upper,
            'DETAILS' in text_upper
        ]
        
        return any(introducer_patterns)
    
    def _is_requirement_item(self, text_lower: str) -> bool:
        """Detect requirement items without hardcoding"""
        requirement_patterns = [
            'credits of' in text_lower,
            'maintain' in text_lower and ('gpa' in text_lower or 'overall' in text_lower),
            'participate in' in text_lower and ('minimum' in text_lower or 'required' in text_lower),
            'at least one' in text_lower,
            'either participate' in text_lower,
            re.search(r'\d+\s+credits', text_lower),
            'attendance' in text_lower and 'year' in text_lower
        ]
        
        return any(requirement_patterns)
    
    def _has_similar_formatting(self, header_text: str, content_text: str, content_block: TextBlock) -> bool:
        """Check if content has similar formatting to header (indicates sub-items)"""
        return (content_text.isupper() and header_text.isupper() and 
                content_block.is_bold and len(content_text.split()) <= 8)
    
    def _is_descriptive_section(self, text: str) -> bool:
        """Identify descriptive content sections"""
        text_lower = text.lower()
        
        descriptive_patterns = [
            text_lower.startswith('mission'),
            text_lower.startswith('goals'),
            text_lower.startswith('objectives'),
            'statement:' in text_lower,
            'to provide' in text_lower,
            'inspire' in text_lower,
            'expose students' in text_lower,
            'encourage students' in text_lower,
            len(text.split()) > 15  # Very long descriptive text
        ]
        
        return any(descriptive_patterns)
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item"""
        return text.startswith(('*', '-', '•')) or text.lower().startswith(('goals:', 'objectives:'))
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        return re.sub(r'\s+', ' ', text.strip())
