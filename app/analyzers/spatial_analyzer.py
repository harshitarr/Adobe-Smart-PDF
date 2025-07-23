import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from models.document_types import TextBlock

logger = logging.getLogger('extraction')

class SpatialAnalyzer:
    def __init__(self):
        self.page_margins = {}
        self.text_alignment = {}
    
    def analyze_layout(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze document layout and spatial patterns"""
        if not text_blocks:
            return {}
        
        # Group blocks by page
        pages = {}
        for block in text_blocks:
            if block.page not in pages:
                pages[block.page] = []
            pages[block.page].append(block)
        
        # Analyze each page
        page_analysis = {}
        for page_num, blocks in pages.items():
            page_analysis[page_num] = self._analyze_page_layout(blocks)
        
        # Overall analysis
        analysis = {
            'pages': page_analysis,
            'total_pages': len(pages),
            'total_blocks': len(text_blocks)
        }
        
        logger.info(f"Spatial analysis completed for {len(pages)} pages")
        return analysis
    
    def _analyze_page_layout(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze layout for a single page"""
        if not blocks:
            return {}
        
        # Calculate page boundaries
        x_coords = []
        y_coords = []
        
        for block in blocks:
            x_coords.extend([block.x0, block.x1])
            y_coords.extend([block.y0, block.y1])
        
        page_bounds = {
            'left': min(x_coords),
            'right': max(x_coords),
            'top': max(y_coords),  # PDF coordinates are bottom-up
            'bottom': min(y_coords),
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords)
        }
        
        # Analyze text alignment
        alignment_analysis = self._analyze_alignment(blocks, page_bounds)
        
        # Detect standalone text (likely headings)
        standalone_blocks = self._detect_standalone_text(blocks)
        
        return {
            'bounds': page_bounds,
            'alignment': alignment_analysis,
            'standalone_blocks': len(standalone_blocks),
            'total_blocks': len(blocks)
        }
    
    def _analyze_alignment(self, blocks: List[TextBlock], page_bounds: Dict[str, float]) -> Dict[str, Any]:
        """Analyze text alignment patterns"""
        left_margin = page_bounds['left']
        center_x = left_margin + page_bounds['width'] / 2
        
        alignment_counts = {'left': 0, 'center': 0, 'right': 0}
        
        for block in blocks:
            block_center = (block.x0 + block.x1) / 2
            
            # Determine alignment
            if abs(block.x0 - left_margin) < 10:  # Within 10 units of left margin
                alignment_counts['left'] += 1
            elif abs(block_center - center_x) < 20:  # Within 20 units of center
                alignment_counts['center'] += 1
            else:
                alignment_counts['right'] += 1
        
        total_blocks = len(blocks)
        alignment_percentages = {
            align: (count / total_blocks) * 100 if total_blocks > 0 else 0
            for align, count in alignment_counts.items()
        }
        
        return {
            'counts': alignment_counts,
            'percentages': alignment_percentages,
            'dominant_alignment': max(alignment_counts, key=alignment_counts.get)
        }
    
    def _detect_standalone_text(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Detect text blocks that are standalone (likely headings)"""
        standalone_blocks = []
        
        # Sort blocks by y-coordinate (top to bottom)
        sorted_blocks = sorted(blocks, key=lambda b: -b.y0)  # Negative for top-to-bottom
        
        for i, block in enumerate(sorted_blocks):
            is_standalone = True
            
            # Check if there are other blocks on the same line
            for j, other_block in enumerate(sorted_blocks):
                if i != j and self._are_on_same_line(block, other_block):
                    is_standalone = False
                    break
            
            if is_standalone and len(block.text.split()) <= 10:  # Short text likely to be heading
                standalone_blocks.append(block)
        
        return standalone_blocks
    
    def _are_on_same_line(self, block1: TextBlock, block2: TextBlock, tolerance: float = 5.0) -> bool:
        """Check if two text blocks are on the same line"""
        # Check if y-coordinates overlap within tolerance
        return (abs(block1.y0 - block2.y0) <= tolerance or 
                abs(block1.y1 - block2.y1) <= tolerance or
                (block1.y0 <= block2.y1 and block2.y0 <= block1.y1))
    
    def is_likely_heading_position(self, block: TextBlock, page_analysis: Dict[str, Any]) -> float:
        """Determine if block position suggests it's a heading"""
        if not page_analysis:
            return 0.5
        
        score = 0.0
        
        # Check if it's standalone
        if page_analysis.get('standalone_blocks', 0) > 0:
            score += 0.4
        
        # Check alignment (center or left-aligned headings are common)
        alignment = page_analysis.get('alignment', {})
        if alignment.get('dominant_alignment') in ['left', 'center']:
            score += 0.3
        
        # Check if it's at the top of the page (y-coordinate is high)
        bounds = page_analysis.get('bounds', {})
        if bounds and block.y0 >= bounds.get('top', 0) * 0.8:
            score += 0.3
        
        return min(1.0, score)
