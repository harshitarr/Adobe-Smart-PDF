import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any
from collections import Counter
import logging

from models.document_types import TextBlock

logger = logging.getLogger('extraction')

class FontAnalyzer:
    def __init__(self):
        self.font_clusters = {}
        self.size_clusters = {}
    
    def analyze_fonts(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze font patterns in document"""
        if not text_blocks:
            return {}
        
        # Extract font information
        font_sizes = [block.font_size for block in text_blocks if block.font_size > 0]
        font_names = [block.font_name for block in text_blocks if block.font_name]
        
        # Analyze font sizes
        size_analysis = self._analyze_font_sizes(font_sizes)
        
        # Analyze font names
        name_analysis = self._analyze_font_names(font_names)
        
        # Combine analysis
        analysis = {
            'size_analysis': size_analysis,
            'name_analysis': name_analysis,
            'total_blocks': len(text_blocks)
        }
        
        logger.info(f"Font analysis completed: {len(font_sizes)} size samples, {len(set(font_names))} unique fonts")
        return analysis
    
    def _analyze_font_sizes(self, font_sizes: List[float]) -> Dict[str, Any]:
        """Analyze font size distribution"""
        if not font_sizes:
            return {}
        
        sizes = np.array(font_sizes)
        
        # Basic statistics
        stats = {
            'min_size': float(np.min(sizes)),
            'max_size': float(np.max(sizes)),
            'mean_size': float(np.mean(sizes)),
            'median_size': float(np.median(sizes)),
            'std_size': float(np.std(sizes))
        }
        
        # Size clustering for hierarchy detection
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        if len(unique_sizes) > 1:
            # Assign hierarchy levels based on size
            size_hierarchy = {}
            
            # Largest sizes are likely titles/H1
            if len(unique_sizes) >= 1:
                size_hierarchy['title'] = unique_sizes[0]
            if len(unique_sizes) >= 2:
                size_hierarchy['H1'] = unique_sizes[0:2]
            if len(unique_sizes) >= 3:
                size_hierarchy['H2'] = unique_sizes[1:3]
            if len(unique_sizes) >= 4:
                size_hierarchy['H3'] = unique_sizes[2:4]
            if len(unique_sizes) >= 5:
                size_hierarchy['H4'] = unique_sizes[3:5]
            
            stats['hierarchy'] = size_hierarchy
        
        stats['unique_sizes'] = unique_sizes
        return stats
    
    def _analyze_font_names(self, font_names: List[str]) -> Dict[str, Any]:
        """Analyze font name distribution"""
        if not font_names:
            return {}
        
        # Count font frequencies
        font_counts = Counter(font_names)
        
        # Identify dominant fonts
        total_count = len(font_names)
        font_analysis = {}
        
        for font_name, count in font_counts.most_common():
            percentage = (count / total_count) * 100
            font_analysis[font_name] = {
                'count': count,
                'percentage': percentage,
                'is_bold': 'bold' in font_name.lower() or 'black' in font_name.lower(),
                'is_italic': 'italic' in font_name.lower()
            }
        
        return {
            'fonts': font_analysis,
            'dominant_font': font_counts.most_common(1)[0][0] if font_counts else None,
            'unique_fonts': len(font_counts)
        }
    
    def classify_text_block_importance(self, block: TextBlock, analysis: Dict[str, Any]) -> float:
        """Classify importance of text block based on font analysis"""
        if not analysis or 'size_analysis' not in analysis:
            return 0.5
        
        size_stats = analysis['size_analysis']
        
        # Size-based importance
        size_importance = 0.0
        if block.font_size >= size_stats.get('max_size', 0) * 0.9:
            size_importance = 1.0  # Likely title or H1
        elif block.font_size >= size_stats.get('mean_size', 0) * 1.2:
            size_importance = 0.8  # Likely H2
        elif block.font_size >= size_stats.get('mean_size', 0):
            size_importance = 0.6  # Likely H3
        else:
            size_importance = 0.3  # Likely content text
        
        # Font style importance
        style_importance = 0.0
        if block.is_bold:
            style_importance += 0.3
        if block.font_name and ('heading' in block.font_name.lower() or 'title' in block.font_name.lower()):
            style_importance += 0.2
        
        return min(1.0, size_importance + style_importance)
