import PyPDF2
import pdfplumber
from typing import List, Dict, Any
import re
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.document_types import TextBlock

logger = logging.getLogger('extraction')

class PDFParser:
    def __init__(self):
        self.pdf_reader = None
        self.pdf_plumber = None
    
    def load_pdf(self, file_path: str) -> bool:
        """Load PDF file using both PyPDF2 and pdfplumber"""
        try:
            with open(file_path, 'rb') as file:
                self.pdf_reader = PyPDF2.PdfReader(file)
                self.pdf_plumber = pdfplumber.open(file_path)
            logger.info(f"PDF loaded successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {str(e)}")
            return False
    
    def extract_text_blocks(self) -> List[TextBlock]:
        """Extract text blocks with formatting information"""
        text_blocks = []
        
        try:
            for page_num, page in enumerate(self.pdf_plumber.pages):
                # Extract text with formatting
                chars = page.chars
                
                # Group characters into text blocks
                current_block = []
                current_font = None
                current_size = None
                
                for char in chars:
                    font_name = char.get('fontname', '')
                    font_size = char.get('size', 0)
                    
                    if (font_name != current_font or font_size != current_size) and current_block:
                        # Create text block from current characters
                        text = ''.join([c['text'] for c in current_block])
                        if text.strip():
                            block = self._create_text_block(current_block, page_num, text, current_font, current_size)
                            if block:
                                text_blocks.append(block)
                        current_block = []
                    
                    current_font = font_name
                    current_size = font_size
                    current_block.append(char)
                
                # Handle remaining characters
                if current_block:
                    text = ''.join([c['text'] for c in current_block])
                    if text.strip():
                        block = self._create_text_block(current_block, page_num, text, current_font, current_size)
                        if block:
                            text_blocks.append(block)
            
            logger.info(f"Extracted {len(text_blocks)} text blocks")
            return text_blocks
            
        except Exception as e:
            logger.error(f"Failed to extract text blocks: {str(e)}")
            return []
    
    def _create_text_block(self, chars: List[Dict], page_num: int, text: str, font_name: str, font_size: float) -> TextBlock:
        """Create TextBlock from character information"""
        if not chars:
            return None
        
        # Calculate bounding box
        x0 = min(c['x0'] for c in chars)
        y0 = min(c['y0'] for c in chars)
        x1 = max(c['x1'] for c in chars)
        y1 = max(c['y1'] for c in chars)
        
        # Detect formatting
        is_bold = 'bold' in font_name.lower() or 'black' in font_name.lower()
        is_italic = 'italic' in font_name.lower() or 'oblique' in font_name.lower()
        
        return TextBlock(
            text=text.strip(),
            page=page_num,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            font_size=font_size,
            font_name=font_name,
            is_bold=is_bold,
            is_italic=is_italic
        )
    
    def get_page_count(self) -> int:
        """Get total number of pages"""
        if self.pdf_reader:
            return len(self.pdf_reader.pages)
        return 0
    
    def close(self):
        """Close PDF resources"""
        if self.pdf_plumber:
            self.pdf_plumber.close()
