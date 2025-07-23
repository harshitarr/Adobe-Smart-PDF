import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from models.document_types import TextBlock, DocumentType, HeadingLevel
from core.title_extractor import TitleExtractor
from core.outline_extractor import OutlineExtractor

class TestTitleExtractor(unittest.TestCase):
    def setUp(self):
        self.title_extractor = TitleExtractor()
    
    def create_mock_text_block(self, text, page=0, font_size=12, font_name="Arial", is_bold=False):
        """Helper method to create mock text blocks"""
        return TextBlock(
            text=text,
            page=page,
            x0=0, y0=0, x1=100, y1=20,
            font_size=font_size,
            font_name=font_name,
            is_bold=is_bold
        )
    
    def test_extract_form_title(self):
        """Test title extraction for form document"""
        text_blocks = [
            self.create_mock_text_block("International Software Testing", font_size=10),
            self.create_mock_text_block("Application form for grant of LTC advance", font_size=16, is_bold=True),
            self.create_mock_text_block("1. Name of Employee:", font_size=12)
        ]
        
        title = self.title_extractor.extract_title(text_blocks, DocumentType.FORM_DOCUMENT)
        self.assertIn("Application form for grant of LTC advance", title)
    
    def test_extract_manual_title(self):
        """Test title extraction for technical manual"""
        text_blocks = [
            self.create_mock_text_block("International Software Testing Qualifications Board", font_size=10),
            self.create_mock_text_block("Overview Foundation Level Extensions", font_size=18, is_bold=True),
            self.create_mock_text_block("Version 2.0", font_size=10)
        ]
        
        title = self.title_extractor.extract_title(text_blocks, DocumentType.TECHNICAL_MANUAL)
        self.assertIn("Overview Foundation Level Extensions", title)
    
    def test_extract_business_title(self):
        """Test title extraction for business document"""
        text_blocks = [
            self.create_mock_text_block("RFP:Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library", font_size=16, is_bold=True),
            self.create_mock_text_block("Summary", font_size=14)
        ]
        
        title = self.title_extractor.extract_title(text_blocks, DocumentType.BUSINESS_DOCUMENT)
        self.assertIn("RFP:", title)
        self.assertIn("Ontario Digital Library", title)

class TestOutlineExtractor(unittest.TestCase):
    def setUp(self):
        self.outline_extractor = OutlineExtractor()
    
    def create_mock_text_block(self, text, page=0, font_size=12, font_name="Arial", is_bold=False):
        """Helper method to create mock text blocks"""
        return TextBlock(
            text=text,
            page=page,
            x0=0, y0=0, x1=100, y1=20,
            font_size=font_size,
            font_name=font_name,
            is_bold=is_bold
        )
    
    def test_extract_manual_outline(self):
        """Test outline extraction for technical manual"""
        text_blocks = [
            self.create_mock_text_block("Overview Foundation Level Extensions", font_size=18),
            self.create_mock_text_block("Revision History", font_size=14, is_bold=True),
            self.create_mock_text_block("Table of Contents", font_size=14, is_bold=True, page=3),
            self.create_mock_text_block("1. Introduction to the Foundation Level Extensions", font_size=14, is_bold=True, page=5),
            self.create_mock_text_block("2.1 Intended Audience", font_size=12, is_bold=True, page=6)
        ]
        
        outline = self.outline_extractor.extract_outline(text_blocks, DocumentType.TECHNICAL_MANUAL)
        
        # Should have headings but not title
        self.assertGreater(len(outline), 0)
        
        # Check for expected headings
        heading_texts = [h.text for h in outline]
        self.assertIn("Revision History", heading_texts)
        self.assertIn("Table of Contents", heading_texts)
    
    def test_extract_form_outline(self):
        """Test outline extraction for form document (should be empty)"""
        text_blocks = [
            self.create_mock_text_block("Application form for grant of LTC advance", font_size=16),
            self.create_mock_text_block("1. Name of Employee:", font_size=12),
            self.create_mock_text_block("Signature:", font_size=12)
        ]
        
        outline = self.outline_extractor.extract_outline(text_blocks, DocumentType.FORM_DOCUMENT)
        self.assertEqual(len(outline), 0)  # Forms should have no outline

if __name__ == '__main__':
    unittest.main()
