import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from models.document_types import TextBlock, DocumentType
from processors.form_processor import FormProcessor
from processors.manual_processor import ManualProcessor
from processors.business_processor import BusinessProcessor

class TestProcessors(unittest.TestCase):
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
    
    def test_form_processor(self):
        """Test form processor"""
        processor = FormProcessor()
        text_blocks = [
            self.create_mock_text_block("Application form for grant of LTC advance", font_size=16, is_bold=True),
            self.create_mock_text_block("1. Name of Employee:", font_size=12),
            self.create_mock_text_block("Signature:", font_size=12),
            self.create_mock_text_block("Date:", font_size=12)
        ]
        
        result = processor.process(text_blocks)
        
        self.assertEqual(result.doc_type, DocumentType.FORM_DOCUMENT)
        self.assertIn("Application form for grant of LTC advance", result.title)
        self.assertEqual(len(result.outline), 0)  # Forms have no outline
    
    def test_manual_processor(self):
        """Test manual processor"""
        processor = ManualProcessor()
        text_blocks = [
            self.create_mock_text_block("Overview Foundation Level Extensions", font_size=18, is_bold=True),
            self.create_mock_text_block("Revision History", font_size=14, is_bold=True, page=2),
            self.create_mock_text_block("Table of Contents", font_size=14, is_bold=True, page=3),
            self.create_mock_text_block("1. Introduction to the Foundation Level Extensions", font_size=14, is_bold=True, page=5),
            self.create_mock_text_block("2.1 Intended Audience", font_size=12, is_bold=True, page=6)
        ]
        
        result = processor.process(text_blocks)
        
        self.assertEqual(result.doc_type, DocumentType.TECHNICAL_MANUAL)
        self.assertIn("Overview Foundation Level Extensions", result.title)
        self.assertGreater(len(result.outline), 0)  # Should have outline items
    
    def test_business_processor(self):
        """Test business processor"""
        processor = BusinessProcessor()
        text_blocks = [
            self.create_mock_text_block("RFP:Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library", font_size=16, is_bold=True),
            self.create_mock_text_block("Ontario's Digital Library", font_size=14, is_bold=True, page=1),
            self.create_mock_text_block("Summary", font_size=12, is_bold=True, page=1),
            self.create_mock_text_block("Background", font_size=12, is_bold=True, page=2),
            self.create_mock_text_block("Timeline:", font_size=11, is_bold=True, page=1)
        ]
        
        result = processor.process(text_blocks)
        
        self.assertEqual(result.doc_type, DocumentType.BUSINESS_DOCUMENT)
        self.assertIn("RFP:", result.title)
        self.assertGreater(len(result.outline), 0)  # Should have outline items

if __name__ == '__main__':
    unittest.main()
