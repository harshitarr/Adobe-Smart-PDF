import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add app directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from models.document_types import TextBlock, DocumentType
from core.document_classifier import DocumentClassifier

class TestDocumentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = DocumentClassifier()
    
    def create_mock_text_block(self, text, page=0, font_size=12, font_name="Arial"):
        """Helper method to create mock text blocks"""
        return TextBlock(
            text=text,
            page=page,
            x0=0, y0=0, x1=100, y1=20,
            font_size=font_size,
            font_name=font_name
        )
    
    def test_classify_form_document(self):
        """Test classification of form document"""
        text_blocks = [
            self.create_mock_text_block("Application form for grant of LTC advance"),
            self.create_mock_text_block("1. Name of Employee:"),
            self.create_mock_text_block("Signature:"),
            self.create_mock_text_block("Date:")
        ]
        
        doc_type, confidence = self.classifier.classify_document(text_blocks)
        self.assertEqual(doc_type, DocumentType.FORM_DOCUMENT)
        self.assertGreater(confidence, 0.3)
    
    def test_classify_technical_manual(self):
        """Test classification of technical manual"""
        text_blocks = [
            self.create_mock_text_block("Overview Foundation Level Extensions"),
            self.create_mock_text_block("Table of Contents"),
            self.create_mock_text_block("1. Introduction to the Foundation Level Extensions"),
            self.create_mock_text_block("2.1 Intended Audience")
        ]
        
        doc_type, confidence = self.classifier.classify_document(text_blocks)
        self.assertEqual(doc_type, DocumentType.TECHNICAL_MANUAL)
        self.assertGreater(confidence, 0.3)
    
    def test_classify_business_document(self):
        """Test classification of business document"""
        text_blocks = [
            self.create_mock_text_block("RFP: Request for Proposal"),
            self.create_mock_text_block("Ontario's Digital Library"),
            self.create_mock_text_block("Appendix A: ODL Envisioned Phases"),
            self.create_mock_text_block("Phase I: Business Planning")
        ]
        
        doc_type, confidence = self.classifier.classify_document(text_blocks)
        self.assertEqual(doc_type, DocumentType.BUSINESS_DOCUMENT)
        self.assertGreater(confidence, 0.3)
    
    def test_classify_program_document(self):
        """Test classification of program document"""
        text_blocks = [
            self.create_mock_text_block("Parsippany -Troy Hills STEM Pathways"),
            self.create_mock_text_block("PATHWAY OPTIONS"),
            self.create_mock_text_block("Regular Pathway"),
            self.create_mock_text_block("Distinction Pathway")
        ]
        
        doc_type, confidence = self.classifier.classify_document(text_blocks)
        self.assertEqual(doc_type, DocumentType.PROGRAM_DOCUMENT)
        self.assertGreater(confidence, 0.3)
    
    def test_classify_invitation_document(self):
        """Test classification of invitation document"""
        text_blocks = [
            self.create_mock_text_block("YOU'RE INVITED TO A PARTY"),
            self.create_mock_text_block("TOPJUMP TRAMPOLINE PARK"),
            self.create_mock_text_block("HOPE To SEE You THERE!"),
            self.create_mock_text_block("RSVP:")
        ]
        
        doc_type, confidence = self.classifier.classify_document(text_blocks)
        self.assertEqual(doc_type, DocumentType.INVITATION_DOCUMENT)
        self.assertGreater(confidence, 0.3)
    
    def test_empty_input(self):
        """Test handling of empty input"""
        doc_type, confidence = self.classifier.classify_document([])
        self.assertEqual(doc_type, DocumentType.FORM_DOCUMENT)
        self.assertEqual(confidence, 0.0)

if __name__ == '__main__':
    unittest.main()
