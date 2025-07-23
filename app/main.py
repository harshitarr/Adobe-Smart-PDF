import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional
import argparse


# Fix the import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Now these imports should work
from config.settings import LOGGING_CONFIG, INPUT_DIR, OUTPUT_DIR, LOGS_DIR
from utils.pdf_utils import PDFParser
from core.document_classifier import DocumentClassifier
from processors.form_processor import FormProcessor
from processors.manual_processor import ManualProcessor
from processors.business_processor import BusinessProcessor
from processors.program_processor import ProgramProcessor
from processors.invitation_processor import InvitationProcessor
from core.json_formatter import JSONFormatter
from models.document_types import DocumentType


# Configure logging
import logging.config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('extraction')


class PDFOutlineExtractor:
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.document_classifier = DocumentClassifier()
        self.json_formatter = JSONFormatter()
        
        # Initialize processors
        self.processors = {
            DocumentType.FORM_DOCUMENT: FormProcessor(),
            DocumentType.TECHNICAL_MANUAL: ManualProcessor(),
            DocumentType.BUSINESS_DOCUMENT: BusinessProcessor(),
            DocumentType.PROGRAM_DOCUMENT: ProgramProcessor(),
            DocumentType.INVITATION_DOCUMENT: InvitationProcessor()
        }
    
    def process_pdf(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """Process a single PDF file"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing PDF: {input_path}")
            
            # Load PDF
            if not self.pdf_parser.load_pdf(input_path):
                logger.error(f"Failed to load PDF: {input_path}")
                return False
            
            # Extract text blocks
            text_blocks = self.pdf_parser.extract_text_blocks()
            if not text_blocks:
                logger.error("No text blocks extracted from PDF")
                return False
            
            logger.info(f"Extracted {len(text_blocks)} text blocks")
            
            # Classify document type
            doc_type, confidence = self.document_classifier.classify_document(text_blocks)
            logger.info(f"Document classified as {doc_type.value} (confidence: {confidence:.2f})")
            
            # Process with appropriate processor
            processor = self.processors.get(doc_type)
            if not processor:
                logger.error(f"No processor found for document type: {doc_type}")
                return False
            
            document_structure = processor.process(text_blocks)
            
            # Format output
            json_output = self.json_formatter.format_output(document_structure)
            
            # Determine output path
            if not output_path:
                input_filename = Path(input_path).stem
                output_path = OUTPUT_DIR / f"{input_filename}_outline.json"
            
            # Save output
            if self.json_formatter.save_json(json_output, str(output_path)):
                processing_time = time.time() - start_time
                logger.info(f"Processing completed in {processing_time:.2f} seconds")
                logger.info(f"Output saved to: {output_path}")
                return True
            else:
                logger.error("Failed to save JSON output")
                return False
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            return False
        
        finally:
            self.pdf_parser.close()
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> int:
        """Process all PDF files in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else OUTPUT_DIR
        
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return 0
        
        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in: {input_dir}")
            return 0
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each file
        processed_count = 0
        for pdf_file in pdf_files:
            output_file = output_path / f"{pdf_file.stem}_outline.json"
            if self.process_pdf(str(pdf_file), str(output_file)):
                processed_count += 1
        
        logger.info(f"Successfully processed {processed_count}/{len(pdf_files)} PDF files")
        return processed_count


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PDF Outline Extractor")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-d", "--directory", action="store_true", 
                       help="Process all PDFs in input directory")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Initialize extractor
    extractor = PDFOutlineExtractor()
    
    # Process input
    if args.directory:
        processed = extractor.process_directory(args.input, args.output)
        print(f"Processed {processed} PDF files")
    else:
        success = extractor.process_pdf(args.input, args.output)
        if success:
            print("PDF processed successfully")
        else:
            print("Failed to process PDF")
            sys.exit(1)


# ------------------------------------------------------------------
# Processing status check - added at the end
# ------------------------------------------------------------------
def _check_processing_status():
    """Check which PDFs were successfully processed"""
    try:
        pdf_dir = Path(INPUT_DIR)
        json_dir = Path(OUTPUT_DIR)
        
        # Get base filenames (without extensions)
        pdfs = {p.stem for p in pdf_dir.glob('*.pdf')}
        jsons = {j.stem.replace('_outline', '') for j in json_dir.glob('*_outline.json')}
        
        processed = pdfs & jsons
        unprocessed = pdfs - jsons
        
        print("\n" + "="*50)
        print("PROCESSING STATUS SUMMARY")
        print("="*50)
        print(f"Processed ({len(processed)}):")
        for name in sorted(processed):
            print(f"  ✓ {name}")
        
        print(f"\nUnprocessed ({len(unprocessed)}):")
        for name in sorted(unprocessed):
            print(f"  ✗ {name}")
        print("="*50)
        
    except Exception as e:
        print(f"Error checking processing status: {e}")


if __name__ == "__main__":
    main()
    
    # Add status check after main processing
    _check_processing_status()
