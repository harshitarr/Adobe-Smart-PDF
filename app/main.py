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
        
        # Initialize processors with error handling
        self.processors = {}
        self.processor_fallbacks = {}
        
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize processors with individual error handling using existing DocumentTypes"""
        processor_configs = [
            (DocumentType.FORM_DOCUMENT, FormProcessor, "FormProcessor"),
            (DocumentType.TECHNICAL_MANUAL, ManualProcessor, "ManualProcessor"),  # Fixed: using existing type
            (DocumentType.BUSINESS_DOCUMENT, BusinessProcessor, "BusinessProcessor"),
            (DocumentType.PROGRAM_DOCUMENT, ProgramProcessor, "ProgramProcessor"),
            (DocumentType.INVITATION_DOCUMENT, InvitationProcessor, "InvitationProcessor")
        ]
        
        for doc_type, processor_class, name in processor_configs:
            try:
                self.processors[doc_type] = processor_class()
                logger.info(f"✓ Successfully initialized {name} for {doc_type.value}")
            except Exception as e:
                logger.error(f"✗ Failed to initialize {name}: {e}")
        
        # Add manual document mapping as alias for technical manual
        try:
            # Check if MANUAL_DOCUMENT exists, if so, map it to ManualProcessor
            if hasattr(DocumentType, 'MANUAL_DOCUMENT'):
                self.processor_fallbacks[DocumentType.MANUAL_DOCUMENT] = ManualProcessor()
                logger.info("✓ MANUAL_DOCUMENT fallback processor initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize MANUAL_DOCUMENT fallback: {e}")
        
        logger.info(f"Total processors initialized: {len(self.processors)}")
        logger.info(f"Available document types: {[dt.value for dt in self.processors.keys()]}")
    
    def process_pdf(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """Process a single PDF file with enhanced error handling"""
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
            
            # Classify document type with enhanced error handling
            doc_type, confidence = self._classify_document_safe(text_blocks)
            
            # Get processor with detailed logging
            processor = self._get_processor_with_logging(doc_type)
            if not processor:
                logger.error(f"No processor found for document type: {doc_type}")
                return False
            
            # Process with appropriate processor
            try:
                logger.info(f"Starting processing with {processor.__class__.__name__}")
                document_structure = processor.process(text_blocks)
                
                if not document_structure:
                    logger.error("Processor returned None/empty result")
                    return False
                
                # Validate result has required fields
                if not hasattr(document_structure, 'title') or not hasattr(document_structure, 'outline'):
                    logger.error("Processor returned invalid document structure")
                    return False
                
                logger.info(f"Processing successful - Title: '{document_structure.title}', Outline items: {len(document_structure.outline)}")
                
            except Exception as e:
                logger.error(f"Processing error with {processor.__class__.__name__}: {e}", exc_info=True)
                return False
            
            # Format output
            try:
                json_output = self.json_formatter.format_output(document_structure)
                logger.info("JSON formatting successful")
            except Exception as e:
                logger.error(f"JSON formatting error: {e}", exc_info=True)
                return False
            
            # Determine output path
            if not output_path:
                input_filename = Path(input_path).stem
                output_path = OUTPUT_DIR / f"{input_filename}_outline.json"
            
            # Save output
            try:
                if self.json_formatter.save_json(json_output, str(output_path)):
                    processing_time = time.time() - start_time
                    logger.info(f"Processing completed in {processing_time:.2f} seconds")
                    logger.info(f"Output saved to: {output_path}")
                    return True
                else:
                    logger.error("Failed to save JSON output")
                    return False
            except Exception as e:
                logger.error(f"Save error: {e}", exc_info=True)
                return False
                
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            return False
        
        finally:
            try:
                self.pdf_parser.close()
            except Exception as e:
                logger.warning(f"Error closing PDF parser: {e}")
    
    def _classify_document_safe(self, text_blocks):
        """Safe document classification with fallbacks using existing types"""
        try:
            doc_type, confidence = self.document_classifier.classify_document(text_blocks)
            logger.info(f"Document classified as {doc_type.value} (confidence: {confidence:.2f})")
            return doc_type, confidence
        except Exception as e:
            logger.error(f"Classification error: {e}")
            # Default to TECHNICAL_MANUAL if classification fails (using existing type)
            doc_type = DocumentType.TECHNICAL_MANUAL
            confidence = 0.5
            logger.info(f"Using default document type: {doc_type.value}")
            return doc_type, confidence
    
    def _get_processor_with_logging(self, doc_type: DocumentType):
        """Get processor with detailed logging and multiple fallback strategies"""
        logger.info(f"Looking for processor for document type: {doc_type.value}")
        logger.info(f"Available processors: {[dt.value for dt in self.processors.keys()]}")
        
        # Try primary mapping
        processor = self.processors.get(doc_type)
        if processor:
            logger.info(f"Found primary processor: {processor.__class__.__name__}")
            return processor
        
        # Try fallback mapping
        processor = self.processor_fallbacks.get(doc_type)
        if processor:
            logger.info(f"Using fallback processor: {processor.__class__.__name__}")
            return processor
        
        # Handle MANUAL_DOCUMENT -> TECHNICAL_MANUAL mapping
        if hasattr(DocumentType, 'MANUAL_DOCUMENT') and doc_type == DocumentType.MANUAL_DOCUMENT:
            processor = self.processors.get(DocumentType.TECHNICAL_MANUAL)
            if processor:
                logger.info(f"Mapping MANUAL_DOCUMENT to TECHNICAL_MANUAL processor")
                return processor
        
        # Handle TECHNICAL_MANUAL as fallback for unknown manual types
        if doc_type.value in ['manual_document', 'technical_manual']:
            processor = self.processors.get(DocumentType.TECHNICAL_MANUAL)
            if processor:
                logger.info(f"Using TECHNICAL_MANUAL processor as fallback")
                return processor
        
        # Final fallback - create new ManualProcessor
        try:
            logger.warning(f"No mapping found for {doc_type}, creating new ManualProcessor")
            return ManualProcessor()
        except Exception as e:
            logger.error(f"Failed to create fallback ManualProcessor: {e}")
            return None
    
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
        
        # Process each file with individual error handling
        processed_count = 0
        for pdf_file in pdf_files:
            try:
                logger.info(f"Starting processing: {pdf_file.name}")
                output_file = output_path / f"{pdf_file.stem}_outline.json"
                
                if self.process_pdf(str(pdf_file), str(output_file)):
                    processed_count += 1
                    logger.info(f"✓ Successfully processed: {pdf_file.name}")
                else:
                    logger.error(f"✗ Failed to process: {pdf_file.name}")
            except Exception as e:
                logger.error(f"✗ Error processing {pdf_file.name}: {e}", exc_info=True)
        
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
    try:
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        logger.info("All directories ensured to exist")
    except Exception as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)
    
    # Initialize extractor with detailed error handling
    try:
        logger.info("Initializing PDF Outline Extractor...")
        extractor = PDFOutlineExtractor()
        logger.info("Extractor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}", exc_info=True)
        print(f"Failed to initialize extractor: {e}")
        sys.exit(1)
    
    # Process input
    try:
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
    except Exception as e:
        logger.error(f"Main processing error: {e}", exc_info=True)
        print(f"Processing failed: {e}")
        sys.exit(1)

# ------------------------------------------------------------------
# Processing status check - added at the end
# ------------------------------------------------------------------
def _check_processing_status():
    """Check which PDFs were successfully processed"""
    try:
        pdf_dir = Path(INPUT_DIR)
        json_dir = Path(OUTPUT_DIR)
        
        # Check if directories exist
        if not pdf_dir.exists():
            print(f"Input directory does not exist: {INPUT_DIR}")
            return
        
        if not json_dir.exists():
            print(f"Output directory does not exist: {OUTPUT_DIR}")
            return
        
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
