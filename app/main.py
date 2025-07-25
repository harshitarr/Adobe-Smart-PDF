import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
import re
from PyPDF2 import PdfReader


# Fix the import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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


import logging.config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('extraction')


class PDFOutlineExtractor:
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.document_classifier = DocumentClassifier()
        self.json_formatter = JSONFormatter()
        self.processors = {}
        self.processor_fallbacks = {}
        self._initialize_processors()


    def _initialize_processors(self):
        processor_configs = [
            (DocumentType.FORM_DOCUMENT, FormProcessor, "FormProcessor"),
            (DocumentType.TECHNICAL_MANUAL, ManualProcessor, "ManualProcessor"),
            (DocumentType.BUSINESS_DOCUMENT, BusinessProcessor, "BusinessProcessor"),
            (DocumentType.PROGRAM_DOCUMENT, ProgramProcessor, "ProgramProcessor"),
            (DocumentType.INVITATION_DOCUMENT, InvitationProcessor, "InvitationProcessor")
        ]
        for doc_type, processor_class, name in processor_configs:
            try:
                self.processors[doc_type] = processor_class()
                logger.info(f"✓ Initialized {name} for {doc_type.value}")
            except Exception as e:
                logger.error(f"✗ Failed to initialize {name}: {e}")


        if hasattr(DocumentType, 'MANUAL_DOCUMENT'):
            try:
                self.processor_fallbacks[DocumentType.MANUAL_DOCUMENT] = ManualProcessor()
                logger.info("✓ Fallback processor for MANUAL_DOCUMENT initialized")
            except Exception as e:
                logger.error(f"✗ Failed to init fallback MANUAL_DOCUMENT: {e}")


    def get_pdf_page_count(self, file_path: str) -> int:
        try:
            reader = PdfReader(file_path)
            return len(reader.pages)
        except Exception as e:
            logger.warning(f"Could not read page count from {file_path}: {e}")
            return -1


    def process_pdf(self, input_path: str, output_path: Optional[str] = None) -> bool:
        filename = Path(input_path).name
        page_count = self.get_pdf_page_count(input_path)


        if page_count == -1:
            logger.warning(f"Skipping {filename} (could not determine page count)")
            print(f"✗ Skipping {filename}: page count could not be determined")
            return False


        print(f"> {filename} | Pages: {page_count}")
        logger.info(f"{filename} has {page_count} pages")


        if page_count > 50:
            logger.warning(f"✗ {filename} exceeds 50 page limit — skipping")
            print(f"✗ Skipping {filename}: page count exceeds 50 (limit)")
            return False


        start_time = time.time()
        try:
            logger.info(f"=== Processing PDF: {filename} ===")


            if not self.pdf_parser.load_pdf(input_path):
                logger.error(f"Failed to load PDF: {input_path}")
                return False


            text_blocks = self.pdf_parser.extract_text_blocks()
            if not text_blocks:
                logger.error("No text blocks extracted from PDF")
                return False


            logger.info(f"Extracted {len(text_blocks)} text blocks from {filename}")
            for i, block in enumerate(text_blocks[:3]):
                logger.debug(f"Block {i}: '{block.text[:50]}...' (page: {getattr(block, 'page', 'unknown')})")


            doc_type, confidence = self._classify_document_enhanced(text_blocks, filename)
            logger.info(f"{filename} classified as: {doc_type.value} (confidence: {confidence:.2f})")


            processor = self._get_processor_with_logging(doc_type, filename)
            if not processor:
                logger.error(f"No processor found for document type: {doc_type}")
                return False


            logger.info(f"Processing {filename} with {processor.__class__.__name__}")
            document_structure = processor.process(text_blocks)


            if not document_structure or not hasattr(document_structure, 'title') or not hasattr(document_structure, 'outline'):
                logger.error("Invalid document structure returned")
                return False


            logger.info(f"{filename} processing successful")
            logger.info(f"  Title: '{document_structure.title}'")
            logger.info(f"  Outline items: {len(document_structure.outline)}")


            json_output = self.json_formatter.format_output(document_structure)


            if not output_path:
                output_path = OUTPUT_DIR / f"{Path(input_path).stem}_outline.json"


            if self.json_formatter.save_json(json_output, str(output_path)):
                logger.info(f"{filename} processing completed in {time.time() - start_time:.2f} seconds")
                logger.info(f"Output saved to: {output_path}")
                return True
            else:
                logger.error("Failed to save JSON")
                return False


        except Exception as e:
            logger.error(f"Exception while processing {filename}: {e}", exc_info=True)
            return False
        finally:
            try:
                self.pdf_parser.close()
            except Exception as e:
                logger.warning(f"Error closing parser for {filename}: {e}")


    def _classify_document_enhanced(self, text_blocks, filename: str = None):
        try:
            doc_type, confidence = self.document_classifier.classify_document(text_blocks)
            if confidence > 0.8:
                return doc_type, confidence
        except Exception:
            pass
        return DocumentType.TECHNICAL_MANUAL, 0.5


    def _get_processor_with_logging(self, doc_type: DocumentType, filename: str = None):
        processor = self.processors.get(doc_type) or self.processor_fallbacks.get(doc_type)
        if not processor and doc_type == DocumentType.MANUAL_DOCUMENT:
            processor = self.processors.get(DocumentType.TECHNICAL_MANUAL)
        return processor


    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> Dict[str, List[str]]:
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else OUTPUT_DIR


        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return {"processed": [], "failed": []}


        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in: {input_dir}")
            return {"processed": [], "failed": []}


        logger.info(f"Found {len(pdf_files)} PDF files to process")
        processed_files = []
        failed_files = []


        for pdf_file in pdf_files:
            try:
                print("\n---")
                output_file = output_path / f"{pdf_file.stem}_outline.json"
                success = self.process_pdf(str(pdf_file), str(output_file))


                if success:
                    processed_files.append(pdf_file.stem)
                    logger.info(f"✓ Successfully processed: {pdf_file.name}")
                else:
                    failed_files.append(pdf_file.stem)
                    logger.error(f"✗ Failed to process: {pdf_file.name}")
            except Exception as e:
                failed_files.append(pdf_file.stem)
                logger.error(f"✗ Error processing {pdf_file.name}: {e}", exc_info=True)


        return {"processed": processed_files, "failed": failed_files, "total": len(pdf_files)}


    def display_processing_summary(self, results: Dict[str, Any]):
        """Display a summary of the processing results"""
        total = results.get("total", 0)
        processed = results.get("processed", [])
        failed = results.get("failed", [])
        
        success_count = len(processed)
        failed_count = len(failed)
        success_rate = (success_count / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 60)
        print(f"Total PDF files found: {total}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed/Unprocessed: {failed_count}")
        print(f"Success rate: {success_rate:.1f}%")
        print("-" * 60)
        
        if processed:
            print(f"Processed ({success_count}):")
            for file in processed:
                print(f"  ✓ {file}")
        
        if failed:
            print(f"\nFailed/Unprocessed ({failed_count}):")
            for file in failed:
                print(f"  ✗ {file}")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PDF Outline Extractor")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-d", "--directory", action="store_true", help="Process all PDFs in directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()


    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose mode enabled")


    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


    extractor = PDFOutlineExtractor()


    if args.directory:
        results = extractor.process_directory(args.input, args.output)
        extractor.display_processing_summary(results)
        print(f"\n✓ Processed {len(results['processed'])} file(s)")
    else:
        success = extractor.process_pdf(args.input, args.output)
        if success:
            print("✓ PDF processed successfully")
        else:
            print("✗ Failed to process PDF")


if __name__ == "__main__":
    main()
