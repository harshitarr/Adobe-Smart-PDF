import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import argparse
import re
from PyPDF2 import PdfReader

# Fix the import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    LOGGING_CONFIG, INPUT_DIR, OUTPUT_DIR, LOGS_DIR,
    LANGUAGE_DETECTION_ENABLED, LANGUAGE_DISPLAY_ENABLED
)
from utils.pdf_utils import PDFParser
from core.document_classifier import DocumentClassifier
from processors.form_processor import FormProcessor
from processors.manual_processor import ManualProcessor
from processors.business_processor import BusinessProcessor
from processors.program_processor import ProgramProcessor
from processors.invitation_processor import InvitationProcessor
from core.json_formatter import JSONFormatter
from models.document_types import DocumentType
from analyzers.language_analyzer import LanguageAnalyzer

import logging.config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('extraction')

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

class LanguageDetectionError(Exception):
    """Custom exception for language detection errors"""
    pass

class ProcessorInitializationError(Exception):
    """Custom exception for processor initialization errors"""
    pass

def suppress_all_console_logging():
    """Completely suppress all console logging"""
    # Disable all logging to console
    logging.disable(logging.CRITICAL)
    
    # Get all existing loggers and remove console handlers
    all_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    all_loggers.append(logging.getLogger())  # Add root logger
    
    for logger_obj in all_loggers:
        # Remove all stream handlers (console handlers)
        handlers_to_remove = []
        for handler in logger_obj.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                handlers_to_remove.append(handler)
        
        for handler in handlers_to_remove:
            logger_obj.removeHandler(handler)
        
        # Disable propagation to prevent messages going to parent loggers
        logger_obj.propagate = False
    
    # Set root logger to highest level to suppress everything
    logging.getLogger().setLevel(logging.CRITICAL + 1)

class PDFOutlineExtractor:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.pdf_parser = None
        self.document_classifier = None
        self.json_formatter = None
        self.processors = {}
        self.processor_fallbacks = {}
        self.language_analyzer = None
        
        try:
            self._initialize_core_components()
            self._initialize_language_analyzer()
            self._initialize_processors()
        except Exception as e:
            logger.error(f"Failed to initialize PDFOutlineExtractor: {e}", exc_info=True)
            raise ProcessorInitializationError(f"Initialization failed: {e}") from e

    def _initialize_core_components(self):
        """Initialize core components with error handling"""
        try:
            self.pdf_parser = PDFParser()
            logger.debug("✓ PDFParser initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PDFParser: {e}")
            raise ProcessorInitializationError(f"PDFParser initialization failed: {e}") from e
        
        try:
            self.document_classifier = DocumentClassifier()
            logger.debug("✓ DocumentClassifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentClassifier: {e}")
            raise ProcessorInitializationError(f"DocumentClassifier initialization failed: {e}") from e
        
        try:
            self.json_formatter = JSONFormatter()
            logger.debug("✓ JSONFormatter initialized")
        except Exception as e:
            logger.error(f"Failed to initialize JSONFormatter: {e}")
            raise ProcessorInitializationError(f"JSONFormatter initialization failed: {e}") from e

    def _initialize_language_analyzer(self):
        """Initialize language analyzer with error handling"""
        try:
            # Initialize language analyzer if enabled
            if LANGUAGE_DETECTION_ENABLED:
                self.language_analyzer = LanguageAnalyzer()
                logger.info("✓ Language detection enabled")
            else:
                self.language_analyzer = None
                logger.info("Language detection disabled")
        except ImportError as e:
            logger.warning(f"Language analyzer dependencies not available: {e}")
            logger.info("Language detection disabled due to missing dependencies")
            self.language_analyzer = None
        except Exception as e:
            logger.error(f"Failed to initialize LanguageAnalyzer: {e}")
            logger.warning("Continuing without language detection")
            self.language_analyzer = None

    def _initialize_processors(self):
        """Initialize document processors with error handling"""
        processor_configs = [
            (DocumentType.FORM_DOCUMENT, FormProcessor, "FormProcessor"),
            (DocumentType.TECHNICAL_MANUAL, ManualProcessor, "ManualProcessor"),
            (DocumentType.BUSINESS_DOCUMENT, BusinessProcessor, "BusinessProcessor"),
            (DocumentType.PROGRAM_DOCUMENT, ProgramProcessor, "ProgramProcessor"),
            (DocumentType.INVITATION_DOCUMENT, InvitationProcessor, "InvitationProcessor")
        ]
        
        initialization_errors = []
        
        for doc_type, processor_class, name in processor_configs:
            try:
                self.processors[doc_type] = processor_class()
                logger.info(f"✓ Initialized {name} for {doc_type.value}")
            except ImportError as e:
                error_msg = f"Missing dependencies for {name}: {e}"
                logger.warning(error_msg)
                initialization_errors.append(error_msg)
            except Exception as e:
                error_msg = f"Failed to initialize {name}: {e}"
                logger.error(error_msg)
                initialization_errors.append(error_msg)

        # Initialize fallback processors
        if hasattr(DocumentType, 'MANUAL_DOCUMENT'):
            try:
                self.processor_fallbacks[DocumentType.MANUAL_DOCUMENT] = ManualProcessor()
                logger.info("✓ Fallback processor for MANUAL_DOCUMENT initialized")
            except Exception as e:
                error_msg = f"Failed to init fallback MANUAL_DOCUMENT: {e}"
                logger.error(error_msg)
                initialization_errors.append(error_msg)
        
        # Check if we have at least one working processor
        if not self.processors:
            error_msg = "No processors could be initialized"
            logger.error(error_msg)
            raise ProcessorInitializationError(error_msg)
        
        if initialization_errors:
            logger.warning(f"Some processors failed to initialize: {len(initialization_errors)} errors")
            for error in initialization_errors:
                logger.debug(f"  - {error}")

    def get_pdf_page_count(self, file_path: str) -> int:
        """Get PDF page count with comprehensive error handling"""
        if not file_path or not os.path.exists(file_path):
            logger.error(f"PDF file does not exist: {file_path}")
            return -1
        
        try:
            # Validate file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"PDF file is empty: {file_path}")
                return -1
            
            if file_size > 500 * 1024 * 1024:  # 500MB limit
                logger.warning(f"PDF file is very large ({file_size / (1024*1024):.1f}MB): {file_path}")
            
            reader = PdfReader(file_path)
            page_count = len(reader.pages)
            
            if page_count <= 0:
                logger.error(f"PDF has no pages: {file_path}")
                return -1
                
            return page_count
            
        except FileNotFoundError:
            logger.error(f"PDF file not found: {file_path}")
            return -1
        except PermissionError:
            logger.error(f"Permission denied accessing PDF: {file_path}")
            return -1
        except Exception as e:
            logger.warning(f"Could not read page count from {file_path}: {e}")
            return -1

    def detect_document_language(self, text_blocks: List) -> Tuple[str, float, Dict[str, Any]]:
        """Detect document language with comprehensive error handling"""
        try:
            if not self.language_analyzer or not LANGUAGE_DETECTION_ENABLED:
                return 'english', 0.0, {'reason': 'disabled'}
            
            if not text_blocks:
                logger.warning("No text blocks provided for language detection")
                return 'english', 0.0, {'reason': 'no_text_blocks'}
            
            return self.language_analyzer.detect_language(text_blocks)
            
        except LanguageDetectionError as e:
            logger.error(f"Language detection error: {e}")
            return 'english', 0.0, {'reason': 'detection_error', 'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}", exc_info=True)
            return 'english', 0.0, {'reason': 'unexpected_error', 'error': str(e)}

    def process_pdf(self, input_path: str, output_path: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Process PDF with comprehensive error handling - returns success and metadata"""
        metadata = {
            'filename': '',
            'pages': 0,
            'language': 'unknown',
            'processing_time': 0.0,
            'error': None
        }
        
        if not input_path:
            logger.error("No input path provided")
            if self.verbose:
                print("✗ ERROR: No input path provided")
            metadata['error'] = "No input path provided"
            return False, metadata
        
        try:
            # Validate input file
            if not os.path.exists(input_path):
                logger.error(f"Input file does not exist: {input_path}")
                if self.verbose:
                    print(f"✗ ERROR: File not found: {input_path}")
                metadata['error'] = "File not found"
                return False, metadata
            
            if not input_path.lower().endswith('.pdf'):
                logger.error(f"Input file is not a PDF: {input_path}")
                if self.verbose:
                    print(f"✗ ERROR: Not a PDF file: {input_path}")
                metadata['error'] = "Not a PDF file"
                return False, metadata
            
            filename = Path(input_path).name
            metadata['filename'] = filename
            
            page_count = self.get_pdf_page_count(input_path)
            metadata['pages'] = page_count

            if page_count == -1:
                logger.warning(f"Skipping {filename} (could not determine page count)")
                if self.verbose:
                    print(f"✗ Skipping {filename}: page count could not be determined")
                metadata['error'] = "Could not determine page count"
                return False, metadata

            logger.info(f"{filename} has {page_count} pages")

            if page_count > 50:
                logger.warning(f"✗ {filename} exceeds 50 page limit — skipping")
                if self.verbose:
                    print(f"✗ SKIPPED: {filename} exceeds 50 page limit")
                metadata['error'] = "Exceeds 50 page limit"
                return False, metadata

            start_time = time.time()
            
            try:
                logger.info(f"=== Processing PDF: {filename} ===")

                # Load PDF with error handling
                if not self._safe_load_pdf(input_path):
                    if self.verbose:
                        print(f"✗ FAILED: Could not load PDF: {filename}")
                    metadata['error'] = "Could not load PDF"
                    return False, metadata

                # Extract text blocks with error handling
                text_blocks = self._safe_extract_text_blocks()
                if not text_blocks:
                    if self.verbose:
                        print(f"✗ FAILED: No text extracted: {filename}")
                    metadata['error'] = "No text extracted"
                    return False, metadata

                logger.info(f"Extracted {len(text_blocks)} text blocks from {filename}")

                # Language detection with error handling
                language, language_confidence, language_details = self._safe_detect_language(text_blocks)
                metadata['language'] = language
                
                # Document classification with error handling
                doc_type, doc_confidence = self._safe_classify_document(text_blocks, filename)

                # Get processor with error handling
                processor = self._safe_get_processor(doc_type, filename)
                if not processor:
                    if self.verbose:
                        print(f"✗ FAILED: No suitable processor found: {filename}")
                    metadata['error'] = "No suitable processor found"
                    return False, metadata

                # Apply language configuration with error handling
                self._safe_apply_language_config(processor, language, language_details)

                # Process document with error handling
                document_structure = self._safe_process_document(processor, text_blocks, filename)
                if not document_structure:
                    if self.verbose:
                        print(f"✗ FAILED: Document processing failed: {filename}")
                    metadata['error'] = "Document processing failed"
                    return False, metadata

                # Add metadata with error handling
                self._safe_add_metadata(document_structure, language, language_confidence, language_details)

                # Format and save output with separate language info
                if not self._safe_save_output(document_structure, input_path, output_path):
                    if self.verbose:
                        print(f"✗ FAILED: Could not save output: {filename}")
                    metadata['error'] = "Could not save output"
                    return False, metadata

                processing_time = time.time() - start_time
                metadata['processing_time'] = processing_time
                
                logger.info(f"{filename} processing completed in {processing_time:.2f} seconds")
                return True, metadata

            except PDFProcessingError as e:
                logger.error(f"PDF processing error for {filename}: {e}")
                if self.verbose:
                    print(f"✗ FAILED: {filename} - {e}")
                metadata['error'] = str(e)
                return False, metadata
            except Exception as e:
                logger.error(f"Unexpected error processing {filename}: {e}", exc_info=True)
                if self.verbose:
                    print(f"✗ FAILED: Unexpected error: {filename}")
                metadata['error'] = "Unexpected error"
                return False, metadata
            finally:
                self._safe_cleanup()

        except Exception as e:
            logger.error(f"Critical error in process_pdf: {e}", exc_info=True)
            if self.verbose:
                print(f"✗ CRITICAL ERROR: {e}")
            metadata['error'] = f"Critical error: {e}"
            return False, metadata

    def _safe_load_pdf(self, input_path: str) -> bool:
        """Safely load PDF with error handling"""
        try:
            if not self.pdf_parser.load_pdf(input_path):
                logger.error(f"Failed to load PDF: {input_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Exception loading PDF {input_path}: {e}")
            raise PDFProcessingError(f"Could not load PDF: {e}") from e

    def _safe_extract_text_blocks(self) -> Optional[List]:
        """Safely extract text blocks with error handling"""
        try:
            text_blocks = self.pdf_parser.extract_text_blocks()
            if not text_blocks:
                logger.error("No text blocks extracted from PDF")
                return None
            return text_blocks
        except Exception as e:
            logger.error(f"Exception extracting text blocks: {e}")
            raise PDFProcessingError(f"Could not extract text: {e}") from e

    def _safe_detect_language(self, text_blocks: List) -> Tuple[str, float, Dict[str, Any]]:
        """Safely detect language with error handling"""
        try:
            language = 'english'
            language_confidence = 0.0
            language_details = {}
            
            if LANGUAGE_DETECTION_ENABLED and self.language_analyzer:
                language, language_confidence, language_details = self.detect_document_language(text_blocks)
                logger.info(f"Language detected: {language} (confidence: {language_confidence:.3f})")
            
            return language, language_confidence, language_details
        except Exception as e:
            logger.warning(f"Language detection failed, using default: {e}")
            return 'english', 0.0, {'reason': 'detection_failed', 'error': str(e)}

    def _safe_classify_document(self, text_blocks: List, filename: str) -> Tuple[DocumentType, float]:
        """Safely classify document with error handling"""
        try:
            doc_type, doc_confidence = self._classify_document_enhanced(text_blocks, filename)
            logger.info(f"{filename} classified as: {doc_type.value} (confidence: {doc_confidence:.2f})")
            return doc_type, doc_confidence
        except Exception as e:
            logger.warning(f"Document classification failed for {filename}, using default: {e}")
            return DocumentType.TECHNICAL_MANUAL, 0.5

    def _safe_get_processor(self, doc_type: DocumentType, filename: str) -> Optional[Any]:
        """Safely get processor with error handling"""
        try:
            processor = self._get_processor_with_logging(doc_type, filename)
            if not processor:
                logger.error(f"No processor found for document type: {doc_type}")
                return None
            return processor
        except Exception as e:
            logger.error(f"Error getting processor for {doc_type}: {e}")
            return None

    def _safe_apply_language_config(self, processor, language: str, language_details: Dict):
        """Safely apply language configuration with error handling"""
        try:
            if hasattr(processor, 'set_language_config') and language != 'english' and self.language_analyzer:
                language_config = self.language_analyzer.get_processing_config(language)
                processor.set_language_config(language, language_config)
                logger.info(f"Applied {language} processing configuration")
        except Exception as e:
            logger.warning(f"Failed to apply language configuration for {language}: {e}")

    def _safe_process_document(self, processor, text_blocks: List, filename: str) -> Optional[Any]:
        """Safely process document with error handling"""
        try:
            logger.info(f"Processing {filename} with {processor.__class__.__name__}")
            document_structure = processor.process(text_blocks)

            if not document_structure or not hasattr(document_structure, 'title') or not hasattr(document_structure, 'outline'):
                logger.error("Invalid document structure returned")
                return None

            logger.info(f"{filename} processing successful")
            logger.info(f"  Title: '{document_structure.title}'")
            logger.info(f"  Outline items: {len(document_structure.outline)}")
            
            return document_structure
        except Exception as e:
            logger.error(f"Error processing document with {processor.__class__.__name__}: {e}")
            raise PDFProcessingError(f"Document processing failed: {e}") from e

    def _safe_add_metadata(self, document_structure, language: str, language_confidence: float, language_details: Dict):
        """Safely add metadata with error handling"""
        try:
            if hasattr(document_structure, 'metadata'):
                document_structure.metadata.update({
                    'detected_language': language,
                    'language_confidence': language_confidence,
                    'language_details': language_details
                })
        except Exception as e:
            logger.warning(f"Failed to add metadata: {e}")

    def _safe_save_output(self, document_structure, input_path: str, output_path: Optional[str]) -> bool:
        """Safely save output with separate language info handling"""
        try:
            # Format and save output with separate language info
            if not output_path:
                output_dir = OUTPUT_DIR
                filename_stem = Path(input_path).stem
            else:
                output_dir = Path(output_path).parent
                filename_stem = Path(output_path).stem.replace('_outline', '')  # Remove _outline if present
            
            # Save both main output and language info separately
            save_results = self.json_formatter.save_complete_output(
                document_structure, 
                str(output_dir), 
                filename_stem
            )
            
            if save_results['main_output']:
                logger.info(f"Output saved successfully for {Path(input_path).name}")
                return True
            else:
                logger.error("Failed to save main output")
                return False
                
        except Exception as e:
            logger.error(f"Error saving output: {e}")
            return False

    def _safe_cleanup(self):
        """Safely cleanup resources"""
        try:
            if self.pdf_parser:
                self.pdf_parser.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def _classify_document_enhanced(self, text_blocks, filename: str = None):
        """Enhanced document classification with error handling"""
        try:
            doc_type, confidence = self.document_classifier.classify_document(text_blocks)
            if confidence > 0.8:
                return doc_type, confidence
        except Exception as e:
            logger.warning(f"Document classification error: {e}")
        return DocumentType.TECHNICAL_MANUAL, 0.5

    def _get_processor_with_logging(self, doc_type: DocumentType, filename: str = None):
        """Get processor with enhanced logging"""
        processor = self.processors.get(doc_type) or self.processor_fallbacks.get(doc_type)
        if not processor and doc_type == DocumentType.MANUAL_DOCUMENT:
            processor = self.processors.get(DocumentType.TECHNICAL_MANUAL)
        return processor

    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process directory with clean terminal output"""
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir) if output_dir else OUTPUT_DIR

            if not input_path.exists():
                logger.error(f"Input directory does not exist: {input_dir}")
                print(f"✗ ERROR: Input directory not found: {input_dir}")
                return {"processed": [], "failed": [], "total": 0, "languages": {}, "errors": ["Directory not found"], "file_details": []}

            if not input_path.is_dir():
                logger.error(f"Input path is not a directory: {input_dir}")
                print(f"✗ ERROR: Not a directory: {input_dir}")
                return {"processed": [], "failed": [], "total": 0, "languages": {}, "errors": ["Not a directory"], "file_details": []}

            # Ensure output directory exists
            try:
                os.makedirs(output_path, exist_ok=True)
            except Exception as e:
                logger.error(f"Cannot create output directory {output_path}: {e}")
                print(f"✗ ERROR: Cannot create output directory: {e}")
                return {"processed": [], "failed": [], "total": 0, "languages": {}, "errors": [f"Output directory error: {e}"], "file_details": []}

            try:
                pdf_files = list(input_path.glob("*.pdf"))
            except Exception as e:
                logger.error(f"Error scanning directory {input_dir}: {e}")
                print(f"✗ ERROR: Cannot scan directory: {e}")
                return {"processed": [], "failed": [], "total": 0, "languages": {}, "errors": [f"Directory scan error: {e}"], "file_details": []}

            if not pdf_files:
                logger.warning(f"No PDF files found in: {input_dir}")
                print(f"⚠ WARNING: No PDF files found in: {input_dir}")
                return {"processed": [], "failed": [], "total": 0, "languages": {}, "errors": ["No PDF files found"], "file_details": []}

            logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            processed_files = []
            failed_files = []
            language_stats = {}
            processing_errors = []
            file_details = []

            for i, pdf_file in enumerate(pdf_files, 1):
                try:
                    # Process file (will save both main output and language info)
                    success, metadata = self.process_pdf(str(pdf_file), None)  # Let it use default output path

                    # Store file details for summary
                    file_detail = {
                        'filename': metadata['filename'],
                        'pages': metadata['pages'],
                        'language': metadata['language'],
                        'success': success,
                        'error': metadata.get('error')
                    }
                    file_details.append(file_detail)

                    if success:
                        processed_files.append(pdf_file.stem)
                        logger.info(f"✓ Successfully processed: {pdf_file.name}")
                        
                        # Track language statistics
                        lang = metadata['language']
                        language_stats[lang] = language_stats.get(lang, 0) + 1
                        
                    else:
                        failed_files.append(pdf_file.stem)
                        error_msg = f"Failed to process {pdf_file.name}: {metadata.get('error', 'Unknown error')}"
                        processing_errors.append(error_msg)
                        logger.error(f"✗ {error_msg}")
                        
                except KeyboardInterrupt:
                    logger.info("Processing interrupted by user")
                    print("\n⚠ Processing interrupted by user")
                    break
                except Exception as e:
                    failed_files.append(pdf_file.stem)
                    error_msg = f"Error processing {pdf_file.name}: {e}"
                    processing_errors.append(error_msg)
                    logger.error(f"✗ {error_msg}", exc_info=True)

            return {
                "processed": processed_files, 
                "failed": failed_files, 
                "total": len(pdf_files),
                "languages": language_stats,
                "errors": processing_errors,
                "file_details": file_details
            }
            
        except Exception as e:
            logger.error(f"Critical error in process_directory: {e}", exc_info=True)
            print(f"✗ CRITICAL ERROR: {e}")
            return {"processed": [], "failed": [], "total": 0, "languages": {}, "errors": [f"Critical error: {e}"], "file_details": []}

    def display_clean_summary(self, results: Dict[str, Any]):
        """Display clean processing summary for terminal - EXACTLY as requested"""
        total = results.get("total", 0)
        processed = results.get("processed", [])
        failed = results.get("failed", [])
        languages = results.get("languages", {})
        file_details = results.get("file_details", [])
        
        success_count = len(processed)
        failed_count = len(failed)
        success_rate = (success_count / total * 100) if total > 0 else 0
        
        print("=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total PDF files found: {total}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed/Unprocessed: {failed_count}")
        print(f"Success rate: {success_rate:.1f}%")
        
        # Display language statistics if available
        if languages and LANGUAGE_DISPLAY_ENABLED:
            print("-" * 60)
            print("Language Distribution:")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                display_name = lang.title()
                if self.language_analyzer:
                    try:
                        display_name = self.language_analyzer.get_language_display_name(lang)
                    except Exception:
                        pass
                percentage = (count / success_count * 100) if success_count > 0 else 0
                print(f"  {display_name}: {count} files ({percentage:.1f}%)")
        
        print("-" * 60)
        
        if processed:
            print(f"SUCCESSFULLY PROCESSED ({success_count}):")
            for detail in file_details:
                if detail['success']:
                    pages = detail['pages']
                    filename = detail['filename'].replace('.pdf', '')
                    print(f"  ✓ {filename} (Pages: {pages})")
        
        if failed:
            print(f"\nFAILED/UNPROCESSED ({failed_count}):")
            for detail in file_details:
                if not detail['success']:
                    pages = detail['pages'] if detail['pages'] > 0 else '?'
                    filename = detail['filename'].replace('.pdf', '')
                    error = detail['error'] or 'Unknown error'
                    print(f"  ✗ {filename} (Pages: {pages} - {error})")
        
        print("=" * 60)

def main():
    """Main function with completely clean terminal output"""
    try:
        parser = argparse.ArgumentParser(description="PDF Outline Extractor with Multilingual Support")
        parser.add_argument("input", help="Input PDF file or directory")
        parser.add_argument("-o", "--output", help="Output file or directory")
        parser.add_argument("-d", "--directory", action="store_true", help="Process all PDFs in directory")
        parser.add_argument("-v", "--verbose", action="store_true", help="Verbose terminal output")
        parser.add_argument("--no-language", action="store_true", help="Disable language detection")
        
        try:
            args = parser.parse_args()
        except SystemExit as e:
            if e.code != 0:
                print("✗ ERROR: Invalid command line arguments")
            return e.code

        # COMPLETELY SUPPRESS CONSOLE LOGGING UNLESS VERBOSE
        if not args.verbose:
            suppress_all_console_logging()

        # Override language detection if disabled via command line
        if args.no_language:
            global LANGUAGE_DETECTION_ENABLED
            LANGUAGE_DETECTION_ENABLED = False
            logger.info("Language detection disabled via command line")

        # Create directories with error handling
        try:
            os.makedirs(INPUT_DIR, exist_ok=True)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            os.makedirs(LOGS_DIR, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create required directories: {e}")
            print(f"✗ ERROR: Cannot create required directories: {e}")
            return 1

        # Initialize extractor with error handling
        try:
            extractor = PDFOutlineExtractor(verbose=args.verbose)
        except ProcessorInitializationError as e:
            logger.error(f"Failed to initialize PDF extractor: {e}")
            print(f"✗ ERROR: Initialization failed: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}", exc_info=True)
            print(f"✗ CRITICAL ERROR: Unexpected initialization error")
            return 1

        # Process files with error handling
        try:
            if args.directory:
                results = extractor.process_directory(args.input, args.output)
                extractor.display_clean_summary(results)
                
                success_count = len(results['processed'])
                if success_count > 0:
                    print(f"\n✓ Successfully processed {success_count} file(s)")
                    return 0
                else:
                    print(f"\n✗ No files were successfully processed")
                    return 1
            else:
                success, metadata = extractor.process_pdf(args.input, args.output)
                if success:
                    print("✓ PDF processed successfully")
                    return 0
                else:
                    print("✗ Failed to process PDF")
                    return 1
                    
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            print("\n⚠ Processing interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            logger.error(f"Unexpected error during processing: {e}", exc_info=True)
            print(f"\n✗ CRITICAL ERROR: Unexpected processing error")
            return 1

    except Exception as e:
        # Catch any other unexpected errors
        try:
            logger.error(f"Critical error in main: {e}", exc_info=True)
        except:
            pass  # If logging fails, continue
        print(f"✗ CRITICAL ERROR: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"✗ FATAL ERROR: {e}")
        sys.exit(1)
