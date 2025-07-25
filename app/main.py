import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
import re

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
            (DocumentType.TECHNICAL_MANUAL, ManualProcessor, "ManualProcessor"),
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
        """Process a single PDF file with enhanced error handling and classification"""
        start_time = time.time()
        filename = Path(input_path).name
        
        try:
            logger.info(f"=== Processing PDF: {filename} ===")
            
            # Load PDF
            if not self.pdf_parser.load_pdf(input_path):
                logger.error(f"Failed to load PDF: {input_path}")
                return False
            
            # Extract text blocks
            text_blocks = self.pdf_parser.extract_text_blocks()
            if not text_blocks:
                logger.error("No text blocks extracted from PDF")
                return False
            
            logger.info(f"Extracted {len(text_blocks)} text blocks from {filename}")
            
            # Log first few blocks for debugging
            for i, block in enumerate(text_blocks[:3]):
                logger.debug(f"Block {i}: '{block.text[:50]}...' (page: {getattr(block, 'page', 'unknown')})")
            
            # Enhanced document classification with content analysis
            doc_type, confidence = self._classify_document_enhanced(text_blocks, filename)
            logger.info(f"{filename} classified as: {doc_type.value} (confidence: {confidence:.2f})")
            
            # Get processor with detailed logging
            processor = self._get_processor_with_logging(doc_type, filename)
            if not processor:
                logger.error(f"No processor found for document type: {doc_type}")
                return False
            
            # Process with appropriate processor
            try:
                logger.info(f"Processing {filename} with {processor.__class__.__name__}")
                document_structure = processor.process(text_blocks)
                
                if not document_structure:
                    logger.error("Processor returned None/empty result")
                    return False
                
                # Validate result has required fields
                if not hasattr(document_structure, 'title') or not hasattr(document_structure, 'outline'):
                    logger.error("Processor returned invalid document structure")
                    return False
                
                logger.info(f"{filename} processing successful:")
                logger.info(f"  Title: '{document_structure.title}'")
                logger.info(f"  Outline items: {len(document_structure.outline)}")
                
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
                    logger.info(f"{filename} processing completed in {processing_time:.2f} seconds")
                    logger.info(f"Output saved to: {output_path}")
                    return True
                else:
                    logger.error("Failed to save JSON output")
                    return False
            except Exception as e:
                logger.error(f"Save error: {e}", exc_info=True)
                return False
                
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}", exc_info=True)
            return False
        
        finally:
            try:
                self.pdf_parser.close()
            except Exception as e:
                logger.warning(f"Error closing PDF parser: {e}")
    
    def _classify_document_enhanced(self, text_blocks, filename: str = None):
        """Enhanced document classification with content analysis fallback"""
        
        logger.info(f"Starting enhanced classification for {filename}")
        
        # Method 1: Try existing document classifier
        try:
            doc_type, confidence = self.document_classifier.classify_document(text_blocks)
            logger.info(f"Document classifier result: {doc_type.value} (confidence: {confidence:.2f})")
            
            # If confidence is high, use it
            if confidence > 0.8:
                return doc_type, confidence
                
            # If confidence is moderate, verify with content analysis
            if confidence > 0.6:
                verified_type, content_confidence = self._classify_by_content_analysis(text_blocks, filename)
                if verified_type == doc_type:
                    logger.info(f"Document classifier result verified by content analysis")
                    return doc_type, confidence
                else:
                    logger.info(f"Document classifier disagreement, using content analysis: {verified_type.value}")
                    return verified_type, content_confidence
                    
        except Exception as e:
            logger.warning(f"Document classifier failed: {e}")
        
        # Method 2: Content-based classification fallback
        logger.info(f"Using content-based classification for {filename}")
        doc_type, confidence = self._classify_by_content_analysis(text_blocks, filename)
        
        return doc_type, confidence
    
    def _classify_by_content_analysis(self, text_blocks, filename: str = None):
        """Dynamic content-based classification without hardcoding"""
        
        if not text_blocks:
            logger.warning(f"No text blocks for classification, defaulting to TECHNICAL_MANUAL")
            return DocumentType.TECHNICAL_MANUAL, 0.5
        
        # Collect sample text for analysis
        sample_texts = []
        total_text_length = 0
        block_count = 0
        
        for block in text_blocks[:20]:  # Analyze first 20 blocks
            try:
                text = block.text.strip()
                if text and len(text) > 2:
                    sample_texts.append(text.lower())
                    total_text_length += len(text)
                    block_count += 1
            except Exception:
                continue
        
        if not sample_texts:
            logger.warning(f"No valid text found for classification")
            return DocumentType.TECHNICAL_MANUAL, 0.5
        
        combined_text = " ".join(sample_texts)
        avg_block_length = total_text_length / max(1, block_count)
        
        # Dynamic content analysis
        content_characteristics = self._analyze_content_characteristics(combined_text, avg_block_length, filename)
        
        # Classification decision based on characteristics
        form_score = content_characteristics['form_score']
        manual_score = content_characteristics['manual_score']
        
        logger.info(f"Content analysis for {filename}:")
        logger.info(f"  Form score: {form_score:.3f}")
        logger.info(f"  Manual score: {manual_score:.3f}")
        logger.info(f"  Avg block length: {avg_block_length:.1f}")
        logger.info(f"  Total text length: {total_text_length}")
        logger.info(f"  Text blocks analyzed: {block_count}")
        
        # Determine classification with confidence
        if form_score > manual_score:
            confidence = min(form_score / max(manual_score, 0.1), 1.0)
            logger.info(f"Classified as FORM_DOCUMENT (confidence: {confidence:.3f})")
            return DocumentType.FORM_DOCUMENT, confidence
        elif manual_score > form_score:
            confidence = min(manual_score / max(form_score, 0.1), 1.0)
            logger.info(f"Classified as TECHNICAL_MANUAL (confidence: {confidence:.3f})")
            return DocumentType.TECHNICAL_MANUAL, confidence
        else:
            # Tie-breaker based on document characteristics
            if total_text_length < 3000 and avg_block_length < 50:
                logger.info(f"Tie-breaker: FORM_DOCUMENT (short content)")
                return DocumentType.FORM_DOCUMENT, 0.6
            else:
                logger.info(f"Tie-breaker: TECHNICAL_MANUAL (long content)")
                return DocumentType.TECHNICAL_MANUAL, 0.6
    
    def _analyze_content_characteristics(self, combined_text: str, avg_block_length: float, filename: str = None) -> Dict[str, float]:
        """Analyze content characteristics dynamically without hardcoding"""
        
        # Form document characteristics (statistical analysis)
        form_indicators = {
            'form_language_density': self._calculate_pattern_density(combined_text, [
                r'\bapplication\b', r'\bform\b', r'\bgrant\b', r'\bgovernment\b', 
                r'\bservant\b', r'\bdesignation\b', r'\bltc\b', r'\bsignature\b'
            ]),
            'field_pattern_density': self._calculate_pattern_density(combined_text, [
                r'\bname of\b', r'\bdate of\b', r'\bwhether\b', r'\bamount of\b',
                r'\bage\b', r'\brelationship\b'
            ]),
            'short_content_indicator': 1.0 if avg_block_length < 30 else 0.0,
            'colon_density': min(combined_text.count(':') / max(1, len(combined_text.split())), 1.0),
            'numbered_field_density': self._calculate_pattern_density(combined_text, [r'^\d+\.']),
            'compact_structure_indicator': 1.0 if len(combined_text) < 2000 else 0.0,
            'form_punctuation_ratio': self._calculate_punctuation_ratio(combined_text)
        }
        
        # Technical manual characteristics (statistical analysis)
        manual_indicators = {
            'technical_language_density': self._calculate_pattern_density(combined_text, [
                r'\boverview\b', r'\bfoundation\b', r'\blevel\b', r'\bextension\b',
                r'\bsyllabus\b', r'\blearning\b', r'\bobjectives\b', r'\bcertification\b'
            ]),
            'structure_element_density': self._calculate_pattern_density(combined_text, [
                r'\btable of contents\b', r'\brevision history\b', r'\backnowledgements\b',
                r'\bintroduction\b', r'\breferences\b', r'\bchapter\b'
            ]),
            'long_content_indicator': 1.0 if avg_block_length > 100 else 0.0,
            'hierarchical_pattern_density': self._calculate_pattern_density(combined_text, [r'\b\d+\.\d+\b']),
            'extensive_text_indicator': 1.0 if len(combined_text) > 5000 else 0.0,
            'academic_language_density': self._calculate_pattern_density(combined_text, [
                r'\btesting\b', r'\bsoftware\b', r'\bdevelopment\b', r'\bqualification\b',
                r'\bexam\b', r'\bassessment\b'
            ]),
            'document_length_score': min(len(combined_text) / 10000, 1.0)
        }
        
        # Calculate weighted scores using dynamic weights
        form_weights = [0.25, 0.25, 0.15, 0.15, 0.1, 0.05, 0.05]
        manual_weights = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]
        
        form_values = list(form_indicators.values())
        manual_values = list(manual_indicators.values())
        
        # Ensure we have enough values for weighting
        if len(form_values) != len(form_weights):
            form_weights = [1.0 / len(form_values)] * len(form_values)
        if len(manual_values) != len(manual_weights):
            manual_weights = [1.0 / len(manual_values)] * len(manual_values)
        
        form_score = sum(value * weight for value, weight in zip(form_values, form_weights))
        manual_score = sum(value * weight for value, weight in zip(manual_values, manual_weights))
        
        logger.debug(f"Form indicators for {filename}: {form_indicators}")
        logger.debug(f"Manual indicators for {filename}: {manual_indicators}")
        
        return {
            'form_score': min(form_score, 1.0),
            'manual_score': min(manual_score, 1.0),
            'form_indicators': form_indicators,
            'manual_indicators': manual_indicators
        }
    
    def _calculate_pattern_density(self, text: str, patterns: List[str]) -> float:
        """Calculate density of patterns in text"""
        total_matches = 0
        total_words = len(text.split())
        
        for pattern in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE))
            total_matches += matches
        
        return min(total_matches / max(1, total_words), 1.0)
    
    def _calculate_punctuation_ratio(self, text: str) -> float:
        """Calculate ratio of punctuation characters"""
        if not text:
            return 0.0
        
        punct_count = len(re.findall(r'[^\w\s]', text))
        return min(punct_count / len(text), 1.0)
    
    def _get_processor_with_logging(self, doc_type: DocumentType, filename: str = None):
        """Get processor with detailed logging and multiple fallback strategies"""
        logger.info(f"Selecting processor for {filename}: document type = {doc_type.value}")
        logger.debug(f"Available processors: {[dt.value for dt in self.processors.keys()]}")
        
        # Try primary mapping
        processor = self.processors.get(doc_type)
        if processor:
            logger.info(f"✓ Found primary processor: {processor.__class__.__name__} for {filename}")
            return processor
        
        # Try fallback mapping
        processor = self.processor_fallbacks.get(doc_type)
        if processor:
            logger.info(f"✓ Using fallback processor: {processor.__class__.__name__} for {filename}")
            return processor
        
        # Handle MANUAL_DOCUMENT -> TECHNICAL_MANUAL mapping
        if hasattr(DocumentType, 'MANUAL_DOCUMENT') and doc_type == DocumentType.MANUAL_DOCUMENT:
            processor = self.processors.get(DocumentType.TECHNICAL_MANUAL)
            if processor:
                logger.info(f"✓ Mapping MANUAL_DOCUMENT to TECHNICAL_MANUAL processor for {filename}")
                return processor
        
        # Handle TECHNICAL_MANUAL as fallback for unknown manual types
        if doc_type.value in ['manual_document', 'technical_manual']:
            processor = self.processors.get(DocumentType.TECHNICAL_MANUAL)
            if processor:
                logger.info(f"✓ Using TECHNICAL_MANUAL processor as fallback for {filename}")
                return processor
        
        # Final fallback - create new processor based on document type
        try:
            if doc_type == DocumentType.FORM_DOCUMENT:
                logger.warning(f"Creating new FormProcessor for {filename}")
                return FormProcessor()
            else:
                logger.warning(f"Creating new ManualProcessor for {filename}")
                return ManualProcessor()
        except Exception as e:
            logger.error(f"✗ Failed to create fallback processor for {filename}: {e}")
            return None
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> int:
        """Process all PDF files in a directory with enhanced status reporting"""
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
        
        # Process each file with detailed tracking
        processed_count = 0
        processing_results = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"\n=== Starting processing: {pdf_file.name} ===")
                output_file = output_path / f"{pdf_file.stem}_outline.json"
                
                success = self.process_pdf(str(pdf_file), str(output_file))
                
                if success:
                    processed_count += 1
                    logger.info(f"✓ Successfully processed: {pdf_file.name}")
                    processing_results.append((pdf_file.name, "SUCCESS"))
                else:
                    logger.error(f"✗ Failed to process: {pdf_file.name}")
                    processing_results.append((pdf_file.name, "FAILED"))
                    
            except Exception as e:
                logger.error(f"✗ Error processing {pdf_file.name}: {e}", exc_info=True)
                processing_results.append((pdf_file.name, f"ERROR: {str(e)}"))
        
        # Log summary
        logger.info(f"\n=== PROCESSING SUMMARY ===")
        logger.info(f"Successfully processed {processed_count}/{len(pdf_files)} PDF files")
        
        for filename, status in processing_results:
            if status == "SUCCESS":
                logger.info(f"  ✓ {filename}")
            else:
                logger.error(f"  ✗ {filename}: {status}")
        
        return processed_count

def main():
    """Main entry point with enhanced error handling"""
    parser = argparse.ArgumentParser(description="PDF Outline Extractor")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-d", "--directory", action="store_true", 
                       help="Process all PDFs in input directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
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
            print(f"\nProcessed {processed} PDF files successfully")
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

def _check_processing_status():
    """Enhanced processing status check with detailed analysis"""
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
        
        print("\n" + "="*60)
        print("PROCESSING STATUS SUMMARY")
        print("="*60)
        print(f"Total PDF files found: {len(pdfs)}")
        print(f"Successfully processed: {len(processed)}")
        print(f"Failed/Unprocessed: {len(unprocessed)}")
        print(f"Success rate: {len(processed)/max(1, len(pdfs))*100:.1f}%")
        print("-"*60)
        
        if processed:
            print(f"Processed ({len(processed)}):")
            for name in sorted(processed):
                print(f"  ✓ {name}")
        
        if unprocessed:
            print(f"\nUnprocessed ({len(unprocessed)}):")
            for name in sorted(unprocessed):
                print(f"  ✗ {name}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error checking processing status: {e}")

if __name__ == "__main__":
    main()
    
    # Add enhanced status check after main processing
    _check_processing_status()
