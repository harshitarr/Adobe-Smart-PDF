import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from typing import Dict, Any, List
from pathlib import Path
import logging

from models.document_types import DocumentStructure, HeadingLevel
from utils.validation_utils import OutputValidator
from config.language_patterns import LANGUAGE_CONFIG

logger = logging.getLogger('extraction')

class JSONFormatter:
    def __init__(self):
        self.validator = OutputValidator()
    
    def format_output(self, document_structure: DocumentStructure) -> Dict[str, Any]:
        """Format document structure as JSON output WITHOUT language information"""
        output = {
            "title": document_structure.title,
            "outline": []
        }
        
        # Format outline items
        for heading in document_structure.outline:
            outline_item = {
                "level": heading.level,
                "text": heading.text,
                "page": heading.page
            }
            output["outline"].append(outline_item)
        
        # Validate output (without language info)
        if self.validator.validate_output(output):
            logger.info("JSON output formatted and validated successfully")
        else:
            logger.warning("JSON output validation failed")
            output = self.validator.clean_output(output)
        
        return output
    
    def format_language_info(self, document_structure: DocumentStructure) -> Dict[str, Any]:
        """Format ONLY language information as separate JSON"""
        language_info = {}
        
        # Extract language information from metadata if available
        if hasattr(document_structure, 'metadata') and document_structure.metadata:
            extracted_info = self._extract_language_info(document_structure.metadata)
            if extracted_info:
                language_info = extracted_info
        
        # If no language info available, create minimal info
        if not language_info:
            language_info = {
                "detected_language": "english",
                "confidence": 0.0,
                "display_name": "English",
                "language_code": "en",
                "confidence_level": "unknown",
                "detection_status": "no_detection_performed"
            }
        
        return language_info
    
    def _extract_language_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format language information from metadata"""
        language_info = {}
        
        # Get detected language
        detected_language = metadata.get('detected_language', 'english')
        language_info['detected_language'] = detected_language
        
        # Get language confidence
        language_confidence = metadata.get('language_confidence', 0.0)
        language_info['confidence'] = round(language_confidence, 3)
        
        # Get display name
        display_name = self._get_language_display_name(detected_language)
        language_info['display_name'] = display_name
        
        # Get language code
        language_code = self._get_language_code(detected_language)
        language_info['language_code'] = language_code
        
        # Get confidence level
        confidence_level = self._get_confidence_level(language_confidence)
        language_info['confidence_level'] = confidence_level
        
        # Add processing information if available
        if metadata.get('language_config_applied', False):
            language_info['language_processing_applied'] = True
            
        if metadata.get('processing_language'):
            language_info['processing_language'] = metadata['processing_language']
            
        # Add detailed analysis if available
        language_details = metadata.get('language_details', {})
        if language_details and isinstance(language_details, dict):
            if 'confidence_level' in language_details:
                language_info['analysis_confidence_level'] = language_details['confidence_level']
            if 'all_scores' in language_details:
                language_info['detection_scores'] = {
                    lang: round(score, 3) for lang, score in language_details['all_scores'].items()
                }
            if 'text_sample_length' in language_details:
                language_info['text_sample_length'] = language_details['text_sample_length']
            if 'threshold_applied' in language_details:
                language_info['threshold_applied'] = language_details['threshold_applied']
        
        return language_info
    
    def _get_language_display_name(self, language_code: str) -> str:
        """Get display name for language"""
        try:
            display_names = LANGUAGE_CONFIG.get('display_names', {})
            return display_names.get(language_code, language_code.title())
        except:
            return language_code.title()
    
    def _get_language_code(self, language: str) -> str:
        """Get ISO language code for language"""
        try:
            language_codes = LANGUAGE_CONFIG.get('language_codes', {})
            return language_codes.get(language, language[:2].lower())
        except:
            return language[:2].lower()
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Determine confidence level based on score"""
        try:
            thresholds = LANGUAGE_CONFIG.get('confidence_thresholds', {})
            
            if confidence >= thresholds.get('very_high', 0.85):
                return 'very_high'
            elif confidence >= thresholds.get('high', 0.7):
                return 'high'
            elif confidence >= thresholds.get('medium', 0.5):
                return 'medium'
            elif confidence >= thresholds.get('low', 0.3):
                return 'low'
            else:
                return 'very_low'
        except:
            # Fallback confidence level determination
            if confidence >= 0.8:
                return 'high'
            elif confidence >= 0.6:
                return 'medium'
            elif confidence >= 0.4:
                return 'low'
            else:
                return 'very_low'
    
    def save_json(self, output: Dict[str, Any], file_path: str) -> bool:
        """Save main JSON output to file (without language info)"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=4, ensure_ascii=False)
            logger.info(f"Main JSON output saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save main JSON output to {file_path}: {str(e)}")
            return False
    
    def save_language_info(self, language_info: Dict[str, Any], file_path: str) -> bool:
        """Save language information to separate file"""
        try:
            # Ensure the language info directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(language_info, f, indent=4, ensure_ascii=False)
            logger.info(f"Language info saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save language info to {file_path}: {str(e)}")
            return False
    
    def save_complete_output(self, document_structure: DocumentStructure, base_path: str, filename_stem: str) -> Dict[str, bool]:
        """Save both main output and language info with proper folder structure"""
        results = {
            'main_output': False,
            'language_info': False
        }
        
        try:
            base_path = Path(base_path)
            
            # Create main output path
            main_output_path = base_path / f"{filename_stem}_outline.json"
            
            # Create language info directory and path
            language_info_dir = base_path / "Language Info"
            
            # Get language info first to determine the filename
            language_info = self.format_language_info(document_structure)
            detected_language = language_info.get('detected_language', 'english')
            
            # Create language info filename with checkmark
            language_filename = f"{filename_stem}_language_{detected_language} ✓.json"
            language_info_path = language_info_dir / language_filename
            
            # Format main output (without language info)
            main_output = self.format_output(document_structure)
            
            # Save main output
            results['main_output'] = self.save_json(main_output, str(main_output_path))
            
            # Save language info
            results['language_info'] = self.save_language_info(language_info, str(language_info_path))
            
            if results['main_output'] and results['language_info']:
                logger.info(f"Complete output saved - Main: {main_output_path}, Language: {language_info_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error saving complete output: {str(e)}")
            return results
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"JSON loaded from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {str(e)}")
            return {}
    
    def load_language_info(self, file_path: str) -> Dict[str, Any]:
        """Load language information from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Language info loaded from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load language info from {file_path}: {str(e)}")
            return {}
    
    def get_language_info_path(self, base_path: str, filename_stem: str, detected_language: str = None) -> str:
        """Get the expected path for language info file"""
        base_path = Path(base_path)
        language_info_dir = base_path / "Language Info"
        
        if detected_language:
            language_filename = f"{filename_stem}_language_{detected_language} ✓.json"
        else:
            # Search for existing language info file
            if language_info_dir.exists():
                pattern = f"{filename_stem}_language_*.json"
                matches = list(language_info_dir.glob(pattern))
                if matches:
                    return str(matches[0])
            language_filename = f"{filename_stem}_language_unknown ✓.json"
        
        return str(language_info_dir / language_filename)
