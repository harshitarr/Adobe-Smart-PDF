import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from typing import Dict, Any, List
import logging

from models.document_types import DocumentStructure, HeadingLevel
from utils.validation_utils import OutputValidator

logger = logging.getLogger('extraction')

class JSONFormatter:
    def __init__(self):
        self.validator = OutputValidator()
    
    def format_output(self, document_structure: DocumentStructure) -> Dict[str, Any]:
        """Format document structure as JSON output"""
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
        
        # Validate output
        if self.validator.validate_output(output):
            logger.info("JSON output formatted and validated successfully")
        else:
            logger.warning("JSON output validation failed")
            output = self.validator.clean_output(output)
        
        return output
    
    def save_json(self, output: Dict[str, Any], file_path: str) -> bool:
        """Save JSON output to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=4, ensure_ascii=False)
            logger.info(f"JSON output saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON output to {file_path}: {str(e)}")
            return False
    
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
