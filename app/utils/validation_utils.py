import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from typing import Dict, Any, List
import logging

logger = logging.getLogger('extraction')

class OutputValidator:
    def __init__(self):
        self.required_fields = ['title', 'outline']
        self.valid_levels = ['H1', 'H2', 'H3', 'H4']
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output JSON structure"""
        try:
            # Check required fields
            for field in self.required_fields:
                if field not in output:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate title
            if not isinstance(output['title'], str):
                logger.error("Title must be a string")
                return False
            
            # Validate outline
            if not isinstance(output['outline'], list):
                logger.error("Outline must be a list")
                return False
            
            # Validate outline items
            for item in output['outline']:
                if not self._validate_outline_item(item):
                    return False
            
            logger.info("Output validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False
    
    def _validate_outline_item(self, item: Dict[str, Any]) -> bool:
        """Validate individual outline item"""
        required_item_fields = ['level', 'text', 'page']
        
        # Check required fields
        for field in required_item_fields:
            if field not in item:
                logger.error(f"Missing required field in outline item: {field}")
                return False
        
        # Validate level
        if item['level'] not in self.valid_levels:
            logger.error(f"Invalid heading level: {item['level']}")
            return False
        
        # Validate text
        if not isinstance(item['text'], str) or not item['text'].strip():
            logger.error("Outline item text must be a non-empty string")
            return False
        
        # Validate page
        if not isinstance(item['page'], int) or item['page'] < 0:
            logger.error("Outline item page must be a non-negative integer")
            return False
        
        return True
    
    def clean_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Clean output to match expected format"""
        cleaned = {
            'title': output.get('title', '').strip(),
            'outline': []
        }
        
        for item in output.get('outline', []):
            if self._validate_outline_item(item):
                cleaned_item = {
                    'level': item['level'],
                    'text': item['text'].strip(),
                    'page': int(item['page'])
                }
                cleaned['outline'].append(cleaned_item)
        
        return cleaned
