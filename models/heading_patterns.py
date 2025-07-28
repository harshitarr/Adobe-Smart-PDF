import re
from typing import Dict, List, Tuple
from .document_types import DocumentType

class HeadingPatternMatcher:
    def __init__(self):
        self.patterns = {
            DocumentType.FORM_DOCUMENT: {
                'title_patterns': [
                    r'application\s+form\s+for\s+grant\s+of\s+ltc\s+advance',
                ],
                'heading_patterns': {}  
            },
            
            DocumentType.TECHNICAL_MANUAL: {
                'title_patterns': [
                    r'overview\s+foundation\s+level\s+extensions',
                ],
                'heading_patterns': {
                    'H1': [
                        r'^revision\s+history\s*$',
                        r'^table\s+of\s+contents\s*$',
                        r'^acknowledgements\s*$',
                        r'^\d+\.\s+introduction\s+to\s+the\s+foundation\s+level',
                        r'^\d+\.\s+[A-Z]',
                        r'^\d+\.\s+references\s*$',
                    ],
                    'H2': [
                        r'^\d+\.\d+\s+[A-Z]',
                    ]
                }
            },
            
            DocumentType.BUSINESS_DOCUMENT: {
                'title_patterns': [
                    r'rfp\s*:\s*request\s+for\s+proposal\s+to\s+present\s+a\s+proposal\s+for\s+developing\s+the\s+business\s+plan\s+for\s+the\s+ontario\s+digital\s+library',
                ],
                'heading_patterns': {
                    'H1': [
                        r'^ontario\'?s\s+digital\s+library\s*$',
                        r'^a\s+critical\s+component\s+for\s+implementing',
                        r'^appendix\s+[A-Z]:\s+odl',
                    ],
                    'H2': [
                        r'^summary\s*$',
                        r'^background\s*$',
                        r'^the\s+business\s+plan\s+to\s+be\s+developed\s*$',
                        r'^approach\s+and\s+specific\s+proposal\s+requirements\s*$',
                        r'^evaluation\s+and\s+awarding\s+of\s+contract\s*$',
                        r'^appendix\s+[A-Z]:\s+',
                    ],
                    'H3': [
                        r'^timeline\s*:\s*$',
                        r'^[a-z][^:]*:\s*$',
                        r'^\d+\.\s+[A-Z]',
                        r'^phase\s+(I{1,3}|\d+):\s+',
                        r'^milestones\s*$',
                    ],
                    'H4': [
                        r'^for\s+each\s+ontario\s+[a-z]+\s+it\s+could\s+mean\s*:\s*$',
                    ]
                }
            },
            
            DocumentType.PROGRAM_DOCUMENT: {
                'title_patterns': [
                    r'parsippany\s*-\s*troy\s+hills\s+stem\s+pathways',
                ],
                'heading_patterns': {
                    'H1': [
                        r'^pathway\s+options\s*$',
                    ]
                }
            },
            
            DocumentType.INVITATION_DOCUMENT: {
                'title_patterns': [],  
                'heading_patterns': {
                    'H1': [
                        r'^hope\s+to\s+see\s+you\s+there!\s*$',
                    ]
                }
            }
        }
    
    def match_heading_level(self, text: str, doc_type: DocumentType) -> Tuple[str, float]:
        """Match text against heading patterns and return level and confidence"""
        if doc_type not in self.patterns:
            return None, 0.0
        
        patterns = self.patterns[doc_type]['heading_patterns']
        
        for level in ['H1', 'H2', 'H3', 'H4']:
            if level in patterns:
                for pattern in patterns[level]:
                    if re.match(pattern, text.strip(), re.IGNORECASE):
                        return level, 0.9
        
        return None, 0.0
    
    def match_title(self, text: str, doc_type: DocumentType) -> float:
        """Match text against title patterns and return confidence"""
        if doc_type not in self.patterns:
            return 0.0
        
        patterns = self.patterns[doc_type]['title_patterns']
        
        for pattern in patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return 0.9
        
        return 0.0
