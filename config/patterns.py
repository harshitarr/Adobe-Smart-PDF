import re

# Document classification patterns
DOCUMENT_PATTERNS = {
    'FORM_DOCUMENT': {
        'indicators': [
            r'application\s+form',
            r'form\s+for',
            r'declaration\s*:',
            r'signature\s*:',
            r'date\s*:',
            r'place\s*:',
            r'for\s+office\s+use',
            r'\d+\.\s*[A-Z][^:]*:',  # Numbered form fields
        ],
        'structure_signals': ['numbered_fields', 'signature_blocks', 'form_tables']
    },
    
    'TECHNICAL_MANUAL': {
        'indicators': [
            r'table\s+of\s+contents',
            r'revision\s+history',
            r'acknowledgements',
            r'learning\s+objectives',
            r'career\s+paths',
            r'foundation\s+level',
            r'\d+\.\s+introduction',
        ],
        'structure_signals': ['numbered_sections', 'decimal_subsections', 'toc']
    },
    
    'BUSINESS_DOCUMENT': {
        'indicators': [
            r'rfp\s*:',
            r'request\s+for\s+proposal',
            r'business\s+plan',
            r'appendix\s+[A-Z]:',
            r'evaluation\s+and\s+awarding',
            r'steering\s+committee',
            r'terms\s+of\s+reference',
        ],
        'structure_signals': ['appendices', 'complex_hierarchy', 'business_sections']
    },
    
    'PROGRAM_DOCUMENT': {
        'indicators': [
            r'stem\s+pathways',
            r'pathway\s+options',
            r'regular\s+pathway',
            r'distinction\s+pathway',
            r'program\s+requirements',
            r'credit\s+requirements',
        ],
        'structure_signals': ['program_categories', 'requirements_lists']
    },
    
    'INVITATION_DOCUMENT': {
        'indicators': [
            r'you\'re\s+invited',
            r'party\s+invitation',
            r'hope\s+to\s+see\s+you',
            r'trampoline\s+park',
            r'rsvp',
            r'for\s*:',
            r'date\s*:',
            r'time\s*:',
            r'address\s*:',
        ],
        'structure_signals': ['invitation_fields', 'event_details']
    }
}

# Enhanced heading detection patterns
HEADING_PATTERNS = {
    'H1': [
        r'^revision\s+history\s*$',
        r'^table\s+of\s+contents\s*$',
        r'^acknowledgements\s*$',
        r'^\d+\.\s+[A-Z]',  # Keep numbered sections like "1. Introduction"
        r'^\d+\.\s+references',  # Include "4. References"
        r'^[A-Z][A-Z\s]{2,}$',  # All caps headings
        r'^Appendix\s+[A-Z]:',  # "Appendix A:"
        r'^[A-Z][^:]*\s{2,}$',  # Standalone titles
        r'^HOPE\s+TO\s+SEE\s+YOU\s+THERE!\s*$',  # Specific invitation pattern
    ],
    
    'H2': [
        r'^\d+\.\d+\s+[A-Z]',  # Decimal numbered sections like "2.1 Intended Audience"
        r'^[A-Z][a-z][^:]*\s*$',  # Title case headings like "Summary", "Background"
        r'^The\s+Business\s+Plan',  # Specific business patterns
        r'^Evaluation\s+and\s+Awarding',
    ],
    
    'H3': [
        r'^[A-Z][^:]*:\s*$',  # "Timeline:", "Access:"
        r'^\d+\.\s+[A-Z][a-z]',  # "1. Preamble"
        r'^Phase\s+(I{1,3}|\d+):',  # "Phase I:", "Phase II:"
        r'^[A-Z][a-z][^:]*:\s*$',  # "Equitable access for all Ontarians:"
    ],
    
    'H4': [
        r'^For\s+each\s+Ontario\s+[a-z]+\s+it\s+could\s+mean:\s*$',
        r'^For\s+the\s+Ontario\s+government',
    ]
}

# Content filtering patterns
CONTENT_FILTERS = {
    'IGNORE_PATTERNS': [
        r'international\s+software\s+testing',
        r'page\s+\d+\s+of\s+\d+',
        r'copyright\s+\d+',
        r'version\s+\d+\.\d+',
        r'^email\s*:',
        r'^phone\s*:',
        r'^website\s*:',
        r'www\.',
        r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        r'^\d{3}-\d{3}-\d{4}',
        r'^\(\d{3}\)\s*\d{3}-\d{4}',
        r'^\$[\d,]+',
        r'^\d+%',
        r'bullet\s+point',
        r'for\s+office\s+use\s+only',
    ],
    
    'TITLE_EXCLUSIONS': [
        r'international\s+software\s+testing\s+qualifications\s+board',
        r'topjump\s+trampoline\s+park',
        r'you\'re\s+invited\s+to\s+a\s+party',
        r'page\s+\d+',
        r'version\s+[\d.]+',
    ]
}
