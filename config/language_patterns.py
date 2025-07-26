from typing import Dict, List, Any

# Language detection patterns - fully configurable with enhanced coverage
LANGUAGE_PATTERNS = {
    'japanese': {
        'scripts': {
            'hiragana': r'[\u3040-\u309F]',
            'katakana': r'[\u30A0-\u30FF]', 
            'kanji': r'[\u4E00-\u9FAF]',
            'fullwidth_chars': r'[\uFF00-\uFFEF]',  # Full-width characters
            'japanese_symbols': r'[\u3000-\u303F]'  # Japanese symbols and punctuation
        },
        'common_words': [
            'です', 'である', 'します', 'について', 'により', 
            '会社', '株式会社', 'こと', 'もの', 'とき', 'から',
            '対して', '場合', '必要', '重要', '問題', '方法',
            'として', 'による', 'その', 'この', 'また', 'ただし',
            '日本', '東京', '年', '月', '日', '時', '分', '秒',
            '開発', '技術', '製品', '販売', '経営', '管理'
        ],
        'punctuation': ['。', '、', '「', '」', '・', '：', '；', '？', '！', '〜', '…'],
        'patterns': {
            'honorifics': r'(さん|様|殿|氏|君|ちゃん)',
            'counters': r'(個|本|枚|回|件|名|人|台|つ|匹|頭|冊)',
            'particles': r'(は|が|を|に|で|と|から|まで|より|へ|や|か)',
            'verb_endings': r'(ます|した|する|である|だった|でした)',
            'adjective_endings': r'(い|な|的|らしい|そうな)',
            'business_terms': r'(株式会社|有限会社|合同会社|一般社団法人)'
        },
        'scoring': {
            'script_threshold_high': 0.3,
            'script_threshold_medium': 0.1,
            'script_score_high': 0.6,
            'script_score_medium': 0.3,
            'common_word_multiplier': 0.08,
            'common_word_max': 0.25,
            'punctuation_multiplier': 0.02,
            'punctuation_max': 0.1,
            'pattern_multiplier': 0.05,
            'pattern_max': 0.15,
            'fullwidth_bonus': 0.05,
            'business_term_bonus': 0.1
        }
    },
    'german': {
        'patterns': {
            'umlauts': r'[äöüÄÖÜß]',
            'compound_words': r'\b\w{12,}\b',  # Very long compounds
            'capitalization': r'\b[A-ZÄÖÜ][a-zäöüß]+\b',
            'articles': r'\b(der|die|das|den|dem|des|einer|eines|einem)\b',
            'prepositions': r'\b(mit|für|von|auf|zu|durch|über|unter|zwischen|bei|nach|vor|hinter)\b',
            'modal_verbs': r'\b(können|müssen|wollen|sollen|dürfen|mögen)\b',
            'german_conjunctions': r'\b(und|oder|aber|denn|sondern|sowie|bzw)\b'
        },
        'common_words': [
            'und', 'der', 'die', 'das', 'ist', 'mit', 'für', 
            'von', 'auf', 'zu', 'werden', 'durch', 'eine', 'ein',
            'auch', 'oder', 'aber', 'nicht', 'nach', 'bei', 
            'sich', 'aus', 'um', 'wenn', 'noch', 'nur',
            'haben', 'sein', 'können', 'werden', 'sollen',
            'Deutschland', 'Unternehmen', 'Jahr', 'Zeit', 'heute'
        ],
        'suffixes': [
            'ung', 'heit', 'keit', 'schaft', 'tion', 'ismus',
            'lich', 'bar', 'sam', 'los', 'voll', 'ig', 'isch',
            'mäßig', 'fähig', 'würdig'
        ],
        'prefixes': [
            'un', 'be', 'ge', 'er', 'ver', 'zer', 'ent', 'emp',
            'miss', 'wider', 'wieder'
        ],
        'scoring': {
            'umlaut_multiplier': 0.04,
            'umlaut_max': 0.25,
            'word_ratio_high': 0.12,
            'word_ratio_medium': 0.06,
            'word_score_high': 0.4,
            'word_score_medium': 0.2,
            'compound_multiplier': 0.06,
            'compound_max': 0.2,
            'capitalization_threshold': 0.15,
            'capitalization_score': 0.15,
            'pattern_multiplier': 0.03,
            'pattern_max': 0.1,
            'suffix_bonus': 0.02,
            'prefix_bonus': 0.02
        }
    },
    'tamil': {
        'scripts': {
            'tamil_main': r'[\u0B80-\u0BFF]',
            'tamil_numerals': r'[\u0BE6-\u0BEF]',
            'tamil_symbols': r'[\u0BD0-\u0BD7]'
        },
        'common_words': [
            'மற்றும்', 'இது', 'அது', 'என்று', 'கூறினார்', 
            'செய்யப்பட்டது', 'உள்ளது', 'இருந்து', 'வரும்',
            'என்பது', 'போது', 'முடியும்', 'வேண்டும்', 'இல்லை',
            'அதன்', 'இந்த', 'அந்த', 'எந்த', 'எல்லா',
            'தமிழ்', 'இந்தியா', 'நாடு', 'மக்கள்', 'அரசு',
            'கல்வி', 'பள்ளி', 'கல்லூரி', 'பல்கலைக்கழகம்'
        ],
        'patterns': {
            'verb_endings': r'(கிறது|கின்றது|ட்டது|யது|ந்தது|வது|க்கும்)',
            'case_markers': r'(இல்|ஆல்|உடன்|க்கு|இன்|அது|ஓடு|விட)',
            'question_words': r'(எப்போது|எங்கே|எதற்கு|எப்படி|ஏன்|யார்|என்ன)',
            'honorifics': r'(அவர்|அவர்கள்|இவர்|திரு|திருமதி)',
            'numerals': r'(ஒன்று|இரண்டு|மூன்று|நான்கு|ஐந்து|ஆறு|ஏழு|எட்டு|ஒன்பது|பத்து)',
            'time_words': r'(இன்று|நேற்று|நாளை|காலை|மாலை|இரவு|பகல்)'
        },
        'punctuation': ['।', '॥', '؍', '؎', '؏'],
        'scoring': {
            'script_threshold_high': 0.4,
            'script_threshold_medium': 0.15,
            'script_score_high': 0.7,
            'script_score_medium': 0.4,
            'common_word_multiplier': 0.1,
            'common_word_max': 0.2,
            'numeral_multiplier': 0.02,
            'numeral_max': 0.08,
            'pattern_multiplier': 0.05,
            'pattern_max': 0.1,
            'honorific_bonus': 0.03,
            'time_word_bonus': 0.02
        }
    },
    'english': {
        'common_words': [
            'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 
            'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as',
            'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have',
            'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not',
            'will', 'can', 'would', 'could', 'should', 'may', 'might',
            'company', 'business', 'document', 'report', 'analysis'
        ],
        'patterns': {
            'latin_chars': r'[a-zA-Z]',
            'contractions': r"(n't|'s|'re|'ve|'ll|'d|'m)",
            'articles': r'\b(the|a|an)\b',
            'prepositions': r'\b(in|on|at|by|for|with|from|to|of|about|through|during)\b',
            'modal_verbs': r'\b(can|could|will|would|shall|should|may|might|must)\b',
            'common_suffixes': r'\b\w+(ing|ed|er|est|ly|tion|sion|ness|ment|ful|less)\b',
            'business_terms': r'\b(company|corporation|inc|ltd|llc|organization|department)\b'
        },
        'scoring': {
            'word_ratio_threshold': 0.15,
            'word_score': 0.5,
            'latin_threshold': 0.85,
            'latin_score': 0.3,
            'fallback_score': 0.2,
            'contraction_multiplier': 0.05,
            'contraction_max': 0.1,
            'preposition_bonus': 0.02,
            'modal_verb_bonus': 0.02,
            'suffix_bonus': 0.03
        }
    }
}

# Language detection configuration with enhanced settings
LANGUAGE_CONFIG = {
    'confidence_thresholds': {
        'very_high': 0.85,
        'high': 0.7,
        'medium': 0.5,
        'low': 0.3,
        'very_low': 0.1
    },
    'analysis_settings': {
        'sample_blocks': 75,
        'min_text_length': 150,
        'max_analysis_chars': 5000,
        'supported_languages': ['japanese', 'german', 'tamil', 'english'],
        'fallback_language': 'english',
        'enable_fuzzy_matching': True,
        'min_confidence_difference': 0.1,  # Minimum difference between top languages
        'max_analysis_time_seconds': 5.0,  # Timeout for analysis
        'enable_statistical_analysis': True
    },
    'display_names': {
        'japanese': 'Japanese (日本語)',
        'german': 'German (Deutsch)', 
        'tamil': 'Tamil (தமிழ்)',
        'english': 'English'
    },
    'language_codes': {
        'japanese': 'ja',
        'german': 'de',
        'tamil': 'ta',
        'english': 'en'
    },
    'native_names': {
        'japanese': '日本語',
        'german': 'Deutsch',
        'tamil': 'தமிழ்',
        'english': 'English'
    },
    'script_families': {
        'japanese': ['hiragana', 'katakana', 'kanji', 'latin'],
        'german': ['latin'],
        'tamil': ['tamil'],
        'english': ['latin']
    },
    'difficulty_levels': {
        'japanese': 'high',    # Complex script mixing
        'german': 'medium',    # Complex grammar but Latin script
        'tamil': 'high',       # Unique script
        'english': 'low'       # Simple patterns
    }
}

# Language-specific processing adjustments with enhanced configuration
LANGUAGE_PROCESSING_CONFIG = {
    'japanese': {
        'title_patterns': [
            r'^[第]\d+[章節条項款目]',  # Chapter/Section markers
            r'^\d+\.\d+',  # Numbered sections
            r'^[●○▲△◆◇■□]',  # Bullet points
            r'^[一二三四五六七八九十]+[、．]',  # Japanese numerals
            r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]+[、．]',  # Roman numerals in Japanese
            r'^[（\(]\d+[）\)]',  # Parenthesized numbers
            r'^[あいうえお]\)',  # Hiragana bullets
            r'^[アイウエオ]\)'   # Katakana bullets
        ],
        'heading_indicators': [
            '章', '節', '項', '条', '款', '目', '編', '部', '巻',
            '序章', '概要', '背景', '方法', '結果', '考察', '結論', '付録',
            'はじめに', 'おわりに', '参考文献', '索引', '謝辞'
        ],
        'font_weight_threshold': 0.7,
        'spacing_multiplier': 1.2,
        'confidence_threshold': 0.4,
        'text_direction': 'mixed',  # Can be vertical or horizontal
        'character_spacing_sensitive': True,
        'script_mixing_tolerance': 0.8
    },
    'german': {
        'title_patterns': [
            r'^\d+\.?\s+[A-ZÄÖÜ]',  # Numbered headings
            r'^[IVXLCDM]+\.?\s+[A-ZÄÖÜ]',  # Roman numerals
            r'^[A-ZÄÖÜ][a-zäöüß\s]+:$',  # Colon endings
            r'^§\s*\d+',  # Paragraph markers
            r'^Art\.\s*\d+',  # Article markers
            r'^Abs\.\s*\d+',  # Absatz (paragraph) markers
            r'^[a-z]\)',  # Lowercase letter bullets
            r'^[A-Z]\)',  # Uppercase letter bullets
            r'^\([a-z]\)',  # Parenthesized letters
            r'^Kapitel\s+\d+',  # Chapter markers
            r'^Abschnitt\s+\d+'  # Section markers
        ],
        'heading_indicators': [
            'Kapitel', 'Abschnitt', 'Teil', 'Punkt', 'Absatz', 'Artikel',
            'Einführung', 'Überblick', 'Zusammenfassung', 'Schlussfolgerung',
            'Hintergrund', 'Methodik', 'Ergebnisse', 'Diskussion',
            'Anhang', 'Literatur', 'Bibliographie', 'Index', 'Inhaltsverzeichnis'
        ],
        'font_weight_threshold': 0.65,
        'spacing_multiplier': 1.0,
        'confidence_threshold': 0.4,
        'text_direction': 'ltr',
        'capitalization_sensitive': True,
        'compound_word_detection': True
    },
    'tamil': {
        'title_patterns': [
            r'^\d+\.',  # Numbered sections
            r'^[௧௨௩௪௫௬௭௮௯௰]+\.',  # Tamil numerals
            r'^அத்தியாயம்\s*\d*',  # Chapter
            r'^பகுதி\s*\d*',  # Section
            r'^பிரிவு\s*\d*',  # Division
            r'^[க-ன][ா-ௌ]*:',  # Colon endings
            r'^[●○▲△]',  # Bullet points
            r'^\([௧௨௩௪௫]\)',  # Parenthesized Tamil numerals
            r'^[அஇஉ]\)',  # Tamil vowel bullets
            r'^[கசதப]\)'   # Tamil consonant bullets
        ],
        'heading_indicators': [
            'அத்தியாயம்', 'பகுதி', 'பிரிவு', 'கட்டுரை', 'பகுதி',
            'அறிமுகம்', 'கண்ணோட்டம்', 'சுருக்கம்', 'முடிவு', 'பின்னணி',
            'முறைமை', 'முடிவுகள்', 'விவாதம்', 'இணைப்பு',
            'குறிப்புகள்', 'பட்டியல்', 'அட்டவணை', 'நன்றி'
        ],
        'font_weight_threshold': 0.7,
        'spacing_multiplier': 1.1,
        'confidence_threshold': 0.4,
        'text_direction': 'ltr',
        'script_detection_required': True,
        'unicode_normalization': True
    },
    'english': {
        'title_patterns': [
            r'^\d+\.?\s+[A-Z]',  # Numbered headings
            r'^[IVXLCDM]+\.?\s+[A-Z]',  # Roman numerals
            r'^[A-Z][a-z\s]+:$',  # Colon endings
            r'^Chapter\s+\d+',  # Chapter markers
            r'^Section\s+\d+',  # Section markers
            r'^Part\s+\d+',  # Part markers
            r'^Article\s+\d+',  # Article markers
            r'^Appendix\s+[A-Z]',  # Appendix markers
            r'^[a-z]\)',  # Lowercase bullets
            r'^[A-Z]\)',  # Uppercase bullets
            r'^\([a-z]\)',  # Parenthesized letters
            r'^\([0-9]+\)',  # Parenthesized numbers
            r'^[•·▪▫]'  # Various bullet points
        ],
        'heading_indicators': [
            'Chapter', 'Section', 'Part', 'Article', 'Paragraph',
            'Introduction', 'Overview', 'Summary', 'Conclusion', 'Background',
            'Methodology', 'Results', 'Discussion', 'Appendix',
            'References', 'Bibliography', 'Index', 'Glossary', 'Abstract',
            'Acknowledgments', 'Table of Contents', 'Preface', 'Foreword'
        ],
        'font_weight_threshold': 0.6,
        'spacing_multiplier': 1.0,
        'confidence_threshold': 0.4,
        'text_direction': 'ltr',
        'case_sensitive_analysis': True,
        'contraction_awareness': True
    }
}

# Additional configuration for specialized document types
DOCUMENT_TYPE_LANGUAGE_CONFIG = {
    'technical_manual': {
        'enhanced_patterns': {
            'japanese': [r'^仕様', r'^手順', r'^注意', r'^警告'],
            'german': [r'^Spezifikation', r'^Verfahren', r'^Hinweis', r'^Warnung'],
            'tamil': [r'^விவரக்குறிப்பு', r'^செயல்முறை', r'^குறிப்பு', r'^எச்சரிக்கை'],
            'english': [r'^Specification', r'^Procedure', r'^Note', r'^Warning']
        },
        'weight_adjustments': {
            'japanese': 1.1,
            'german': 1.05,
            'tamil': 1.1,
            'english': 1.0
        }
    },
    'business_document': {
        'enhanced_patterns': {
            'japanese': [r'^会議', r'^報告', r'^提案', r'^計画'],
            'german': [r'^Besprechung', r'^Bericht', r'^Vorschlag', r'^Plan'],
            'tamil': [r'^கூட்டம்', r'^அறிக்கை', r'^பரிந்துரை', r'^திட்டம்'],
            'english': [r'^Meeting', r'^Report', r'^Proposal', r'^Plan']
        },
        'weight_adjustments': {
            'japanese': 1.0,
            'german': 1.0,
            'tamil': 1.05,
            'english': 1.0
        }
    },
    'form_document': {
        'enhanced_patterns': {
            'japanese': [r'^氏名', r'^住所', r'^電話', r'^記入'],
            'german': [r'^Name', r'^Adresse', r'^Telefon', r'^Ausfüllen'],
            'tamil': [r'^பெயர்', r'^முகவரி', r'^தொலைபேசி', r'^நிரப்பு'],
            'english': [r'^Name', r'^Address', r'^Phone', r'^Fill']
        },
        'weight_adjustments': {
            'japanese': 1.0,
            'german': 1.0,
            'tamil': 1.0,
            'english': 1.0
        }
    }
}

# Performance optimization settings
PERFORMANCE_CONFIG = {
    'max_pattern_cache_size': 1000,
    'enable_pattern_compilation': True,
    'parallel_analysis_threshold': 10000,  # Character threshold for parallel processing
    'memory_limit_mb': 100,
    'timeout_seconds': 30,
    'enable_profiling': False
}

# Validation and quality assurance settings
VALIDATION_CONFIG = {
    'minimum_pattern_matches': 3,
    'cross_validation_enabled': True,
    'statistical_significance_threshold': 0.05,
    'enable_sanity_checks': True,
    'fallback_on_detection_failure': True,
    'log_detection_details': True
}

# Export configuration for easy access
__all__ = [
    'LANGUAGE_PATTERNS',
    'LANGUAGE_CONFIG', 
    'LANGUAGE_PROCESSING_CONFIG',
    'DOCUMENT_TYPE_LANGUAGE_CONFIG',
    'PERFORMANCE_CONFIG',
    'VALIDATION_CONFIG'
]

# Version information
LANGUAGE_PATTERNS_VERSION = '1.2.0'
LAST_UPDATED = '2025-01-27'
SUPPORTED_LANGUAGES_COUNT = len(LANGUAGE_PATTERNS)

# Quick access functions for external modules
def get_supported_languages() -> List[str]:
    """Get list of all supported languages"""
    return list(LANGUAGE_PATTERNS.keys())

def get_language_display_name(language_code: str) -> str:
    """Get display name for a language code"""
    return LANGUAGE_CONFIG.get('display_names', {}).get(language_code, language_code.title())

def get_language_native_name(language_code: str) -> str:
    """Get native name for a language code"""
    return LANGUAGE_CONFIG.get('native_names', {}).get(language_code, language_code.title())

def is_language_supported(language_code: str) -> bool:
    """Check if a language is supported"""
    return language_code in LANGUAGE_PATTERNS

def get_confidence_thresholds() -> Dict[str, float]:
    """Get confidence level thresholds"""
    return LANGUAGE_CONFIG.get('confidence_thresholds', {})

def validate_language_config() -> Dict[str, Any]:
    """Validate the language configuration integrity"""
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    try:
        # Check if all languages have required components
        for lang in LANGUAGE_PATTERNS:
            if lang not in LANGUAGE_PROCESSING_CONFIG:
                validation_results['warnings'].append(f"Missing processing config for {lang}")
            
            if lang not in LANGUAGE_CONFIG.get('display_names', {}):
                validation_results['warnings'].append(f"Missing display name for {lang}")
        
        # Check pattern integrity
        for lang, patterns in LANGUAGE_PATTERNS.items():
            if 'scoring' not in patterns:
                validation_results['errors'].append(f"Missing scoring config for {lang}")
                validation_results['valid'] = False
        
        validation_results['summary'] = {
            'total_languages': len(LANGUAGE_PATTERNS),
            'total_errors': len(validation_results['errors']),
            'total_warnings': len(validation_results['warnings'])
        }
        
    except Exception as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f'Validation failed: {str(e)}')
    
    return validation_results

if __name__ == '__main__':
    # Self-validation when run as script
    print(f"Language Patterns Configuration v{LANGUAGE_PATTERNS_VERSION}")
    print(f"Last Updated: {LAST_UPDATED}")
    print(f"Supported Languages: {get_supported_languages()}")
    
    validation = validate_language_config()
    if validation['valid']:
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
        for error in validation['errors']:
            print(f"  Error: {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  Warning: {warning}")
