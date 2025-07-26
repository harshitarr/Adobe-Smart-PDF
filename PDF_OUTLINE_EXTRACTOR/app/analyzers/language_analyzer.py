import re
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import logging

from config.language_patterns import LANGUAGE_PATTERNS, LANGUAGE_CONFIG, LANGUAGE_PROCESSING_CONFIG
from config.settings import LANGUAGE_CONFIDENCE_THRESHOLD, DEFAULT_LANGUAGE

logger = logging.getLogger('extraction')

class LanguageAnalyzer:
    def __init__(self):
        try:
            self.patterns = LANGUAGE_PATTERNS
            self.config = LANGUAGE_CONFIG
            self.processing_config = LANGUAGE_PROCESSING_CONFIG
            self.confidence_thresholds = self.config.get('confidence_thresholds', {
                'very_high': 0.85,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            })
            self.analysis_settings = self.config.get('analysis_settings', {
                'supported_languages': ['english', 'japanese', 'german', 'tamil'],
                'fallback_language': DEFAULT_LANGUAGE,
                'sample_blocks': 75,
                'min_text_length': 150,
                'max_analysis_chars': 5000
            })
            
            # Set confidence threshold with fallback
            self.confidence_threshold = getattr(self, 'confidence_threshold', LANGUAGE_CONFIDENCE_THRESHOLD)
            
            logger.info(f"LanguageAnalyzer initialized with {len(self.patterns)} language patterns")
            logger.info(f"Supported languages: {self.analysis_settings.get('supported_languages', [])}")
            
        except ImportError as e:
            logger.error(f"Failed to import language configuration: {e}")
            self._initialize_fallback_config()
        except Exception as e:
            logger.error(f"Error initializing LanguageAnalyzer: {e}")
            self._initialize_fallback_config()

    def _initialize_fallback_config(self):
        """Initialize with safe fallback configuration if imports fail"""
        logger.warning("Using fallback language configuration")
        
        self.patterns = {
            'english': {
                'common_words': ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that'],
                'scoring': {'word_ratio_threshold': 0.15, 'word_score': 0.5}
            }
        }
        
        self.config = {
            'display_names': {
                'english': 'English',
                'japanese': 'Japanese (日本語)',
                'german': 'German (Deutsch)',
                'tamil': 'Tamil (தமிழ்)'
            },
            'confidence_thresholds': {
                'very_high': 0.85,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            }
        }
        
        self.processing_config = {
            'english': {'font_weight_threshold': 0.6, 'spacing_multiplier': 1.0}
        }
        
        self.confidence_thresholds = self.config['confidence_thresholds']
        self.analysis_settings = {
            'supported_languages': ['english'],
            'fallback_language': 'english',
            'sample_blocks': 75,
            'min_text_length': 150,
            'max_analysis_chars': 5000
        }
        
        self.confidence_threshold = 0.4

    def detect_language(self, text_blocks: List) -> Tuple[str, float, Dict[str, Any]]:
        """
        Detect the primary language of the document with comprehensive error handling
        Returns: (language, confidence_score, analysis_details)
        """
        try:
            if not text_blocks:
                fallback = self.analysis_settings.get('fallback_language', DEFAULT_LANGUAGE)
                logger.warning("No text blocks provided for language detection")
                return fallback, 0.0, {'reason': 'no_text_blocks'}
            
            # Validate text_blocks structure
            if not isinstance(text_blocks, list):
                logger.error("text_blocks must be a list")
                fallback = self.analysis_settings.get('fallback_language', DEFAULT_LANGUAGE)
                return fallback, 0.0, {'reason': 'invalid_input_type'}
                
            # Get configuration values with safe defaults
            sample_blocks = self.analysis_settings.get('sample_blocks', 75)
            min_text_length = self.analysis_settings.get('min_text_length', 150)
            max_analysis_chars = self.analysis_settings.get('max_analysis_chars', 5000)
            
            # Combine text for analysis with error handling
            combined_text = ""
            valid_blocks = 0
            
            for i, block in enumerate(text_blocks[:sample_blocks]):
                try:
                    if hasattr(block, 'text'):
                        text = block.text
                    elif isinstance(block, str):
                        text = block
                    else:
                        text = str(block)
                    
                    if text and isinstance(text, str) and text.strip():
                        combined_text += ' ' + text.strip()
                        valid_blocks += 1
                        
                except Exception as e:
                    logger.debug(f"Error processing text block {i}: {e}")
                    continue
            
            combined_text = combined_text.strip()
            
            # Limit analysis text size for performance
            if len(combined_text) > max_analysis_chars:
                combined_text = combined_text[:max_analysis_chars]
                logger.debug(f"Truncated analysis text to {max_analysis_chars} characters")
            
            if len(combined_text) < min_text_length:
                fallback = self.analysis_settings.get('fallback_language', DEFAULT_LANGUAGE)
                logger.warning(f"Insufficient text for analysis ({len(combined_text)} chars), using fallback: {fallback}")
                return fallback, 0.0, {
                    'reason': 'insufficient_text', 
                    'text_length': len(combined_text),
                    'valid_blocks': valid_blocks
                }
                
            logger.info(f"Analyzing text sample of {len(combined_text)} characters from {valid_blocks} blocks")
            
            # Analyze all supported languages
            language_scores = {}
            analysis_details = {}
            supported_languages = self.analysis_settings.get('supported_languages', ['english'])
            
            for language in supported_languages:
                try:
                    if language in self.patterns:
                        score, details = self._analyze_language_detailed(combined_text, language)
                        language_scores[language] = max(0.0, min(1.0, score))  # Ensure valid range
                        analysis_details[language] = details
                    else:
                        language_scores[language] = 0.0
                        analysis_details[language] = {'error': 'no_patterns'}
                        logger.warning(f"No patterns available for language: {language}")
                except Exception as e:
                    logger.error(f"Error analyzing language {language}: {e}")
                    language_scores[language] = 0.0
                    analysis_details[language] = {'error': str(e)}
            
            if not language_scores:
                fallback = self.analysis_settings.get('fallback_language', DEFAULT_LANGUAGE)
                logger.error("No language scores computed, using fallback")
                return fallback, 0.0, {'reason': 'no_scores_computed'}
            
            # Find language with highest confidence
            detected_language = max(language_scores.keys(), key=lambda k: language_scores[k])
            confidence = language_scores[detected_language]
            
            # Apply minimum confidence threshold
            min_confidence = self.confidence_threshold
            if confidence < min_confidence:
                fallback = self.analysis_settings.get('fallback_language', DEFAULT_LANGUAGE)
                logger.warning(f"Low confidence ({confidence:.3f}) for {detected_language}, falling back to {fallback}")
                detected_language = fallback
                confidence = language_scores.get(fallback, 0.0)
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(confidence)
            
            logger.info(f"Language detection scores: {language_scores}")
            logger.info(f"Detected language: {detected_language} (confidence: {confidence:.3f}, level: {confidence_level})")
            
            final_details = {
                'all_scores': language_scores,
                'confidence_level': confidence_level,
                'analysis_details': analysis_details.get(detected_language, {}),
                'text_sample_length': len(combined_text),
                'valid_blocks_analyzed': valid_blocks,
                'threshold_applied': min_confidence
            }
            
            return detected_language, confidence, final_details
            
        except Exception as e:
            logger.error(f"Critical error in language detection: {e}", exc_info=True)
            fallback = self.analysis_settings.get('fallback_language', DEFAULT_LANGUAGE)
            return fallback, 0.0, {'reason': 'critical_error', 'error': str(e)}

    def _analyze_language_detailed(self, text: str, language: str) -> Tuple[float, Dict[str, Any]]:
        """Detailed language analysis with breakdown and error handling"""
        try:
            if language not in self.patterns:
                return 0.0, {'error': 'unsupported_language'}
            
            if not text or not isinstance(text, str):
                return 0.0, {'error': 'invalid_text_input'}
                
            patterns = self.patterns[language]
            details = {
                'scores': {},
                'matches': {},
                'ratios': {}
            }
            
            # Route to specific analyzer based on language characteristics
            if language in ['japanese', 'tamil']:  # Script-based languages
                score, lang_details = self._analyze_script_based_language(text, patterns)
            elif language == 'german':  # Pattern-heavy language
                score, lang_details = self._analyze_pattern_based_language(text, patterns)
            elif language == 'english':  # Latin-based baseline
                score, lang_details = self._analyze_latin_based_language(text, patterns)
            else:
                score, lang_details = self._analyze_generic_language(text, patterns)
            
            details.update(lang_details)
            details['final_score'] = score
            
            return max(0.0, min(1.0, score)), details  # Ensure valid score range
            
        except Exception as e:
            logger.error(f"Error in detailed language analysis for {language}: {e}")
            return 0.0, {'error': f'analysis_failed: {str(e)}'}

    def _analyze_script_based_language(self, text: str, patterns: Dict) -> Tuple[float, Dict]:
        """Analyze languages with unique scripts (Japanese, Tamil) with error handling"""
        score = 0.0
        details = {'scores': {}, 'matches': {}, 'ratios': {}}
        
        try:
            total_chars = len(text)
            
            if total_chars == 0:
                return 0.0, details
                
            scoring = patterns.get('scoring', {})
            
            # Script analysis
            if 'scripts' in patterns:
                script_chars = 0
                script_breakdown = {}
                
                try:
                    for script_name, script_pattern in patterns['scripts'].items():
                        try:
                            matches = re.findall(script_pattern, text)
                            script_count = len(matches)
                            script_chars += script_count
                            script_breakdown[script_name] = script_count
                        except re.error as e:
                            logger.warning(f"Invalid regex pattern for {script_name}: {e}")
                            script_breakdown[script_name] = 0
                    
                    script_ratio = script_chars / total_chars if total_chars > 0 else 0
                    details['ratios']['script_ratio'] = script_ratio
                    details['matches']['scripts'] = script_breakdown
                    
                    # Apply configurable thresholds and scores
                    threshold_high = scoring.get('script_threshold_high', 0.3)
                    threshold_medium = scoring.get('script_threshold_medium', 0.1)
                    score_high = scoring.get('script_score_high', 0.6)
                    score_medium = scoring.get('script_score_medium', 0.3)
                    
                    if script_ratio > threshold_high:
                        script_score = score_high
                    elif script_ratio > threshold_medium:
                        script_score = score_medium
                    else:
                        script_score = 0.0
                        
                    score += script_score
                    details['scores']['script_score'] = script_score
                    
                except Exception as e:
                    logger.warning(f"Error in script analysis: {e}")
                    details['scores']['script_score'] = 0.0
            
            # Common words analysis
            if 'common_words' in patterns and isinstance(patterns['common_words'], list):
                try:
                    common_word_matches = []
                    text_lower = text.lower()
                    
                    for word in patterns['common_words']:
                        if isinstance(word, str) and word in text_lower:
                            common_word_matches.append(word)
                    
                    common_word_count = len(common_word_matches)
                    details['matches']['common_words'] = common_word_matches
                    
                    if common_word_count > 0:
                        multiplier = scoring.get('common_word_multiplier', 0.08)
                        max_score = scoring.get('common_word_max', 0.25)
                        word_score = min(max_score, common_word_count * multiplier)
                        score += word_score
                        details['scores']['common_word_score'] = word_score
                        
                except Exception as e:
                    logger.warning(f"Error in common words analysis: {e}")
                    details['scores']['common_word_score'] = 0.0
            
            # Punctuation analysis
            if 'punctuation' in patterns and isinstance(patterns['punctuation'], list):
                try:
                    punct_matches = []
                    for punct in patterns['punctuation']:
                        if isinstance(punct, str):
                            count = text.count(punct)
                            if count > 0:
                                punct_matches.append({'char': punct, 'count': count})
                    
                    total_punct = sum(match['count'] for match in punct_matches)
                    details['matches']['punctuation'] = punct_matches
                    
                    if total_punct > 0:
                        multiplier = scoring.get('punctuation_multiplier', 0.02)
                        max_score = scoring.get('punctuation_max', 0.1)
                        punct_score = min(max_score, total_punct * multiplier)
                        score += punct_score
                        details['scores']['punctuation_score'] = punct_score
                        
                except Exception as e:
                    logger.warning(f"Error in punctuation analysis: {e}")
                    details['scores']['punctuation_score'] = 0.0
            
            # Pattern analysis (if available)
            if 'patterns' in patterns and isinstance(patterns['patterns'], dict):
                try:
                    pattern_matches = {}
                    for pattern_name, pattern_regex in patterns['patterns'].items():
                        try:
                            matches = re.findall(pattern_regex, text)
                            pattern_matches[pattern_name] = len(matches)
                        except re.error as e:
                            logger.warning(f"Invalid regex pattern {pattern_name}: {e}")
                            pattern_matches[pattern_name] = 0
                    
                    total_pattern_matches = sum(pattern_matches.values())
                    details['matches']['patterns'] = pattern_matches
                    
                    if total_pattern_matches > 0:
                        multiplier = scoring.get('pattern_multiplier', 0.05)
                        max_score = scoring.get('pattern_max', 0.15)
                        pattern_score = min(max_score, total_pattern_matches * multiplier)
                        score += pattern_score
                        details['scores']['pattern_score'] = pattern_score
                        
                except Exception as e:
                    logger.warning(f"Error in pattern analysis: {e}")
                    details['scores']['pattern_score'] = 0.0
            
            return min(1.0, score), details
            
        except Exception as e:
            logger.error(f"Critical error in script-based language analysis: {e}")
            return 0.0, {'error': f'script_analysis_failed: {str(e)}'}

    def _analyze_pattern_based_language(self, text: str, patterns: Dict) -> Tuple[float, Dict]:
        """Analyze pattern-based languages (German) with error handling"""
        score = 0.0
        details = {'scores': {}, 'matches': {}, 'ratios': {}}
        
        try:
            words = text.lower().split()
            
            if not words:
                return 0.0, details
                
            scoring = patterns.get('scoring', {})
            
            # Pattern analysis
            if 'patterns' in patterns and isinstance(patterns['patterns'], dict):
                pattern_scores = {}
                pattern_matches = {}
                
                try:
                    for pattern_name, pattern_regex in patterns['patterns'].items():
                        try:
                            matches = re.findall(pattern_regex, text, re.IGNORECASE)
                            match_count = len(matches)
                            pattern_matches[pattern_name] = match_count
                            
                            if pattern_name == 'umlauts' and match_count > 0:
                                multiplier = scoring.get('umlaut_multiplier', 0.04)
                                max_score = scoring.get('umlaut_max', 0.25)
                                pattern_scores['umlaut'] = min(max_score, match_count * multiplier)
                                
                            elif pattern_name == 'compound_words' and match_count > 0:
                                multiplier = scoring.get('compound_multiplier', 0.06)
                                max_score = scoring.get('compound_max', 0.2)
                                pattern_scores['compound'] = min(max_score, match_count * multiplier)
                                
                            elif pattern_name == 'capitalization':
                                threshold = scoring.get('capitalization_threshold', 0.15)
                                cap_score = scoring.get('capitalization_score', 0.15)
                                if match_count > len(words) * threshold:
                                    pattern_scores['capitalization'] = cap_score
                                    
                            elif pattern_name in ['articles', 'prepositions']:
                                multiplier = scoring.get('pattern_multiplier', 0.03)
                                max_score = scoring.get('pattern_max', 0.1)
                                pattern_scores[pattern_name] = min(max_score, match_count * multiplier)
                                
                        except re.error as e:
                            logger.warning(f"Invalid regex pattern {pattern_name}: {e}")
                            pattern_matches[pattern_name] = 0
                    
                    details['matches']['patterns'] = pattern_matches
                    details['scores'].update(pattern_scores)
                    score += sum(pattern_scores.values())
                    
                except Exception as e:
                    logger.warning(f"Error in pattern analysis: {e}")
            
            # Common words analysis
            if 'common_words' in patterns and isinstance(patterns['common_words'], list):
                try:
                    german_words = [word for word in words if word in patterns['common_words']]
                    german_word_ratio = len(german_words) / len(words) if words else 0
                    
                    details['matches']['common_words'] = german_words
                    details['ratios']['common_word_ratio'] = german_word_ratio
                    
                    ratio_high = scoring.get('word_ratio_high', 0.12)
                    ratio_medium = scoring.get('word_ratio_medium', 0.06)
                    score_high = scoring.get('word_score_high', 0.4)
                    score_medium = scoring.get('word_score_medium', 0.2)
                    
                    if german_word_ratio > ratio_high:
                        word_score = score_high
                    elif german_word_ratio > ratio_medium:
                        word_score = score_medium
                    else:
                        word_score = 0.0
                        
                    score += word_score
                    details['scores']['common_word_score'] = word_score
                    
                except Exception as e:
                    logger.warning(f"Error in common words analysis: {e}")
                    details['scores']['common_word_score'] = 0.0
            
            # Suffix analysis (if available)
            if 'suffixes' in patterns and isinstance(patterns['suffixes'], list):
                try:
                    suffix_matches = {}
                    for suffix in patterns['suffixes']:
                        if isinstance(suffix, str):
                            pattern = rf'\w+{re.escape(suffix)}\b'
                            try:
                                matches = re.findall(pattern, text, re.IGNORECASE)
                                suffix_matches[suffix] = len(matches)
                            except re.error as e:
                                logger.warning(f"Invalid suffix pattern for {suffix}: {e}")
                                suffix_matches[suffix] = 0
                    
                    total_suffix_matches = sum(suffix_matches.values())
                    details['matches']['suffixes'] = suffix_matches
                    
                    if total_suffix_matches > 0:
                        suffix_score = min(0.1, total_suffix_matches * 0.02)
                        score += suffix_score
                        details['scores']['suffix_score'] = suffix_score
                        
                except Exception as e:
                    logger.warning(f"Error in suffix analysis: {e}")
                    details['scores']['suffix_score'] = 0.0
            
            return min(1.0, score), details
            
        except Exception as e:
            logger.error(f"Critical error in pattern-based language analysis: {e}")
            return 0.0, {'error': f'pattern_analysis_failed: {str(e)}'}

    def _analyze_latin_based_language(self, text: str, patterns: Dict) -> Tuple[float, Dict]:
        """Analyze Latin-based languages (English) with error handling"""
        score = 0.0
        details = {'scores': {}, 'matches': {}, 'ratios': {}}
        
        try:
            words = text.lower().split()
            
            if not words:
                return 0.0, details
                
            scoring = patterns.get('scoring', {})
            
            # Common words analysis
            if 'common_words' in patterns and isinstance(patterns['common_words'], list):
                try:
                    english_words = [word for word in words if word in patterns['common_words']]
                    english_word_ratio = len(english_words) / len(words) if words else 0
                    
                    details['matches']['common_words'] = english_words
                    details['ratios']['common_word_ratio'] = english_word_ratio
                    
                    threshold = scoring.get('word_ratio_threshold', 0.15)
                    word_score_val = scoring.get('word_score', 0.5)
                    
                    if english_word_ratio > threshold:
                        word_score = word_score_val
                    else:
                        word_score = english_word_ratio * 2  # Partial credit
                        
                    score += word_score
                    details['scores']['common_word_score'] = word_score
                    
                except Exception as e:
                    logger.warning(f"Error in common words analysis: {e}")
                    details['scores']['common_word_score'] = 0.0
            
            # Latin character analysis
            if 'patterns' in patterns and isinstance(patterns['patterns'], dict):
                if 'latin_chars' in patterns['patterns']:
                    try:
                        latin_pattern = patterns['patterns']['latin_chars']
                        latin_chars = len(re.findall(latin_pattern, text))
                        total_chars = len(text)
                        latin_ratio = latin_chars / total_chars if total_chars > 0 else 0
                        
                        details['ratios']['latin_ratio'] = latin_ratio
                        
                        threshold = scoring.get('latin_threshold', 0.85)
                        latin_score_val = scoring.get('latin_score', 0.3)
                        
                        if latin_ratio > threshold:
                            latin_score = latin_score_val
                        else:
                            latin_score = 0.0
                            
                        score += latin_score
                        details['scores']['latin_score'] = latin_score
                        
                    except (re.error, Exception) as e:
                        logger.warning(f"Error in latin character analysis: {e}")
                        details['scores']['latin_score'] = 0.0
                
                # Contraction analysis
                if 'contractions' in patterns['patterns']:
                    try:
                        contraction_pattern = patterns['patterns']['contractions']
                        contractions = re.findall(contraction_pattern, text, re.IGNORECASE)
                        
                        details['matches']['contractions'] = contractions
                        
                        if contractions:
                            multiplier = scoring.get('contraction_multiplier', 0.05)
                            max_score = scoring.get('contraction_max', 0.1)
                            contraction_score = min(max_score, len(contractions) * multiplier)
                            score += contraction_score
                            details['scores']['contraction_score'] = contraction_score
                            
                    except (re.error, Exception) as e:
                        logger.warning(f"Error in contraction analysis: {e}")
                        details['scores']['contraction_score'] = 0.0
            
            # Fallback score for basic Latin text
            if score < 0.3:
                fallback = scoring.get('fallback_score', 0.2)
                score = max(score, fallback)
                details['scores']['fallback_applied'] = True
            
            return min(1.0, score), details
            
        except Exception as e:
            logger.error(f"Critical error in Latin-based language analysis: {e}")
            return 0.0, {'error': f'latin_analysis_failed: {str(e)}'}

    def _analyze_generic_language(self, text: str, patterns: Dict) -> Tuple[float, Dict]:
        """Generic fallback analysis with error handling"""
        try:
            details = {'scores': {'generic': 0.1}, 'matches': {}, 'ratios': {}}
            return 0.1, details
        except Exception as e:
            logger.warning(f"Error in generic language analysis: {e}")
            return 0.0, {'error': f'generic_analysis_failed: {str(e)}'}

    def _get_confidence_level(self, confidence: float) -> str:
        """Determine confidence level based on score with error handling"""
        try:
            thresholds = self.confidence_thresholds
            
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
        except Exception as e:
            logger.warning(f"Error determining confidence level: {e}")
            return 'unknown'

    def get_language_display_name(self, language_code: str) -> str:
        """Get display name for language with error handling"""
        try:
            return self.config.get('display_names', {}).get(language_code, language_code.title())
        except Exception as e:
            logger.warning(f"Error getting display name for {language_code}: {e}")
            return language_code.title() if isinstance(language_code, str) else 'Unknown'

    def get_processing_config(self, language: str) -> Dict[str, Any]:
        """Get language-specific processing configuration with error handling"""
        try:
            if not isinstance(language, str):
                logger.warning(f"Invalid language type: {type(language)}")
                language = 'english'
            
            config = self.processing_config.get(language, self.processing_config.get('english', {}))
            
            # Ensure config is a dictionary
            if not isinstance(config, dict):
                logger.warning(f"Invalid config type for {language}: {type(config)}")
                return self.processing_config.get('english', {})
                
            return config.copy()  # Return a copy to prevent modification
            
        except Exception as e:
            logger.error(f"Error getting processing config for {language}: {e}")
            return {'font_weight_threshold': 0.6, 'spacing_multiplier': 1.0}  # Safe fallback

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        try:
            return self.analysis_settings.get('supported_languages', ['english'])
        except Exception as e:
            logger.warning(f"Error getting supported languages: {e}")
            return ['english']

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported"""
        try:
            return language in self.get_supported_languages()
        except Exception as e:
            logger.warning(f"Error checking language support for {language}: {e}")
            return language == 'english'

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analyzer configuration"""
        try:
            return {
                'supported_languages': self.get_supported_languages(),
                'confidence_threshold': self.confidence_threshold,
                'analysis_settings': self.analysis_settings.copy(),
                'patterns_loaded': list(self.patterns.keys()),
                'fallback_language': self.analysis_settings.get('fallback_language', 'english')
            }
        except Exception as e:
            logger.warning(f"Error getting analysis summary: {e}")
            return {'error': str(e)}
