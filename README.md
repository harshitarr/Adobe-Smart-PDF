# PDF Outline Extractor

## Challenge Overview
This solution addresses **Adobe's "Connecting the Dots" Challenge - Round 1A: Document Structure Extraction**

### Challenge Requirements
- Extract hierarchical document structure from PDF files
- Output JSON format with title and headings (H1, H2, H3, H4)
- Process PDFs up to 50 pages
- Performance constraint: ≤10 seconds per document
- CPU-only execution with ≤200MB model size
- Offline operation (no internet access)
- Multilingual support with bonus scoring

## Solution Architecture

### Core Components

#### 1. **Document Analysis Pipeline**
```bash
PDF Input → Text Extraction → Language Detection → Document Classification → Structure Analysis → JSON Output
```

#### 2. **Multilingual Language Support**
- **English**: Pattern-based analysis with business/technical vocabulary
- **Japanese**: Hiragana/Katakana/Kanji script detection with honorifics
- **German**: Umlaut patterns, compound words, capitalization rules
- **Tamil**: Script-specific Unicode range analysis

#### 3. **Document Type Classification**
- **Form Documents**: Application forms, declarations
- **Technical Manuals**: Structured documentation with TOC
- **Business Documents**: RFPs, proposals, appendices
- **Program Documents**: Educational pathways, curricula
- **Invitation Documents**: Events, announcements

#### 4. **Hierarchical Structure Detection**
- **Font Analysis**: Size clustering, formatting detection
- **Spatial Analysis**: Layout patterns, positioning
- **Semantic Classification**: Content vs. structure differentiation
- **Pattern Matching**: Numbering schemes, formatting conventions

## Installation & Setup

### Prerequisites
```bash
Python 3.9+
Docker (for containerized execution)
```

### Dependencies
```bash
pip install -r requirements.txt
```


**Core Libraries:**
- `PyPDF2`: PDF parsing
- `pdfplumber`: Advanced text extraction
- `scikit-learn`: Font clustering analysis
- `numpy`: Numerical operations

### Directory Structure
```bash
PDF_OUTLINE_EXTRACTOR/
│
├── 📁 app/
│   ├── __init__.py
│   ├── main.py                          # Entry point
│   │
│   ├── 📁 core/
│   │   ├── __init__.py
│   │   ├── document_classifier.py       # Document type classification
│   │   ├── title_extractor.py          # Smart title extraction
│   │   ├── outline_extractor.py        # Hierarchical outline extraction
│   │   └── json_formatter.py           # Output formatting
│   │
│   ├── 📁 analyzers/
│   │   ├── __init__.py
│   │   ├── font_analyzer.py            # Font clustering and analysis
│   │   ├── spatial_analyzer.py         # Layout and positioning analysis
│   │   ├── text_analyzer.py            # Text pattern recognition
│   │   ├── structure_analyzer.py       # Document structure detection
│   │   └── language_analyzer.py        # Language detection and analysis
│   │
│   ├── 📁 classifiers/
│   │   ├── __init__.py
│   │   ├── content_filter.py           # Content vs structure filtering
│   │   ├── heading_detector.py         # Multi-level heading detection
│   │   ├── pattern_matcher.py          # Numbering and formatting patterns
│   │   └── semantic_classifier.py      # Semantic content classification
│   │
│   ├── 📁 processors/
│   │   ├── __init__.py
│   │   ├── base_processor.py           # Base processor class
│   │   ├── form_processor.py           # Form document processing
│   │   ├── manual_processor.py         # Technical manual processing
│   │   ├── business_processor.py       # Business document processing
│   │   ├── program_processor.py        # Program/curriculum processing
│   │   └── invitation_processor.py     # Event/invitation processing
│   │
│   └── 📁 utils/
│       ├── __init__.py
│       ├── pdf_utils.py               # PDF parsing utilities
│       ├── text_utils.py              # Text processing utilities
│       └── validation_utils.py        # Output validation
│
├── 📁 models/
│   ├── __init__.py
│   ├── document_types.py              # Document type definitions
│   ├── heading_patterns.py            # Heading pattern definitions
│   └── extraction_rules.py            # Extraction rule engine 
│
├── 📁 config/
│   ├── __init__.py
│   ├── settings.py                    # Configuration settings
│   └── patterns.py                    # Pattern matching configurations
│   └── language_patterns.py
│
├── 📁 input/                          # Input PDF files
│
├── 📁 output/                         # Output JSON files
│   └── 📁 Language Info/              # Language-specific outputs
│
├── 📁 logs/                           # Logging directory
│   ├── extraction.log                 # Extraction process logs
│   ├── classification.log             # Document classification logs
│   └── errors.log                     # Error logs
│
├── 📁 tests/
│   ├── __init__.py
│   ├── test_classifier.py            # Classification tests
│   ├── test_extractors.py            # Extraction tests
│   └── test_processors.py            # Processor tests
│
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
└── README.md                          # Project documentationPDF_OUTLINE_EXTRACTOR/
```

### Build container
```bash
docker build --no-cache -t pdf_outline_extractor .
```

### Run with volume mounting
```bash
docker run --rm -v "${PWD}\input:/app/input" -v "${PWD}\output:/app/output" -v "${PWD}\logs:/app/logs" pdf_outline_extractor
```


### Performance Specifications

| Metric | Requirement | Our Solution |
|--------|-------------|--------------|
| **Processing Time** | ≤10 seconds | ✅ 3-8 seconds avg |
| **Hardcode** | Required | ✅ Zero-Hardcoding |
| **Page Limit** | 50 pages | ✅ Enforced |
| **CPU Only** | Required | ✅ No GPU dependencies |
| **Offline** | Required | ✅ No internet calls |
| **Multilingual Support** | Bonus | ✅ Implemented |

## Output Format

### Main Output (JSON)
```bash
{
"title": "Document Title Here",
"outline": [
{
  "level": "H1",
  "text": "Introduction",
  "page": 1
},
{
  "level": "H2",
  "text": "Background Information",
  "page": 2
}
  ]
    }
```

### Language Information (Separate File)

```bash
{
    "detected_language": "japanese",
    "confidence": 1.0,
    "display_name": "Japanese (日本語)",
    "language_code": "ja",
    "confidence_level": "very_high",
    "language_processing_applied": true,
    "analysis_confidence_level": "very_high",
    "detection_scores": {
        "japanese": 1.0,
        "german": 0.2,
        "tamil": 0.0,
        "english": 0.2
    },
    "text_sample_length": 807,
    "threshold_applied": 0.4
}

```

## Algorithm Details

### 1. **Language Detection Pipeline**

def detect_language(text_blocks):
 - Script-based analysis for Japanese/Tamil
 - Pattern matching for German umlauts/compounds
 - Statistical analysis for English
 - Confidence scoring with thresholds


### 2. **Heading Detection Strategy**

def detect_headings(blocks, doc_type, font_analysis):
 - Multi-factor scoring:
 - Font size (25 points max)
 - Visual prominence (spatial analysis)
 - Pattern matching (numbering, formatting)
 - Semantic classification
 - Language-specific adjustments


### 3. **Title Extraction Logic**
- def extract_title(blocks, doc_type):
 - Document-type specific patterns
 - Font size hierarchy analysis
 - Position-based scoring
 - Language-aware processing


## Multilingual Capabilities

### Language-Specific Features

#### **Japanese Support**
- **Script Detection**: Hiragana (ひらがな), Katakana (カタカナ), Kanji (漢字)
- **Pattern Recognition**: Chapter markers (第1章), section numbering
- **Honorifics**: Recognition of さん, 様, 殿 patterns
- **Business Terms**: 株式会社, 有限会社 identification

#### **German Support** 
- **Umlaut Processing**: ä, ö, ü, ß character analysis
- **Compound Words**: Detection of long German compounds (12+ chars)
- **Capitalization**: German noun capitalization patterns
- **Grammar Patterns**: Articles (der, die, das), modal verbs

#### **Tamil Support**
- **Unicode Range**: Tamil script block (U+0B80-U+0BFF)
- **Numerals**: Tamil numeral recognition (௧௨௩...)
- **Honorifics**: திரு, திருமதி identification
- **Question Words**: எப்போது, எங்கே, எப்படி patterns

## Performance Optimizations

### 1. **Font Clustering Algorithm**
 - K-means clustering for font hierarchy
 - Threshold-based size classification
 - Bold/italic formatting detection
 - Language-adjusted font weight scoring


### 2. **Memory Management**
- **Text Block Limiting**: Process max 10,000 blocks
- **Memory Cap**: 512MB maximum usage
- **Timeout Protection**: 30-second analysis limit
- **Resource Cleanup**: PDF handle management

### 3. **Processing Efficiency**
- **Early Termination**: Skip invalid/oversized PDFs
- **Lazy Loading**: On-demand component initialization
- **Pattern Compilation**: Regex pre-compilation
- **Caching**: Language detection result caching



## Scoring Alignment

### Adobe Challenge Scoring (45 points)

#### **Heading Detection Accuracy (25 points)**
-  **Pattern-based detection**: Numbering, formatting, positioning
-  **Multi-level hierarchy**: H1, H2, H3, H4 classification
-  **Document-type awareness**: Adaptive extraction rules
-  **Font analysis integration**: Size-based importance scoring

#### **Performance Compliance (10 points)**
-  **Processing speed**: 3-8 seconds average (≤10 required)
-  **Memory efficiency**: ~45MB model size (≤200MB limit)
-  **Page handling**: 50-page limit enforcement
-  **CPU-only execution**: No GPU dependencies

#### **Multilingual Handling Bonus (10 points)**
-  **4 Languages**: English, Japanese, German, Tamil
-  **Script detection**: Unicode range analysis
-  **Pattern adaptation**: Language-specific rules
-  **Confidence scoring**: Threshold-based validation

## Architecture Highlights

### **Modular Design**
- **Processors**: Document-type specific handling
- **Analyzers**: Font, spatial, text, structure analysis
- **Classifiers**: Content filtering, semantic classification
- **Language Support**: Pluggable language detection

### **Error Handling**
- **Graceful Degradation**: Fallback to English processing
- **Resource Protection**: Memory and timeout limits
- **Input Validation**: PDF format and size checking
- **Exception Logging**: Comprehensive error tracking

### **Extensibility**
- **New Languages**: Easy addition via pattern configuration
- **Document Types**: Pluggable processor architecture
- **Analysis Methods**: Modular analyzer components
- **Output Formats**: Configurable JSON formatting

## Contributors
- Harshita Ravindran Revathi
- Jayanth.C.R

**Note**: This solution prioritizes **accuracy**, **performance**, and **multilingual support** as per Adobe's Challenge requirements while maintaining clean, maintainable code architecture.



