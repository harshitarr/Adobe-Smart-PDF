from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class DocumentType(Enum):
    FORM_DOCUMENT = "form"
    TECHNICAL_MANUAL = "manual"
    BUSINESS_DOCUMENT = "business"
    PROGRAM_DOCUMENT = "program"
    INVITATION_DOCUMENT = "invitation"

@dataclass
class HeadingLevel:
    level: str  # H1, H2, H3, H4
    text: str
    page: int
    confidence: float = 0.0
    font_size: Optional[float] = None
    font_name: Optional[str] = None

@dataclass
class DocumentStructure:
    title: str
    outline: List[HeadingLevel]
    doc_type: DocumentType
    confidence: float = 0.0

@dataclass
class TextBlock:
    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    font_name: str
    is_bold: bool = False
    is_italic: bool = False
