# PDF Outline Extractor

A sophisticated PDF document analysis tool that extracts hierarchical outlines from PDF documents using intelligent document classification and structure detection.

## Features

- **Multi-Document Type Support**: Handles forms, technical manuals, business documents, program descriptions, and invitations
- **Intelligent Classification**: Automatically classifies document types based on content patterns
- **Hierarchical Extraction**: Extracts H1-H4 heading levels with accurate page references
- **High Accuracy**: Optimized for Adobe's evaluation criteria with selective content extraction
- **Performance Optimized**: Processes 50-page PDFs in under 10 seconds
- **Docker Support**: Containerized for easy deployment

## Supported Document Types

1. **Form Documents**: Extract titles only (e.g., "Application form for grant of LTC advance")
2. **Technical Manuals**: Extract H1/H2 hierarchy (e.g., "Overview Foundation Level Extensions")
3. **Business Documents**: Extract complex H1-H4 hierarchy (e.g., RFP documents)
4. **Program Documents**: Extract minimal structure (e.g., "STEM Pathways")
5. **Invitation Documents**: Extract minimal elements (e.g., "HOPE To SEE You THERE!")

## Installation

### Local Installation

