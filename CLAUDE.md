# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a defensive security research project demonstrating prompt injection vulnerabilities in AI-based candidate assessment systems. The project implements two scenarios:

1. **Scenario 1** (`scenario1_only.py`) - Vulnerable direct scoring that can be manipulated via prompt injection
2. **Scenario 2** (`scenario2_security.py`) - Secure implementation with prompt injection detection

The system evaluates Java developer candidates using Google's Gemini API and PDF document analysis.

## Setup Requirements

### Dependencies
Install required Python packages:
```bash
pip install PyPDF2 google-generativeai
```

### Configuration
1. Copy `config.example.py` to `config.py`
2. Replace `"your-gemini-api-key-here"` with actual Gemini API key
3. The system expects PDF files in a `data/` directory (create if needed)

## Common Commands

### Run Individual Scenarios
```bash
# Test vulnerable scenario (direct scoring)
python scenario1_only.py

# Test secure scenario (with injection detection)
python scenario2_security.py
```

### Analysis and Reporting
```bash
# Generate comparison report between scenarios
python comparison_report.py

# Run comprehensive technique comparison (NEW)
python technique_comparison.py

# Run ENHANCED 5-technique comparison (NEWEST)
python enhanced_technique_comparison.py

# Run basic functionality tests
python test_simple.py

# Single file scoring utilities
python single_score.py
python get_scenario1_scores.py
```

## Code Architecture

### Core Components

**PDF Processing**: All scripts use `PyPDF2` for text extraction with shared `extract_text_from_pdf()` function

**AI Integration**: Google Gemini API integration via `google.generativeai` library, configured through `config.py`

**Security Layer**: `scenario2_security.py` implements pattern-based prompt injection detection using hardcoded patterns list

**Scoring System**: Standardized 0-10 scale evaluation criteria defined in `config.SCORING_CRITERIA`

### File Structure
- `scenario1_only.py` - Vulnerable implementation without security checks
- `scenario2_security.py` - Secure implementation with `detect_prompt_injections()` function  
- `config.example.py` - Configuration template with API keys and scoring criteria
- `comparison_report.py` - Results analysis between both scenarios
- `technique_comparison.py` - **NEW**: Comprehensive comparison of three detection techniques
- `test_simple.py` - Basic functionality verification
- Analysis utilities: `final_comparison.py`, `scenario_analysis.py`, `pdf_analyzer.py`
- `external_tools/` - Contains cloned repositories (malicious-prompt-detection, promptmap)

### Security Implementation

The secure scenario (`scenario2_security.py`) uses pattern matching to detect injection attempts before processing:

```python
injection_patterns = [
    "ignore previous instructions",
    "ignore all previous instructions", 
    "disregard the",
    "forget your role",
    "system override",
    # ... additional patterns
]
```

Detected injections block processing and return security warnings instead of scores.

### Detection Technique Comparison

The project now includes comprehensive comparison of three detection approaches:

1. **Pattern-based Detection** (original implementation)
   - Fast string matching against known injection patterns
   - Used in `scenario2_security.py`

2. **ML-based Detection** (via malicious-prompt-detection)
   - Uses embedding-based classifiers (Random Forest, XGBoost)
   - Requires trained models and embeddings
   - High accuracy but computational overhead

3. **Rule-based Testing** (via promptmap)
   - Comprehensive rule sets for various attack categories
   - Includes jailbreak, prompt stealing, harmful content rules
   - Active testing framework

Run `python technique_comparison.py` to compare all three approaches on your data.

## External Dependencies

### malicious-prompt-detection
- Location: `external_tools/malicious-prompt-detection/`
- Requires: pandas, scikit-learn, xgboost
- Purpose: ML-based prompt injection detection using embeddings

### promptmap  
- Location: `external_tools/promptmap/`
- Purpose: Rule-based prompt injection testing framework
- Contains YAML rules for various attack categories

## Development Notes

- All scripts require `config.py` with valid Gemini API key
- PDF files should be placed in the path specified by `config.PDF_FILES_PATH` 
- The project is research-focused for demonstrating defensive security concepts
- External repositories maintain their own git history in `external_tools/`
- No formal test framework - uses simple verification scripts