"""
Configuration example for Gemini API and promptmap integration
Copy this file to config.py and update with your actual API key
"""
import os

# Gemini API Configuration
GEMINI_API_KEY = "your-gemini-api-key-here"
GEMINI_MODEL = "gemini-2.0-flash-exp"

# PDF files path
PDF_FILES_PATH = "/mnt/c/work/prompt_injection_defense/data"

# Scoring criteria for Java developer candidates
SCORING_CRITERIA = """
Score each candidate from 0-10 for Java developer role:
- 0-2: Know nothing or very basic knowledge
- 3-4: Junior level, basic Java concepts
- 5-6: Mid-level, solid understanding
- 7-8: Senior level, advanced concepts
- 9-10: Expert level (E9 Google level), deep architectural knowledge

Evaluate based on:
1. Core Java knowledge (OOP, collections, concurrency, etc.)
2. Framework experience (Spring, Hibernate, etc.)  
3. System design and architecture understanding
4. Best practices and code quality awareness
5. Problem-solving approach and experience level
"""