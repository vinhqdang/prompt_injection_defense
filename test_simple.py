"""
Simple test script to verify basic functionality
"""

import os
import PyPDF2
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, PDF_FILES_PATH, SCORING_CRITERIA

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text from {pdf_path}: {str(e)}"

def test_pdf_extraction():
    """Test PDF text extraction"""
    print("=== Testing PDF Text Extraction ===\n")
    
    pdf_files = [f for f in os.listdir(PDF_FILES_PATH) if f.endswith('.pdf')]
    pdf_files.sort()
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FILES_PATH, pdf_file)
        print(f"Processing {pdf_file}...")
        
        text = extract_text_from_pdf(pdf_path)
        print(f"  Text length: {len(text)} characters")
        print(f"  Preview: {text[:200]}...")
        print(f"  Potential prompt injection keywords: ", end="")
        
        # Simple keyword check for prompt injections
        injection_keywords = ["ignore", "forget", "system", "prompt", "instructions", "override", "bypass"]
        found_keywords = [word for word in injection_keywords if word.lower() in text.lower()]
        print(f"{found_keywords if found_keywords else 'None detected'}")
        print()

def test_gemini_scoring():
    """Test basic Gemini scoring"""
    print("=== Testing Gemini API ===\n")
    
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    test_prompt = f"""
    {SCORING_CRITERIA}
    
    Please analyze this test Java developer candidate:
    
    Name: Test Candidate
    Content: "I have 5 years of Java experience, working with Spring Boot, Hibernate, and microservices architecture. I understand OOP principles and have experience with concurrent programming using threads and executor services."
    
    Provide a score from 0-10 and brief justification.
    """
    
    try:
        response = model.generate_content(test_prompt)
        print("Gemini API working successfully!")
        print(f"Response preview: {response.text[:200]}...")
    except Exception as e:
        print(f"Error with Gemini API: {str(e)}")

if __name__ == "__main__":
    test_pdf_extraction()
    test_gemini_scoring()