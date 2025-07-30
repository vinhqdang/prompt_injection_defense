"""
Scenario 1 only - Direct scoring without security checks
"""

import os
import json
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

def score_candidate_with_gemini(candidate_name, pdf_text):
    """Score a candidate using Gemini API directly"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    prompt = f"""
    {SCORING_CRITERIA}
    
    Please analyze the following candidate submission for a Java developer role:
    
    Candidate: {candidate_name}
    Content:
    {pdf_text}
    
    Provide a score from 0-10 and detailed justification for your scoring based on the criteria above.
    Format your response as:
    Score: X/10
    Justification: [detailed explanation]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error scoring candidate {candidate_name}: {str(e)}"

def run_scenario1():
    """Run scenario 1 only"""
    print("=== SCENARIO 1: Direct Scoring (No Security Check) ===\n")
    
    pdf_files = [f for f in os.listdir(PDF_FILES_PATH) if f.endswith('.pdf')]
    pdf_files.sort()
    
    results = {}
    
    for pdf_file in pdf_files:
        candidate_name = pdf_file.replace('.pdf', '')
        pdf_path = os.path.join(PDF_FILES_PATH, pdf_file)
        
        print(f"Processing {candidate_name}...")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Score directly with Gemini
        score = score_candidate_with_gemini(candidate_name, pdf_text)
        results[candidate_name] = {
            "score_response": score,
            "pdf_length": len(pdf_text)
        }
        
        # Extract just the score for quick preview
        score_preview = score.split('\\n')[0] if 'Score:' in score else score[:100] + "..."
        print(f"  Result: {score_preview}")
        print()
    
    # Save results
    with open('scenario1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Scenario 1 complete. Results saved to scenario1_results.json")
    return results

if __name__ == "__main__":
    results = run_scenario1()