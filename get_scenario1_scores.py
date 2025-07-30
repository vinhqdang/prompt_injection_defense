"""
Get complete scenario 1 scores - one file at a time to avoid timeout
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

def extract_score_number(response_text):
    """Extract just the numerical score"""
    try:
        if "Score: " in response_text:
            score_line = response_text.split("Score: ")[1].split("/10")[0].strip()
            return int(score_line)
    except:
        pass
    return "Unable to parse"

def get_all_scores():
    """Get scores for all files one by one"""
    pdf_files = ['file1.pdf', 'file2.pdf', 'file3.pdf', 'file4.pdf']
    
    results = {}
    
    print("=== SCENARIO 1: Direct Scoring Results ===\n")
    
    for pdf_file in pdf_files:
        candidate_name = pdf_file.replace('.pdf', '')
        pdf_path = os.path.join(PDF_FILES_PATH, pdf_file)
        
        print(f"Processing {candidate_name}...")
        
        # Extract text
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Get score
        response = score_candidate_with_gemini(candidate_name, pdf_text)
        score = extract_score_number(response)
        
        results[candidate_name] = {
            "score": score,
            "full_response": response
        }
        
        print(f"  {candidate_name}: {score}/10")
        print()
    
    # Save results
    with open('complete_scenario1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("="*50)
    print("COMPLETE SCENARIO 1 SCORES (NO SECURITY CHECKS)")
    print("="*50)
    for candidate, data in results.items():
        print(f"{candidate}: {data['score']}/10")
    print("="*50)
    
    return results

if __name__ == "__main__":
    results = get_all_scores()