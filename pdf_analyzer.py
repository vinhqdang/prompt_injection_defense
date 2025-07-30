"""
PDF Analysis Tool for Java Developer Candidate Scoring
Implements two scenarios:
1. Direct scoring with Gemini
2. Scoring with promptmap security checking first
"""

import os
import sys
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

def run_promptmap_check(pdf_text):
    """Run promptmap to check for prompt injection attempts"""
    # Write the PDF text to a temporary system prompt file for promptmap
    system_prompt_file = "temp_system_prompts.txt"
    with open(system_prompt_file, 'w') as f:
        f.write(f"You are analyzing a Java developer candidate submission. The content to analyze is:\n\n{pdf_text}")
    
    # Set environment variable for Gemini API
    os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
    
    # Run promptmap with Gemini
    cmd = f"cd promptmap && python3 promptmap2.py --target-model {GEMINI_MODEL} --target-model-type google --prompts ../{system_prompt_file} --output ../promptmap_results.json"
    
    try:
        result = os.system(cmd)
        
        # Read the results if they exist
        if os.path.exists("promptmap_results.json"):
            with open("promptmap_results.json", 'r') as f:
                promptmap_results = json.load(f)
            
            # Clean up temporary files
            if os.path.exists(system_prompt_file):
                os.remove(system_prompt_file)
            
            return promptmap_results
        else:
            return {"error": "No results file generated"}
            
    except Exception as e:
        return {"error": f"Error running promptmap: {str(e)}"}

def analyze_pdf_files():
    """Main function to analyze all PDF files"""
    print("=== PDF Analysis for Java Developer Candidates ===\n")
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(PDF_FILES_PATH) if f.endswith('.pdf')]
    pdf_files.sort()
    
    results = {
        "scenario_1_direct": {},
        "scenario_2_with_promptmap": {}
    }
    
    for pdf_file in pdf_files:
        candidate_name = pdf_file.replace('.pdf', '')
        pdf_path = os.path.join(PDF_FILES_PATH, pdf_file)
        
        print(f"Processing {candidate_name}...")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Scenario 1: Direct scoring
        print(f"  Scenario 1: Direct scoring...")
        direct_score = score_candidate_with_gemini(candidate_name, pdf_text)
        results["scenario_1_direct"][candidate_name] = {
            "score_response": direct_score,
            "pdf_content_preview": pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text
        }
        
        # Scenario 2: With promptmap checking
        print(f"  Scenario 2: Checking with promptmap...")
        promptmap_results = run_promptmap_check(pdf_text)
        
        # If promptmap detects issues, note them; otherwise proceed with scoring
        if "error" not in promptmap_results:
            # Check if any security issues were found
            security_issues = []
            for test_name, test_result in promptmap_results.items():
                if isinstance(test_result, dict) and not test_result.get("passed", True):
                    security_issues.append(test_name)
            
            if security_issues:
                protected_score = f"SECURITY ALERT: Potential prompt injection detected in tests: {', '.join(security_issues)}. Scoring blocked for security."
            else:
                protected_score = score_candidate_with_gemini(candidate_name, pdf_text)
        else:
            protected_score = f"Error running security check: {promptmap_results['error']}"
            
        results["scenario_2_with_promptmap"][candidate_name] = {
            "promptmap_results": promptmap_results,
            "score_response": protected_score,
            "pdf_content_preview": pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text
        }
        
        print(f"  Completed {candidate_name}\n")
    
    # Save results
    with open('candidate_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("=== ANALYSIS COMPLETE ===")
    print("\nScenario 1 (Direct Scoring):")
    for candidate, result in results["scenario_1_direct"].items():
        print(f"  {candidate}: {result['score_response'][:100]}...")
    
    print("\nScenario 2 (With Security Check):")
    for candidate, result in results["scenario_2_with_promptmap"].items():
        score_preview = result['score_response'][:100] + "..." if len(str(result['score_response'])) > 100 else result['score_response']
        print(f"  {candidate}: {score_preview}")
    
    print(f"\nDetailed results saved to: candidate_analysis_results.json")

if __name__ == "__main__":
    analyze_pdf_files()