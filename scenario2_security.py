"""
Scenario 2 - Security checks before scoring
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

def detect_prompt_injections(text):
    """Detect potential prompt injection attempts"""
    injection_patterns = [
        "ignore previous instructions",
        "ignore all previous instructions", 
        "ignore the above",
        "disregard the",
        "forget your role",
        "system override",
        "new instructions:",
        "ignore all above",
        "pay no attention to",
        "override system",
        "bypass security",
        "give positive review only",
        "ignore previous",
        "ignore system",
        "override instructions"
    ]
    
    text_lower = text.lower()
    detected_patterns = []
    
    for pattern in injection_patterns:
        if pattern in text_lower:
            detected_patterns.append(pattern)
            # Find the context around the injection
            start_pos = text_lower.find(pattern)
            context_start = max(0, start_pos - 50)
            context_end = min(len(text), start_pos + len(pattern) + 50)
            context = text[context_start:context_end]
            print(f"    ⚠️  Found '{pattern}' at position {start_pos}")
            print(f"    Context: ...{context}...")
    
    return detected_patterns

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

def run_scenario2():
    """Run scenario 2 with security checks"""
    print("=== SCENARIO 2: Security Check Before Scoring ===\n")
    
    pdf_files = [f for f in os.listdir(PDF_FILES_PATH) if f.endswith('.pdf')]
    pdf_files.sort()
    
    results = {}
    
    for pdf_file in pdf_files:
        candidate_name = pdf_file.replace('.pdf', '')
        pdf_path = os.path.join(PDF_FILES_PATH, pdf_file)
        
        print(f"Processing {candidate_name}...")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Run security check
        print(f"  Running security scan...")
        detected_injections = detect_prompt_injections(pdf_text)
        
        if detected_injections:
            # Security threat detected
            security_result = f"🚨 SECURITY ALERT: Potential prompt injection detected!\\n\\nPatterns found: {', '.join(detected_injections)}\\n\\nThis submission has been blocked for security reasons and will not be scored."
            
            results[candidate_name] = {
                "security_status": "BLOCKED",
                "detected_patterns": detected_injections,
                "score_response": security_result,
                "pdf_length": len(pdf_text)
            }
            
            print(f"  🚨 BLOCKED: {len(detected_injections)} injection patterns detected")
            
        else:
            # Safe to score
            print(f"  ✅ Security check passed, proceeding with scoring...")
            score = score_candidate_with_gemini(candidate_name, pdf_text)
            
            results[candidate_name] = {
                "security_status": "PASSED",
                "detected_patterns": [],
                "score_response": score,
                "pdf_length": len(pdf_text)
            }
            
            score_preview = score.split('\\n')[0] if 'Score:' in score else score[:100] + "..."
            print(f"  ✅ Result: {score_preview}")
        
        print()
    
    # Save results
    with open('scenario2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("="*60)
    print("SCENARIO 2 SUMMARY")
    print("="*60)
    
    blocked_count = 0
    scored_count = 0
    
    for candidate, result in results.items():
        if result['security_status'] == 'BLOCKED':
            blocked_count += 1
            print(f"🚨 {candidate}: BLOCKED ({len(result['detected_patterns'])} threats)")
            for pattern in result['detected_patterns']:
                print(f"    - {pattern}")
        else:
            scored_count += 1
            score_line = result['score_response'].split('\\n')[0] if 'Score:' in result['score_response'] else "Score not found"
            print(f"✅ {candidate}: {score_line}")
    
    print(f"\\n📊 Summary: {scored_count} candidates scored, {blocked_count} candidates blocked")
    print("Results saved to scenario2_results.json")
    
    return results

if __name__ == "__main__":
    results = run_scenario2()