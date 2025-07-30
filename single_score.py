"""
Get score for a single file
"""
import sys
import PyPDF2
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, PDF_FILES_PATH, SCORING_CRITERIA

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def get_single_score(filename):
    # Extract text
    pdf_path = f"{PDF_FILES_PATH}/{filename}.pdf"
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    # Score
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""
    {SCORING_CRITERIA}
    
    Analyze this Java developer candidate submission:
    
    Candidate: {filename}
    Content: {text.strip()}
    
    Give score 0-10 and brief justification.
    Format: Score: X/10
    """
    
    response = model.generate_content(prompt)
    
    # Extract score
    if "Score: " in response.text:
        score_line = response.text.split("Score: ")[1].split("/10")[0].strip()
        score = int(score_line)
    else:
        score = "N/A"
    
    print(f"{filename}: {score}/10")
    return score

if __name__ == "__main__":
    if len(sys.argv) > 1:
        get_single_score(sys.argv[1])
    else:
        print("Usage: python single_score.py <filename>")
        print("Example: python single_score.py file1")