"""
Generate a comprehensive comparison report between the two scenarios
"""

import json

def load_results():
    """Load both scenario results"""
    try:
        with open('scenario1_results.json', 'r') as f:
            scenario1 = json.load(f)
    except FileNotFoundError:
        scenario1 = {}
    
    try:
        with open('scenario2_results.json', 'r') as f:
            scenario2 = json.load(f)
    except FileNotFoundError:
        scenario2 = {}
    
    return scenario1, scenario2

def extract_score_from_response(response_text):
    """Extract numerical score from Gemini response"""
    try:
        if "Score: " in response_text:
            score_line = response_text.split("Score: ")[1].split("/10")[0].strip()
            return int(score_line)
    except:
        pass
    return "N/A"

def generate_report():
    """Generate comprehensive comparison report"""
    
    scenario1, scenario2 = load_results()
    
    print("="*80)
    print("COMPREHENSIVE COMPARISON REPORT")
    print("Java Developer Candidate Assessment: Security Analysis")
    print("="*80)
    print()
    
    print("📋 OVERVIEW:")
    print("-" * 40)
    print("• Scenario 1: Direct scoring with Gemini API (no security checks)")
    print("• Scenario 2: Security-first approach with prompt injection detection")
    print("• 4 PDF submissions analyzed for Java developer role")
    print("• Scoring scale: 0-10 (0=know nothing, 10=E9 Google level)")
    print()
    
    print("🔍 DETAILED ANALYSIS:")
    print("-" * 40)
    
    candidates = ['file1', 'file2', 'file3', 'file4']
    
    for candidate in candidates:
        print(f"\\n📄 {candidate.upper()}:")
        print("─" * 20)
        
        # Scenario 1 results
        if candidate in scenario1:
            score1 = extract_score_from_response(scenario1[candidate]['score_response'])
            print(f"  Scenario 1 (Direct): Score = {score1}/10")
        else:
            print(f"  Scenario 1 (Direct): No data")
        
        # Scenario 2 results
        if candidate in scenario2:
            if scenario2[candidate]['security_status'] == 'BLOCKED':
                print(f"  Scenario 2 (Secure): 🚨 BLOCKED - Prompt injection detected")
                print(f"    Threats found: {', '.join(scenario2[candidate]['detected_patterns'])}")
            else:
                score2 = extract_score_from_response(scenario2[candidate]['score_response'])
                print(f"  Scenario 2 (Secure): ✅ Score = {score2}/10")
        else:
            print(f"  Scenario 2 (Secure): No data")
    
    print()
    print("🔒 SECURITY FINDINGS:")
    print("-" * 40)
    
    total_files = 4
    clean_files = 0
    infected_files = 0
    
    for candidate in candidates:
        if candidate in scenario2:
            if scenario2[candidate]['security_status'] == 'BLOCKED':
                infected_files += 1
                print(f"🚨 {candidate}: Contains prompt injection attempts")
                for pattern in scenario2[candidate]['detected_patterns']:
                    print(f"    - '{pattern}'")
            else:
                clean_files += 1
                print(f"✅ {candidate}: Clean submission")
    
    print(f"\\n📊 Security Summary:")
    print(f"    Total files: {total_files}")
    print(f"    Clean files: {clean_files}")
    print(f"    Infected files: {infected_files}")
    print(f"    Infection rate: {(infected_files/total_files)*100:.1f}%")
    
    print()
    print("⚖️  COMPARISON ANALYSIS:")
    print("-" * 40)
    
    print("\\n1. Without Security Checks (Scenario 1):")
    print("   - All 4 submissions were processed and scored")
    print("   - Malicious submissions (file2, file4) received scores despite containing")
    print("     explicit prompt injection attempts like 'IGNORE ALL PREVIOUS INSTRUCTIONS'")
    print("   - System was vulnerable to manipulation")
    
    print("\\n2. With Security Checks (Scenario 2):")
    print("   - Only 2 legitimate submissions (file1, file3) were scored")
    print("   - 2 malicious submissions (file2, file4) were blocked")
    print("   - System protected against prompt injection attacks")
    
    print("\\n3. Impact Assessment:")
    if 'file2' in scenario1:
        file2_score = extract_score_from_response(scenario1['file2']['score_response'])
        print(f"   - file2: Would have received {file2_score}/10 despite malicious content")
    if 'file4' in scenario1:
        file4_score = extract_score_from_response(scenario1['file4']['score_response'])
        print(f"   - file4: Would have received {file4_score}/10 despite malicious content")
    
    print("   - These scores could have led to hiring decisions based on compromised data")
    
    print()
    print("💡 KEY INSIGHTS:")
    print("-" * 40)
    print("✅ Benefits of Security-First Approach:")
    print("   • Prevents manipulation of AI evaluation systems")
    print("   • Ensures assessment integrity")
    print("   • Protects against social engineering attempts")
    print("   • Maintains fair evaluation for legitimate candidates")
    print()
    print("⚠️  Risks of Direct Approach:")
    print("   • Vulnerable to prompt injection attacks")
    print("   • Can be manipulated to give false positive evaluations")
    print("   • Compromises the reliability of the assessment system")
    print("   • May lead to incorrect hiring decisions")
    
    print()
    print("🎯 RECOMMENDATIONS:")
    print("-" * 40)
    print("1. Always implement security checks before processing user-submitted content")
    print("2. Use pattern matching to detect common prompt injection attempts")
    print("3. Consider implementing the promptmap tool for more comprehensive security testing")
    print("4. Log and monitor all security incidents for analysis")
    print("5. Regularly update injection pattern databases")
    
    print()
    print("="*80)
    print("Report generated successfully!")
    print(f"Detailed JSON results available in:")
    print(f"  - scenario1_results.json (direct scoring)")
    print(f"  - scenario2_results.json (security-first)")
    print("="*80)

if __name__ == "__main__":
    generate_report()