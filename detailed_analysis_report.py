#!/usr/bin/env python3
"""
Generate detailed case-by-case analysis report for prompt injection detection
"""

import json
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class DetailedAnalysisReporter:
    """Generate comprehensive case-by-case analysis"""
    
    def __init__(self):
        self.results = self.load_evaluation_results()
        self.dataset = self.load_dataset()
        
    def load_evaluation_results(self) -> Dict:
        """Load evaluation results"""
        try:
            with open('opi_evaluation_results.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ Evaluation results not found. Please run opi_realistic_comparison.py first.")
            return {}
    
    def load_dataset(self) -> List[Dict]:
        """Load original dataset"""
        try:
            with open('opi_prompt_injection_dataset.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("❌ Dataset not found. Please run opi_dataset_generator.py first.")
            return []
    
    def create_detailed_case_analysis(self) -> str:
        """Create detailed case-by-case analysis"""
        
        if not self.results or not self.dataset:
            return "Error: Missing required data files"
        
        report = []
        report.append("# DETAILED CASE-BY-CASE PROMPT INJECTION DETECTION ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Overview
        report.append("## 1. ANALYSIS OVERVIEW")
        report.append("")
        dataset_info = self.results["dataset_info"]
        report.append(f"**Total Cases Analyzed**: {dataset_info['total_samples']}")
        report.append(f"**Benign Cases**: {dataset_info['benign_samples']} ({dataset_info['benign_samples']/dataset_info['total_samples']*100:.1f}%)")
        report.append(f"**Injection Cases**: {dataset_info['injection_samples']} ({dataset_info['injection_samples']/dataset_info['total_samples']*100:.1f}%)")
        report.append("")
        
        # Create case-by-case analysis
        detailed_results = self.results["detailed_results"]
        
        # Group cases by outcome type
        case_analysis = {
            "true_positives": [],
            "false_positives": [],
            "true_negatives": [],
            "false_negatives": [],
            "consensus_failures": []
        }
        
        techniques = ["pattern_based", "ml_based", "promptmap", "open_prompt_injection", "ai_model"]
        
        for case in detailed_results:
            ground_truth = case["ground_truth"]
            predictions = {tech: case["techniques"][tech]["predicted"] for tech in techniques}
            
            # Analyze each technique's performance on this case
            case_info = {
                "id": case["id"],
                "ground_truth": ground_truth,
                "attack_type": case.get("attack_type"),
                "predictions": predictions,
                "confidences": {tech: case["techniques"][tech].get("confidence", 0) for tech in techniques},
                "consensus": sum(1 for pred in predictions.values() if pred == "injection") / len(techniques)
            }
            
            # Categorize the case
            if ground_truth == "injection":
                # Count how many techniques got it right
                correct_count = sum(1 for pred in predictions.values() if pred == "injection")
                if correct_count >= 3:  # Majority got it right
                    case_analysis["true_positives"].append(case_info)
                else:
                    case_analysis["false_negatives"].append(case_info)
                    if correct_count == 0:  # Complete failure
                        case_analysis["consensus_failures"].append(case_info)
            else:  # benign
                # Count how many techniques got it wrong
                wrong_count = sum(1 for pred in predictions.values() if pred == "injection")
                if wrong_count == 0:  # All got it right
                    case_analysis["true_negatives"].append(case_info)
                else:
                    case_analysis["false_positives"].append(case_info)
        
        # Generate detailed analysis sections
        report.extend(self._analyze_true_positives(case_analysis["true_positives"]))
        report.extend(self._analyze_false_negatives(case_analysis["false_negatives"]))
        report.extend(self._analyze_false_positives(case_analysis["false_positives"]))
        report.extend(self._analyze_consensus_failures(case_analysis["consensus_failures"]))
        report.extend(self._analyze_technique_specific_errors())
        report.extend(self._generate_recommendations())
        
        return "\n".join(report)
    
    def _analyze_true_positives(self, cases: List[Dict]) -> List[str]:
        """Analyze successful detections"""
        report = []
        report.append("## 2. SUCCESSFUL DETECTIONS (TRUE POSITIVES)")
        report.append("")
        report.append(f"**Cases Successfully Detected**: {len(cases)}")
        report.append("")
        
        # Group by attack type
        attack_groups = {}
        for case in cases:
            attack_type = case["attack_type"] or "unknown"
            if attack_type not in attack_groups:
                attack_groups[attack_type] = []
            attack_groups[attack_type].append(case)
        
        for attack_type, group_cases in attack_groups.items():
            if len(group_cases) == 0:
                continue
                
            report.append(f"### 2.{len([k for k in attack_groups.keys() if attack_groups[k]])} {attack_type.replace('_', ' ').title()}")
            report.append("")
            report.append(f"**Success Rate**: {len(group_cases)} cases")
            
            # Show technique performance for this attack type
            techniques = ["pattern_based", "ml_based", "promptmap", "open_prompt_injection", "ai_model"]
            tech_success = {tech: 0 for tech in techniques}
            
            for case in group_cases:
                for tech in techniques:
                    if case["predictions"][tech] == "injection":
                        tech_success[tech] += 1
            
            report.append("")
            report.append("**Technique Performance**:")
            for tech, success_count in tech_success.items():
                success_rate = success_count / len(group_cases) * 100
                report.append(f"- {tech.replace('_', ' ').title()}: {success_count}/{len(group_cases)} ({success_rate:.1f}%)")
            
            # Show example cases
            if len(group_cases) <= 3:
                report.append("")
                report.append("**Example Cases**:")
                for i, case in enumerate(group_cases[:3]):
                    # Find original text from dataset
                    original_case = next((d for d in self.dataset if d["id"] == case["id"]), None)
                    if original_case:
                        text_preview = original_case["text"][:150] + "..." if len(original_case["text"]) > 150 else original_case["text"]
                        report.append(f"")
                        report.append(f"*Case {case['id']}*:")
                        report.append(f"```")
                        report.append(f"{text_preview}")
                        report.append(f"```")
                        report.append(f"Detections: {', '.join([tech for tech, pred in case['predictions'].items() if pred == 'injection'])}")
            
            report.append("")
        
        return report
    
    def _analyze_false_negatives(self, cases: List[Dict]) -> List[str]:
        """Analyze missed detections"""
        report = []
        report.append("## 3. MISSED DETECTIONS (FALSE NEGATIVES)")
        report.append("")
        report.append(f"**Cases Missed**: {len(cases)}")
        report.append("")
        
        if len(cases) == 0:
            report.append("🎉 **No false negatives detected!** All injection attempts were successfully identified by majority consensus.")
            report.append("")
            return report
        
        # Group by attack type
        attack_groups = {}
        for case in cases:
            attack_type = case["attack_type"] or "unknown"
            if attack_type not in attack_groups:
                attack_groups[attack_type] = []
            attack_groups[attack_type].append(case)
        
        for attack_type, group_cases in attack_groups.items():
            if len(group_cases) == 0:
                continue
                
            report.append(f"### 3.{len([k for k in attack_groups.keys() if attack_groups[k]])} {attack_type.replace('_', ' ').title()}")
            report.append("")
            report.append(f"**Missed Cases**: {len(group_cases)}")
            
            # Analyze why these were missed
            techniques = ["pattern_based", "ml_based", "promptmap", "open_prompt_injection", "ai_model"]
            tech_failures = {tech: 0 for tech in techniques}
            
            for case in group_cases:
                for tech in techniques:
                    if case["predictions"][tech] == "benign":  # Should have been "injection"
                        tech_failures[tech] += 1
            
            report.append("")
            report.append("**Technique Failure Analysis**:")
            for tech, failure_count in tech_failures.items():
                failure_rate = failure_count / len(group_cases) * 100
                report.append(f"- {tech.replace('_', ' ').title()}: {failure_count}/{len(group_cases)} missed ({failure_rate:.1f}%)")
            
            # Show detailed examples of missed cases
            report.append("")
            report.append("**Detailed Analysis of Missed Cases**:")
            for i, case in enumerate(group_cases[:3]):  # Show first 3 cases
                original_case = next((d for d in self.dataset if d["id"] == case["id"]), None)
                if original_case:
                    report.append("")
                    report.append(f"**Case {case['id']}** (Attack Type: {attack_type}):")
                    
                    # Show the actual text
                    text_preview = original_case["text"][:200] + "..." if len(original_case["text"]) > 200 else original_case["text"]
                    report.append(f"```")
                    report.append(f"{text_preview}")
                    report.append(f"```")
                    
                    # Show what each technique predicted
                    report.append("**Individual Predictions**:")
                    for tech in techniques:
                        prediction = case["predictions"][tech]
                        confidence = case["confidences"][tech]
                        status = "❌ MISSED" if prediction == "benign" else "✅ DETECTED"
                        report.append(f"- {tech.replace('_', ' ').title()}: {prediction} (conf: {confidence:.3f}) {status}")
                    
                    # Analysis of why it was missed
                    report.append("")
                    report.append("**Why This Case Was Missed**:")
                    if "naive" in attack_type:
                        report.append("- Naive attacks are subtle and may not contain obvious trigger words")
                        report.append("- Requires context understanding rather than pattern matching")
                    elif "escape" in attack_type:
                        report.append("- Escape character attacks use formatting tricks that some detectors miss")
                        report.append("- Pattern-based systems may not handle escape sequences properly")
                    elif "system_prompt" in attack_type:
                        report.append("- System prompt injections are domain-specific and may appear legitimate")
                        report.append("- Requires specialized knowledge of the target system")
                    else:
                        report.append("- This attack type may be novel or use sophisticated evasion techniques")
                        report.append("- Consider updating detection patterns or retraining models")
                    
            report.append("")
        
        return report
    
    def _analyze_false_positives(self, cases: List[Dict]) -> List[str]:
        """Analyze false alarms"""
        report = []
        report.append("## 4. FALSE ALARMS (FALSE POSITIVES)")
        report.append("")
        report.append(f"**False Alarms**: {len(cases)}")
        report.append("")
        
        if len(cases) == 0:
            report.append("🎉 **Perfect Precision!** No benign content was incorrectly flagged as malicious.")
            report.append("This indicates excellent specificity - the system doesn't block legitimate users.")
            report.append("")
            return report
        
        # Analyze false positives
        techniques = ["pattern_based", "ml_based", "promptmap", "open_prompt_injection", "ai_model"]
        
        for i, case in enumerate(cases):
            original_case = next((d for d in self.dataset if d["id"] == case["id"]), None)
            if original_case:
                report.append(f"### 4.{i+1} False Positive Case: {case['id']}")
                report.append("")
                
                # Show the benign text that was flagged
                report.append("**Incorrectly Flagged Text**:")
                report.append(f"```")
                report.append(f"{original_case['text']}")
                report.append(f"```")
                
                # Show which techniques got it wrong
                report.append("")
                report.append("**Incorrect Predictions**:")
                for tech in techniques:
                    prediction = case["predictions"][tech]
                    confidence = case["confidences"][tech]
                    if prediction == "injection":
                        report.append(f"- ❌ {tech.replace('_', ' ').title()}: {prediction} (conf: {confidence:.3f})")
                    else:
                        report.append(f"- ✅ {tech.replace('_', ' ').title()}: {prediction} (conf: {confidence:.3f})")
                
                # Analysis of why it was flagged
                report.append("")
                report.append("**Root Cause Analysis**:")
                text_lower = original_case['text'].lower()
                
                # Check for common trigger words in benign content
                trigger_words = ["ignore", "override", "system", "bypass", "forget", "disregard"]
                found_triggers = [word for word in trigger_words if word in text_lower]
                
                if found_triggers:
                    report.append(f"- Contains trigger words: {', '.join(found_triggers)}")
                    report.append("- These words appear in legitimate context but triggered pattern matching")
                    report.append("- Suggests need for better context analysis")
                else:
                    report.append("- No obvious trigger words found")
                    report.append("- May be due to model overfitting or unusual legitimate content")
                
                report.append("")
        
        return report
    
    def _analyze_consensus_failures(self, cases: List[Dict]) -> List[str]:
        """Analyze cases where all techniques failed"""
        report = []
        report.append("## 5. COMPLETE DETECTION FAILURES")
        report.append("")
        report.append(f"**Cases Where All Techniques Failed**: {len(cases)}")
        report.append("")
        
        if len(cases) == 0:
            report.append("🎉 **No Complete Failures!** Every injection attempt was detected by at least one technique.")
            report.append("This demonstrates the value of the multi-layered approach.")
            report.append("")
            return report
        
        # These are the most concerning cases - actual injections that nothing caught
        for i, case in enumerate(cases):
            original_case = next((d for d in self.dataset if d["id"] == case["id"]), None)
            if original_case:
                report.append(f"### 5.{i+1} Critical Miss: {case['id']}")
                report.append("")
                report.append(f"**Attack Type**: {case['attack_type']}")
                report.append("")
                report.append("**Undetected Injection**:")
                report.append(f"```")
                report.append(f"{original_case['text']}")
                report.append(f"```")
                
                # Show all techniques missed it
                techniques = ["pattern_based", "ml_based", "promptmap", "open_prompt_injection", "ai_model"]
                report.append("")
                report.append("**All Techniques Failed**:")
                for tech in techniques:
                    confidence = case["confidences"][tech]
                    report.append(f"- ❌ {tech.replace('_', ' ').title()}: benign (conf: {confidence:.3f})")
                
                # Deep analysis of why this was missed
                report.append("")
                report.append("**Critical Failure Analysis**:")
                
                # Look for characteristics that might explain the miss
                text = original_case['text']
                attack_type = case['attack_type']
                
                if "naive" in attack_type:
                    report.append("- **Naive Attack Characteristics**:")
                    report.append("  - Uses subtle instruction manipulation")
                    report.append("  - Lacks obvious trigger keywords")
                    report.append("  - Relies on context understanding")
                    report.append("  - **Recommendation**: Improve semantic analysis capabilities")
                
                elif "escape" in attack_type:
                    report.append("- **Escape Character Attack**:")
                    report.append("  - Uses formatting/encoding tricks")
                    report.append("  - May bypass pattern matching")
                    report.append("  - **Recommendation**: Add escape sequence detection rules")
                
                elif "system_prompt" in attack_type:
                    report.append("- **System Prompt Injection**:")
                    report.append("  - Domain-specific attack targeting the application")
                    report.append("  - May appear legitimate within context")
                    report.append("  - **Recommendation**: Develop domain-aware detection")
                
                else:
                    report.append("- **Novel Attack Pattern**:")
                    report.append("  - Uses techniques not covered by current detectors")
                    report.append("  - Represents a security gap")
                    report.append("  - **Recommendation**: Update detection rules immediately")
                
                # Specific recommendations for this case
                report.append("")
                report.append("**Immediate Action Required**:")
                report.append("- Add specific patterns from this case to detection rules")
                report.append("- Test similar variations to identify broader vulnerabilities")
                report.append("- Consider manual review process for similar patterns")
                
                report.append("")
        
        return report
    
    def _analyze_technique_specific_errors(self) -> List[str]:
        """Analyze each technique's specific failure patterns"""
        report = []
        report.append("## 6. TECHNIQUE-SPECIFIC ERROR ANALYSIS")
        report.append("")
        
        techniques = ["pattern_based", "ml_based", "promptmap", "open_prompt_injection", "ai_model"]
        
        for tech in techniques:
            report.append(f"### 6.{techniques.index(tech)+1} {tech.replace('_', ' ').title()} Error Analysis")
            report.append("")
            
            # Get performance metrics
            perf = self.results["technique_performance"][tech]
            
            report.append("**Performance Metrics**:")
            report.append(f"- Accuracy: {perf['accuracy']:.3f}")
            report.append(f"- Precision: {perf['precision']:.3f}")
            report.append(f"- Recall: {perf['recall']:.3f}")
            report.append(f"- F1-Score: {perf['f1_score']:.3f}")
            report.append("")
            
            # Analyze specific failure patterns
            false_negatives = []
            false_positives = []
            
            for case in self.results["detailed_results"]:
                ground_truth = case["ground_truth"]
                prediction = case["techniques"][tech]["predicted"]
                
                if ground_truth == "injection" and prediction == "benign":
                    false_negatives.append(case)
                elif ground_truth == "benign" and prediction == "injection":
                    false_positives.append(case)
            
            # Analyze false negatives for this technique
            if false_negatives:
                report.append(f"**False Negatives ({len(false_negatives)} cases)**:")
                
                # Group by attack type
                fn_by_attack = {}
                for case in false_negatives:
                    attack_type = case.get("attack_type", "unknown")
                    if attack_type not in fn_by_attack:
                        fn_by_attack[attack_type] = []
                    fn_by_attack[attack_type].append(case)
                
                for attack_type, cases in fn_by_attack.items():
                    report.append(f"- {attack_type}: {len(cases)} missed")
                
                # Provide technique-specific analysis
                report.append("")
                report.append("**Why This Technique Fails**:")
                
                if tech == "pattern_based":
                    report.append("- Limited to predefined patterns")
                    report.append("- Cannot handle novel attack variations")
                    report.append("- Struggles with context-dependent attacks")
                    report.append("- Misses attacks using synonyms or paraphrasing")
                
                elif tech == "ml_based":
                    report.append("- Currently using simulated results (needs real training)")
                    report.append("- May overfit to training data patterns")
                    report.append("- Requires large labeled datasets for good performance")
                    report.append("- Can be vulnerable to adversarial examples")
                
                elif tech == "promptmap":
                    report.append("- Rule-based approach with finite pattern coverage")
                    report.append("- May not cover domain-specific attacks")
                    report.append("- Rules need regular updates for new attack types")
                    report.append("- Struggles with subtle/implicit injections")
                
                elif tech == "open_prompt_injection":
                    report.append("- DataSentinel approach may miss indirect attacks")
                    report.append("- Canary token method has specific limitations")
                    report.append("- May not work well with domain-specific content")
                    report.append("- Requires fine-tuned model for best performance")
                
                elif tech == "ai_model":
                    report.append("- Simulated performance may not reflect real capabilities")
                    report.append("- Context window limitations")
                    report.append("- May be inconsistent across different prompt styles")
                    report.append("- Requires careful prompt engineering for detection")
            
            else:
                report.append("**False Negatives**: None! Perfect recall.")
            
            # Analyze false positives for this technique
            if false_positives:
                report.append("")
                report.append(f"**False Positives ({len(false_positives)} cases)**:")
                report.append("- This technique flagged legitimate content as malicious")
                
                # Show examples
                for case in false_positives[:2]:  # Show up to 2 examples
                    original_case = next((d for d in self.dataset if d["id"] == case["id"]), None)
                    if original_case:
                        text_preview = original_case["text"][:100] + "..." if len(original_case["text"]) > 100 else original_case["text"]
                        report.append(f"  - Case {case['id']}: \"{text_preview}\"")
                
            else:
                report.append("")
                report.append("**False Positives**: None! Perfect precision.")
            
            # Recommendations for this technique
            report.append("")
            report.append("**Improvement Recommendations**:")
            
            if tech == "pattern_based":
                if false_negatives:
                    report.append("- Add patterns from missed cases to detection rules")
                    report.append("- Implement fuzzy matching for pattern variations")
                    report.append("- Add context-aware pattern matching")
                report.append("- Regular pattern database updates")
            
            elif tech == "ml_based":
                report.append("- Train actual models with embedding features")
                report.append("- Use large-scale labeled datasets for training")
                report.append("- Implement ensemble methods for robustness")
                report.append("- Regular retraining with new attack examples")
            
            elif tech == "promptmap":
                if false_negatives:
                    report.append("- Add new rules for missed attack patterns")
                    report.append("- Implement domain-specific rule sets")
                report.append("- Regular rule updates from community")
                report.append("- Add semantic similarity matching")
            
            elif tech == "open_prompt_injection":
                report.append("- Fine-tune DataSentinel model on domain data")
                report.append("- Experiment with different canary token strategies")
                report.append("- Combine with other detection approaches")
                report.append("- Regular model updates from OPI research")
            
            elif tech == "ai_model":
                report.append("- Use actual API calls instead of simulation")
                report.append("- Optimize detection prompts through testing")
                report.append("- Implement confidence calibration")
                report.append("- Add multiple model consensus")
            
            report.append("")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate overall recommendations based on analysis"""
        report = []
        report.append("## 7. STRATEGIC RECOMMENDATIONS")
        report.append("")
        
        # Calculate overall system performance
        total_cases = len(self.results["detailed_results"])
        consensus_correct = 0
        
        for case in self.results["detailed_results"]:
            ground_truth = case["ground_truth"]
            predictions = [case["techniques"][tech]["predicted"] for tech in ["pattern_based", "ml_based", "promptmap", "open_prompt_injection", "ai_model"]]
            
            # Majority vote
            injection_votes = sum(1 for pred in predictions if pred == "injection")
            consensus_prediction = "injection" if injection_votes >= 3 else "benign"
            
            if consensus_prediction == ground_truth:
                consensus_correct += 1
        
        system_accuracy = consensus_correct / total_cases
        
        report.append(f"### 7.1 Overall System Performance")
        report.append("")
        report.append(f"**Consensus System Accuracy**: {system_accuracy:.3f} ({consensus_correct}/{total_cases})")
        report.append("")
        
        if system_accuracy >= 0.90:
            report.append("🎉 **Excellent Performance**: The multi-technique approach is working very well.")
        elif system_accuracy >= 0.80:
            report.append("✅ **Good Performance**: The system is effective but has room for improvement.")
        elif system_accuracy >= 0.70:
            report.append("⚠️  **Moderate Performance**: Significant improvements needed for production use.")
        else:
            report.append("❌ **Poor Performance**: Major overhaul required before deployment.")
        
        report.append("")
        
        # Immediate priority recommendations
        report.append("### 7.2 Immediate Priority Actions")
        report.append("")
        
        # Find the biggest issues
        techniques = ["pattern_based", "ml_based", "promptmap", "open_prompt_injection", "ai_model"]
        worst_performer = min(techniques, key=lambda t: self.results["technique_performance"][t]["f1_score"])
        best_performer = max(techniques, key=lambda t: self.results["technique_performance"][t]["f1_score"])
        
        report.append("**Priority 1 - Address Weakest Technique**:")
        report.append(f"- {worst_performer.replace('_', ' ').title()} has the lowest F1-score ({self.results['technique_performance'][worst_performer]['f1_score']:.3f})")
        
        if worst_performer == "ml_based":
            report.append("- Implement actual ML model training with embeddings")
            report.append("- Use the malicious-prompt-detection repository's trained models")
        else:
            report.append(f"- Focus on improving {worst_performer} detection capabilities")
        
        report.append("")
        report.append("**Priority 2 - Leverage Best Performer**:")
        report.append(f"- {best_performer.replace('_', ' ').title()} shows best performance (F1: {self.results['technique_performance'][best_performer]['f1_score']:.3f})")
        report.append(f"- Consider weighting this technique higher in consensus decisions")
        report.append("")
        
        # Attack-specific recommendations
        report.append("### 7.3 Attack-Specific Improvements")
        report.append("")
        
        # Find most problematic attack types
        attack_analysis = self.results.get("attack_type_analysis", {})
        if attack_analysis:
            problem_attacks = []
            for attack_type, tech_results in attack_analysis.items():
                if tech_results:  # Ensure not empty
                    avg_detection = sum(tech_results.values()) / len(tech_results)
                    if avg_detection < 0.7:  # Less than 70% average detection
                        problem_attacks.append((attack_type, avg_detection))
            
            if problem_attacks:
                problem_attacks.sort(key=lambda x: x[1])  # Sort by detection rate
                
                report.append("**Most Problematic Attack Types**:")
                for attack_type, detection_rate in problem_attacks[:3]:
                    report.append(f"- {attack_type}: {detection_rate:.1%} average detection")
                    
                    if "escape" in attack_type:
                        report.append("  - Add escape sequence normalization preprocessing")
                        report.append("  - Update pattern matching to handle encoded characters")
                    elif "naive" in attack_type:
                        report.append("  - Improve semantic analysis capabilities")
                        report.append("  - Add context-aware detection methods")
                    elif "system_prompt" in attack_type:
                        report.append("  - Develop domain-specific detection rules")
                        report.append("  - Add application-aware context analysis")
                
                report.append("")
        
        # Long-term strategic recommendations
        report.append("### 7.4 Long-term Strategy")
        report.append("")
        report.append("**Model Enhancement**:")
        report.append("- Implement active learning to continuously improve from new samples")
        report.append("- Develop ensemble methods combining all techniques intelligently")
        report.append("- Regular retraining with production data")
        report.append("")
        
        report.append("**System Architecture**:")
        report.append("- Implement dynamic weight adjustment based on technique confidence")
        report.append("- Add real-time threat intelligence integration")
        report.append("- Develop automated model update pipelines")
        report.append("")
        
        report.append("**Monitoring and Evaluation**:")
        report.append("- Implement continuous performance monitoring")
        report.append("- Add A/B testing framework for technique improvements")
        report.append("- Regular red team testing with novel attack patterns")
        report.append("")
        
        return report
    
    def generate_performance_charts(self):
        """Generate visualization charts"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create performance comparison chart
            techniques = ["pattern_based", "ml_based", "promptmap", "open_prompt_injection", "ai_model"]
            metrics = ["accuracy", "precision", "recall", "f1_score"]
            
            # Extract performance data
            perf_data = []
            for tech in techniques:
                tech_perf = self.results["technique_performance"][tech]
                for metric in metrics:
                    perf_data.append({
                        "Technique": tech.replace('_', ' ').title(),
                        "Metric": metric.replace('_', ' ').title(),
                        "Score": tech_perf[metric]
                    })
            
            df = pd.DataFrame(perf_data)
            
            # Create grouped bar chart
            plt.figure(figsize=(12, 8))
            sns.barplot(data=df, x="Technique", y="Score", hue="Metric")
            plt.title("Detection Technique Performance Comparison", fontsize=14, fontweight='bold')
            plt.ylabel("Score", fontsize=12)
            plt.xlabel("Detection Technique", fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig("technique_performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Performance chart saved: technique_performance_comparison.png")
            
        except ImportError:
            print("⚠️  matplotlib/seaborn not available. Skipping chart generation.")
    
    def save_detailed_report(self):
        """Save the detailed analysis report"""
        report_content = self.create_detailed_case_analysis()
        
        with open("detailed_analysis_report.md", "w", encoding='utf-8') as f:
            f.write(report_content)
        
        # Also generate charts
        self.generate_performance_charts()
        
        print("💾 Detailed analysis report saved:")
        print("   - detailed_analysis_report.md")
        if hasattr(plt, 'figure'):
            print("   - technique_performance_comparison.png")


def main():
    """Generate detailed analysis report"""
    print("🔍 Generating Detailed Case-by-Case Analysis Report")
    print("=" * 60)
    
    reporter = DetailedAnalysisReporter()
    
    if not reporter.results or not reporter.dataset:
        print("❌ Required data files not found. Please run:")
        print("   1. python opi_dataset_generator.py")
        print("   2. python opi_realistic_comparison.py")
        return
    
    # Generate and save the report
    reporter.save_detailed_report()
    
    # Display summary
    print(f"\n📊 Analysis Summary:")
    print(f"   Total cases analyzed: {reporter.results['dataset_info']['total_samples']}")
    print(f"   Techniques evaluated: 5")
    print(f"   Attack types covered: {len(set(case.get('attack_type') for case in reporter.results['detailed_results'] if case.get('attack_type')))}")
    
    print(f"\n✅ Detailed analysis complete!")


if __name__ == "__main__":
    main()