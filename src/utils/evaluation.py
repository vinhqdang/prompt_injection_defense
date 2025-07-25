import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.comparison_data = []
    
    def add_result(self, model_name, result_dict):
        """Add evaluation result for a model"""
        self.results[model_name] = result_dict
        
        # Add to comparison data
        comparison_entry = {
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': result_dict['accuracy'],
            'Precision': result_dict['precision'],
            'Recall': result_dict['recall'],
            'F1-Score': result_dict['f1_score'],
            'Inference Time (s)': result_dict.get('inference_time', 0)
        }
        self.comparison_data.append(comparison_entry)
    
    def plot_confusion_matrices(self, y_test, save_path='results/confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        if n_models == 0:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 10))
        if n_models == 1:
            axes = [axes]
        elif n_models <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(self.results.items()):
            if 'predictions' in result:
                cm = confusion_matrix(y_test, result['predictions'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           ax=axes[i], cbar=True)
                axes[i].set_title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrices saved to {save_path}")
    
    def plot_roc_curves(self, y_test, save_path='results/roc_curves.png'):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            if 'probabilities' in result:
                try:
                    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, linewidth=2, 
                            label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
                except Exception as e:
                    print(f"Could not plot ROC for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ROC curves saved to {save_path}")
    
    def plot_performance_comparison(self, save_path='results/performance_comparison.png'):
        """Plot performance comparison bar chart"""
        if not self.comparison_data:
            print("No comparison data available")
            return
        
        df = pd.DataFrame(self.comparison_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for metric, pos in zip(metrics, positions):
            ax = axes[pos]
            bars = ax.bar(df['Model'], df[metric], alpha=0.7)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Performance comparison saved to {save_path}")
    
    def plot_inference_time_comparison(self, save_path='results/inference_time_comparison.png'):
        """Plot inference time comparison"""
        if not self.comparison_data:
            print("No comparison data available")
            return
        
        df = pd.DataFrame(self.comparison_data)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df['Model'], df['Inference Time (s)'], alpha=0.7, color='skyblue')
        plt.title('Inference Time Comparison')
        plt.ylabel('Inference Time (seconds)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, df['Inference Time (s)']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Inference time comparison saved to {save_path}")
    
    def create_interactive_dashboard(self, save_path='results/interactive_dashboard.html'):
        """Create interactive dashboard using Plotly"""
        if not self.comparison_data:
            print("No comparison data available")
            return
        
        df = pd.DataFrame(self.comparison_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'Precision vs Recall', 
                          'F1-Score Comparison', 'Inference Time'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Accuracy'], name='Accuracy',
                   text=[f'{x:.3f}' for x in df['Accuracy']],
                   textposition='auto'),
            row=1, col=1
        )
        
        # Precision vs Recall scatter
        fig.add_trace(
            go.Scatter(x=df['Precision'], y=df['Recall'], 
                      mode='markers+text',
                      text=df['Model'],
                      textposition='top center',
                      marker=dict(size=10),
                      name='Precision vs Recall'),
            row=1, col=2
        )
        
        # F1-Score comparison
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['F1-Score'], name='F1-Score',
                   text=[f'{x:.3f}' for x in df['F1-Score']],
                   textposition='auto'),
            row=2, col=1
        )
        
        # Inference time
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Inference Time (s)'], name='Inference Time',
                   text=[f'{x:.4f}s' for x in df['Inference Time (s)']],
                   textposition='auto'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Model Performance Dashboard',
            height=800,
            showlegend=False
        )
        
        # Update x-axis labels
        for i in range(1, 3):
            for j in range(1, 3):
                if not (i == 1 and j == 2):  # Skip scatter plot
                    fig.update_xaxes(tickangle=45, row=i, col=j)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def generate_detailed_report(self, y_test, save_path='results/detailed_report.txt'):
        """Generate detailed evaluation report"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HOMOMORPHIC ENCRYPTION INTRUSION DETECTION SYSTEM\n")
            f.write("DETAILED EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Set Size: {len(y_test)} samples\n")
            f.write(f"Attack Distribution: {np.sum(y_test)} attacks, {len(y_test) - np.sum(y_test)} benign\n\n")
            
            # Model comparison table
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-" * 80 + "\n")
            
            df = pd.DataFrame(self.comparison_data)
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Detailed results for each model
            for model_name, result in self.results.items():
                f.write(f"\n{model_name.upper().replace('_', ' ')} - DETAILED RESULTS\n")
                f.write("-" * 50 + "\n")
                
                if 'predictions' in result:
                    y_pred = result['predictions']
                    
                    # Classification report
                    f.write("Classification Report:\n")
                    f.write(classification_report(y_test, y_pred, 
                                                target_names=['Benign', 'Attack']))
                    f.write("\n")
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    f.write("Confusion Matrix:\n")
                    f.write(f"True Negatives: {cm[0,0]}\n")
                    f.write(f"False Positives: {cm[0,1]}\n")
                    f.write(f"False Negatives: {cm[1,0]}\n")
                    f.write(f"True Positives: {cm[1,1]}\n\n")
                    
                    # Additional metrics
                    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
                    f.write(f"Specificity (True Negative Rate): {specificity:.4f}\n")
                    
                    false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
                    f.write(f"False Positive Rate: {false_positive_rate:.4f}\n")
                    
                    false_negative_rate = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
                    f.write(f"False Negative Rate: {false_negative_rate:.4f}\n\n")
        
        print(f"Detailed report saved to {save_path}")
    
    def get_summary_table(self):
        """Get summary comparison table"""
        if not self.comparison_data:
            return None
        
        df = pd.DataFrame(self.comparison_data)
        return df
    
    def plot_all_visualizations(self, y_test, save_dir='results'):
        """Generate all visualization plots"""
        print("Generating all visualizations...")
        
        self.plot_confusion_matrices(y_test, f'{save_dir}/confusion_matrices.png')
        self.plot_roc_curves(y_test, f'{save_dir}/roc_curves.png')
        self.plot_performance_comparison(f'{save_dir}/performance_comparison.png')
        self.plot_inference_time_comparison(f'{save_dir}/inference_time_comparison.png')
        self.create_interactive_dashboard(f'{save_dir}/interactive_dashboard.html')
        self.generate_detailed_report(y_test, f'{save_dir}/detailed_report.txt')
        
        print(f"All visualizations saved to {save_dir}/")
    
    def save_results(self, save_path='results/evaluation_results.pkl'):
        """Save evaluation results"""
        import pickle
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            'results': self.results,
            'comparison_data': self.comparison_data
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Evaluation results saved to {save_path}")
    
    def load_results(self, save_path='results/evaluation_results.pkl'):
        """Load evaluation results"""
        import pickle
        
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        
        self.results = data['results']
        self.comparison_data = data['comparison_data']
        
        print(f"Evaluation results loaded from {save_path}")