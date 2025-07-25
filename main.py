#!/usr/bin/env python3
"""
Homomorphic Encryption for Intrusion Detection System
Main execution script for training and evaluating models
"""

import sys
import os
import numpy as np
import pandas as pd
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.cic_ids_loader import CICIDSLoader
from models.baseline_models import BaselineModels
from homomorphic.he_ids import HomomorphicIDS
from utils.evaluation import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description='Homomorphic Encryption IDS')
    parser.add_argument('--data-path', default='data/raw', 
                       help='Path to raw CIC-IDS2017 dataset')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline model training')
    parser.add_argument('--skip-homomorphic', action='store_true',
                       help='Skip homomorphic model training')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for testing (use subset of data)')
    args = parser.parse_args()

    print("=" * 80)
    print("HOMOMORPHIC ENCRYPTION FOR INTRUSION DETECTION SYSTEM")
    print("=" * 80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize components
    loader = CICIDSLoader(data_path=args.data_path)
    evaluator = ModelEvaluator()
    
    # Step 1: Load and preprocess data
    print("\n" + "="*50)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*50)
    
    try:
        if os.path.exists('data/processed/X_train.npy'):
            print("Loading preprocessed data...")
            X_train, X_test, y_train, y_test = loader.load_processed_data()
        else:
            print("Loading raw data...")
            df = loader.load_data()
            X_train, X_test, y_train, y_test = loader.preprocess_data(
                df, test_size=args.test_size
            )
            loader.save_processed_data(X_train, X_test, y_train, y_test)
            
    except FileNotFoundError:
        print("Dataset not found. Please download the CIC-IDS2017 dataset.")
        loader.download_dataset()
        return
    
    # Sample data if specified
    if args.sample_size and args.sample_size < len(X_test):
        print(f"Sampling {args.sample_size} samples for testing...")
        indices = np.random.choice(len(X_test), args.sample_size, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    print(f"Final dataset shapes:")
    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
    print(f"Training labels: {np.bincount(y_train)}")
    print(f"Test labels: {np.bincount(y_test)}")
    
    # Step 2: Baseline Models
    if not args.skip_baseline:
        print("\n" + "="*50)
        print("STEP 2: BASELINE MODELS TRAINING")
        print("="*50)
        
        baseline_models = BaselineModels()
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        rf_model = baseline_models.train_random_forest(X_train, y_train)
        rf_results = baseline_models.evaluate_model('random_forest', X_test, y_test)
        evaluator.add_result('random_forest', rf_results)
        
        # Train XGBoost
        print("\nTraining XGBoost...")
        xgb_model = baseline_models.train_xgboost(X_train, y_train, X_test, y_test)
        xgb_results = baseline_models.evaluate_model('xgboost', X_test, y_test)
        evaluator.add_result('xgboost', xgb_results)
        
        # Train Deep Learning
        print("\nTraining Deep Learning model...")
        dl_model = baseline_models.train_deep_learning(
            X_train, y_train, X_test, y_test, epochs=50
        )
        dl_results = baseline_models.evaluate_model('deep_learning', X_test, y_test)
        evaluator.add_result('deep_learning', dl_results)
        
        # Save baseline models
        baseline_models.save_models()
        
        print("\nBaseline Models Summary:")
        summary_table = baseline_models.get_comparison_table()
        print(summary_table.to_string(index=False))
    
    # Step 3: Homomorphic Encryption IDS
    if not args.skip_homomorphic:
        print("\n" + "="*50)
        print("STEP 3: HOMOMORPHIC ENCRYPTION IDS")
        print("="*50)
        
        # Use smaller sample for homomorphic evaluation due to computational cost
        he_sample_size = min(1000, len(X_test))
        he_indices = np.random.choice(len(X_test), he_sample_size, replace=False)
        X_test_he = X_test[he_indices]
        y_test_he = y_test[he_indices]
        
        print(f"Using {he_sample_size} samples for homomorphic evaluation")
        
        # Initialize Homomorphic IDS
        he_ids = HomomorphicIDS()
        
        # Train linear model
        print("\nTraining linear model for homomorphic evaluation...")
        he_ids.train_linear_model(X_train, y_train, epochs=100)
        
        # Encrypt model
        print("\nEncrypting model weights...")
        he_ids.encrypt_model()
        
        # Evaluate homomorphic model (simple method)
        print("\nEvaluating homomorphic model (simple threshold)...")
        he_simple_results = he_ids.evaluate_homomorphic_model(
            X_test_he, y_test_he, method='simple'
        )
        evaluator.add_result('homomorphic_simple', he_simple_results)
        
        # Evaluate homomorphic model (polynomial approximation)
        print("\nEvaluating homomorphic model (polynomial approximation)...")
        he_poly_results = he_ids.evaluate_homomorphic_model(
            X_test_he, y_test_he, method='polynomial'
        )
        evaluator.add_result('homomorphic_polynomial', he_poly_results)
        
        # Compare with plaintext
        print("\nComparing homomorphic vs plaintext computation...")
        comparison = he_ids.compare_with_plaintext(X_test_he, y_test_he)
        print("Comparison Results:")
        for key, value in comparison.items():
            print(f"{key}: {value:.4f}")
        
        # Save homomorphic model
        he_ids.save_context()
    
    # Step 4: Comprehensive Evaluation
    print("\n" + "="*50)
    print("STEP 4: COMPREHENSIVE EVALUATION")
    print("="*50)
    
    # Generate all visualizations and reports
    test_labels = y_test_he if not args.skip_homomorphic else y_test
    evaluator.plot_all_visualizations(test_labels)
    
    # Print final summary
    print("\nFINAL PERFORMANCE SUMMARY:")
    print("="*50)
    summary_df = evaluator.get_summary_table()
    if summary_df is not None:
        print(summary_df.to_string(index=False))
    
    # Save evaluation results
    evaluator.save_results()
    
    print(f"\nExperiment completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Results and visualizations saved to 'results/' directory")
    print("Models saved to 'models/saved/' directory")

if __name__ == "__main__":
    main()