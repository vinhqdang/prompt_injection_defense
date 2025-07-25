#!/usr/bin/env python3
"""
Generate synthetic network traffic data similar to CIC-IDS2017 for demonstration
"""

import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def generate_network_features(n_samples, n_features=78, random_state=42):
    """Generate synthetic network traffic features"""
    
    # Create base classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.1),
        n_clusters_per_class=3,
        class_sep=0.8,
        random_state=random_state
    )
    
    # Make features non-negative and scale appropriately for network data
    X = np.abs(X)
    
    # Scale different feature groups to realistic ranges
    feature_groups = {
        'flow_duration': (0, 1),      # Duration features (0-10 indices)
        'packet_counts': (10, 20),    # Packet count features
        'byte_counts': (20, 30),      # Byte count features  
        'packet_lengths': (30, 40),   # Packet length features
        'flow_rates': (40, 50),       # Flow rate features
        'timing_features': (50, 60),  # Inter-arrival time features
        'tcp_flags': (60, 70),        # TCP flag features
        'protocol_features': (70, 78) # Protocol-specific features
    }
    
    # Apply different scaling to different feature groups
    for group, (start, end) in feature_groups.items():
        if group == 'flow_duration':
            X[:, start:end] *= np.random.uniform(0.001, 120, size=(n_samples, end-start))
        elif group == 'packet_counts':
            X[:, start:end] = np.random.poisson(50, size=(n_samples, end-start)) * X[:, start:end]
        elif group == 'byte_counts':
            X[:, start:end] *= np.random.uniform(64, 65535, size=(n_samples, end-start))
        elif group == 'packet_lengths':
            X[:, start:end] = np.random.normal(800, 200, size=(n_samples, end-start)) * np.abs(X[:, start:end])
        elif group == 'flow_rates':
            X[:, start:end] *= np.random.uniform(0.1, 1000, size=(n_samples, end-start))
        elif group == 'timing_features':
            X[:, start:end] *= np.random.uniform(0.001, 10, size=(n_samples, end-start))
        elif group == 'tcp_flags':
            X[:, start:end] = np.random.randint(0, 2, size=(n_samples, end-start))
        else:  # protocol_features
            X[:, start:end] *= np.random.uniform(0, 100, size=(n_samples, end-start))
    
    return X, y

def create_feature_names(n_features=78):
    """Create realistic feature names similar to CIC-IDS2017"""
    base_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
        'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
        'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
        'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
        'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
        'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
        'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
        'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
        'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
        'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
        'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ]
    
    # Extend or truncate to match n_features
    if len(base_features) < n_features:
        for i in range(len(base_features), n_features):
            base_features.append(f'Feature_{i+1}')
    elif len(base_features) > n_features:
        base_features = base_features[:n_features]
    
    return base_features

def generate_synthetic_dataset(n_samples=10000, output_file='data/synthetic_ids_data.csv'):
    """Generate and save synthetic IDS dataset"""
    
    print(f"Generating synthetic network traffic dataset with {n_samples} samples...")
    
    # Generate features and labels
    X, y = generate_network_features(n_samples)
    
    # Create feature names
    feature_names = create_feature_names(X.shape[1])
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add labels (convert to BENIGN/ATTACK format)
    df['Label'] = ['BENIGN' if label == 0 else 'ATTACK' for label in y]
    
    # Add some variety to attack types
    attack_indices = df[df['Label'] == 'ATTACK'].index
    attack_types = ['DoS', 'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack', 'Brute Force']
    
    for i, idx in enumerate(attack_indices):
        if np.random.random() < 0.3:  # 30% chance to specify attack type
            df.loc[idx, 'Label'] = np.random.choice(attack_types)
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Synthetic dataset saved to: {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:")
    print(df['Label'].value_counts())
    
    return df

if __name__ == "__main__":
    # Generate synthetic dataset
    df = generate_synthetic_dataset(n_samples=10000, output_file='data/raw/synthetic_ids_data.csv')
    
    # Also create a smaller version for quick testing
    df_small = df.sample(n=2000, random_state=42)
    df_small.to_csv('data/raw/synthetic_ids_small.csv', index=False)
    
    print("\nGenerated datasets:")
    print("- data/raw/synthetic_ids_data.csv (10,000 samples)")
    print("- data/raw/synthetic_ids_small.csv (2,000 samples)")