import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import warnings
warnings.filterwarnings('ignore')

class CICIDSLoader:
    def __init__(self, data_path='data/raw'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.features = None
        self.labels = None
        self.feature_names = None
        
    def download_dataset(self):
        """Download CIC-IDS2017 dataset"""
        print("Please download the CIC-IDS2017 dataset from:")
        print("https://www.unb.ca/cic/datasets/ids-2017.html")
        print("Extract CSV files to:", os.path.abspath(self.data_path))
        
    def load_data(self, file_pattern=None):
        """Load CIC-IDS2017 dataset from CSV files"""
        csv_files = [
            'Monday-WorkingHours.pcap_ISCX.csv',
            'Tuesday-WorkingHours.pcap_ISCX.csv', 
            'Wednesday-workingHours.pcap_ISCX.csv',
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
        ]
        
        dataframes = []
        
        # Check both locations: direct data_path and MachineLearningCVE subdirectory
        search_paths = [
            self.data_path,
            os.path.join(self.data_path, 'MachineLearningCVE')
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                files_found = 0
                for file in csv_files:
                    file_path = os.path.join(search_path, file)
                    if os.path.exists(file_path):
                        print(f"Loading {file}...")
                        df = pd.read_csv(file_path)
                        dataframes.append(df)
                        files_found += 1
                        # Load at least 3 files to ensure we get attacks
                        if files_found >= 3:
                            break
                if dataframes:
                    break  # Found files in this search path
        
        if not dataframes:
            # Try to find any CSV files in the data directory
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.csv') and any(csv_file in file for csv_file in csv_files):
                        print(f"Found CSV file: {file}")
                        file_path = os.path.join(root, file)
                        df = pd.read_csv(file_path)
                        dataframes.append(df)
            
        if not dataframes:
            raise FileNotFoundError("No CSV files found. Please download the dataset first.")
            
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined dataset shape: {combined_df.shape}")
        
        return combined_df
    
    def preprocess_data(self, df, test_size=0.2, balance_data=True):
        """Preprocess the dataset"""
        print("Preprocessing data...")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Handle label column
        label_col = 'Label'
        if label_col not in df.columns:
            label_col = ' Label'  # Some files have space before Label
        
        # Remove rows with missing labels
        df = df.dropna(subset=[label_col])
        
        # Separate features and labels
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Clean feature data
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode labels (BENIGN -> 0, attacks -> 1)
        y_binary = (y != 'BENIGN').astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Balance training data using SMOTE
        if balance_data:
            print("Balancing training data with SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        print(f"Training label distribution: {np.bincount(y_train)}")
        print(f"Test label distribution: {np.bincount(y_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        os.makedirs('data/processed', exist_ok=True)
        
        np.save('data/processed/X_train.npy', X_train)
        np.save('data/processed/X_test.npy', X_test)
        np.save('data/processed/y_train.npy', y_train)
        np.save('data/processed/y_test.npy', y_test)
        
        # Save feature names
        with open('data/processed/feature_names.txt', 'w') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        
        print("Processed data saved to data/processed/")
    
    def load_processed_data(self):
        """Load preprocessed data"""
        X_train = np.load('data/processed/X_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
        
        with open('data/processed/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    loader = CICIDSLoader()
    
    # Check if processed data exists
    if os.path.exists('data/processed/X_train.npy'):
        print("Loading preprocessed data...")
        X_train, X_test, y_train, y_test = loader.load_processed_data()
    else:
        # Load and preprocess raw data
        try:
            df = loader.load_data()
            X_train, X_test, y_train, y_test = loader.preprocess_data(df)
            loader.save_processed_data(X_train, X_test, y_train, y_test)
        except FileNotFoundError:
            loader.download_dataset()