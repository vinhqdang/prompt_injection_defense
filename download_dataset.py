#!/usr/bin/env python3
"""
Download and extract CIC-IDS2017 dataset
"""

import os
import requests
import zipfile
from io import BytesIO
import pandas as pd

def download_file(url, filename):
    """Download file with progress"""
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size})", end='', flush=True)
    
    print(f"\nDownloaded: {filename}")
    return filename

def extract_and_examine_dataset():
    """Extract and examine the CIC-IDS2017 dataset"""
    
    # Create directory
    os.makedirs('data/raw', exist_ok=True)
    
    # URLs for CIC-IDS2017 dataset
    urls = {
        'MachineLearningCSV.zip': 'http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip',
        'GeneratedLabelledFlows.zip': 'http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip'
    }
    
    # Try to use existing files first
    for filename in urls.keys():
        filepath = f'data/raw/{filename}'
        if os.path.exists(filepath):
            print(f"Found existing file: {filepath}")
            
            # Try to extract and examine
            try:
                print(f"Attempting to extract {filename}...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    print(f"Files in {filename}:")
                    for file in file_list[:10]:  # Show first 10 files
                        print(f"  - {file}")
                    
                    if len(file_list) > 10:
                        print(f"  ... and {len(file_list) - 10} more files")
                    
                    # Extract files
                    zip_ref.extractall('data/raw')
                    print(f"Extracted {len(file_list)} files from {filename}")
                    
                    # If this is the MachineLearningCSV, examine the first CSV
                    if 'MachineLearningCSV' in filename:
                        csv_files = [f for f in file_list if f.endswith('.csv')]
                        if csv_files:
                            first_csv = csv_files[0]
                            print(f"\nExamining first CSV file: {first_csv}")
                            
                            df = pd.read_csv(f'data/raw/{first_csv}', nrows=5)
                            print(f"Shape: {df.shape}")
                            print(f"Columns: {list(df.columns)}")
                            print("\nFirst few rows:")
                            print(df.head())
                            
                            # Check label distribution
                            full_df = pd.read_csv(f'data/raw/{first_csv}')
                            print(f"\nFull dataset shape: {full_df.shape}")
                            if 'Label' in full_df.columns:
                                print(f"Label distribution:")
                                print(full_df['Label'].value_counts())
                            elif ' Label' in full_df.columns:
                                print(f"Label distribution:")
                                print(full_df[' Label'].value_counts())
                    
                    return True
                    
            except zipfile.BadZipFile:
                print(f"Error: {filename} appears to be corrupted or incomplete")
                continue
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
                continue
    
    return False

if __name__ == "__main__":
    success = extract_and_examine_dataset()
    if success:
        print("\nDataset extraction completed successfully!")
    else:
        print("\nFailed to extract dataset. Please check the files.")