# CIC-IDS2017 Dataset Setup Instructions

Due to GitHub's file size limitations, the CIC-IDS2017 dataset files are not included in this repository. Follow these instructions to set up the dataset for reproducible experiments.

## Automatic Download (Recommended)

Run the provided download script:

```bash
conda activate homomorphic-ids
python download_dataset.py
```

This will automatically:
- Download the MachineLearningCSV.zip file (224MB)
- Extract all 8 CSV files to `data/raw/MachineLearningCVE/`
- Verify the dataset integrity
- Display dataset statistics

## Manual Download

If the automatic download fails, manually download the dataset:

1. **Download the dataset zip file:**
   ```
   URL: http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip
   Size: ~224MB
   ```

2. **Extract to the correct location:**
   ```bash
   mkdir -p data/raw
   # Extract MachineLearningCSV.zip to data/raw/
   # This creates data/raw/MachineLearningCVE/ with 8 CSV files
   ```

## Expected Dataset Structure

After successful setup, you should have:

```
data/raw/MachineLearningCVE/
├── Monday-WorkingHours.pcap_ISCX.csv                    (168.73 MB, 529,918 samples)
├── Tuesday-WorkingHours.pcap_ISCX.csv                   (128.82 MB, 445,909 samples)
├── Wednesday-workingHours.pcap_ISCX.csv                 (214.74 MB, 692,703 samples)
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv   (34.86 MB, 170,366 samples)
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv (79.25 MB, 288,566 samples)
├── Friday-WorkingHours-Morning.pcap_ISCX.csv            (55.62 MB, 191,033 samples)
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv     (73.55 MB, 225,745 samples)
└── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv (73.34 MB, 289,191 samples)
```

## Dataset Information

- **Total Samples:** ~2.8 million network flow records
- **Features:** 79 network flow characteristics
- **Labels:** BENIGN and various attack types
  - DoS attacks (Hulk, GoldenEye, slowloris, Slowhttptest)
  - DDoS attacks
  - PortScan
  - Web Attacks (Brute Force, XSS, SQL Injection)
  - Infiltration
  - Heartbleed

## Verification

Verify your setup by running:

```bash
conda activate homomorphic-ids
python -c "
import pandas as pd
import os

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

base_path = 'data/raw/MachineLearningCVE'
total_samples = 0

for csv_file in csv_files:
    filepath = os.path.join(base_path, csv_file)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        total_samples += len(df)
        print(f'✓ {csv_file}: {len(df):,} samples')
    else:
        print(f'✗ Missing: {csv_file}')

print(f'\nTotal samples: {total_samples:,}')
print('Dataset setup completed successfully!' if total_samples > 2000000 else 'Dataset setup incomplete.')
"
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{sharafaldin2018toward,
  title={Toward generating a new intrusion detection dataset and intrusion traffic characterization},
  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A},
  booktitle={4th International Conference on Information Systems Security and Privacy (ICISSP)},
  pages={108--116},
  year={2018}
}
```

## Troubleshooting

**Issue:** Download timeout or connection errors
- **Solution:** The server may be slow. Try multiple times or download manually from a browser.

**Issue:** Zip extraction fails  
- **Solution:** The download may be incomplete. Delete the zip file and re-download.

**Issue:** CSV files are missing or corrupted
- **Solution:** Re-extract the zip file or re-download the dataset.

**Issue:** Permission errors on Windows/WSL
- **Solution:** Ensure you have write permissions in the project directory.

For additional help, check the dataset provider's documentation at: https://www.unb.ca/cic/datasets/ids-2017.html